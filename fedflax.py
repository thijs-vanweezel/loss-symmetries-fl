import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
os.environ["XLA_FLAGS"] += " --xla_gpu_strict_conv_algorithm_picker=false"
import jax, numpy as np
from functools import reduce
from npy_append_array import NpyAppendArray
from jax import numpy as jnp
from flax import nnx
from functools import partial
from tqdm.auto import tqdm
from utils import save_model, load_model

def save(models, filename, n, overwrite):
        with NpyAppendArray(filename, delete_if_exists=overwrite) as f:
            f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.split(models, (nnx.Param, nnx.BatchStat), ...)[1])], axis=1))

# Parallelized train step
def return_train_step(ell, n_inputs, vectorize=True):
    def train_step(model, model_g, opt, y, *xs):
        loss, grad = nnx.value_and_grad(ell)(model, model_g, y, *xs)
        opt.update(grad)
        return loss
    if vectorize:
        train_step = nnx.vmap(train_step, in_axes=(0,None,0,0)+(0,)*n_inputs)
    else:
        train_step = nnx.vmap(train_step, in_axes=(None,None,None,0)+(0,)*n_inputs)
    train_step = nnx.jit(train_step)
    return train_step

# Get updates, i.e., difference between initial model and locally converged models
def get_updates(model_g, models):
    params = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)[1]
    params_g = nnx.split(model_g, (nnx.Param, nnx.BatchStat), ...)[1]
    return jax.tree.map(lambda pg, p: p-pg, params_g, params) # state of length n_layers with arrays of shape (n_clients, *layer_shape)

def aggregate(model_g, updates):
    # Get global model's structure, along with its parameters/rest (which have no client dimension)
    struct, params_g, rest = nnx.split(model_g, (nnx.Param, nnx.BatchStat), ...)
    # Average updates
    update = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
    # Apply to global model
    params_g = jax.tree.map(lambda pg, u: pg + u, params_g, update)
    # Convert to model
    model_g = nnx.merge(struct, params_g, rest)
    return model_g

# Broadcast global model to clients
def cast(module_g, n):
    struct, params_g = nnx.split(module_g, ...)
    params_all = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, 0), params_g)
    models = nnx.merge(struct, params_all)
    return models

# Local training loop
def train_local(models, opts, n_clients, ds_train, ds_val, ell, model_g=None, local_epochs="early", max_patience=None, filename=None, r=0, rounds="N/A", local_val_fn=None, parallel=True):
        local_val_losses = []
        local_patience = 1
        epoch = 0
        while (epoch!=local_epochs) if isinstance(local_epochs, int) else (local_patience<=max_patience):
            # Collect and save params for visualization
            if filename: save(models, filename, n_clients, overwrite=(r==0 and epoch==0))
            # Iterate over batches
            for batch, (y, *xs) in enumerate(bar := tqdm(ds_train, leave=False)):
                # Create train step function if first iteration
                if batch==0 and epoch==0 and r==0:
                    train_step = return_train_step(ell, n_inputs:=len(xs), vectorize=parallel)
                # Train step
                loss = train_step(models, model_g, opts, y, *xs)
                # Inform user
                bar.set_description(f"Round {r}/{rounds}, epoch {epoch}/{local_epochs} (local validation score: {'N/A' if epoch==0 or isinstance(local_epochs, int) else val}, local batch loss: {loss.mean():.4f})")
            # Evaluate on local validation
            if not isinstance(local_epochs, int):
                val = reduce(lambda a, batch: a+local_val_fn(models, *batch).mean(), ds_val, 0.)
                val /= len(ds_val)
                local_val_losses.append(val)
                # Check if local models are converged
                if epoch>=1 and val>=local_val_losses[-local_patience-1]:
                    local_patience += 1
                else:
                    local_patience = 1
            epoch += 1
        return models, n_inputs, epoch

def train(model_g, opt_create, ds_train, ell, ds_val=None, local_epochs:int|str="early", filename:str=None, n_clients=4, 
          max_patience:int=None, rounds:int|str="early", val_fn=None, tmp_file:str=None):
    """
    Federated training loop. 
    Args:
        model_g: Initialized global model
        opt_create: Function that creates optimizer when given a model
        ds_train: Iterable training dataset with signature ( y, *xs ), of which both arguments have shape ( n_clients, batch_size, ... )
        ell: Loss function with signature ( model, model_g, y, *xs ) -> loss
        ds_val: Optional iterable validation dataset with signature ( y, *xs ), of which both arguments have shape ( n_clients, batch_size, ... )
        local_epochs: Number of local epochs per communication round, or "early" for early stopping based on validation loss
        filename: If provided, saves model parameters to this file at each epoch
        n_clients: Number of clients
        max_patience: Maximum patience for early stopping (if local_epochs or rounds is "early")
        rounds: Number of communication rounds, or "early" for early stopping based on validation loss
        val_fn: Validation function, necessarily a minimization metric, with signature ( model, y, *xs ) -> loss
    Returns:
        The final local models before aggregation and the number of rounds it took to converge. To aggregate;
        ```
        from fedflax import train, get_updates, aggregate
        model_init = ... # initialize global model
        models, _rounds = train(model_init, ...) # train with desired arguments
        updates = get_updates(model_init, models)
        model_g = aggregate(model_init, updates)
        ```
    """
    tmp_file = tmp_file or f"tmp_fedflax_{hash(model_g.__repr__())}.pkl"
    # Validation function for local early stopping that can be used as stand-alone
    if isinstance(local_epochs, str):
        local_val_fn = nnx.jit(nnx.vmap(
            val_fn, in_axes=0
        ))

    # Communication rounds
    val_losses = []
    r = 0
    patience = 1
    while (r!=rounds) if isinstance(rounds, int) else (patience<=max_patience):
        # Parallelize global model and optimizers
        models = cast(model_g, n_clients)
        opts = nnx.vmap(opt_create)(models)
        
        # Local training
        models, n_inputs, epochs = train_local(models, opts, n_clients, ds_train, ds_val, ell, model_g, local_epochs, max_patience, filename, r, rounds, local_val_fn)
        
        # Aggregate
        updates = get_updates(model_g, models)
        model_g = aggregate(model_g, updates)
        
        if rounds=="early":
            # Initialize validation function for global early stopping
            if r==0:
                val_fn = nnx.jit(nnx.vmap(
                    val_fn, in_axes=(None,0)+(0,)*n_inputs
                ))
            # Evaluate aggregated model
            val = reduce(lambda a, batch: a+val_fn(model_g, *batch).mean(), ds_val, 0.)
            val /= len(ds_val)
            val_losses.append(val)       
            print(f"round {r} ({epochs} local epochs); global validation score: {val:.4f}")
            # Check whether model has converged
            if r>=1 and val>=val_losses[-patience-1]:
                patience += 1
            else:
                # ... otherwise continue training and save best model
                patience = 1
                save_model(models, tmp_file)
        # If not early stopping, just print progress and save model
        else:
            print(f"round {r} ({epochs} local epochs)")
        r += 1
    
    # Save final params
    if filename: save(cast(model_g, n_clients), filename, n_clients, overwrite=False)

    # Return the final local models, i.e., before aggregation
    if isinstance(rounds, str):
        models = load_model(lambda: models, tmp_file)
        os.remove(tmp_file)
    return models, r-patience-1