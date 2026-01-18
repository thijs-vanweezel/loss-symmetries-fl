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
def return_train_step(ell, n_inputs):
    @nnx.jit
    @nnx.vmap(in_axes=(0,None,0,0)+(0,)*n_inputs)
    def train_step(model, model_g, opt, y, *xs):
        loss, grad = nnx.value_and_grad(ell)(model, model_g, y, *xs)
        opt.update(grad)
        return loss
    return train_step

# Get updates, i.e., difference between initial model and locally converged models
def get_updates(model_g, models):
    params = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)[1]
    params_g = nnx.split(model_g, (nnx.Param, nnx.BatchStat), ...)[1]
    return jax.tree.map(lambda pg, p: p-pg, params_g, params) # state of length n_layers with arrays of shape (n_clients, *layer_shape)

def aggregate(model_g, updates):
    # Get global model's structure, along with its parameters/rest
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

def train(model_g, opt_create, ds_train, ell, ds_val=None, local_epochs:int|str="early", ckpt_fp:str=None, n_clients=4, 
          max_patience:int=None, rounds:int|str="early", val_fn=None):
    """
    Federated training loop. 
    Args:
        model_g: Initialized global model
        opt_create: Function that creates optimizer when given a model
        ds_train: Iterable training dataset with signature ( y, *xs ), of which both arguments have shape ( n_clients, batch_size, ... )
        ell: Loss function with signature ( model, model_g, y, *xs ) -> loss
        ds_val: Optional iterable validation dataset with signature ( y, *xs ), of which both arguments have shape ( n_clients, batch_size, ... )
        local_epochs: Number of local epochs per communication round, or "early" for early stopping based on validation loss
        ckpt_fp: If provided, saves model parameters to this path at each epoch
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
    ckpt = bool(ckpt_fp)
    ckpt_fp, ext = os.path.splitext(ckpt_fp or f"./tmp_fedflax_{hash(model_g.__repr__())}.pkl")
    os.makedirs(os.path.split(ckpt_fp)[0], exist_ok=True)
    # Validation function for local early stopping that can be used as stand-alone
    if isinstance(local_epochs, str):
        local_val_fn = nnx.jit(nnx.vmap(
            val_fn, in_axes=0
        ))

    # Communication rounds
    val_losses = []
    r = 0
    patience = 1
    while (r!=rounds) if rounds!="early" else (patience<=max_patience):
        # Parallelize global model and optimizers
        models = cast(model_g, n_clients)
        opts = nnx.vmap(opt_create)(models)
        
        # Local training
        local_val_losses = []
        local_patience = 1
        epoch = 0
        while (epoch!=local_epochs) if local_epochs!="early" else (local_patience<=max_patience):
            # Collect and save params for visualization
            if ckpt: save_model(models, f"{ckpt_fp}_{r}_{epoch}{ext}")
            # Iterate over batches
            for batch, (y, *xs) in enumerate(bar := tqdm(ds_train, leave=False)):
                # Create train step function if first iteration
                if batch==0 and epoch==0 and r==0:
                    train_step = return_train_step(ell, n_inputs:=len(xs))
                # Train step
                loss = train_step(models, model_g, opts, y, *xs)
                # Inform user
                bar.set_description(f"Round {r}/{rounds}, epoch {epoch}/{local_epochs} (local validation score: {'N/A' if epoch==0 or local_epochs!='early' else val}, local batch loss: {loss.mean():.4f})")
            # Evaluate on local validation
            if local_epochs=="early":
                val = reduce(lambda a, batch: a+local_val_fn(models, *batch).mean(), ds_val, 0.)
                val /= len(ds_val)
                local_val_losses.append(val)
                # Check if local models are converged
                if epoch>=1 and val>=local_val_losses[-local_patience-1]:
                    local_patience += 1
                else:
                    # ... otherwise continue training and save best model
                    local_patience = 1
                    save_model(models, ckpt_fp+ext)
            epoch += 1
        
        # Aggregate
        if local_epochs=="early":
            models = load_model(lambda: models, ckpt_fp+ext)
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
            print(f"round {r} ({epoch} local epochs); global validation score: {val:.4f}")
            # Check whether model has converged
            if r>=1 and val>=val_losses[-patience-1]:
                patience += 1
            else:
                # ... otherwise continue training and save best model
                patience = 1
                save_model(models, ckpt_fp+ext)
        r += 1
    
    # Save final params
    if ckpt: save_model(cast(model_g, n_clients), f"{ckpt_fp}_{r}_{epoch}{ext}")

    # Return the final local models, i.e., before aggregation
    if rounds=="early" or local_epochs=="early":
        models = load_model(lambda: models, ckpt_fp+ext)
        os.remove(ckpt_fp+ext)
    return models, r-patience-1