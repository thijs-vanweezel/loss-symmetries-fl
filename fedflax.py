import jax, numpy as np
from functools import reduce
from npy_append_array import NpyAppendArray
from jax import numpy as jnp
from flax import nnx
from functools import partial
from tqdm.auto import tqdm
from utils import err_fn

def save(models, filename, n, overwrite):
        with NpyAppendArray(filename, delete_if_exists=overwrite) as f:
            f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.split(models, (nnx.Param, nnx.BatchStat), ...)[1])], axis=1))

# Parallelized train step
def return_train_step(ell):
    @nnx.jit
    @nnx.vmap(in_axes=(0,None,0,0,0,0))
    def train_step(model, model_g, opt, x_batch, z_batch, y_batch):
        loss, grad = nnx.value_and_grad(ell)(model, model_g, x_batch, z_batch, y_batch)
        opt.update(grad)
        return loss
    return train_step

# Get updates, i.e., difference between initial model and locally converged models
def get_updates(model_g, models):
    params = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)[1]
    params_g = nnx.split(model_g, (nnx.Param, nnx.BatchStat), ...)[1]
    return jax.tree.map(lambda pg, p: p-pg, params_g, params) # state of length n_layers with arrays of shape (n_clients, *layer_shape)

def aggregate(model_g, updates):
    # Get model structure
    struct, params_g, rest = nnx.split(model_g, (nnx.Param, nnx.BatchStat), ...)
    # Average updates
    update = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
    rest = jax.tree.map(lambda r: r if r.ndim==0 else r[-1], rest)
    # Apply to global model
    params_g = jax.tree.map(lambda pg, u: pg + u, params_g, update)
    # Convert to model
    model_g = nnx.merge(struct, params_g, rest)
    return model_g

# Broadcast global model to clients
def cast(model_g, n):
    params_g, struct = jax.tree.flatten(nnx.to_tree(model_g))
    params_all = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, 0), params_g)
    models = nnx.from_tree(jax.tree.unflatten(struct, params_all))
    return models

def train(model_g, opt_create, ds_train, ds_val, ell, local_epochs:int|str="early", filename:str=None, n=4, max_patience:int=None, rounds:int|str="early", val_fn=None):
    # Parallelize train step
    train_step = return_train_step(ell)
    # Validation function that can be used as stand-alone
    # NOTE: For patience purposes, must be a Minimization metric
    local_val_fn = nnx.jit(nnx.vmap(
        val_fn or err_fn, in_axes=0
    ))
    val_fn = nnx.jit(nnx.vmap(
        val_fn or err_fn, in_axes=(None,0,0,0)
    ))

    # Communication rounds
    val_losses = []
    r = 0
    patience = 1
    while (r!=rounds) if isinstance(rounds, int) else (patience<=max_patience):
        # Parallelize global model and optimizers
        models = cast(model_g, n)
        if r==0:
            opts = nnx.vmap(opt_create)(models)
        
        # Local training
        local_val_losses = []
        local_patience = 1
        epoch = 0
        while (epoch!=local_epochs) if isinstance(rounds, int) else (local_patience<=max_patience):
            # Collect and save params for visualization
            if filename: save(models, filename, n, overwrite=(r==0 and epoch==0))
            # Iterate over batches
            for x_batch, z_batch, y_batch in (bar := tqdm(ds_train, leave=False)):
                loss = train_step(models, model_g, opts, x_batch, z_batch, y_batch)
                bar.set_description(f"Batch loss: {loss.mean():.4f}. Round {r} Epoch {epoch+1}/{local_epochs}")
            # Evaluate on local validation
            if local_epochs=="early":
                globals()['ms'] = models  # For debugging
                globals()['ds'] = ds_val  # For debugging
                globals()['vf'] = local_val_fn  # For debugging
                val = reduce(lambda a,b: a+local_val_fn(models, *b).mean(), ds_val, 0.)
                local_val_losses.append(val / len(ds_val))
                # Check if local models are converged
                if epoch>1 and val>=local_val_losses[-local_patience-1]:
                    local_patience += 1
                else:
                    local_patience = 1
            epoch += 1
        
        # Aggregate
        updates = get_updates(model_g, models)
        model_g = aggregate(model_g, updates)
        
        if rounds=="early":
            # Evaluate aggregated model
            val = reduce(lambda a,b: a+val_fn(model_g, *b).mean(), ds_val, 0.)
            val /= len(ds_val)
            val_losses.append(val)       
            print(f"round {r} ({epoch} local epochs); global validation score: {val:.4f}")
            # Check if model is converged
            r += 1
            if r>1 and val>=val_losses[-patience-1]:
                patience += 1
            else:
                patience = 1
    
    # Save final params
    if filename: save(cast(model_g, n), filename, n, overwrite=False)

    # Returns for the various analyses
    return updates, models