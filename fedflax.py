import jax, numpy as np
from functools import reduce
from npy_append_array import NpyAppendArray
from jax import numpy as jnp
from flax import nnx
from functools import partial
from tqdm.auto import tqdm

def save(models, filename, n, overwrite):
        with NpyAppendArray(filename, delete_if_exists=overwrite) as f:
            f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.split(models, nnx.Param, ...)[1])], axis=1))

# Parallelized train step
def return_train_step(ell):
    @nnx.jit
    @nnx.vmap(in_axes=(0,None,0,0,0,0))
    def train_step(model, model_g, opt, x_batch, z_batch, y_batch):
        loss, grad = nnx.value_and_grad(partial(ell, train=True))(model, model_g, x_batch, z_batch, y_batch)
        opt.update(grad)
        return loss
    return train_step

# Get updates, i.e., difference between initial model and locally converged models
def get_updates(model_g, models):
    params = nnx.split(models, nnx.Param, ...)[1]
    params_g = nnx.split(model_g, nnx.Param, ...)[1]
    return jax.tree.map(lambda pg, p: p-pg, params_g, params) # state of length n_layers with arrays of shape (n_clients, *layer_shape)

def aggregate(model_g, updates):
    # Get model structure
    struct, params_g, rest = nnx.split(model_g, nnx.Param, ...)
    # Average updates
    update = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
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

def train(model_g, opt_create, ds_train, ds_val, ell, local_epochs, filename=None, n=4, max_patience=None, rounds=None):
    # Parallelize train step
    train_step = return_train_step(ell)
    # Acc function that can be used as stand-alone
    acc_fn = nnx.jit(nnx.vmap(lambda m,x,z,y: (m(x,z,train=False).argmax(-1)==y.argmax(-1)).mean()))

    # Communication rounds
    losses = jnp.zeros((0,n+1)) # last column for validation loss
    r = 0
    patience = 1
    while r!=rounds and (max_patience is None or r<=1 or patience<=max_patience):
        # Parallelize global model and optimizers
        models = cast(model_g, n)
        if r==0:
            opts = nnx.vmap(opt_create)(models)
        
        # Local training
        losses = jnp.concat([losses, jnp.zeros((1,n+1))])
        for epoch in range(local_epochs): # TODO: should allow for early stopping
            # Collect and save params for visualization
            if filename: save(models, filename, n, overwrite=(r==0 and epoch==0))
            # Iterate over batches
            for x_batch, z_batch, y_batch in (bar := tqdm(ds_train, leave=False)):
                loss = train_step(models, model_g, opts, x_batch, z_batch, y_batch)
                losses = losses.at[-1,:-1].set(losses[-1,:-1] + loss)
                bar.set_description(f"Round {r} Epoch {epoch+1}/{local_epochs}. Batch loss: {loss.mean():.4f}")

        # Evaluate
        losses = losses.at[-1,:-1].set(losses[-1,:-1]/local_epochs/len(ds_train))    
        val_acc = reduce(lambda a,b: a+acc_fn(models, *b).mean(), ds_val, 0.)
        val_acc /= len(ds_val)
        losses = losses.at[-1, -1].set(val_acc)
        print(f"round {r} validation accuracy (mean over clients): {val_acc}")
        
        # Aggregate
        globals()["model_g"] = model_g  # For debugging
        globals()["models"] = models  # For debugging
        updates = get_updates(model_g, models)
        model_g = aggregate(model_g, updates)
        
        # Check if model is converged
        r += 1
        if r>1 and val_acc<=losses[-patience-1,-1]:
            patience += 1
        else:
            patience = 1
    
    # Save final params
    if filename: save(cast(model_g, n), filename, n, overwrite=False)

    # Returns for the various analyses
    return jax.tree.leaves(updates), models