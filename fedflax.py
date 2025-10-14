import jax, numpy as np
from functools import reduce
from npy_append_array import NpyAppendArray
from jax import numpy as jnp
from flax import nnx
from functools import partial
from tqdm.auto import tqdm

# Parallelized train step
def return_train_step(ell):
    @nnx.jit
    @nnx.vmap(in_axes=(0,None,0,0,0,0))
    def train_step(model, model_g, opt, x_batch, z_batch, y_batch):
        (loss, (prox, ce)), grads = nnx.value_and_grad(partial(ell, train=True), has_aux=True)(model, model_g, x_batch, z_batch, y_batch)
        # grads = jax.tree.map(lambda g: g/2**15, grads) # assumes scaled loss for numerical stability with float16
        opt.update(grads)
        return loss
    return train_step

# Get updates, i.e., difference between initial model and locally converged models
def get_updates(model_g, models):
    params = jax.tree.leaves(nnx.to_tree(models))
    params_g = jax.tree.leaves(nnx.to_tree(model_g))
    return jax.tree.map(lambda pg, p: p-pg, params_g, params) # list of length n_layers with arrays of shape (n_clients, layer_shape)

def aggregate(model_g, updates):
    # Get model structure
    params_g, struct = jax.tree.flatten(nnx.to_tree(model_g))
    # Average updates
    update = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
    # Apply to global model
    params_g = jax.tree.map(lambda pg, u: pg + u, params_g, update)
    # Convert to model
    model_g = nnx.from_tree(jax.tree.unflatten(struct, params_g))
    return model_g

# Broadcast global model to clients
def cast(model_g, n):
    params_g, struct = jax.tree.flatten(nnx.to_tree(model_g))
    params_all = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, 0), params_g)
    models = nnx.from_tree(jax.tree.unflatten(struct, params_all))
    return models

def train(model, opt_create, ds_train, ds_val, ell, local_epochs, filename=None, n=4, max_patience=None, rounds=None):
    # Identically initialized models, interpretable as collection by nnx
    if isinstance(model, type):
        keys = nnx.vmap(lambda k: nnx.Rngs(k))(jnp.array([jax.random.key(42)]*n))
        models = nnx.vmap(model)(keys)
    else:
        print("Using provided initial models")
        models = model
    # Ditto for optimizers
    opts = nnx.vmap(opt_create)(models)
    train_step = return_train_step(ell)
    # Init and save
    params, struct = jax.tree.flatten(nnx.to_tree(models))
    model_g = nnx.from_tree(jax.tree.unflatten(struct, jax.tree.map(lambda x: jnp.mean(x, axis=0), params)))
    # Adjust loss function so that it can be used as stand-alone
    ell_val = nnx.jit(nnx.vmap(partial(ell, train=False), in_axes=(0,None,0,0,0)))

    # Communication rounds
    losses = jnp.zeros((0,n+1)) # last column for validation loss
    r = 0
    patience = 1
    while r!=rounds and (max_patience is None or r<=1 or patience<=max_patience):
        # Re-initialize models to global model
        models = cast(model_g, n)
        # Local training
        losses = jnp.concat([losses, jnp.zeros((1,n+1))])
        for epoch in range(local_epochs):
            # Collect and save params for visualization
            if filename:
                with NpyAppendArray(filename, delete_if_exists=True if epoch+r==0 else False) as f:
                    f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.to_tree(models))], axis=1))
            # Iterate over batches
            for b, (x_batch, z_batch, y_batch) in enumerate(tqdm(ds_train, leave=False, desc=f"Round {r} Epoch {epoch+1}/{local_epochs}")):
                loss = train_step(models, model_g, opts, x_batch, z_batch, y_batch)
                if jnp.isnan(loss).any():
                    print(losses)
                    raise ValueError("NaN encountered")
                losses = losses.at[-1,:-1].set(losses[-1,:-1] + loss)
        # Evaluate
        losses = losses.at[-1,:-1].set(losses[-1,:-1]/local_epochs/(b+1))
        val_loss = reduce(lambda a,b: a+ell_val(models, model_g, *b)[0].mean(), ds_val, 0.)
        val_loss /= len(ds_val)
        losses = losses.at[-1, -1].set(val_loss)
        print(f"round {r} global validation loss: {val_loss}")
        # Aggregate
        updates = get_updates(model_g, models)
        model_g = aggregate(model_g, updates)
        # Check if model is converged
        r += 1
        if r>1 and losses[-1,-1]>losses[-patience-1,-1]:
            patience += 1
        else:
            patience = 1
    # Save final params
    if filename:
        with NpyAppendArray(filename) as f:
            f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.to_tree(cast(models, n)))], axis=1))

    # Returns all kinds of output for the various analyses
    return updates, models