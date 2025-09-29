import jax, numpy as np
from functools import reduce
from npy_append_array import NpyAppendArray
from jax import numpy as jnp
from flax import nnx
from functools import partial
from tqdm.auto import tqdm

# Parallelized train step (DOES NOT assumes scaled loss for numerical stability with float16)
def return_train_step(ell, train=True):
    @nnx.jit
    @nnx.vmap(in_axes=(0,None,0,0,0))
    def train_step(model, model_g, opt, x_batch, y_batch):
        (loss, (prox, ce)), grads = nnx.value_and_grad(partial(ell, train=train), has_aux=True)(model, model_g, x_batch, y_batch)
        # grads = jax.tree.map(lambda g: g/2**15, grads)
        opt.update(grads)
        return loss
    return train_step

# Get updates, i.e., difference between initial model and locally converged models
def get_updates(model_g, models):
    params = jax.tree.leaves(nnx.to_tree(models))
    params_g = jax.tree.leaves(nnx.to_tree(model_g))
    return jax.tree.map(lambda pg, p: p-pg, params_g, params) # list of length n_layers with arrays of shape (n_clients, layer_shape)

def aggregate(model_g, updates, n):
    # Get model structure
    params_g, struct = jax.tree.flatten(nnx.to_tree(model_g))
    # Average updates
    update = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)
    # Apply to global model
    params_g = jax.tree.map(lambda pg, u: pg + u, params_g, update)
    # Convert to model
    model_g = nnx.from_tree(jax.tree.unflatten(struct, params_g))
    # Broadcast to n models
    params_all = jax.tree.map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, 0), params_g)
    models = nnx.from_tree(jax.tree.unflatten(struct, params_all))
    return model_g, models

def train(Model, opt, ds_train, ds_val, ell, local_epochs, filename=None, n=4, max_patience=5):
    # Identically initialized models, interpretable as collection by nnx 
    keys = nnx.vmap(lambda k: nnx.Rngs(k))(jnp.array([jax.random.key(42)]*n))
    models = nnx.vmap(Model)(keys)
    # Ditto for optimizers
    opts = nnx.vmap(opt)(models)
    train_step = return_train_step(ell, train=True)
    # Init and save
    params, struct = jax.tree.flatten(nnx.to_tree(models))
    model_g = nnx.from_tree(jax.tree.unflatten(struct, jax.tree.map(lambda x: jnp.mean(x, axis=0), params)))
    # Adjust loss function so that it can be used as stand-alone
    ell_val = nnx.jit(nnx.vmap(return_train_step(ell, train=False), in_axes=(0,None,0,0)))

    # Communication rounds
    losses = jnp.zeros((0,n+1)) # last column for validation loss
    r = 0
    patience = 1
    while r<=(1-(0 if max_patience else 1)) or patience<=(max_patience or -1):
        # Local training
        losses = jnp.concat([losses, jnp.zeros((1,n+1))])
        for epoch in range(local_epochs):
            # Collect and save params for visualization
            if filename:
                with NpyAppendArray(filename, delete_if_exists=True if epoch+r==0 else False) as f:
                    f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.to_tree(models))], axis=1))
            # Iterate over batches
            for b, (x_batch, y_batch) in enumerate(tqdm(ds_train, leave=False, desc=f"Round {r} Epoch {epoch}/{local_epochs}")):
                loss = train_step(models, model_g, opts, x_batch, y_batch)
                if jnp.isnan(loss).any():
                    print(losses)
                    raise ValueError("NaN encountered")
                losses = losses.at[-1,:-1].set(losses[-1,:-1] + loss)
        # Aggregate and evaluate
        updates = get_updates(model_g, models)
        model_g, models = aggregate(model_g, updates, n)
        losses = losses.at[-1,:-1].set(losses[-1,:-1]/local_epochs/(b+1))
        val_loss = reduce(lambda a,b: a+ell_val(models, model_g, *b)[0].mean(), ds_val, 0.)
        val_loss /= len(ds_val)
        losses = losses.at[-1, -1].set(val_loss)
        print(f"round {r} global validation loss: {val_loss}")
        # Check if model is converged
        r += 1
        if r>1 and losses[-1,-1]>losses[-patience-1,-1]:
            patience += 1
        else:
            patience = 1
    # Save final params
    if filename:
        with NpyAppendArray(filename) as f:
            f.append(np.concat([p.reshape(n,-1) for p in jax.tree.leaves(nnx.to_tree(models))], axis=1))

    # Returns all kinds of output for the various analyses
    return updates