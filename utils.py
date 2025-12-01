import jax, optax
from jax import numpy as jnp
from flax import nnx

# Optimizer
opt_create = lambda model, learning_rate, **kwargs: nnx.Optimizer(
    model,
    optax.adamw(learning_rate=learning_rate, **kwargs),
    wrt=nnx.Param
)

# Regression loss
def return_l2(omega):
    def ell(model, model_g, y, *xs):
        prox = sum(jax.tree.map(lambda a, b: jnp.sum((a-b)**2), jax.tree.leaves(nnx.to_tree(model)), jax.tree.leaves(nnx.to_tree(model_g))))
        l2 = jnp.square(model(*xs, train=True) - y).mean()
        return l2 + omega/2 * prox
    return ell

# Angular error for regression validation
to_vec = lambda pitch, yaw: jnp.stack([
    jnp.cos(pitch) * jnp.sin(yaw),
    jnp.sin(pitch),
    jnp.cos(pitch) * jnp.cos(yaw)
])
def angle_err(model, y, *xs):
    pred = model(*xs, train=False)
    pred_vec = to_vec(pred[:,0], pred[:,1])
    y_vec = to_vec(y[:,0], y[:,1])
    rads = jnp.arccos(optax.losses.cosine_similarity(pred_vec, y_vec, axis=0))
    return jnp.nanmean(jnp.rad2deg(rads))

# Classification loss including softmax layer
def return_ce(omega):
    def ell(model, model_g, y, *xs):
        prox = sum(jax.tree.map(lambda a, b: jnp.sum((a-b)**2), jax.tree.leaves(nnx.to_tree(model)), jax.tree.leaves(nnx.to_tree(model_g))))
        ce = optax.softmax_cross_entropy(model(*xs, train=True), y).mean()
        return omega/2*prox + ce
    return ell

# Inverse accuracy function for classification validation
err_fn = lambda m,y,*xs: 1-(m(*xs,train=False).argmax(-1)==y.argmax(-1)).mean()

top_5_err = lambda m,y,*xs: 1 - jnp.any(y.argmax(-1, keepdims=True) == jnp.argsort(m(*xs,train=False), axis=-1)[:,-5:], axis=-1).mean()