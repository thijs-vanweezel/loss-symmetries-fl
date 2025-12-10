import jax, optax, pickle
from jax import numpy as jnp
from flax import nnx
from functools import reduce

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

# Client drift in function space, by comparing logits to avg logits
def functional_drift(models, ds_test):
    vcall = nnx.vmap(lambda model, *batch: model(*batch, train=False))
    logits = reduce(lambda acc, batch: acc + [vcall(models, *batch[1:])], ds_test, [])
    logits = jnp.concatenate(logits, axis=1)
    logits_mean = logits.mean(0)
    drift = jnp.abs(logits - logits_mean).mean((1,2))
    return drift

def save_model(model, filename):
    _struct, state = nnx.split(model, ...)
    pickle.dump(state, open(filename, "wb"))

def load_model(model_initializer, filename, **kwargs):
    abstract_model = nnx.eval_shape(model_initializer, **kwargs)
    struct, _state = nnx.split(abstract_model, ...)
    state = pickle.load(open(filename, "rb"))
    model = nnx.merge(struct, state)
    return model