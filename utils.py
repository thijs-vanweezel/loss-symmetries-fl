import jax, optax, pickle
from jax import numpy as jnp
from flax import nnx
from functools import reduce

# Regression loss
def return_l2(omega):
    def ell(model, model_g, y, *xs):
        prox = sum(jax.tree.map(
            lambda a, b: jnp.sum((a-b)**2), 
            jax.tree.leaves(nnx.state(model, nnx.Param)), 
            jax.tree.leaves(nnx.state(model_g, nnx.Param))
        ))
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

# Classification loss including softmax layer over last dimension
def return_ce(omega):
    def ell(model, model_g, y, *xs):
        """`model(*xs)` should have shape (batch, ..., n_classes), whereas `y` is should have shape (batch,)"""
        prox = sum(jax.tree.map(
            lambda a, b: jnp.sum((a-b)**2), 
            jax.tree.leaves(nnx.state(model, nnx.Param)), 
            jax.tree.leaves(nnx.state(model_g, nnx.Param))
        ))
        ce = optax.softmax_cross_entropy_with_integer_labels(model(*xs, train=True), y, axis=-1).mean()
        return omega/2*prox + ce
    return ell

# Inverse accuracy function for classification validation
err_fn = lambda m,y,*xs: 1-(m(*xs,train=False).argmax(-1)==y).mean()
top_5_err = lambda m,y,*xs: 1 - jnp.any(jnp.expand_dims(y, -1) == jnp.argsort(m(*xs,train=False), axis=-1)[:,-5:], axis=-1).mean()

def miou(y_pred, y):
    """`y_pred` and `y` should be one-hot encoded and have shape (batch, *, n_classes)"""
    # Flatten image
    b, *_, c = y_pred.shape
    y_pred = y_pred.reshape((b, -1, c))
    y = nnx.one_hot(y, num_classes=c, axis=-1)
    y = y.reshape((b, -1, c))
    # Intersection and union without double counting intersection
    intersection = jnp.sum(y_pred * y, axis=1)
    union = jnp.sum(y_pred + y, axis=1) - intersection
    # Set classes with no ground truth to 1 (~valid)
    valid = jnp.sum(y, axis=1)>0
    iou = jnp.where(valid, intersection/union, 1.)
    # Avg over classes and batch, subtracting invalid ones
    return (jnp.sum(iou) - jnp.sum(~valid)) / jnp.sum(valid)

# Client drift in function space, by comparing logits to avg logits
def functional_drift(models, ds_test):
    vcall = nnx.vmap(lambda model, *batch: model(*batch, train=False))
    logits = reduce(lambda acc, batch: acc + [vcall(models, *batch[1:])], ds_test, [])
    logits = jnp.concatenate(logits, axis=1)
    logits_mean = logits.mean(0)
    drift = jnp.abs(logits - logits_mean).mean((1,2))
    return drift

def save_model(model, filename):
    state = nnx.state(model, ...)
    with open(filename, "wb") as f:
        pickle.dump(state, f)

def load_model(return_model, filename, **kwargs):
    """Args:
        return_model: Function or class that returns the model when called with **kwargs
        filename: File from which to load the model state
    """
    abstract_model = nnx.eval_shape(return_model, **kwargs)
    struct = nnx.graphdef(abstract_model)
    with open(filename, "rb") as f:
        state = pickle.load(f)
    model = nnx.merge(struct, state)
    return model

def nnx_norm(state1:nnx.State, state2:nnx.State|float=0., order:float=2., n_clients:int=1):
    """Compute norm between two pytrees. When `n_clients>1`, accounts for a client dimension."""
    if isinstance(state2, (int, float)): squares = jax.tree.map(lambda pl: jnp.abs(state2 - pl)**order, state1)
    else: squares = jax.tree.map(lambda pl, pg: jnp.abs(pg - pl)**order, state1, state2)
    sumofsquares = jax.tree.reduce(lambda acc, d: acc+jnp.sum(d.reshape(n_clients,-1), -1), squares, jnp.zeros(n_clients))
    return jnp.power(sumofsquares, 1/order).squeeze()