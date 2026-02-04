# Imports
import jax, optax, pickle
from fedflax import train
from unetr import UNETR
from data import fetch_data
from utils import miou
from jax import numpy as jnp
from flax import nnx
from functools import reduce

# Configuration
n_clients = 1
asymkwargs = {"key":jax.random.key(42)}

# Initialize model
model = UNETR(20, img_size=224, **asymkwargs)
lr = optax.warmup_exponential_decay_schedule(1e-4, .5, 2000, 1000, .9, end_value=1e-5)
opt = nnx.Optimizer(
    model,
    optax.adamw(lr),
    wrt=nnx.Param
)

# Load data
ds_train = fetch_data(beta=1., dataset=2, n_clients=n_clients, batch_size=16)
ds_val = fetch_data(beta=1., dataset=2, partition="val", n_clients=n_clients, batch_size=16)

# Train (fixed number of epochs since test data is not available)
def loss_fn(model, model_g, y, *xs):
    logits = model(*xs, train=True)
    ce = optax.softmax_cross_entropy(logits, y, axis=-1).mean()
    miou_err = 1. - miou(jax.nn.softmax(logits, axis=-1), y)
    return ce + miou_err
models, rounds = train(
    model,
    opt,
    ds_train,
    loss_fn, 
    local_epochs=50,
    n_clients=n_clients,
    rounds=1
)

# Aggregate
struct, state, rest = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)
state_g = jax.tree.map(lambda x: x.mean(0), state)
model_g = nnx.merge(struct, state_g, rest)

# Save client models
state = nnx.state(models, ...)
pickle.dump(state, open("models/cs_central.pkl", "wb"))

# Evaluate aggregated model
model_g.eval()
vval_fn = nnx.jit(nnx.vmap(
    lambda model, y, *xs: miou(jax.nn.one_hot(jnp.argmax(model(*xs, train=False), axis=-1), y.shape[-1]), y), 
    in_axes=(None,0,0)))
miou_g = reduce(lambda acc, batch: acc + vval_fn(model_g, *batch), ds_val, 0.) / len(ds_val)
print("Global mIoU: ", miou_g.mean().item())

# Evaluate client models separately
models.eval()
vval_fn = nnx.jit(nnx.vmap(
    lambda model, y, *xs: miou(jax.nn.one_hot(jnp.argmax(model(*xs, train=False), axis=-1), y.shape[-1]), y), 
    ))
miou_l = reduce(lambda acc, batch: acc + vval_fn(models, *batch), ds_val, 0.) / len(ds_val)
print("Local mIoU: ", miou_l.mean().item())