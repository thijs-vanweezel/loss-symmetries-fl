# Imports
import jax, optax, pickle, subprocess, os
from fedflax import train, get_updates, aggregate
from unetr import UNETR
from data import fetch_data
from utils import miou, save_model, nnx_norm
from jax import numpy as jnp
from flax import nnx
from functools import reduce
import argparse

parser = argparse.ArgumentParser(description="Train a UNETR on Oxford Pets in a federated setting")
parser.add_argument("--n_clients", type=int, default=4, help="Number of clients to simulate")
parser.add_argument("--asymtype", type=str, default="", choices=["", "wasym", "syre", "normweights", "orderbias"], help="Type of symmetry elimination to use")
args = parser.parse_args()
n_clients = args.n_clients
asymkwargs = {"key":jax.random.key(42)}
# Asymmetry parameters
if args.asymtype == "wasym":
    asymkwargs["wasym"] = True
    asymkwargs["kappa"] = 1
elif args.asymtype == "syre":
    asymkwargs["ssigma"] = 1e-4
elif args.asymtype == "normweights":
    asymkwargs["normweights"] = True
elif args.asymtype == "orderbias":
    asymkwargs["orderbias"] = True
model_name = f"models/oxford_{args.asymtype or ('central' if n_clients==1 else 'base')}.pkl"

# Initialize model
model_init = UNETR(20, img_size=224, **asymkwargs)
lr = optax.warmup_exponential_decay_schedule(1e-4, .5, 500, 250, .9, end_value=1e-5)
opt = nnx.Optimizer(
    model_init,
    optax.adam(lr),
    wrt=nnx.Param
)

# Loss function with jaccard term
def _loss_fn(model, model_g, y, x):
    logits = model(x, train=True)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y, axis=-1).mean()
    miou_err = 1. - miou(jax.nn.softmax(logits, axis=-1), y)
    return ce + miou_err
if args.asymtype == "syre":
    loss_fn = lambda model, model_g, y, x: _loss_fn(model, model_g, y, x) + 1e-4*nnx_norm(nnx.state(model, nnx.Param), n_clients=n_clients)
else:
    loss_fn = _loss_fn

# Load data
ds_train = fetch_data(beta=1., dataset=2, n_clients=n_clients)
ds_val = fetch_data(beta=1., dataset=2, partition="val", n_clients=n_clients)
ds_test = fetch_data(beta=1., dataset=2, partition="test", n_clients=n_clients)

# Train (fixed number of epochs since test data is not available)
models, rounds = train(
    model_init,
    opt,
    ds_train,
    loss_fn, 
    local_epochs=50,
    n_clients=n_clients,
    rounds=1 if n_clients==1 else "ealy",
    max_patience=None if n_clients==1 else 3,
    val_fn=lambda model, y, x: 1-miou(jax.nn.one_hot(jnp.argmax(model(x, train=False), axis=-1), y.shape[-1]), y),
    ds_val=ds_val
)
# Save client models
save_model(models, model_name)

# Aggregate
model_g = aggregate(model_init, get_updates(model_init, models))

# Evaluate aggregated model
model_g.eval()
vval_fn = nnx.jit(nnx.vmap(
    lambda model, y, x: miou(jax.nn.one_hot(jnp.argmax(model(x, train=False), axis=-1), y.shape[-1]), y), 
    in_axes=(None,0,0)))
miou_g = reduce(lambda acc, batch: acc + vval_fn(model_g, *batch), ds_test, 0.) / len(ds_test)
print("Global mIoU: ", miou_g.mean().item())

# Evaluate client models separately
models.eval()
vval_fn = nnx.jit(nnx.vmap(
    lambda model, y, x: miou(jax.nn.one_hot(jnp.argmax(model(x, train=False), axis=-1), y.shape[-1]), y), 
    ))
miou_l = reduce(lambda acc, batch: acc + vval_fn(models, *batch), ds_test, 0.) / len(ds_test)
print("Local mIoU: ", miou_l.mean().item())
