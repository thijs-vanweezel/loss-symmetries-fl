import os
os.environ["XLA_FLAGS"] = " --xla_gpu_strict_conv_algorithm_picker=false"
from fedflax import train, aggregate, get_updates
import jax, optax, argparse
from jax import numpy as jnp
from flax import nnx
from models import ResNet
from data import fetch_data
from functools import reduce
from utils import return_ce, top_5_err, functional_drift, save_model, load_model, err_fn, nnx_norm

# Allow user-specified arguments
parser = argparse.ArgumentParser(description="Train a ResNet on ImageNet in a federated setting")
parser.add_argument("--n_clients", type=int, default=4, help="Number of clients to simulate")
parser.add_argument("--asymtype", type=str, default="", choices=["", "wasym", "syre", "normweights", "dimexp"], help="Type of symmetry elimination to use")
parser.add_argument("--n_classes", type=int, default=1000, help="Number of classes to use from ImageNet")
args = parser.parse_args()
n_clients = args.n_clients
# Fill asymkwargs
asymkwargs = {}
ell = lambda m, mg, y, x: optax.softmax_cross_entropy_with_integer_labels(m(x, train=True), y).mean()
if args.asymtype == "wasym":
    asymkwargs["wasym"] = True
    asymkwargs["kappa"] = 1
elif args.asymtype == "syre":
    asymkwargs["ssigma"] = 1e-4
    ell = lambda m, mg, y, x: optax.softmax_cross_entropy_with_integer_labels(m(x, train=True), y).mean() \
        + 1e-4*nnx_norm(nnx.state(m, nnx.Param), n_clients=n_clients)
elif args.asymtype == "normweights":
    asymkwargs["normweights"] = True
elif args.asymtype == "dimexp":
    asymkwargs["dimexp"] = 1
# Model name based on asymmetry type
model_name = f"/data/bucket/traincombmodels/models/imagenet{args.n_classes}_{args.asymtype or ('central' if n_clients==1 else 'base')}.pkl"
# Model complexity based on data complexity
if (n_classes:=args.n_classes)>100:
    layers = [3,4,6,3]
else:
    layers = [2,2,2,2]

# Get data
ds_train = fetch_data(beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=n_classes)
ds_val = fetch_data(partition="val", beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=n_classes)

# Initialize model and optimizer
model_init = ResNet(jax.random.key(42), dim_out=n_classes, layers=layers, **asymkwargs)
lr = optax.warmup_exponential_decay_schedule(1e-4, .1, 4000, 1000, .9, end_value=1e-5)
opt = nnx.Optimizer(
    model_init,
    optax.adam(lr),
    wrt=nnx.Param
)

# Train
models, _ = train(model_init, opt, ds_train, return_ce(0.), ds_val, 
                  local_epochs="early", rounds="early" if n_clients>1 else 1, max_patience=3, val_fn=top_5_err, n_clients=n_clients);

# Save model
save_model(models, model_name)

# Load
models = load_model(
    lambda: ResNet(layers=layers, dim_out=n_classes, **asymkwargs), 
    model_name
)

# Aggregate
updates = get_updates(model_init, models)
model_g = aggregate(model_init, updates)

# Load test data
del ds_train, ds_val
ds_test = fetch_data(partition="test", beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=n_classes)

# Evaluate global performance
vval_fn = nnx.jit(nnx.vmap(top_5_err, in_axes=(None,0,0)))
err_test = reduce(lambda e, batch: e + vval_fn(model_g, *batch), ds_test, 0.) / len(ds_test)
print(f"Top-5 err test, measured after aggregation: {err_test.mean()}")
vval_fn = nnx.jit(nnx.vmap(err_fn, in_axes=(None,0,0)))
err_test = reduce(lambda e, batch: e + vval_fn(model_g, *batch), ds_test, 0.) / len(ds_test)
print(f"Top-1 test err, measured after aggregation: {err_test.mean()}")

# Evaluate local performance
vval_fn = nnx.jit(nnx.vmap(top_5_err, in_axes=(0,0,0)))
err_sep_test = reduce(lambda e, batch: e + vval_fn(models, *batch), ds_test, 0.) / len(ds_test)
print(f"Top-5 err test, measured separately: {err_sep_test.mean()}")
vval_fn = nnx.jit(nnx.vmap(err_fn, in_axes=(0,0,0)))
err_test = reduce(lambda e, batch: e + vval_fn(models, *batch), ds_test, 0.) / len(ds_test)
print(f"Top-1 test err, measured separately: {err_test.mean()}")

# Measure angular drift
updates_flat = jnp.concatenate(jax.tree.map(lambda x: jnp.reshape(x, (n_clients,-1)), jax.tree.leaves(updates)), axis=1)
update_g = updates_flat.mean(0)
print("angular drift: ", jnp.degrees(jnp.arccos(optax.losses.cosine_similarity(update_g, updates_flat)).mean()).item())

# Measure functional drift
func_drift = functional_drift(models, ds_test)
print("functional drift: ", func_drift.mean().item())

# Measure absolute drift
print("L1 drift: ", jnp.sum(jnp.abs(update_g - updates_flat), axis=-1).mean().item())
print("L2 drift: ", jnp.sqrt(jnp.sum((update_g - updates_flat)**2, axis=-1)).mean().item())