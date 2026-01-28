# Imports
from fedflax import train
from models import UNETR
from data import fetch_data
from utils import return_ce, mean_iou_err
import jax, optax, pickle
from flax import nnx
from functools import reduce

# Configuration
n_clients = 3
asymkwargs = {}

# Initialize model
model = UNETR(20, img_size=224, **asymkwargs)
lr = optax.warmup_exponential_decay_schedule(1e-4, .1, 2000, 1000, .9, end_value=1e-5)
opt = nnx.Optimizer(
    model,
    optax.adamw(lr),
    wrt=nnx.Param
)

# Load data
ds_train = fetch_data(beta=1., dataset=2, n_clients=n_clients, batch_size=16)
ds_val = fetch_data(beta=1., dataset=2, partition="val", n_clients=n_clients, batch_size=16)

# Train (fixed number of epochs since test data is not available)
models, rounds = train(
    model,
    opt,
    ds_train,
    return_ce(0.), 
    local_epochs=50,
    n_clients=n_clients,
    rounds=1
)

# Aggregate
struct, state, rest = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)
state_g = jax.tree.map(lambda x: x.mean(0), state)
model_g = nnx.merge(struct, state_g, rest)

# Evaluate aggregated model
vval_fn = nnx.jit(nnx.vmap(mean_iou_err, in_axes=(None,0,0)))
err_g = reduce(lambda acc, batch: acc + vval_fn(model_g, *batch), ds_val, 0.) / len(ds_val)
print("Global mIoU: ", err_g.mean().item())

# Evaluate client models separately
vval_fn = nnx.jit(nnx.vmap(mean_iou_err))
err_l = reduce(lambda acc, batch: acc + vval_fn(models, *batch), ds_val, 0.) / len(ds_val)
print("Local mIoU: ", err_l.mean().item())

# Save client models
state = nnx.state(models, ...)
pickle.dump(state, open("models/cs_central.pkl", "wb"))