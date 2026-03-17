import os, sys
from pathlib import Path
os.environ["XLA_FLAGS"] = " --xla_gpu_strict_conv_algorithm_picker=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Allow direct script execution from this subfolder while using package imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax, optax, argparse, jax.experimental
from orbax import checkpoint
from jax import numpy as jnp
from flax import nnx
from backend.utils import nnx_norm, miou
from backend.data import fetch_data
from backend.models import ResNet
from backend.unetr import UNETR
from methodology.finetune import Classifier
from tqdm.auto import tqdm
from backend.fedflax import cast
from functools import partial

parser = argparse.ArgumentParser(
    description="Load model and dataset, and calculate the dominant eigenvalue of the Hessian of the loss using power iteration."
)
parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "oxford", "imagenet"], help="Dataset to evaluate on. Also determines the model to load.")
parser.add_argument("--asymtype", type=str, default="", choices=["", "wasym", "syre", "normweights", "dimexp"], help="Type of symmetry elimination to use")
parser.add_argument("--n_clients", type=int, default=4, help="Number of clients to simulate (only relevant for oxford)")
parser.add_argument("--n_classes", type=int, default=1000, help="Number of classes to use from ImageNet (only relevant for imagenet)")
args = parser.parse_args()

# Setup model, dataset, and loss
kwargs = {"key":jax.random.key(0)}
n_clients = args.n_clients
if args.dataset == "celeba":
    model_name = f"/data/bucket/traincombmodels/models/celeba_{args.asymtype or ('central' if n_clients==1 else 'base')}"
    modelclass = Classifier
    dataloader = fetch_data(skew="feature", partition="test", n_clients=1, beta=1., dataset=3, batch_size=64)
    _loss_fn = lambda m, y, x: optax.sigmoid_binary_cross_entropy(m(x, train=False), y).mean()
elif args.dataset == "oxford":
    model_name = f"/data/bucket/traincombmodels/models/oxford_{args.asymtype or ('central' if n_clients==1 else 'base')}_{args.n_clients}clients"
    modelclass = UNETR
    dataloader = fetch_data(skew="feature", partition="test", n_clients=1, beta=1., dataset=2, batch_size=64)
    def _loss_fn(model, y, x):
        logits = model(x, train=False)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, y, axis=-1).mean()
        miou_err = 1. - miou(jax.nn.softmax(logits, axis=-1), y)
        return ce + miou_err
elif args.dataset == "imagenet":
    kwargs["layers"] = [3,4,6,3] if args.n_classes>100 else [2,2,2,2]
    kwargs["dim_out"] = args.n_classes
    model_name = f"/data/bucket/traincombmodels/models/imagenet{args.n_classes}_{args.asymtype or ('central' if n_clients==1 else 'base')}"
    modelclass = ResNet
    dataloader = fetch_data(skew="label", partition="test", n_clients=1, beta=1., dataset=1, batch_size=64, n_classes=args.n_classes)
    _loss_fn = lambda m, y, x: optax.softmax_cross_entropy_with_integer_labels(m(x, train=False), y).mean()

# Asymmetry parameters
loss_fn = _loss_fn
if args.asymtype == "wasym":
    kwargs["wasym"] = True
    kwargs["kappa"] = 1
elif args.asymtype == "syre":
    kwargs["sigma"] = 1e-4
    loss_fn = lambda m, y, x: _loss_fn(m, y, x) \
        + 1e-4*nnx_norm(nnx.state(m, nnx.Param), n_clients=n_clients).mean()
elif args.asymtype == "normweights":
    kwargs["normweights"] = True
elif args.asymtype == "dimexp":
    kwargs["dimexp"] = 1

# Load model
abstract_model = nnx.eval_shape(lambda: cast(modelclass(**kwargs), n_clients))
struct, stateref = nnx.split(abstract_model)
with checkpoint.StandardCheckpointer() as cptr:
    state = cptr.restore(os.path.abspath(model_name), stateref)
state = jax.tree.map(lambda p: p if not (hasattr(p, "dtype") and jnp.issubdtype(p.dtype, jnp.floating)) else jnp.astype(p, jnp.float32), state)
models = nnx.merge(struct, state)

# Convert dataloader to a jax stack
ys, xs = [], []
for y, x in dataloader:
    ys.append(y)
    xs.append(x)
xs = jnp.stack(xs, axis=0)
ys = jnp.stack(ys, axis=0)

# Calculate the dominant eigenvalue (lambda_max) of an nnx model using the power iteration method
# Note that each iteration only considers one batch
@nnx.vmap(in_axes=(0,None,None))
def lambda_max(model, key, max_iter):
    # Convenience (jvp must act on a parameter-only pytree)
    struct, theta, rest = nnx.split(model, nnx.Param, ...)
    reconstruct = lambda th: nnx.merge(struct, th, rest)

    # Gradient of loss on this data
    _grad_fn = nnx.grad(lambda th, x, y: loss_fn(reconstruct(th), y, x), argnums=0)

    # Random normalized vector
    def random_like(arr): nonlocal key; _, key = jax.random.split(key); return jax.random.normal(key, arr.shape, arr.dtype)
    rand = jax.tree.map(lambda p: random_like(p), theta)

    # Perform iteration avoiding python bools
    def true_fun(val):
        # Unpack
        hv_prev, *_, i = val
        # Get batch
        y, x = ys[i%len(ys)], xs[i%len(xs)]
        # Fix data
        grad_fn = partial(_grad_fn, x=x.squeeze(0), y=y.squeeze(0))
        # Power iteration step
        norm = nnx_norm(hv_prev)
        v = jax.tree.map(lambda hv_: hv_ / norm, hv_prev)
        hv = jax.jvp(
            grad_fn,
            (theta,),
            (v,)
        )[1]
        return hv, hv_prev, v, i+1
    def cond_fun(val):
        # Check convergence
        hv, hv_prev, _, i = val
        return jnp.logical_or(i<2, jnp.logical_and(i<max_iter, nnx_norm(hv, hv_prev)>=1e-3))
    # Iterate
    i0 = jnp.array(0)
    hv, _, v, _ = jax.lax.while_loop(cond_fun, true_fun, (rand, rand, rand, i0))
    # Return Rayleigh quotient (manual dot product between hv and v)
    return jax.tree.reduce(
        lambda acc, prod: acc+prod, 
        jax.tree.map(lambda hv_, v_: jnp.sum(hv_*v_), hv, v)
        )

# Run
key = jax.random.key(42)
hessian = lambda_max(models, key, 100)
print(f"Dominant eigenvalue of Hessian matrix for {args.dataset} with asymmetry {args.asymtype}: ", hessian)
