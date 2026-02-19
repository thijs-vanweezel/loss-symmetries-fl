import jax, optax, argparse
from jax import numpy as jnp
from flax import nnx
from utils import nnx_norm, miou, load_model
from data import fetch_data
from models import ResNet

parser = argparse.ArgumentParser(
    description="Load model and dataset, and calculate the dominant eigenvalue of the Hessian of the loss using power iteration. Assumes four clients."
)
parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "oxford", "imagenet"], help="Dataset to evaluate on. Also determines the model to load.")
parser.add_argument("--asymtype", type=str, default="", choices=["", "wasym", "syre", "normweights", "dimexp"], help="Type of symmetry elimination to use")
args = parser.parse_args()

# Setup model, dataset, and loss
asymkwargs = {"key":jax.random.key(0)}
n_clients = 4
if args.dataset == "celeba":
    modelclass = ...
    dataloader = iter(fetch_data(skew="feature", partition="test", n_clients=n_clients, beta=1., dataset=3))
    _loss_fn = lambda m, y, x: optax.sigmoid_binary_cross_entropy(m(x, train=False), y).mean()
elif args.dataset == "oxford":
    modelclass = UNETR
    dataloader = iter(fetch_data(skew="feature", partition="test", n_clients=n_clients, beta=1., dataset=2))
    def _loss_fn(model, y, x):
        logits = model(x, train=False)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, y, axis=-1).mean()
        miou_err = 1. - miou(jax.nn.softmax(logits, axis=-1), y)
        return ce + miou_err
elif args.dataset == "imagenet":
    modelclass = ResNet
    dataloader = iter(fetch_data(skew="feature", partition="test", n_clients=n_clients, beta=1., dataset=1))
    _loss_fn = lambda m, y, x: optax.softmax_cross_entropy_with_integer_labels(m(x, train=False), y).mean()

# Asymmetry parameters
loss_fn = _loss_fn
if args.asymtype == "wasym":
    asymkwargs["wasym"] = True
    asymkwargs["kappa"] = 1
elif args.asymtype == "syre":
    asymkwargs["sigma"] = 1e-4
    loss_fn = lambda m, y, x: _loss_fn(m, y, x) \
        + 1e-4*nnx_norm(nnx.state(m, nnx.Param), n_clients=n_clients).mean()
elif args.asymtype == "normweights":
    asymkwargs["normweights"] = True
elif args.asymtype == "dimexp":
    asymkwargs["dimexp"] = 1

# Load model
model_name = f"../models/{args.dataset}_{args.asymtype or 'base'}.pkl"
model = load_model(lambda: modelclass(**asymkwargs), model_name)
key = jax.random.key(42)

# Calculate the dominant eigenvalue (lambda_max) of an nnx model using the power iteration method
# Note that each iteration only considers one batch
@nnx.vmap(in_axes=(0,None,None,None))
def lambda_max(model, dataloader, key, max_iter=100):
    # Convenience (jvp must act on a parameter-only pytree)
    struct, theta, rest = nnx.split(model, nnx.Param, ...)
    reconstruct = lambda th: nnx.merge(struct, th, rest)

    # Gradient of loss on this data
    grad_fn = nnx.grad(lambda th, x, y: loss_fn(reconstruct(th), y, x), argnums=0)

    # Random normalized vector
    def random_like(arr): nonlocal key; _, key = jax.random.split(key); return jax.random.normal(key, arr.shape)
    rand = jax.tree.map(lambda p: random_like(p), theta)

    # Perform iteration avoiding python bools
    def true_fun(val):
        # Power iteration step
        hv_prev, *_, y, x, i = val
        norm = nnx_norm(hv_prev)
        v = jax.tree.map(lambda hv_: hv_ / norm, hv_prev)
        hv = jax.jvp(
            grad_fn,
            (theta,y,x),
            (v,)
        )[1]
        return hv, hv_prev, v, *next(dataloader), i+1
    def cond_fun(val):
        # Check convergence
        hv, hv_prev, *_, i = val
        return jnp.logical_or(i<2, jnp.logical_and(i<max_iter, nnx_norm(hv, hv_prev)>=1e-3))
    # Iterate
    i0 = jnp.array(0)
    hv, _, v, *_ = jax.lax.while_loop(cond_fun, true_fun, (rand, rand, rand, *next(dataloader), i0))
    # Return Rayleigh quotient (manual dot product between hv and v)
    return jax.tree.reduce(
        lambda acc, prod: acc+prod, 
        jax.tree.map(lambda hv_, v_: jnp.sum(hv_*v_), hv, v)
        )

# Run
hessian = lambda_max(model, dataloader, key)
print("Dominant eigenvalue of Hessian:", hessian)