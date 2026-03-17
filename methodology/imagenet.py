import os, sys
from pathlib import Path
os.environ["XLA_FLAGS"] = " --xla_gpu_strict_conv_algorithm_picker=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Allow direct script execution from this subfolder while using package imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax, optax, argparse, multiprocessing as mp
from backend.fedflax import train, aggregate, get_updates
from flax import nnx
from backend.models import ResNet
from backend.data import fetch_data, seed_worker
from functools import reduce
from backend.utils import return_ce, top_5_err, err_fn, nnx_norm
from orbax import checkpoint

if __name__ == "__main__":
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
        asymkwargs["sigma"] = 1e-4
        ell = lambda m, mg, y, x: optax.softmax_cross_entropy_with_integer_labels(m(x, train=True), y).mean() \
            + 1e-4*nnx_norm(nnx.state(m, nnx.Param))
    elif args.asymtype == "normweights":
        asymkwargs["normweights"] = True
    elif args.asymtype == "dimexp":
        asymkwargs["dimexp"] = 1
    # Model name based on asymmetry type
    model_name = f"/data/bucket/traincombmodels/models/imagenet{args.n_classes}_{args.asymtype or ('central' if n_clients==1 else 'base')}"
    # Model complexity based on data complexity
    if (n_classes:=args.n_classes)>100:
        layers = [3,4,6,3]
    else:
        layers = [2,2,2,2]

    # Get data
    ds_train = fetch_data(beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=n_classes, num_workers=8,
                          multiprocessing_context=mp.get_context("spawn"), persistent_workers=True, worker_init_fn=seed_worker)
    ds_val = fetch_data(partition="val", beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=n_classes, num_workers=4,
                        prefetch_factor=1, multiprocessing_context=mp.get_context("spawn"), persistent_workers=True, worker_init_fn=seed_worker)

    # Initialize model and optimizer
    model_init = ResNet(jax.random.key(42), dim_out=n_classes, layers=layers, **asymkwargs)
    opt = nnx.Optimizer(
        model_init,
        optax.adam(1e-3),
        wrt=nnx.Param
    )

    # Train
    models, _ = train(model_init, opt, ds_train, return_ce(0.), ds_val, 
                      local_epochs="early", rounds="early" if n_clients>1 else 1, 
                      max_patience=3, val_fn=top_5_err, n_clients=n_clients)

    # Save model
    _, state = nnx.split(models)
    with checkpoint.StandardCheckpointer() as cptr:
        cptr.save(os.path.abspath(model_name), state, force=True)

    # Aggregate
    updates = get_updates(model_init, models)
    model_g = aggregate(model_init, updates)

    # Load test data
    del ds_train, ds_val
    ds_test = fetch_data(partition="test", beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=n_classes,
                         num_workers=4, multiprocessing_context=mp.get_context("spawn"), worker_init_fn=seed_worker)

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