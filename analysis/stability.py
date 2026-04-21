# Four potential outcomes in a scenario where client drift is induced with label heterogeneity:
# 1. FedAvg exhibits high barrier, but W-Asym does not. According to Entezari (and to logic), the local solutions are stochastic, and removing permutation symmetries resolves this.
# 2. FedAvg does not exhibit high barrier, neither does W-Asym. Client drift was then not a problem in the first place, so W-Asym does not help.
# 3. FedAvg does not exhibit high barrier, but W-Asym does. Simply will not happen.
# 4. FedAvg exhibits high barrier, as well as W-Asym. This prompts more possibilities, in which _homogeneous_ client drift is assessed, meaning that solutions are certainly stochastic symmetries:
# 4a. FedAvg exhibits high barrier, but W-Asymmetry does not. According to Entezari (and logic), W-Asymmetry removes all symmetries, but in the previous scenario, the local solutions were not stochastic.
# 4b. FedAvg and W-Asym both exhibit high barrier. According to Entezari, W-Asymmetry does not remove all permutation symmetries.
# 4c. FedAvg and W-Asym both exhibit low barrier. Then stochastic client drift is not a problem, in contrast to the previous scenario.
# 4d. FedAvg does not exhibit high barrier, but W-Asymmetry does. This will not happen.

import os, sys
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Allow direct script execution from this subfolder while using package imports.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax, optax, argparse, shutil, json, multiprocessing as mp
from backend.fedflax import train, cast
from backend.models import ResNet
from backend.data import fetch_data, seed_worker
from backend.utils import top_5_err, err_fn as _err_fn, load_model, nnx_norm
from jax import numpy as jnp
from flax import nnx
from functools import reduce
from tqdm.auto import tqdm

def partial_aggregate(models, alpha):
    struct, params, rest = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)
    avg_params = jax.tree.map(lambda p: jnp.mean(p, axis=0), params)
    new_params = jax.tree.map(lambda p, ap: (1-alpha)*p + alpha*ap, params, avg_params)
    return nnx.merge(struct, new_params, rest)

if __name__ == "__main__":
    # Allow user-specified arguments
    parser = argparse.ArgumentParser(description="Analyze stability of a ResNet trained on ImageNet in a federated setting")
    parser.add_argument("--n_clients", type=int, default=4, help="Number of clients to simulate")
    parser.add_argument("--asymtype", type=str, default="", help="Type of asymmetry to use", choices=["", "wasym", "syre", "normweights", "dimexp"])
    parser.add_argument("--heterogeneous", action=argparse.BooleanOptionalAction, default=False, help="Type of client drift to induce")
    parser.add_argument("--key", type=int, default=42, help="Value of the random seed.")
    args = parser.parse_args()
    n_clients = args.n_clients
    # Fill kwargs
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
        asymkwargs["dimexp"] = 2
    if args.heterogeneous:
        skew = "label"
        beta = 1.
    else:
        skew = "overlap"
        beta = 0.

    # Get datasets
    ds_train = fetch_data(skew, beta=beta, n_clients=n_clients, dataset=1, n_classes=100, num_workers=8,
                          multiprocessing_context=mp.get_context("spawn"), persistent_workers=True, worker_init_fn=seed_worker)
    ds_test = fetch_data(skew, partition="test", beta=beta, n_clients=n_clients, dataset=1, n_classes=100, num_workers=4, prefetch_factor=1, 
                         multiprocessing_context=mp.get_context("spawn"), persistent_workers=True, worker_init_fn=seed_worker)
    ds_val = fetch_data(skew, partition="val", beta=beta, n_clients=n_clients, dataset=1, n_classes=100, num_workers=4,
                        multiprocessing_context=mp.get_context("spawn"), persistent_workers=True, worker_init_fn=seed_worker)

    # Get model
    model_g = ResNet(key=jax.random.key(args.key), layers=[2,2,2,2], dim_out=100, **asymkwargs)
    lr = optax.warmup_exponential_decay_schedule(1e-6, 3e-3, 1000, 2000, 0.9, end_value=1e-5)
    opt = nnx.Optimizer(
        model_g,
        optax.adam(learning_rate=lr),
        wrt=nnx.Param
    )

    # Get federated models at each round
    dirname = f"/data/bucket/traincombmodels/logs/checkpoints/imagenet100_{args.asymtype or 'base'}_{'heterogeneous' if args.heterogeneous else 'homogeneous'}_key{args.key}"
    os.makedirs(dirname, exist_ok=True)
    train(model_g, opt, ds_train, ell, ds_val, local_epochs=30, 
          max_patience=3, val_fn=top_5_err, rounds=10, n_clients=n_clients, ckpt_fp=os.path.join(dirname, "r_e_"))
    
    # Iterate over rounds to check stability
    log = {}
    paths = os.listdir(dirname)
    paths = map(lambda s: (
        int(s.split("_")[-2]), int(s.split("_")[-1].split(".")[0]), os.path.join(dirname, s)
    ), paths)
    paths = sorted(paths)
    for i, (r, epoch, models_path) in enumerate(paths):
        # not very last file    and    not last epoch    and    round is the same as previous file 
        if (i!=len(paths)-1) and (not epoch>paths[i+1][1]) and (epoch==0 or r==paths[i-1][0]):
            shutil.rmtree(models_path)
            continue
        else:
            print(f"Checking stability at round {r} (epoch {epoch})", end="")
            fl_models = load_model(lambda: cast(ResNet(layers=[2,2,2,2], dim_out=100, **asymkwargs), n_clients), models_path)
        # Check LMC
        err_fn = nnx.jit(nnx.vmap(_err_fn))
        lmc:dict[float, jax.Array] = {}
        for alpha in tqdm(jnp.linspace(0.,1.,10).tolist(), leave=False):
            models_agg = partial_aggregate(fl_models, alpha)
            err = reduce(lambda acc, b: acc + err_fn(models_agg,*b), ds_test, 0.) / len(ds_test)
            lmc[alpha] = err
        # Get instability measure inspired by Frankle
        max_alpha = max(lmc, key=lambda k: lmc[k].mean().item())
        instability = lmc[max_alpha] - lmc[0.]
        instability = instability.mean().item()
        print(f": instability found at alpha {max_alpha} = {instability*100}%")
        # Log results
        log[r] = instability
        json.dump(log, open(f"/data/bucket/traincombmodels/logs/stability_{args.asymtype or 'base'}_{'heterogeneous' if args.heterogeneous else 'homogeneous'}_key{args.key}.json", "w"))