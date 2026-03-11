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
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import jax, optax, argparse, shutil, json
from fedflax import train, cast
from models import ResNet
from data import fetch_data, seed_worker
from utils import return_ce, top_5_err, err_fn, load_model
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
    parser.add_argument("--wasym", action=argparse.BooleanOptionalAction, default=False, help="Whether to use W-Asymmetry or not")
    parser.add_argument("--drift-type", type=str, default="heterogeneous", choices=["heterogeneous", "homogeneous"], help="Type of client drift to induce")
    args = parser.parse_args()
    n_clients = args.n_clients
    # Fill kwargs
    asymkwargs = {}
    asymkwargs["wasym"] = args.wasym
    asymkwargs["kappa"] = 1
    if args.drift_type == "heterogeneous":
        skew = "label"
        beta = 1.
    else:
        skew = "overlap"
        beta = 0.

    # Get datasets
    ds_train = fetch_data(skew, beta=beta, n_clients=n_clients, dataset=1, n_classes=100)
    ds_test = fetch_data(skew, partition="test", beta=beta, n_clients=n_clients, dataset=1, n_classes=100)
    ds_val = fetch_data(skew, partition="val", beta=beta, n_clients=n_clients, dataset=1, n_classes=100)

    # Get model
    model_g = ResNet(key=jax.random.key(42), layers=[2,2,2,2], dim_out=100, **asymkwargs)
    lr = optax.warmup_exponential_decay_schedule(1e-6, 3e-3, 1000, 2000, 0.9, end_value=1e-5)
    opt = nnx.Optimizer(
        model_g,
        optax.adam(learning_rate=lr),
        wrt=nnx.Param
    )

    # Get federated models at each round
    ckpt_fp = f"analysis/checkpoints/imagenet100{'wasym' if args.wasym else 'fedavg'}_{args.drift_type}"
    train(model_g, opt, ds_train, return_ce(0.), ds_val, local_epochs="early", 
                         max_patience=3, val_fn=top_5_err, rounds=10, n_clients=n_clients, 
                         ckpt_fp=ckpt_fp)
    
    # Iterate over rounds to check stability
    log = {}
    for models_path in filter(lambda fp: fp.startswith(os.path.split(ckpt_fp)[-1]), os.listdir("analysis/checkpoints/")):
        *_, r, epoch = models_path.split("_")
        if not epoch=="0":
            shutil.rmtree(os.path.join("analysis/checkpoints/", models_path))
            continue
        else:
            fl_models = load_model(
                lambda: cast(ResNet(layers=[2,2,2,2], dim_out=100), n_clients), 
                os.path.join("analysis/checkpoints/", models_path)
            )
        # Check LMC
        err_fn = nnx.jit(nnx.vmap(err_fn))
        lmc:dict[float, jax.Array] = {}
        for alpha in tqdm(jnp.linspace(0.,1.,30).tolist(), leave=False):
            models_agg = partial_aggregate(fl_models, alpha)
            err= reduce(lambda acc, b: acc + err_fn(models_agg,*b), ds_test, 0.) / len(ds_test)
            lmc[alpha] = err
        # Get instability measure inspired by Frankle
        max_alpha = max(lmc, key=lambda k: lmc[k].mean().item())
        instability = lmc[max_alpha] - lmc[0.].mean()
        instability = instability.mean().item()
        # Log results
        log[r] = instability
        json.dump(log, open(f"analysis/stability_{'wasym' if args.wasym else 'fedavg'}_{args.drift_type}.json", "w"))