import os
os.environ["XLA_FLAGS"] = " --xla_gpu_strict_conv_algorithm_picker=false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax, optax, multiprocessing as mp, argparse
from fedflax import train
from models import fetch_vit, AsymLinear, interleave
from data import fetch_data
from orbax import checkpoint
from utils import nnx_norm
from flax import nnx
from functools import reduce
from jax import numpy as jnp

# Load Google's ViT as backbone and attach head, mixing flax.linen and flax.nnx
class Classifier(nnx.Module):
    def __init__(self, key=jax.random.key(0), **asymkwargs):
        super().__init__()
        self.dimexp = asymkwargs.get("dimexp", 1)
        img_size = 224*self.dimexp
        self.backbone, self.bbparams = fetch_vit(img_size=img_size)
        self.bbparams = nnx.data(jax.tree.map(nnx.Param, self.bbparams))
        keys = jax.random.split(key, 3)
        self.fc1 = AsymLinear(768, 512, key=keys[0], **asymkwargs)
        self.fc2 = AsymLinear(512, 128, key=keys[1], **asymkwargs)
        self.fc3 = AsymLinear(128, 40, key=keys[2], **asymkwargs)

    def __call__(self, x, train=False):
        if self.dimexp>1: x = interleave(x, k=self.dimexp)
        x = self.backbone.apply({"params": self.bbparams}, x, train=train)
        x = x.reshape(x.shape[0], -1)
        x, norm = self.fc1(x)
        x = nnx.relu(x)
        x, norm = self.fc2(x, norm)
        x = nnx.relu(x)
        x, norm = self.fc3(x, norm)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ViT on CelebA with federated learning")
    parser.add_argument("--n_clients", type=int, default=4, help="Number of clients to simulate")
    parser.add_argument("--asymtype", type=str, default="", choices=["", "wasym", "syre", "normweights"], help="Type of symmetry elimination to use")
    args = parser.parse_args()
    n_clients = args.n_clients
    asymkwargs = {}
    ell = lambda m, mg, y, x: optax.sigmoid_binary_cross_entropy(m(x, train=True), y).mean()
    if args.asymtype == "wasym":
        asymkwargs["wasym"] = True
        asymkwargs["kappa"] = 1
    elif args.asymtype == "syre":
        asymkwargs["sigma"] = 1e-4
        ell = lambda m, mg, y, x: optax.sigmoid_binary_cross_entropy(m(x, train=True), y).mean() \
            + 1e-4*nnx_norm(nnx.state(m, nnx.Param), n_clients=n_clients).mean()
    elif args.asymtype == "normweights":
        asymkwargs["normweights"] = True
    elif args.asymtype == "dimexp":
        asymkwargs["dimexp"] = 1
    model_name = f"/data/bucket/traincombmodels/models/celeba_{args.asymtype or ('central' if n_clients==1 else 'base')}"

    # Model and optimizer
    model = Classifier(**asymkwargs)
    lr = optax.warmup_exponential_decay_schedule(1e-4, .1, 4000, 1000, .9, end_value=1e-5)
    opt = nnx.Optimizer(
        model,
        optax.chain(
            optax.masked(optax.adam(1e-5), lambda ptree: jax.tree.map_with_path(lambda path, _p: any("bbparams" in str(part) for part in path), ptree)),
            optax.masked(optax.adam(lr), lambda ptree: jax.tree.map_with_path(lambda path, _p: not any("bbparams" in str(part) for part in path), ptree))
        ),
        wrt=nnx.Param
    )

    # CelebA data
    ds_train = fetch_data(beta=1., dataset=3, n_clients=n_clients, skew="label", batch_size=64,
                          num_workers=6, multiprocessing_context=mp.get_context("spawn"))
    ds_val = fetch_data(beta=1., dataset=3, partition="val", n_clients=n_clients, skew="label", batch_size=64,
                        num_workers=2, multiprocessing_context=mp.get_context("spawn"))

    # Train
    err_fn = lambda m, y, x: 1-jnp.mean(round(nnx.sigmoid(m(x, train=False)))==y)
    models, rounds = train(
        model,
        opt,
        ds_train,
        ell, 
        ds_val,
        local_epochs="early",
        n_clients=n_clients,
        max_patience=3,
        rounds=1 if n_clients==1 else "early",
        val_fn=err_fn
    )

    # Save model
    _, state = nnx.split(models)
    with checkpoint.StandardCheckpointer() as cptr:
        cptr.save(os.path.abspath(model_name), state, force=True)

    # Load test data
    del ds_train, ds_val
    ds_test = fetch_data(beta=1., dataset=3, partition="test", n_clients=n_clients, skew="label",
                         num_workers=4, multiprocessing_context=mp.get_context("spawn"))

    # Evaluate local
    vtest_fn = nnx.jit(nnx.vmap(err_fn, in_axes=(0,0,0)))
    error = reduce(lambda acc, batch: acc + vtest_fn(models, *batch), ds_test, 0.) / len(ds_test)
    print(f"Local test error (local model): {error.mean().item()*100:.2f}%")

    # Evaluate global
    struct, state, rest = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)
    model = nnx.merge(struct, jax.tree.map(lambda p: p.mean(0), state), rest)
    vtest_fn = nnx.jit(nnx.vmap(err_fn, in_axes=(None,0,0)))
    error = reduce(lambda acc, batch: acc + vtest_fn(model, *batch), ds_test, 0.) / len(ds_test)
    print(f"Global test error (aggregated model): {error.mean().item()*100:.2f}%")
