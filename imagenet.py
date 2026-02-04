from fedflax import train, aggregate, get_updates
import jax, optax, pickle
from jax import numpy as jnp
from flax import nnx
from models import ResNet
from data import fetch_data
from functools import partial, reduce
from utils import opt_create, return_ce, top_5_err, functional_drift
n_clients = 4
asymmetries = {"wasym":True, "kappa": 1}

ds_train = fetch_data(beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=1000)
ds_val = fetch_data(partition="val", beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=1000)
ds_test = fetch_data(partition="test", beta=1., skew="label", dataset=1, n_clients=n_clients, n_classes=1000)

model_init = ResNet(jax.random.key(42), **asymmetries, dim_out=1000, layers=[3,4,6,3])
lr = optax.warmup_exponential_decay_schedule(1e-4, .1, 4000, 1000, .9, end_value=1e-5)
models, _ = train(model_init, opt_create(model_init, learning_rate=lr), ds_train, return_ce(0.), ds_val, local_epochs="early", rounds="early", max_patience=3, val_fn=top_5_err, n_clients=n_clients);
_struct, models_state = nnx.split(models)
pickle.dump(models_state, open("resnet34_syre_imagenet1000.pkl", "wb"))

# models_state = pickle.load(open("resnet34_baseline_imagenet1000.pkl", "rb"))
# struct, _abstract_state = nnx.split(nnx.eval_shape(lambda: ResNet(jax.random.key(42), layers=[3,4,6,3], dim_out=1000)))
# models = nnx.merge(struct, models_state)

updates = get_updates(model_init, models)
model_g = aggregate(model_init, updates)

vval_fn = nnx.jit(nnx.vmap(top_5_err, in_axes=(None,0,0)))
err_test = reduce(lambda e, batch: e + vval_fn(model_g, *batch), ds_test, 0.) / len(ds_test)
err_val = reduce(lambda e, batch: e + vval_fn(model_g, *batch), ds_val, 0.) / len(ds_val)
print('wasym')
print(f"err test, measured after aggregation: {err_test.mean()}, err val: {err_val.mean()}")
vval_fn = nnx.jit(nnx.vmap(top_5_err, in_axes=(0,0,0)))
err_sep_test = reduce(lambda e, batch: e + vval_fn(models, *batch), ds_test, 0.) / len(ds_test)
err_sep_val = reduce(lambda e, batch: e + vval_fn(models, *batch), ds_val, 0.) / len(ds_val)
print(f"error test, measured separately: {err_sep_test.mean()}, error val: {err_sep_val.mean()}")

updates_flat = jnp.concatenate(jax.tree.map(lambda x: jnp.reshape(x, (n_clients,-1)), jax.tree.leaves(updates)), axis=1)
update_g = updates_flat.mean(0)
print("angular drift: ", jnp.degrees(jnp.arccos(optax.losses.cosine_similarity(update_g, updates_flat)).mean()).item())

func_drift = functional_drift(models, ds_test)
print("functional drift: ", func_drift.mean().item())

l1 = jnp.mean(jnp.abs(update_g - updates_flat), axis=-1)
print("L1 drift: ", l1.mean().item())

print("True L1 drift: ", jnp.sum(jnp.abs(update_g - updates_flat), axis=-1).mean().item())
print("True L2 drift: ", jnp.sqrt(jnp.sum((update_g - updates_flat)**2, axis=-1)).mean().item())