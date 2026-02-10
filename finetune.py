{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "261b7e2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "from fedflax import train\n",
    "from models import fetch_vit, NonTrainable, AsymLinear\n",
    "from data import fetch_data\n",
    "from utils import load_model, save_model, nnx_norm\n",
    "import jax, optax, pickle\n",
    "from flax import nnx\n",
    "from functools import reduce\n",
    "from jax import numpy as jnp\n",
    "\n",
    "n_clients = 4\n",
    "asymkwargs = {}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "01ff9fa3",
   "metadata": {},
   "source": [
    "## Load Google's ViT as backbone and attach head\n",
    "Mix flax.linen and flax.nnx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c272767e",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Classifier(nnx.Module):\n",
    "    def __init__(self, key=jax.random.key(0), **asymkwargs):\n",
    "        super().__init__()\n",
    "        self.backbone, self.bbparams = fetch_vit()\n",
    "        self.bbparams = jax.tree.map_with_path(lambda path, p: nnx.Param(p) if not any(\"LayerNorm\" in part.key for part in path) else nnx.BatchStat(p), self.bbparams)\n",
    "        self.head = AsymLinear(768, 40, key=key, **asymkwargs)\n",
    "\n",
    "    def __call__(self, x, train=False):\n",
    "        x = self.backbone.apply({\"params\": self.bbparams}, x, train=train)\n",
    "        x = x.reshape(x.shape[0], -1)\n",
    "        x = self.head(x)\n",
    "        return x\n",
    "ell = lambda m, mg, y, x: optax.sigmoid_binary_cross_entropy(m(x, train=True), y).mean() + nnx_norm(nnx.state(m, nnx.Param), n_clients=n_clients)\n",
    "\n",
    "# Optimizer which applies only to nnx.Param, leaving linen backbone untouched\n",
    "model = Classifier(**asymkwargs)\n",
    "lr = optax.warmup_exponential_decay_schedule(1e-4, .1, 4000, 1000, .9, end_value=1e-5)\n",
    "opt = nnx.Optimizer(\n",
    "    model,\n",
    "    optax.chain(\n",
    "        optax.masked(optax.adam(1e-5), lambda ptree: jax.tree.map_with_path(lambda path, _p: any(\"bbparams\" in str(part) for part in path), ptree)),\n",
    "        optax.masked(optax.adam(lr), lambda ptree: jax.tree.map_with_path(lambda path, _p: not any(\"bbparams\" in str(part) for part in path), ptree))\n",
    "    )\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9ad6551",
   "metadata": {},
   "source": [
    "## Finetune using asymmetries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3bfe7e20",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cityscapes data\n",
    "ds_train = fetch_data(beta=1., dataset=3, n_clients=n_clients, batch_size=16, skew=\"label\")\n",
    "ds_val = fetch_data(beta=1., dataset=3, partition=\"val\", n_clients=n_clients, batch_size=16, skew=\"label\")\n",
    "\n",
    "# Train\n",
    "err_fn = lambda m, y, x: jnp.mean((m(x, train=False)>0)==(y>0.5))\n",
    "models, rounds = train(\n",
    "    model,\n",
    "    opt,\n",
    "    ds_train,\n",
    "    ell, \n",
    "    ds_val,\n",
    "    local_epochs=\"early\",\n",
    "    n_clients=n_clients,\n",
    "    max_patience=3,\n",
    "    rounds=\"early\",\n",
    "    val_fn=err_fn\n",
    ")\n",
    "\n",
    "# Save decoder\n",
    "save_model(models, \"models/celeba_base.pkl\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "58c6b15d",
   "metadata": {},
   "source": [
    "## Reload & evaluate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ef3ef29",
   "metadata": {},
   "outputs": [],
   "source": [
    "models = load_model(lambda: Classifier(**asymkwargs), \"models/celeba_base.pkl\")\n",
    "struct, state, rest = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)\n",
    "model = nnx.merge(struct, jax.tree.map(lambda p: p.mean(0), state), rest)\n",
    "ds_val = fetch_data(beta=1., dataset=3, partition=\"test\", n_clients=1, batch_size=16)\n",
    "vtest_fn = nnx.jit(err_fn)\n",
    "error = reduce(lambda acc, model: acc + vtest_fn(model, ds_val), model, 0.) / len(models)\n",
    "print(f\"Global test error (aggregated model): {error*100:.2f}%\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
