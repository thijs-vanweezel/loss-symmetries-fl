# %%
from fedflax import train
from models import ResNet, ResNetAutoEncoder
from data import fetch_data
from utils import load_model, return_ce, mean_iou_err
import jax, optax, pickle
from flax import nnx

# %% [markdown]
# ## Fetch foundation model
# Trained using the imagenet script.

# %%
# Reload backbone and use as encoder
models = load_model(
    lambda: ResNet(layers=[2,2,2,2], dim_out=100), 
    "models/resnet18_central_imagenet100.pkl"
)
struct, params, rest = nnx.split(models, (nnx.Param, nnx.BatchStat), ...)
model = nnx.merge(
    struct,
    jax.tree.map(lambda p: p.mean(0), params),
    rest
)

# %% [markdown]
# ## Alternatively, fetch ViT-224 foundation model
# Google repo: https://github.com/google-research/vision_transformer

# %%
pass

# %% [markdown]
# ## Finetune using asymmetries
# Compare with and without asymmetries in the ResNetAutoEncoder.

# %%
# Cityscapes data
n_clients = 3
ds_train = fetch_data(beta=1., dataset=2, n_clients=n_clients, batch_size=32)
ds_val = fetch_data(beta=1., dataset=2, partition="val", n_clients=n_clients, batch_size=16)

# Autoencoder model for segmentation via image reconstruction
asymkwargs = {}
ae = ResNetAutoEncoder(backboneencoder=model, key=jax.random.key(43), **asymkwargs)

# Optimizer with lower lr for pretrained backbone
lr = optax.warmup_exponential_decay_schedule(1e-4, .1, 4000, 1000, .9, end_value=1e-5)
def opt_create(ae:ResNetAutoEncoder):
    return nnx.Optimizer(
        ae,
        optax.chain(
            optax.masked(optax.adamw(1e-4), lambda ptree: jax.tree.map_with_path(lambda path, _p: "backboneencoder" in path, ptree)),
            optax.masked(optax.adamw(lr), lambda ptree: jax.tree.map_with_path(lambda path, _p: not "backboneencoder" in path, ptree))
        )
    )

# Train
aes, rounds = train(
    ae,
    opt_create,
    ds_train,
    return_ce(0.), 
    ds_val,
    local_epochs="early",
    n_clients=n_clients,
    max_patience=3,
    rounds="early",
    val_fn=mean_iou_err
)

# Save decoder
state = nnx.state(aes, nnx.Not(nnx.PathContains("backboneencoder")))
pickle.dump(state, open("models/cs_rn18_decoder.pkl", "wb"))

# Reload and aggregate it
load_fn = lambda: ResNetAutoEncoder(
    backboneencoder=load_model(
        lambda: ResNet(layers=[2,2,2,2], dim_out=100), 
        "models/resnet18_central_imagenet100.pkl"
    ),
    **asymkwargs
)
aes = load_model(load_fn, "models/cs_rn18_autoencoder.pkl")


