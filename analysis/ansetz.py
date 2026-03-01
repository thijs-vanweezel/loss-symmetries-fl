import os, sys
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import jax, optax, argparse
from jax import numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from models import mask_linear_densest

parser = argparse.ArgumentParser()
parser.add_argument("--wasym", action=argparse.BooleanOptionalAction, default=False, 
                    help="Whether to apply W-Asymmetry for eliminating permutation symmetries")
args = parser.parse_args()
wasym:bool = args.wasym

plt.style.use("seaborn-v0_8-pastel")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"""
        \usepackage{amsmath, amssymb}
        \usepackage{mathptmx}
    """
})
fig, ax = plt.subplots(figsize=(6,6), dpi=400)

# The himmelblau function, of which we want to find one of the four minima
def himmelblau(x):
    total = 0.
    for i in range(0, len(x)//2):
        total += (x[2*i]**2 + x[2*i+1] - 11)**2 + (x[2*i] + x[2*i+1]**2 - 7)**2 
    return total

keys = jax.random.split(jax.random.key(42), 4)
for inputkey in range(50):
    # Initialize parameters
    paramstree = {"fc1": {"w": jax.nn.initializers.he_normal()(keys[0], (10,8)), "b": jnp.zeros((8,))}, 
                "fc2": {"w": jax.nn.initializers.he_normal()(keys[1], (8,2)), "b": jnp.zeros((2,))}}
    if wasym:
        mask = {"fc1": mask_linear_densest(*paramstree["fc1"]["w"].shape), "fc2": mask_linear_densest(*paramstree["fc2"]["w"].shape)}
        randk = {"fc1": jax.random.normal(keys[2], (10,8)), "fc2": jax.random.normal(keys[3], (8,2))}
    else:
        mask = {"fc1": jnp.ones((10,8)), "fc2": jnp.ones((8,2))}
        randk = {"fc1": jnp.zeros((10,8)), "fc2": jnp.zeros((8,2))}
    # Setup two-layer MLP
    fixed_input = jax.random.uniform(jax.random.key(inputkey), (10,))
    call = lambda ptree: jnp.dot(
        jax.nn.relu(jnp.dot(fixed_input, ptree["fc1"]["w"]*mask["fc1"] + randk["fc1"]*(1-mask["fc1"]))\
            +ptree["fc1"]["b"]), 
        ptree["fc2"]["w"]*mask["fc2"] + randk["fc2"]*(1-mask["fc2"])
    )+ptree["fc2"]["b"]
    # Initialize optimizer
    opt = optax.adam(1e-3)
    opt_state = opt.init(paramstree)
    # Precalculate shift so that starting point is where there is equal directional pull between minima
    shift = jnp.asarray([-0.48693585, -0.5074425]) - call(paramstree)

    # Train step is straightforward
    @jax.jit
    def train_step(params, opt_state):
        (loss, pred), grad = jax.value_and_grad(lambda ptree: (himmelblau(pred:=call(ptree) + shift), pred), has_aux=True)(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    # Train
    trajectory = []
    for step in range(300):
        paramstree, opt_state, loss, pred = train_step(paramstree, opt_state)
        trajectory.append(pred)
    
    # Plot trajectory
    trajectory = jnp.asarray(trajectory)
    ax.plot(*trajectory.T, color="gray", zorder=1, alpha=.65)

# Plot the himmelblau function
loss_grid = jnp.empty((100,100))
alpha_grid = jnp.linspace(-6, 6, 100)
beta_grid = jnp.linspace(-6, 6, 100)
for i, alpha_ in enumerate(alpha_grid):
    for j, beta_ in enumerate(beta_grid):
        loss_grid = loss_grid.at[i,j].set(himmelblau([alpha_, beta_]))
norm = mpl.colors.LogNorm(vmin=1e-3, vmax=1e3)
plot = ax.pcolormesh(
    alpha_grid,
    beta_grid,
    loss_grid,
    shading="auto",
    cmap="inferno",
    norm=norm,
    zorder=0
)
bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plot.cmap), ax=ax, ticks=None)
fig.savefig(f"himmelblau_{wasym=}.png", bbox_inches="tight")