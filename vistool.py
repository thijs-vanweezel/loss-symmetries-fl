import jax, scipy, numpy as np, matplotlib as mpl, pickle
from jax import numpy as jnp
from flax import nnx
from functools import reduce
from matplotlib import pyplot as plt
from utils import load_model
from models import LeNet
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

def get_reconstruct(model, client_dim=True):
    # Function to reconstruct model from flat params
    params, struct = jax.tree.flatten(nnx.to_tree(model))
    shapes = [p.shape[int(client_dim):] for p in params]+[None]
    def reconstruct(flat_params):
        # Indices of kernels in flat vector
        slices = [slice
            (sum(map(lambda s: np.prod(s), shapes[:i])),
            sum(map(lambda s: np.prod(s), shapes[:i+1])))
        for i in range(len(shapes)-1)]
        # Get kernels as correct shape
        params = [flat_params[sl] for sl in slices]
        params = [jnp.array(p).reshape(s) for p, s in zip(params, shapes)]
        # Revert to model
        return nnx.from_tree(jax.tree.unflatten(struct, params))
    return reconstruct

def compute_surface(alpha_grid, beta_grid, pca, reconstruct, ds, val_fn, interpolate=True):
    val_fn = nnx.jit(nnx.vmap(val_fn, in_axes=(None,0,0,0)))
    errs = jnp.zeros((len(alpha_grid), len(beta_grid)))
    for i, alpha_ in enumerate(alpha_grid):
        for j, beta_ in enumerate(beta_grid):
            # Reconstruct the model for some point in the 2d plane
            params = pca.inverse_transform(jnp.array([[alpha_, beta_]])).reshape(-1)
            model = reconstruct(params)
            # Compute accuracy
            score = reduce(lambda score, b: score + val_fn(model,*b), ds, 0.) / len(ds)
            errs = errs.at[i,j].set(score.mean()) # mean over clients' data, i.e., global data error rate
    
    if interpolate:
        alpha_grid_fine = np.linspace(alpha_grid.min(), alpha_grid.max(), 1000)
        beta_grid_fine = np.linspace(beta_grid.min(), beta_grid.max(), 1000)
        mesh = np.meshgrid(alpha_grid_fine, beta_grid_fine)
        errs = scipy.interpolate.interpn((alpha_grid, beta_grid), errs, np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T, method='cubic').reshape(1000,1000)
        return errs, alpha_grid_fine, beta_grid_fine
    return errs

def plot_trajectory(errs, pca, fps, alpha_grid, beta_grid, filename, labels=True, n_clients=4):
    # Plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=700)
    fig.tight_layout()
    ax.set_box_aspect(1)
    # Plot the level sets, exponential scale
    norm = mpl.colors.LogNorm()
    plot = ax.pcolormesh(
        alpha_grid,
        beta_grid,
        errs,
        shading="auto",
        cmap="inferno",
        norm=norm,
        zorder=0
    )
    if labels:
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plot.cmap), ax=ax, shrink=0.8, ticks=None)
        bar.set_label("Error")
    # Plot the model paths (assume params are sorted by optimization step)
    prev_coords = None
    r = 0
    for fp in fps:
        # Flatten parameters
        params = pickle.load(open(fp, "rb"))
        params = jax.tree.reduce(lambda acc, p: jnp.concatenate([acc, p.reshape(p.shape[0],-1)], axis=1), params, jnp.empty((n_clients,0)))
        coords = pca.transform(params) - 1.
        # Plot as line
        if prev_coords is not None:
            ax.add_collection(mpl.collections.LineCollection(
                zip(prev_coords, coords), 
                colors=[f"C{c+1}" for c in range(coords.shape[0])],
                zorder=1
            ))
        # Plot points
        aggregate = jnp.allclose(coords[0], coords[1])
        if aggregate:
            ax.annotate(r, xy=coords[0]+.07, c="w", fontsize=10)
            r += 1; lw = 1; size = 10; colors = "black"
        else: 
            lw = 0; size = 7
            colors = [f"C{c+1}" for c in range(coords.shape[0])]
        ax.scatter(coords[:,0], coords[:,1], c=colors, s=size, linewidths=lw, edgecolors="k", zorder=2)
        prev_coords = coords
    # Display accuracy of final aggregated model
    coords_discrete = jnp.argmin(jnp.abs(alpha_grid - coords[0,0])), jnp.argmin(jnp.abs(beta_grid - coords[0,1]))
    ax.annotate(f"{int(errs[coords_discrete])}°", xy=coords_discrete, c="k", fontsize=25)
    # Show
    if labels:
        handles, labels = ax.get_legend_handles_labels()
        handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="black", label="Aggr point")
        ax.legend(handles + [handle], labels + ["Aggr point"], loc="upper left")
    plt.xticks([])
    plt.yticks([])
    fig.savefig(filename, bbox_inches="tight")
