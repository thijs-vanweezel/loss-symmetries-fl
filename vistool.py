import jax, scipy, numpy as np, matplotlib as mpl
from jax import numpy as jnp
from flax import nnx
from functools import reduce
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-pastel")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.sans-serif": ["Helvetica"],
    "text.latex.preamble": r"""
        \usepackage{amsmath, amssymb}
        \usepackage{mathptmx}  % Safe fallback for Times + math
    """
})

# Utils
acc_fn = nnx.jit(nnx.vmap(lambda m,x,z,y: (m(x,z,train=False).argmax(-1)==y.argmax(-1)).mean(), in_axes=(None,0,0,0)))

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

def compute_surface(alpha_grid, beta_grid, pca, reconstruct, ds, acc_fn=acc_fn, interpolate=True):
    errs = jnp.zeros((len(alpha_grid), len(beta_grid)))
    for i, alpha_ in enumerate(alpha_grid):
        for j, beta_ in enumerate(beta_grid):
            # Reconstruct the model for some point in the 2d plane
            params = pca.inverse_transform(jnp.array([[alpha_, beta_]])).reshape(-1)
            model = reconstruct(params)
            # Compute accuracy
            acc = reduce(lambda acc, b: acc + acc_fn(model,*b), ds, 0.) / len(ds)
            errs = errs.at[i,j].set(1-acc.mean()) # mean over clients' data, i.e., global data error rate
    
    if interpolate:
        alpha_grid_fine = np.linspace(alpha_grid.min(), alpha_grid.max(), 1000)
        beta_grid_fine = np.linspace(beta_grid.min(), beta_grid.max(), 1000)
        mesh = np.meshgrid(alpha_grid_fine, beta_grid_fine)
        errs = scipy.interpolate.interpn((alpha_grid, beta_grid), errs, np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T, method='cubic').reshape(1000,1000)
        return errs, alpha_grid_fine, beta_grid_fine
    return errs

def plot_trajectory(errs, model_idx, filename, epochs, reduced_params, alpha_grid, beta_grid, labels=True):
    # Plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    fig.tight_layout()
    ax.set_box_aspect(1)
    # Plot the level sets, exponential scale
    norm = mpl.colors.Normalize(vmin=.5, vmax=1.) # TODO: 0.5 error is pretty bad
    plot = ax.pcolormesh(
        alpha_grid,
        beta_grid,
        errs,
        shading="auto",
        cmap="cividis",
        norm=norm,
    )
    if labels:
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plot.cmap), ax=ax, shrink=0.8, ticks=None)
        bar.set_label("Accuracy")
    # Plot the model paths (assume params are sorted by optimization step)
    for i in jnp.unique(model_idx):
        idx = jnp.where(model_idx==i)[0]
        # Path
        ax.plot(reduced_params[idx,0], reduced_params[idx,1], c=f"C{i.item()+1}", label=f"Client {i.item()}", lw=1)
        # Color uniformly at aggregation points
        colors = [f"C{c.item()}" for c in jnp.array([i+1]*len(idx))*(1-jnp.maximum(1-jnp.arange(len(idx))%epochs, 0))]
        # Points
        ax.scatter(reduced_params[idx,0], reduced_params[idx,1], c=colors, s=10)
    # Show
    if labels:
        handles, labels = ax.get_legend_handles_labels()
        handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="C0", label="Aggr point")
        ax.legend(handles + [handle], labels + ["Aggr point"], loc="upper left")
    plt.xticks([])
    plt.yticks([])
    fig.savefig(filename)