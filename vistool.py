import optax, scipy, numpy as np, matplotlib as mpl
from jax import numpy as jnp
from flax import nnx
from functools import reduce
from matplotlib import pyplot as plt
plt.style.use("seaborn-v0_8-pastel")

def pca_plot(pca, model_idx, ds, reconstruct, filename, epochs, reduced_params=None, all_params=None, beta_min=None, alpha_min=None, alpha_max=None, beta_max=None, points=20, levels=15, type="density", labels=True, errs=None):
    if reduced_params is None:
        # Perform pca
        reduced_params = pca.transform(all_params)

    # Define grid
    alpha_min = alpha_min or reduced_params[:,0].min()-.1
    alpha_max = alpha_max or reduced_params[:,0].max()+.1
    alpha_grid = jnp.linspace(alpha_min, alpha_max, points)
    beta_min = beta_min or reduced_params[:,1].min()-.1
    beta_max = beta_max or reduced_params[:,1].max()+.1
    beta_grid = jnp.linspace(beta_min, beta_max, points)

    # For sampled points on the 2d plane, compute the accuracy
    if errs is None:
        acc_fn = nnx.jit(nnx.vmap(lambda m,x,z,y: (m(x,z,train=False).argmax(-1)==y.argmax(-1)).mean(), in_axes=(None,0,0,0)))
        errs = jnp.zeros((points, points))
        for i, alpha_ in enumerate(alpha_grid):
            for j, beta_ in enumerate(beta_grid):
                # Reconstruct the model for some point in the 2d plane
                params = pca.inverse_transform(jnp.array([[alpha_, beta_]])).reshape(-1)
                model = reconstruct(params)
                # Compute accuracy
                acc = reduce(lambda acc, b: acc + acc_fn(model,*b), ds, 0.) / len(ds)
                errs = errs.at[i,j].set(1-acc.mean()) # mean over clients' data, i.e., global data error rate

    # Plot
    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
    fig.tight_layout()
    ax.set_box_aspect(1)
    # Plot the level sets, exponential scale
    maxi, mini = errs.max(), errs.min()
    norm = mpl.colors.Normalize(vmin=.5, vmax=1.) # TODO: 0.5 error is pretty bad
    if type=="density":
        alpha_grid_fine = np.linspace(alpha_grid.min(), alpha_grid.max(), 1000) # using alpha_min and alpha_max directly causes issues with pcolormesh
        beta_grid_fine = np.linspace(beta_grid.min(), beta_grid.max(), 1000)
        mesh = np.meshgrid(alpha_grid_fine, beta_grid_fine)
        plot = ax.pcolormesh(
            alpha_grid_fine,
            beta_grid_fine,
            scipy.interpolate.interpn((alpha_grid, beta_grid), errs, np.vstack([mesh[0].ravel(), mesh[1].ravel()]).T, method='cubic').reshape(1000,1000),
            shading="auto",
            cmap="magma", 
            norm=norm,
        )
    elif type=="contour":
        plot = ax.contour(
        alpha_grid,
        beta_grid,
        errs.T,
        levels=jnp.log(jnp.linspace(jnp.exp(mini), jnp.exp(maxi), levels)),
        cmap="magma",
        norm=norm
        )
    if labels:
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plot.cmap), ax=ax, shrink=0.8, ticks=None if type=="density"else plot.levels)
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
    # ax.set_title("Accuracy surface along top two principal directions in parameter space")
    if labels:
        handles, labels = ax.get_legend_handles_labels()
        handle = plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="C0", label="Aggr point")
        ax.legend(handles + [handle], labels + ["Aggr point"], loc="upper left")
    plt.xticks([])
    plt.yticks([])
    fig.savefig(filename)

    return errs