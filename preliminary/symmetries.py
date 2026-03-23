from matplotlib import pyplot as plt
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
import jax
from jax import numpy as jnp

def mlp(x, w1, b1, w2):
    assert x.shape == (1,) and w1.shape == (1, 2) and b1.shape == (2,) and w2.shape == (2, 1)
    return jax.nn.tanh(x @ w1 + b1) @ w2

def make_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return (
        jax.random.normal(k1, (1, 2)),
        jax.random.normal(k2, (2,)),
        jax.random.normal(k3, (2, 1)),
    )

def forward(params, xs):
    return jax.vmap(mlp, in_axes=(0, None, None, None))(xs, *params)

x = jnp.linspace(-4, 4, 100)[:, None]
y = jnp.sin(x)

fn1_params = make_params(jax.random.key(7))
opt1 = jax.jit(jax.value_and_grad(lambda p: jnp.square(forward(p, x[:50]) - y[:50]).mean()))
for _ in range(20_000):
    loss, grad = opt1(fn1_params)
    fn1_params = jax.tree.map(lambda p, g: p - 1e-3 * g, fn1_params, grad)

fn2_params = (jnp.flip(fn1_params[0], axis=1), jnp.flip(fn1_params[1], axis=0), jnp.flip(fn1_params[2], axis=0))

fn3_params = make_params(jax.random.key(7))
opt3 = jax.jit(jax.value_and_grad(lambda p: jnp.square(forward(p, x[50:]) - y[50:]).mean()))
for _ in range(20_000):
    loss, grad = opt3(fn3_params)
    fn3_params = jax.tree.map(lambda p, g: p - 1e-3 * g, fn3_params, grad)

fn1_out = forward(fn1_params, x)
fn2_out = forward(fn2_params, x)
fn3_out = forward(fn3_params, x)
loss1 = float(jnp.square(fn1_out - y).mean())
loss2 = float(jnp.square(fn2_out - y).mean())
loss3 = float(jnp.square(fn3_out - y).mean())
print(f"loss1={loss1:.4f}  loss2={loss2:.4f}  loss3={loss3:.4f}")

fig = plt.figure(figsize=(7, 5))
ax = fig.add_axes([0.05, 0.05, 0.75, 0.9], projection="3d")

ax.zaxis.set_rotate_label(False)

curves = [
    ("fn1", fn1_out.squeeze(), 0.0, "C0"),
    ("fn2", fn2_out.squeeze(), 1.0, "C1"),
    ("fn3", fn3_out.squeeze(), 2.0, "C2"),
]
xv = x.squeeze()
ax.scatter(jnp.full_like(xv, 3.0)[::2], xv[::2], y.squeeze()[::2],
           color="grey", label="target", s=10)
for name, out, z0, color in curves:
    ax.plot(jnp.full_like(xv, z0), xv, out, color=color, label=name, linewidth=2)

ax.set_xlabel("Parameters", labelpad=8, fontsize=11)
ax.set_ylabel("Input",      labelpad=8, fontsize=11)
# ax.set_zlabel(r"Output $f(x)$",     labelpad=20, rotation=0)  # horizontal, pushed out
fig.text(0.80, 0.55, "Output", rotation=90, va="center", fontsize=11)


ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels([r"$\theta_1$", r"$\theta_2$", r"$\theta_3$", r"$y$"], fontsize=11)
ax.view_init(elev=30, azim=-65)
fig.savefig("symmetries.png", dpi=800)