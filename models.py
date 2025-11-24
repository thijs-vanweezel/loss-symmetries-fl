from flax import nnx
from jax import numpy as jnp
from itertools import chain
from flax.nnx.nn.linear import _conv_dimension_numbers
import jax

# Dimension expansion
@jax.vmap
def interleave(img):
    img = jnp.repeat(img, 2, axis=0)
    img = jnp.repeat(img, 2, axis=1)
    img = img.at[::2].set(.5)
    img = img.at[:, ::2].set(.5)
    return img

class AsymLinear(nnx.Linear):
    def __init__(self, key:jax.dtypes.prng_key, pfree:float=1., sigma:float=0., **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(rngs=nnx.Rngs(keys[0]), **kwargs)
        # Check if asymmetry is to be applied
        self.ssigma = sigma
        self.wasym = pfree<1.
        # Create params
        if self.wasym: self.wmask = jax.random.bernoulli(keys[1], p=pfree, shape=self.kernel.shape).astype(jnp.float32)
        if sigma>0. or self.wasym: self.randk = jax.random.normal(keys[2], self.kernel.shape)
        if sigma>0.: self.randb = jax.random.normal(keys[3], self.bias.shape)

    def __call__(self, inputs:jax.Array) -> jax.Array:
        kernel = self.kernel.value
        bias = self.bias.value
        # Apply SyRe (before wasym to avoid biasing the masked weights)
        if self.ssigma>0.:
            bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Apply w-asymmetry
        if self.wasym:
            kernel = kernel * self.wmask + (1-self.wmask) * self.randk # TODO: decrease std
        # Implementation directly copied from nnx.Linear
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

class AsymConv(nnx.Conv):
    def __init__(self, key:jax.dtypes.prng_key, pfree:float=1., sigma:float=0., **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(rngs=nnx.Rngs(keys[0]), **kwargs)
        # Check if asymmetry is to be applied
        self.ssigma = sigma
        self.wasym = pfree<1.
        # Create params
        if self.wasym: self.wmask = jax.random.bernoulli(keys[1], p=pfree, shape=self.kernel.shape[-1]).astype(jnp.float32)
        if sigma>0. or self.wasym: self.randk = jax.random.normal(keys[2], self.kernel.shape)
        if sigma>0.: self.randb = jax.random.normal(keys[3], self.bias.shape)

    def __call__(self, inputs:jax.Array) -> jax.Array:
        kernel = self.kernel.value
        bias = self.bias.value
        # Apply SyRe (before wasym to avoid biasing the masked weights)
        if self.ssigma>0.:
            bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Apply w-asymmetry
        if self.wasym:
            kernel = kernel * self.wmask + (1-self.wmask) * self.randk # TODO: decrease std
        # Implementation directly copied from nnx.Conv
        def maybe_broadcast(x: int | tuple[int, ...]):
            if isinstance(x, int):
                return (x,) * len(self.kernel_size)
            return tuple(x)
        y = self.conv_general_dilated(
            inputs,
            kernel,
            maybe_broadcast(self.strides),
            self.padding, # NOTE: restricted to "SAME" or "VALID"
            lhs_dilation=maybe_broadcast(self.input_dilation),
            rhs_dilation=maybe_broadcast(self.kernel_dilation),
            dimension_numbers=_conv_dimension_numbers(inputs.shape),
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )
        y += bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
        return y

# Used in resnet
class ResNetBlock(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, in_kernels:int, out_kernels:int, stride:int=1, pfree=1., sigma=0.):
        super().__init__()
        keys = jax.random.split(key, 5)
        self.stride = stride
        self.conv1 = AsymConv(
            keys[0],
            pfree,
            sigma,
            in_features=in_kernels,
            out_features=out_kernels,
            kernel_size=(3,3),
            strides=(stride,stride),
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm1 = nnx.BatchNorm(out_kernels, rngs=nnx.Rngs(keys[1]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv2 = AsymConv(
            keys[2],
            pfree,
            sigma,
            in_features=out_kernels,
            out_features=out_kernels,
            kernel_size=(3,3),
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm2 = nnx.BatchNorm(out_kernels, rngs=nnx.Rngs(keys[3]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        if stride>1 and in_kernels!=out_kernels:
            self.id_conv = AsymConv(keys[4], pfree, sigma, in_features=in_kernels, out_features=out_kernels, 
                                    kernel_size=(1,1), strides=(stride, stride), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

    def __call__(self, x, train=True):
        res = x if self.stride==1 else self.id_conv(x)
        x = self.conv1(x)
        x = self.norm1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.norm2(x, use_running_average=not train)
        x = res+x
        return x

# Resnet for ImageNet ([3,4,6,3] for 34 layers, [2,2,2,2] for 18 layers)
class ResNet(nnx.Module): # TODO: 36/(2**5) is a small shape for conv
    def __init__(self, key:jax.dtypes.prng_key, block=ResNetBlock, layers=[2,2,2,2], kernels=[64,128,256,512], channels_in=1, dim_out=9, dimexp=False, pfree=1., sigma=0., **kwargs):
        super().__init__(**kwargs)
        # Dimension expansion params
        self.dimexp = dimexp
        # Keys
        keys = jax.random.split(key, sum(layers)+2)
        # Layers
        self.conv = AsymConv(keys[0], pfree, sigma, in_features=channels_in, out_features=64, kernel_size=(7,7), strides=(2,2), padding="SAME", param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(block(keys[j+(i*j)], k_in, k_out, stride=s, pfree=pfree, sigma=sigma))
        self.fc = AsymLinear(keys[-1], pfree, sigma, in_features=kernels[-1]+3, out_features=dim_out, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)

    def __call__(self, x, z, train=True):
        # Apply asymmetries (syre before wasym to avoid biasing the masks)
        x = x if not self.dimexp else interleave(x)
        # Forward pass
        x = self.conv(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        for layer in self.layers:
            x = layer(x, train=train)
        x = jnp.mean(x, axis=(1,2))
        x = self.fc(jnp.concatenate([x, z], axis=-1))
        return x

# LeNet-5 for 36X60 images + 3 auxiliary features
class LeNet(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, dimexp=False, pfree=1., dim_out=9, sigma=0.):
        super().__init__()
        # Dimension expansion params
        flat_shape = 15*27 if dimexp else 6*12
        self.dimexp = dimexp
        # Layers
        keys = jax.random.split(key, 5)
        self.conv1 = AsymConv(keys[0], pfree, sigma, in_features=1, out_features=8, kernel_size=(4,4), padding="VALID")
        self.conv2 = AsymConv(keys[1], pfree, sigma, in_features=8, out_features=16, kernel_size=(4,4), padding="VALID")
        self.fc1 = AsymLinear(keys[2], pfree, sigma, in_features=flat_shape*16+3, out_features=128)
        self.fc2 = AsymLinear(keys[3], pfree, sigma, in_features=128, out_features=64)
        self.fc3 = AsymLinear(keys[4], pfree, sigma, in_features=64, out_features=dim_out)
    
    def __call__(self, x, z, train=None):
        # Apply dimension expansion if necessary
        x = x if not self.dimexp else interleave(x)
        # Forward pass
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.concatenate([x, z], axis=-1)
        x = self.fc1(x)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        x = self.fc3(x)
        return x

def teleport_lenet(model, key, tau_range=.1):
    assert isinstance(model, LeNet), "Teleportation must be (slightly) adjusted for models other than LeNet-5"
    # Extract params
    model_tree = nnx.to_tree(model)
    params, struct = jax.tree.flatten(model_tree)

    # Assign tau to the output channels of each kernel
    def random(size):
        nonlocal key
        _, key = jax.random.split(key)
        return jax.random.uniform(key, size, minval=1.-tau_range, maxval=1.+tau_range)
    tau = jax.tree.map(lambda p: random(p.shape[-1]), params[::2])
    # "Input neurons" require tau=1
    tau_a = [jnp.ones(1)] + jax.tree.leaves(tau) 
    tau_a[2] = jnp.tile(tau_a[2], (6,12,1)).flatten() # Transition from conv to flattened fc
    tau_a[2] = jnp.concat([tau_a[2], jnp.ones(3)])  # Account for auxiliary input neurons
    # Output neurons require tau=1
    tau_b = jax.tree.leaves(tau)[:-1] + [jnp.ones(16)]
    coefs_kernel = jax.tree.map(lambda t_a, t_b: jnp.outer(1/t_a, t_b), tau_a[:-1], tau_b)

    # Due to bias requiring tau=1, this is equivalent to tau + ones(channels_out)
    coefs_bias = tau_b

    # Interleave kernel and bias coefs, as they are originally
    coefs = list(chain(*zip(coefs_bias, coefs_kernel)))
    # Teleport the model
    params_tele = jax.tree.map(lambda p, c: c*p, params, coefs)
    # Rebuild the model
    return nnx.from_tree(jax.tree.unflatten(struct, params_tele))