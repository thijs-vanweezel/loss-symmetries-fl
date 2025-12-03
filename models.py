from flax import nnx
from jax import numpy as jnp
from itertools import chain, combinations
from flax.nnx.nn.linear import _conv_dimension_numbers
import jax

# Dimension expansion
@jax.vmap
def interleave(img, fill_value=.5):
    img = jnp.repeat(img, 2, axis=0)
    img = jnp.repeat(img, 2, axis=1)
    img = img.at[::2].set(fill_value)
    img = img.at[:, ::2].set(fill_value)
    return img

# Creates mask with the maximum number of non-zero weights while ensuring that each row is unique
# https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_mlp.py#L224
def mask_linear_densest(in_dim, out_dim, **kwargs):
    mask = jnp.ones((in_dim, out_dim), **kwargs)
    row_idx = 1
    for num_zeros in range(1, in_dim):
        for cols_idx in combinations(range(in_dim), num_zeros):
            mask = mask.at[cols_idx, row_idx].set(0.)
            row_idx += 1
            if row_idx >= out_dim:
                return mask
def mask_linear_random(in_dim, out_dim, pfix, key, **kwargs):
    mask = jnp.ones((in_dim, out_dim), **kwargs)
    nfix = max(1, int(in_dim**(1/4))) #int(pfix*in_dim)
    for row_idx, key in zip(range(out_dim), jax.random.split(key, out_dim)):
        zeros_in_row = jax.random.permutation(key, jnp.arange(in_dim))[:nfix]
        mask = mask.at[zeros_in_row, row_idx].set(0)
    return mask

# W-Asymmetry implementation consistent with https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_mlp.py#L169 # TODO: is it strange that updates for masked weights are not zero?
# SyRe implementation consistent with https://github.com/xu-yz19/syre/blob/main/MLP.ipynb
# And kernel normalization consistent with https://github.com/o-laurent/bayes_posterior_symmetry_exploration/blob/main/symmetries/scale_resnet.py#L166
class AsymLinear(nnx.Linear):
    def __init__(self, in_features:int, out_features:int, key:jax.dtypes.prng_key, wasym:str|None=None, 
                 kappa:float=1., sigma:float=0., orderbias:bool=False, normweights:bool=False, **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(in_features, out_features, rngs=nnx.Rngs(keys[0]), use_bias=True, **kwargs)
        # Check if asymmetry is to be applied
        self.ssigma = sigma
        self.wasym = bool(wasym)
        self.kappa = kappa
        self.orderbias = orderbias
        self.normweights = normweights
        # Create W-Assymmetry and SyRe params
        if wasym=="densest": self.wmask = mask_linear_densest(*self.kernel.shape, dtype=self.param_dtype)
        elif wasym=="random": self.wmask = mask_linear_random(*self.kernel.shape, key=keys[1], pfix=1/3, dtype=self.param_dtype)
        if sigma>0. or self.wasym: self.randk = jax.random.normal(keys[2], self.kernel.shape, dtype=self.param_dtype)
        if sigma>0.: self.randb = jax.random.normal(keys[3], self.bias.shape, dtype=self.param_dtype)

    def __call__(self, inputs:jax.Array) -> jax.Array:
        kernel = self.kernel.value
        bias = self.bias.value
        # Apply SyRe (before wasym to avoid biasing the masked weights)
        if self.ssigma>0.:
            bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Order bias to counter permutation symmetry
        if self.orderbias: bias = jnp.concatenate([
            bias[0:1],
            jnp.cumsum(jnp.exp(bias[1:])) + bias[0:1]
        ])
        # Apply w-asymmetry
        if self.wasym:
            kernel = kernel * self.wmask + (1-self.wmask) * self.randk * self.kappa
        # Normalize kernel to unit norm per output neuron
        if self.normweights:
            norm = jnp.linalg.norm(kernel, axis=0, keepdims=True)
            kernel /= norm
            bias /= norm.squeeze()
        # Implementation directly copied from nnx.Linear
        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

# Creates mask with the maximum number of non-zero weights while ensuring that each row is unique
# Supports only square kernels
# https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_resnet.py#L173
def mask_conv_densest(kernel_size, in_channels, out_channels, **kwargs):
    mask = jnp.ones((kernel_size, kernel_size, in_channels, out_channels), **kwargs)
    weights_per_out_channel = in_channels * kernel_size**2
    flat_idx_to_3d_idx = jax.vmap(lambda idx : [idx%kernel_size, (idx//kernel_size)%kernel_size, idx//kernel_size**2])
    out_channel_idx = 1
    for num_zeros in range(1, weights_per_out_channel):
        flat_cols_idxs = combinations(range(weights_per_out_channel), num_zeros) # not actually column, but rather (k_h, k_w, in_c)
        cols_idxs = map(flat_idx_to_3d_idx, map(jnp.array, flat_cols_idxs))
        for cols_idx in cols_idxs:
            if out_channel_idx >= out_channels:
                return mask
            mask = mask.at[tuple(cols_idx) + (out_channel_idx,)].set(0)
            out_channel_idx += 1
def mask_conv_random(kernel_size, in_channels, out_channels, key, pfix:float, **kwargs):
    mask = jnp.ones((kernel_size, kernel_size, in_channels, out_channels), **kwargs)
    weights_per_out_channel = in_channels * kernel_size**2
    nfix = max(1, int(weights_per_out_channel**(1/4))) #int(pfix*weights_per_out_channel)
    flat_idx_to_3d_idx = jax.vmap(lambda idx : [idx%kernel_size, (idx//kernel_size)%kernel_size, idx//kernel_size**2])
    for out_channel_idx, key in zip(range(out_channels), jax.random.split(key, out_channels)):
        flat_col_idxs = jax.random.permutation(key, jnp.arange(weights_per_out_channel))[:nfix]
        col_idxs = flat_idx_to_3d_idx(flat_col_idxs)
        mask = mask.at[tuple(col_idxs) + (out_channel_idx,)].set(0)
    return mask

# W-Asymmetry implementation consistent with https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_resnet.py#L22
# And kernel normalization consistent with https://github.com/o-laurent/bayes_posterior_symmetry_exploration/blob/main/symmetries/scale_resnet.py#L166
class AsymConv(nnx.Conv):
    def __init__(self, in_features:int, out_features:int, key:jax.dtypes.prng_key, wasym:str|None=None, 
                 kappa:float=1., sigma:float=0., orderbias:bool=False, normweights:bool=False, **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(in_features, out_features, rngs=nnx.Rngs(keys[0]), use_bias=True, **kwargs)
        # Check if asymmetry is to be applied
        self.ssigma = sigma
        self.wasym = bool(wasym)
        self.kappa = kappa
        self.orderbias = orderbias
        self.normweights = normweights
        # Create W-Assymmetry and SyRe params
        if wasym=="densest": self.wmask = mask_conv_densest(*self.kernel.shape[1:], dtype=self.param_dtype)
        elif wasym=="random": self.wmask = mask_conv_random(*self.kernel.shape[1:], key=keys[1], pfix=1/3, dtype=self.param_dtype)
        if sigma>0. or self.wasym: self.randk = jax.random.normal(keys[2], self.kernel.shape, dtype=self.param_dtype)
        if sigma>0.: self.randb = jax.random.normal(keys[3], self.bias.shape, dtype=self.param_dtype)

    def __call__(self, inputs:jax.Array) -> jax.Array:
        kernel = self.kernel.value
        bias = self.bias.value
        # Apply SyRe (before wasym to avoid biasing the masked weights)
        if self.ssigma>0.:
            bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Order bias to counter permutation symmetry
        if self.orderbias: bias = jnp.concatenate([
            bias[0:1],
            jnp.cumsum(jnp.exp(bias[1:])) + bias[0:1]
        ])
        # Apply w-asymmetry
        if self.wasym:
            kernel = kernel * self.wmask + (1-self.wmask) * self.randk * self.kappa
        # Normalize kernel to unit norm per output neuron
        if self.normweights:
            norm = jnp.linalg.norm(kernel.reshape(-1, self.out_features), axis=0, keepdims=True)
            kernel /= norm
            bias /= norm.squeeze()
        # Implementation directly copied from nnx.Conv
        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        broadcast = lambda x: (x,) * len(self.kernel_size) if isinstance(x, int) else tuple(x)
        y = self.conv_general_dilated(
            inputs,
            kernel,
            broadcast(self.strides), 
            self.padding, # restricted to "SAME" or "VALID"
            lhs_dilation=broadcast(self.input_dilation),
            rhs_dilation=broadcast(self.kernel_dilation),
            dimension_numbers=_conv_dimension_numbers(inputs.shape),
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )
        y += bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
        return y

# Used in resnet
class ResNetBlock(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, in_kernels:int, out_kernels:int, stride:int=1, wasym:bool=False, 
                 kappa:float=1., sigma:float=0., activation=nnx.relu, orderbias:bool=False, normweights:bool=False):
        super().__init__()
        keys = jax.random.split(key, 5)
        self.stride = stride
        self.activation = activation
        self.norm1 = nnx.BatchNorm(in_kernels, rngs=nnx.Rngs(keys[1]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv1 = AsymConv(
            in_kernels, 
            out_kernels,
            keys[0],
            wasym,
            kappa,
            sigma,
            orderbias,
            normweights,
            kernel_size=(3,3),
            strides=(stride,stride),
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm2 = nnx.BatchNorm(out_kernels, rngs=nnx.Rngs(keys[3]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv2 = AsymConv(
            out_kernels,
            out_kernels,
            keys[2],
            wasym,
            kappa,
            sigma,
            orderbias,
            normweights,
            kernel_size=(3,3),
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        if stride>1 and in_kernels!=out_kernels:
            self.id_conv = AsymConv(
                in_kernels, 
                out_kernels, 
                keys[4], 
                wasym, 
                kappa, 
                sigma, 
                orderbias, 
                normweights,
                kernel_size=(1,1), 
                strides=(stride, stride), 
                dtype=jnp.bfloat16, 
                param_dtype=jnp.bfloat16
            )

    def __call__(self, x, train=True):
        res = x if self.stride==1 else self.id_conv(x)
        # Pre-activation implementation
        x = self.norm1(x, use_running_average=not train)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x, use_running_average=not train)
        x = self.activation(x)
        x = self.conv2(x)
        x = res+x
        return x

# Resnet for ImageNet ([3,4,6,3] for 34 layers, [2,2,2,2] for 18 layers)
class ResNet(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, layers:tuple[int,...]=[2,2,2,2], kernels:tuple[int,...]=[64,128,256,512], 
                 channels_in:int=3, dim_out:int=1000, dimexp:bool=False, wasym:str|None=None, kappa:float=1., sigma:float=0., 
                 activation=nnx.relu, orderbias:bool=False, normweights:bool=False, **kwargs):
        # Set some params
        super().__init__(**kwargs)
        self.dimexp = dimexp
        # Keys
        keys = iter(jax.random.split(key, sum(layers)+2))
        # Layers
        self.conv = AsymConv(channels_in, 64, next(keys), wasym, kappa, sigma, orderbias, normweights, kernel_size=(7,7),
                             strides=(2,2), padding="SAME", param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.activation = activation
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(ResNetBlock(next(keys), k_in, k_out, stride=s, wasym=wasym, kappa=kappa, sigma=sigma, 
                                               activation=activation, orderbias=orderbias, normweights=normweights))
        self.fc = AsymLinear(kernels[-1], dim_out, next(keys), wasym, kappa, sigma, orderbias, normweights=False, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)

    def __call__(self, x, z=None, train=True):
        # Apply dimension expansion if desired
        x = x if not self.dimexp else interleave(x)
        # Forward pass
        x = self.conv(x)
        x = self.activation(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        for layer in self.layers:
            x = layer(x, train=train)
        x = jnp.mean(x, axis=(1,2))
        x = self.fc(x)
        return x

# LeNet-5 for 36X60 images + 3 auxiliary features
class LeNet(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, dimexp=False, wasym=None, kappa=1., dim_out=2, sigma=0., 
                 activation=nnx.relu, orderbias=False, channels_in=1, normweights=False):
        # Some params
        super().__init__()
        self.activation = activation
        # Dimension expansion params
        flat_shape = 15*27 if dimexp else 6*12
        self.dimexp = dimexp
        # Layers
        keys = jax.random.split(key, 5)
        self.conv1 = AsymConv(channels_in, 8, keys[0], wasym, kappa, sigma, orderbias, normweights, kernel_size=(4,4), padding="VALID")
        self.conv2 = AsymConv(8, 16, keys[1], wasym, kappa, sigma, orderbias, normweights, kernel_size=(4,4), padding="VALID")
        self.fc1 = AsymLinear(flat_shape*16+3, 128, keys[2], wasym, kappa, sigma, orderbias, normweights)
        self.fc2 = AsymLinear(128, 64, keys[3], wasym, kappa, sigma, orderbias, normweights)
        self.fc3 = AsymLinear(64, dim_out, keys[4], wasym, kappa, sigma, orderbias, normweights=False)
    
    def __call__(self, x, z, train=None):
        # Apply dimension expansion if desired
        x = x if not self.dimexp else interleave(x)
        # Forward pass
        x = self.conv1(x)
        x = self.activation(x)
        x = nnx.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = self.conv2(x)
        x = self.activation(x)
        x = nnx.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.concatenate([x, z], axis=-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
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