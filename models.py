from flax import nnx
from jax import numpy as jnp
from itertools import chain, combinations
from flax.nnx.nn.linear import _conv_dimension_numbers
import jax, flax, sys, importlib, warnings
from functools import partial
from ml_collections.config_dict import ConfigDict

# Dimension expansion
@jax.vmap
def interleave(img, fill_value=.5):
    img = jnp.repeat(img, 2, axis=0)
    img = jnp.repeat(img, 2, axis=1)
    img = img.at[::2].set(fill_value)
    img = img.at[:, ::2].set(fill_value)
    return img

class NonTrainable(nnx.Variable):
    """Necessary for filtering and abstract evaluation."""
    pass

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

# W-Asymmetry implementation consistent with https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_mlp.py#L169 # TODO: is it strange that updates for masked weights are not zero?
# SyRe implementation consistent with https://github.com/xu-yz19/syre/blob/main/MLP.ipynb
# And kernel normalization consistent with https://github.com/o-laurent/bayes_posterior_symmetry_exploration/blob/main/symmetries/scale_resnet.py#L166
class AsymLinear(nnx.Linear):
    def __init__(self, in_features:int, out_features:int, key:jax.dtypes.prng_key, wasym:bool=False, 
                 kappa:float=1., sigma:float=0., orderbias:bool=False, normweights:bool=False, **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(in_features, out_features, rngs=kwargs.pop("rngs",nnx.Rngs(keys[0])), use_bias=True, **kwargs)
        # Check if asymmetry is to be applied
        self.ssigma = sigma
        self.wasym = bool(wasym)
        self.kappa = kappa
        self.orderbias = orderbias
        self.normweights = normweights
        # Create asymmetry params
        if self.wasym: self.wmask = NonTrainable(mask_linear_densest(*self.kernel.shape, dtype=self.param_dtype))
        if sigma>0. or self.wasym: self.randk = NonTrainable(jax.random.normal(keys[2], self.kernel.shape, dtype=self.param_dtype))
        if sigma>0.: self.randb = NonTrainable(jax.random.normal(keys[3], self.bias.shape, dtype=self.param_dtype))

    def __call__(self, inputs:jax.Array) -> jax.Array:
        bias = self.bias
        kernel = self.kernel
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
    warnings.warn("Mask is identifiable")
    return mask

# W-Asymmetry implementation consistent with https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_resnet.py#L22
# And kernel normalization consistent with https://github.com/o-laurent/bayes_posterior_symmetry_exploration/blob/main/symmetries/scale_resnet.py#L166
class AsymConv(nnx.Conv):
    def __init__(self, in_features:int, out_features:int, kernel_size:tuple[int,...], key:jax.dtypes.prng_key, wasym:bool=False, 
                 kappa:float=1., sigma:float=0., orderbias:bool=False, normweights:bool=False, **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(in_features, out_features, kernel_size, rngs=kwargs.pop("rngs", nnx.Rngs(keys[0])), **kwargs)
        # Check if asymmetry is to be applied
        self.ssigma = sigma
        self.wasym = bool(wasym)
        self.kappa = kappa
        assert not orderbias or self.use_bias, "Order bias requires use_bias=True"
        self.orderbias = orderbias
        self.normweights = normweights
        self.maybe_broadcast = lambda x: (x,) * len(self.kernel_size) if isinstance(x, int) else tuple(x)
        # Create asymmetry params
        if self.wasym: self.wmask = NonTrainable(mask_conv_densest(*self.kernel.shape[1:], dtype=self.param_dtype))
        if sigma>0. or self.wasym: self.randk = NonTrainable(jax.random.normal(keys[2], self.kernel.shape, dtype=self.param_dtype))
        if sigma>0. and self.use_bias: self.randb = NonTrainable(jax.random.normal(keys[3], self.bias.shape, dtype=self.param_dtype))

    def __call__(self, inputs:jax.Array) -> jax.Array:
        bias = self.bias
        kernel = self.kernel
        # Apply SyRe (before wasym to avoid biasing the masked weights)
        if self.ssigma>0.:
            if self.use_bias: bias = bias + self.randb * self.ssigma
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
            if self.use_bias: bias /= norm.squeeze()
        # Implementation directly copied from nnx.Conv
        inputs, kernel, bias = self.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        y = self.conv_general_dilated(
            inputs,
            kernel,
            self.maybe_broadcast(self.strides), 
            self.padding, # restricted to "SAME" or "VALID"
            lhs_dilation=self.maybe_broadcast(self.input_dilation),
            rhs_dilation=self.maybe_broadcast(self.kernel_dilation),
            dimension_numbers=_conv_dimension_numbers(inputs.shape),
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )
        if self.use_bias:
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
            (3,3),
            keys[0],
            wasym,
            kappa,
            sigma,
            False, # no bias due to subsequent BN
            normweights,
            strides=(stride,stride),
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            use_bias=False
        )
        self.norm2 = nnx.BatchNorm(out_kernels, rngs=nnx.Rngs(keys[3]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv2 = AsymConv(
            out_kernels,
            out_kernels,
            (3,3),
            keys[2],
            wasym,
            kappa,
            sigma,
            False,
            normweights,
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            use_bias=False
        )
        if stride>1 and in_kernels!=out_kernels:
            self.id_conv = AsymConv(
                in_kernels, 
                out_kernels, 
                (1,1),
                keys[4], 
                wasym, 
                kappa, 
                sigma, 
                False, 
                normweights,
                strides=(stride, stride), 
                dtype=jnp.bfloat16, 
                param_dtype=jnp.bfloat16,
                use_bias=False
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
    def __init__(self, key=jax.random.key(0), layers:tuple[int,...]=[2,2,2,2], kernels:tuple[int,...]=[64,128,256,512], 
                 channels_in:int=3, dim_out:int=1000, dimexp:bool=False, wasym:bool=False, kappa:float=1., sigma:float=0., 
                 activation=nnx.relu, orderbias:bool=False, normweights:bool=False, **kwargs):
        assert len(layers)==len(kernels)
        # Set some params
        super().__init__(**kwargs)
        self.dimexp = dimexp
        # Keys
        keys = iter(jax.random.split(key, sum(layers)+3))
        # Layers
        self.conv = AsymConv(channels_in, 64, (7,7), next(keys), wasym, kappa, sigma, orderbias, normweights,
                             strides=(2,2), padding="SAME", param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(ResNetBlock(next(keys), k_in, k_out, stride=s, wasym=wasym, kappa=kappa, sigma=sigma, 
                                               activation=activation, orderbias=orderbias, normweights=normweights))
        self.bn = nnx.BatchNorm(kernels[-1], rngs=nnx.Rngs(next(keys)), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.activation = activation
        self.fc = AsymLinear(kernels[-1], dim_out, next(keys), wasym, kappa, sigma, orderbias, normweights=False, 
                             param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
    def __call__(self, x, z=None, train=True):
        # Apply dimension expansion if desired
        x = x if not self.dimexp else interleave(x)
        # Forward pass
        x = self.conv(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        for layer in self.layers:
            x = layer(x, train=train)
        x = self.bn(x, use_running_average=not train)
        x = self.activation(x)
        x = jnp.mean(x, axis=(1,2), dtype=jnp.float32)
        x = self.fc(x)
        return x

class ResNetAutoEncoder(nnx.Module):
    def __init__(self, backboneencoder:ResNet, key=jax.random.key(0), **asymkwargs):
        keys = jax.random.split(key, 2*len(backboneencoder.layers)+2)
        self.backboneencoder = backboneencoder
        self.spp = AsymConv( # TODO: placeholder for spatial pyramid pooling
            backboneencoder.layers[-1].conv2.out_features, 128, (3,3), keys[-1], param_dtype=jnp.bfloat16, dtype=jnp.bfloat16, **asymkwargs
        ) 
        self.id_convs = []
        self.convs = []
        for i, layer in enumerate(reversed(backboneencoder.layers[:-1])):
            self.id_convs.append(AsymConv(
                layer.conv2.out_features,
                128,
                (1,1),
                keys[2*i],
                param_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                **asymkwargs
            ))
            self.convs.append(AsymConv(
                128,
                128,
                (3,3),
                keys[2*i+1],
                param_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                **asymkwargs
            ))
        self.final_conv = AsymConv(
            128,
            20,
            (3,3),
            keys[-2],
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            **asymkwargs
        )
    def __call__(self, x, train=True):
        # Encode using backbone's forward pass and save intermediate representations
        x = x if not self.backboneencoder.dimexp else interleave(x)
        x = self.backboneencoder.conv(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        laterals = []
        for layer in self.backboneencoder.layers:
            x = layer(x, train=train)
            laterals.append(x)
        # Increase receptive field with SPP
        x = self.spp(x)
        # Decode using lateral representations
        for z, id_conv, conv in zip(reversed(laterals[:-1]), self.id_convs, self.convs):
            # Match encoded material's size with lateral's
            x = jax.image.resize(x, (*z.shape[:-1], x.shape[-1]), method="bilinear", precision=jax.lax.Precision.HIGHEST)
            # Match lateral's channels with desired number of channels
            z = id_conv(z)
            # Sum
            x = x + z
            # Convolution over intermediate representation
            x = conv(x)
            x = self.backboneencoder.activation(x)
        # Final convolution to get desired number of output channels
        x = self.final_conv(x)
        x = jax.image.resize(x, (x.shape[0], 224, 224, x.shape[-1]), method="bilinear", precision=jax.lax.Precision.HIGHEST)
        return x

# LeNet-5 for 36X60 images + 3 auxiliary features
class LeNet(nnx.Module):
    def __init__(self, key=jax.random.key(0), dimexp=False, wasym=False, kappa=1., dim_out=2, sigma=0., 
                 activation=nnx.relu, orderbias=False, channels_in=1, normweights=False):
        # Some params
        super().__init__()
        self.activation = activation
        # Dimension expansion params
        flat_shape = 15*27 if dimexp else 6*12
        self.dimexp = dimexp
        # Layers
        keys = jax.random.split(key, 5)
        self.conv1 = AsymConv(channels_in, 8, (4,4), keys[0], wasym, kappa, sigma, orderbias, normweights,  padding="VALID")
        self.conv2 = AsymConv(8, 16, (4,4), keys[1], wasym, kappa, sigma, orderbias, normweights, padding="VALID")
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

# Config copied from the ViT-B_16 at https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/models.py#L113
defaultconfig = ConfigDict({
    "num_classes": 0, # No classification head
    "patches": ConfigDict({"size": (16, 16)}),
    "model_name": "ViT-B_16",
    "transformer": ConfigDict(
        {"mlp_dim": 3072, "num_heads": 12, "num_layers": 12, "attention_dropout_rate": 0.0, "dropout_rate": 0.0, "add_position_embedding": False}
    ),
    "classifier": "token",
    "representation_size": None,
    "hidden_size": 768
})
def fetch_vit(path="models/ViT-B_16.npz", config=defaultconfig):
    """
    The weights for the backbone are available at https://console.cloud.google.com/storage/browser/vit_models/imagenet21k. 
    Any version should do, if you change the config accordingly.
    """
    try: 
        if not importlib.util.find_spec("tensorflow"):
            from flax import io as fio
            fio.gfile = fio
            sys.modules["tensorflow.io"] = fio
        from vit_jax.models_vit import VisionTransformer
        from vit_jax.checkpoint import load, inspect_params, _fix_groupnorm
    except ImportError as e:
        e.msg += "\nInstall vit_jax with flag `--no-deps` from https://github.com/google-research/vision_transformer"
    
    model = VisionTransformer(**config)
    reference_params = model.init(jax.random.key(42), jnp.ones((1,224,224,3), jnp.bfloat16), train=False)["params"]
    # Dumbed down version of `load_pretrained` removing unused parameters
    # Also converts to bfloat16
    # Case where posemb_new.shape!=posemb.shape is not handled
    params = _fix_groupnorm(inspect_params(
        params=load(path),
        expected=reference_params,
        fail_if_extra=False,
        fail_if_missing=False))
    params = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype=jnp.bfloat16), params)
    for key in set(params.keys()).difference(reference_params.keys()):
        params.pop(key)
    params = flax.core.freeze(params)
    # Return linen model
    return model, params

class ViTAutoEncoder(nnx.Module):
    def __init__(self, n_layers=5, key=jax.random.key(0), out_shape=(224,224), **asymkwargs):
        super().__init__()
        backbone, bbparams = fetch_vit()
        self.bbparams = NonTrainable(bbparams)
        self.encode_fn = jax.jit(partial(backbone.apply, train=False))
        keys = jax.random.split(key, n_layers)
        self.layers = []
        self.n_layers = n_layers
        self.out_shape = out_shape
        for i in range(n_layers):
            self.layers.append(AsymConv(
                1 if i==0 else 128,
                128 if i<n_layers-1 else 20,
                (3,3),
                key=keys[i], 
                param_dtype=jnp.bfloat16,
                dtype=jnp.bfloat16,
                **asymkwargs
            ))

    def __call__(self, x, train=None):
        # Encode with ViT
        x = self.encode_fn({"params": self.bbparams.value}, x)
        x = jnp.expand_dims(x, -1)
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x)
            # Interpolate between original size and final size
            if i==0: original_shape = x.shape[1:3]
            progress = (i+1)/self.n_layers
            new_shape = tuple([int(progress*final+(1-progress)*orig) for orig, final in zip(original_shape, self.out_shape)])
            # Interpolate image to new size
            x = jax.image.resize(x, (x.shape[0], *new_shape, x.shape[-1]), method="bilinear", precision=jax.lax.Precision.HIGHEST)
        return x