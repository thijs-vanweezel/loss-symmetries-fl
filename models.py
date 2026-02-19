from flax import nnx
from jax import numpy as jnp
from itertools import combinations
from functools import partial, reduce
from flax.nnx.nn.linear import _conv_dimension_numbers
import jax, flax, sys, importlib, warnings
from ml_collections.config_dict import ConfigDict

# Dimension expansion
@partial(jax.vmap, in_axes=(0,None))
def interleave(img, k):
    fill_value = .5
    img = (fill_value*jnp.ones(
        (img.shape[0]*k, img.shape[1]*k, img.shape[2])
    )).at[::k, ::k].set(img)
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
                 kappa:float=1., sigma:float=0., normweights:bool=False, **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(in_features, out_features, rngs=kwargs.pop("rngs",nnx.Rngs(keys[0])), use_bias=True, **kwargs)
        # Check if asymmetry is to be applied
        assert (bool(wasym) + bool(sigma) + bool(normweights))<2, "Currently only tested with one symmetry elimination method."
        self.syre = bool(sigma)
        self.ssigma = sigma
        self.wasym = wasym
        self.kappa = kappa
        self.normweights = bool(normweights)
        # Create asymmetry params
        if self.wasym: self.wmask = NonTrainable(mask_linear_densest(*self.kernel.shape, dtype=self.param_dtype))
        if self.syre or self.wasym: self.randk = NonTrainable(jax.random.normal(keys[2], self.kernel.shape, dtype=self.param_dtype))
        if self.syre: self.randb = NonTrainable(jax.random.normal(keys[3], self.bias.shape, dtype=self.param_dtype))

    def __call__(self, inputs:jax.Array, norm_prev:jax.Array=None) -> jax.Array:
        # Apply w-asymmetry (overwriting the param's value to bypass the graph)
        if self.wasym:
            self.kernel.value = self.kernel.value * self.wmask + (1-self.wmask) * self.randk * self.kappa
        # Normalize kernel to unit norm per output neuron
        if norm_prev is not None:
            self.kernel.value = self.kernel.value*norm_prev[:,None]
        if self.normweights:
            norm = jnp.linalg.norm(self.kernel.value, axis=0)
            self.kernel.value = self.kernel.value / norm
            if self.use_bias: self.bias.value /= norm
        else: norm = None
        # Convert to computation dtype without overwriting internal dtype
        inputs, kernel, bias = self.promote_dtype(
            (inputs, self.kernel, self.bias), dtype=self.dtype
        )        
        # Apply SyRe (not overwriting the param's value to avoid re-biasing)
        if self.syre:
            if self.use_bias: bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Implementation directly copied from nnx.Linear
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if self.use_bias: 
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y, norm

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
    warnings.warn("Mask is not identifiable")
    return mask

# W-Asymmetry implementation consistent with https://github.com/cptq/asymmetric-networks/blob/main/lmc/models/models_resnet.py#L22
# And kernel normalization consistent with https://github.com/o-laurent/bayes_posterior_symmetry_exploration/blob/main/symmetries/scale_resnet.py#L166
class AsymConv(nnx.Conv):
    def __init__(self, in_features:int, out_features:int, kernel_size:tuple[int,...], key:jax.dtypes.prng_key, wasym:bool=False, 
                 kappa:float=1., sigma:float=0., normweights:bool=False, **kwargs):
        keys = jax.random.split(key, 4)
        super().__init__(in_features, out_features, kernel_size, rngs=kwargs.pop("rngs", nnx.Rngs(keys[0])), **kwargs)
        # Check if asymmetry is to be applied
        assert (bool(wasym) + bool(sigma) + bool(normweights))<2, "Currently only tested with one symmetry elimination method."
        self.syre = bool(sigma)
        self.ssigma = sigma
        self.wasym = wasym
        self.kappa = kappa
        self.normweights = normweights
        self.maybe_broadcast = lambda x: (x,) * len(self.kernel_size) if isinstance(x, int) else tuple(x)
        # Create asymmetry params
        if self.wasym: self.wmask = NonTrainable(mask_conv_densest(*self.kernel.shape[1:], dtype=self.param_dtype))
        if self.syre or self.wasym: self.randk = NonTrainable(jax.random.normal(keys[2], self.kernel.shape, dtype=self.param_dtype))
        if self.syre and self.use_bias: self.randb = NonTrainable(jax.random.normal(keys[3], self.bias.shape, dtype=self.param_dtype))

    def __call__(self, inputs:jax.Array, norm_prev:jax.Array=None) -> jax.Array:
        # Apply w-asymmetry (overwriting the param's value to bypass the graph)
        if self.wasym:
            self.kernel.value = self.kernel.value * self.wmask + (1-self.wmask) * self.randk * self.kappa
        # Normalize kernel to unit norm per output neuron
        if norm_prev is not None:
            self.kernel.value = self.kernel.value*norm_prev[:,None]
        if self.normweights:
            norm = jnp.linalg.norm(jnp.reshape(self.kernel.value, (-1, self.out_features)), axis=0)
            self.kernel.value = self.kernel.value / norm
            if self.use_bias: self.bias.value = self.bias.value / norm
        else: norm = None
        # Convert to computation dtype without overwriting internal dtype
        inputs, kernel, bias = self.promote_dtype(
            (inputs, self.kernel, self.bias), dtype=self.dtype
        )
        # Apply SyRe (not overwriting the param's value to avoid re-biasing)
        if self.syre:
            if self.use_bias: bias = bias + self.randb * self.ssigma
            kernel = kernel + self.randk * self.ssigma
        # Implementation directly copied from nnx.Conv
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
        return y, norm

class AsymBatchNorm(nnx.BatchNorm):
    def __init__(self, normweights=False, **kwargs):
        super().__init__(**kwargs)
        self.normweights = normweights
    def __call__(self, x, use_running_average=None, norm_prev=None):
        if norm_prev is not None:
            self.mean.value /= norm_prev
            self.var.value /= norm_prev**2
        if self.normweights:
            norm = self.scale.value
            if self.use_scale: self.scale.value /= norm
            if self.use_bias: self.bias.value /= norm
        else: norm = None
        x = super().__call__(x, use_running_average)
        return x, norm

# Used in resnet
class ResNetBlock(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, in_kernels:int, out_kernels:int, stride:int=1, 
                 wasym:bool=False, kappa:float=1., sigma:float=0., normweights:bool=False):
        super().__init__()
        keys = jax.random.split(key, 5)
        self.stride = stride
        self.norm1 = AsymBatchNorm(normweights, num_features=in_kernels, rngs=nnx.Rngs(keys[1]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv1 = AsymConv(
            in_kernels, 
            out_kernels,
            (3,3),
            keys[0],
            wasym,
            kappa,
            sigma,
            normweights,
            strides=(stride,stride),
            padding="SAME",
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
            use_bias=False # due to subsequent BN
        )
        self.norm2 = AsymBatchNorm(normweights, num_features=out_kernels, rngs=nnx.Rngs(keys[3]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv2 = AsymConv(
            out_kernels,
            out_kernels,
            (3,3),
            keys[2],
            wasym,
            kappa,
            sigma,
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
                normweights=False,
                strides=(stride, stride), 
                dtype=jnp.bfloat16, 
                param_dtype=jnp.bfloat16,
                use_bias=False
            )

    def __call__(self, x_in, train=True, norm_prev=None):
        # Pre-activation implementation
        h, norm = self.norm1(x_in, use_running_average=not train, norm_prev=norm_prev)
        h = nnx.relu(h)
        h, norm = self.conv1(h, norm)
        h, norm = self.norm2(h, use_running_average=not train, norm_prev=norm)
        h = nnx.relu(h)
        h, norm = self.conv2(h, norm)
        # Apply identity downsampling if needed and perform scaling
        if norm_prev is not None:
            x_in = x_in*norm_prev
        if self.stride>1:
            x_in, _ = self.id_conv(x_in)
        if norm is not None:
            x_in /= norm
        # Add residual
        out = x_in+h
        return out, norm

# Resnet for ImageNet ([3,4,6,3] for 34 layers, [2,2,2,2] for 18 layers)
class ResNet(nnx.Module):
    def __init__(self, key=jax.random.key(0), layers:tuple[int,...]=[2,2,2,2], kernels:tuple[int,...]=[64,128,256,512], 
                 channels_in:int=3, dim_out:int=1000, dimexp:int=1, wasym:bool=False, kappa:float=1., sigma:float=0., 
                 normweights:bool=False, **kwargs):
        assert len(layers)==len(kernels)
        # Set some params
        super().__init__(**kwargs)
        self.dimexp = dimexp
        # Keys
        keys = iter(jax.random.split(key, sum(layers)+3))
        # Layers
        self.conv = AsymConv(channels_in, 64, (7,7), next(keys), wasym, kappa, sigma, normweights,
                             strides=(2,2), padding="SAME", param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.layers = nnx.List([])
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([kernels[0]]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(ResNetBlock(next(keys), k_in, k_out, stride=s, wasym=wasym, kappa=kappa, 
                                               sigma=sigma, normweights=normweights))
        self.bn = AsymBatchNorm(normweights, num_features=kernels[-1], rngs=nnx.Rngs(next(keys)), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.fc = AsymLinear(kernels[-1], dim_out, next(keys), wasym, kappa, sigma, normweights=False, 
                             param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
    def __call__(self, x, train=True):
        # Apply dimension expansion if desired
        if self.dimexp>1: x=interleave(x, k=self.dimexp)
        # Forward pass
        x, norm = self.conv(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        for layer in self.layers:
            x, norm = layer(x, train=train, norm_prev=norm)
        x, norm = self.bn(x, use_running_average=not train, norm_prev=norm)
        x = nnx.relu(x)
        x = jnp.mean(x, axis=(1,2), dtype=jnp.float32)
        x, _ = self.fc(x, norm)
        return x

# LeNet-5 for 36X60 images + 3 auxiliary features
class LeNet(nnx.Module):
    def __init__(self, key=jax.random.key(0), dimexp=1, wasym=False, kappa=1., dim_out=2, 
                 sigma=0., channels_in=1, normweights=False):
        # Some params
        super().__init__()
        # Dimension expansion params
        self.dimexp = dimexp
        # Layers
        keys = jax.random.split(key, 5)
        self.conv1 = AsymConv(channels_in, 8, (4,4), keys[0], wasym, kappa, sigma, normweights,  padding="VALID")
        self.conv2 = AsymConv(8, 16, (4,4), keys[1], wasym, kappa, sigma, normweights, padding="VALID")
        # Calculate flat shape
        flat_shape = jax.eval_shape(self.conv1, jnp.empty((1,36*dimexp,60*dimexp,channels_in)))[0].shape
        flat_shape = jax.eval_shape(partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2)), jnp.empty(flat_shape)).shape
        flat_shape = jax.eval_shape(self.conv2, jnp.empty(flat_shape))[0].shape
        self.flat_shape = jax.eval_shape(partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2)), jnp.empty(flat_shape)).shape
        # Use flat shape as in_features
        self.fc1 = AsymLinear(reduce(lambda x,y: x*y, self.flat_shape)+3 , 128, keys[2], wasym, kappa, sigma, normweights)
        self.fc2 = AsymLinear(128, 64, keys[3], wasym, kappa, sigma, normweights)
        self.fc3 = AsymLinear(64, dim_out, keys[4], wasym, kappa, sigma, normweights=False)
    
    def __call__(self, x, z, train=None):
        # Apply dimension expansion if desired
        if self.dimexp>1: x=interleave(x, self.dimexp)
        # Forward pass
        x, norm = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x, norm = self.conv2(x, norm)
        x = nnx.relu(x)
        x = nnx.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jnp.concatenate([x, z], axis=-1)
        if norm is not None: norm = jnp.concat([jnp.tile(norm, (*self.flat_shape,1)).flatten(), jnp.ones(3)])
        x, norm = self.fc1(x, norm)
        x = nnx.relu(x)
        x, norm = self.fc2(x, norm)
        x = nnx.relu(x)
        x, _ = self.fc3(x, norm)
        return x

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
def fetch_vit(img_size=224, path="/data/bucket/traincombmodels/models/ViT-B_16.npz", config=defaultconfig):
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
    reference_params = model.init(jax.random.key(42), jnp.ones((1,img_size,img_size,3), jnp.bfloat16), train=False)["params"]
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