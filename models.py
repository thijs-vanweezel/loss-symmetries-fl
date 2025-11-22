from flax import nnx
from jax import numpy as jnp
from itertools import chain
from functools import partial
import jax, dataclasses

# Dimension expansion
@jax.vmap
def interleave(img):
    img = jnp.repeat(img, 2, axis=0)
    img = jnp.repeat(img, 2, axis=1)
    img = img.at[::2].set(.5)
    img = img.at[:, ::2].set(.5)
    return img

convert_pathpart = lambda p: f".{p}" if isinstance(p, str) else f"[{p}]" if isinstance(p, int) else ""
class Asymmetric(nnx.Module):
    def create_asym_vars(self, key, pfix, sigma):
        # Skip if no asymmetry is required
        self.wasym = pfix<1.
        self.syre = sigma>0.
        if (not self.wasym) and (not self.syre): return
        # Init masks
        if self.wasym: self.masks = {}
        self.randk = {}
        if self.syre: self.randb = {}
        # Create mask for each layer
        for path, layer in self.iter_modules():
            # Stop if layer has neither kernel nor bias
            if not ((bias:=hasattr(layer, "bias")) | (kernel:=hasattr(layer, "kernel"))): continue
            k1, k2, k3 = jax.random.split(key, 3) 
            if bias and self.syre: bshape = layer.bias.value.shape
            if kernel: kshape = layer.kernel.value.shape
            # Create Gaussian values for SyRe and wasym
            if bias and self.syre: self.randb[path] = jax.random.normal(k1, bshape)
            if kernel: self.randk[path] = jax.random.normal(k2, kshape)
            # Create masks for w-asymmetry 
            if (not self.wasym) or (not kernel): continue
            # Mask per filter if conv
            mask_shape = kshape if path[-1][:-1]=="fc" else kshape[-1]
            self.masks[path] = jax.random.bernoulli(k3, p=pfix, shape=mask_shape).astype(jnp.float32)

    def apply_masks(self):
        if not self.wasym: return
        # Apply W-asymmetry per layer
        for path, layer in self.iter_modules():
            # Stop if layer has no kernel (masking bias is not necessary)
            if not hasattr(layer, "kernel"): continue
            # Mask layer
            mask = self.masks[path]
            layer.kernel.value = layer.kernel.value * mask + (1-mask) * self.randk[path]
            # Re-assign masked layer TODO: anything better than eval?
            eval(f"self{''.join(convert_pathpart(p) for p in path[:-1])}").__setattr__(path[-1], layer)
    
    def apply_syre(self, sigma, application=1): # TODO: Is `application` a bit hacky?
        if not self.syre: return
        # Apply symmetry removal (SyRe) per layer (NOTE: requires a weight decay optimizer such as adamw)
        for path, layer in self.iter_modules():
            # Stop if layer has no kernel nor bias
            if not ((bias:=hasattr(layer, "bias")) | (kernel:=hasattr(layer, "kernel"))): continue
            # Apply static bias to layer's values
            if bias: layer.bias.value = layer.bias.value + application*self.randb[path]*sigma
            if kernel: layer.kernel.value = layer.kernel.value + application*self.randk[path]*sigma
            # Re-assign layer
            eval(f"self{''.join(convert_pathpart(p) for p in path[:-1])}").__setattr__(path[-1], layer)

# Used in resnet
class ResNetBlock(nnx.Module):
    def __init__(self, key:jax.dtypes.prng_key, in_kernels:int, out_kernels:int, stride:int=1):
        super().__init__()
        keys = jax.random.split(key, 5)
        self.stride = stride
        self.conv1 = nnx.Conv(
            in_features=in_kernels,
            out_features=out_kernels,
            kernel_size=(3,3),
            strides=(stride,stride),
            padding="SAME",
            rngs=nnx.Rngs(keys[0]),
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm1 = nnx.BatchNorm(out_kernels, rngs=nnx.Rngs(keys[1]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv2 = nnx.Conv(
            in_features=out_kernels,
            out_features=out_kernels,
            kernel_size=(3,3),
            padding="SAME",
            rngs=nnx.Rngs(keys[2]),
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm2 = nnx.BatchNorm(out_kernels, rngs=nnx.Rngs(keys[3]), param_dtype=jnp.float32, dtype=jnp.bfloat16)
        if stride>1 and in_kernels!=out_kernels:
            self.id_conv = nnx.Conv(in_kernels, out_kernels, kernel_size=(1,1), strides=(stride, stride), rngs=nnx.Rngs(keys[4]), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

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
class ResNet(Asymmetric): # TODO: 36/(2**5) is a small shape for conv
    def __init__(self, key:jax.dtypes.prng_key, block=ResNetBlock, layers=[2,2,2,2], kernels=[64,128,256,512], channels_in=1, dim_out=9, dimexp=False, pfix=1., syre_sigma=0., **kwargs):
        super().__init__(**kwargs)
        # Asymmetry params
        self.dimexp = dimexp
        self.pfix = pfix
        self.sigma = syre_sigma
        # Keys
        keys = jax.random.split(key, sum(layers)+3)
        # Layers
        self.conv = nnx.Conv(channels_in, 64, kernel_size=(7,7), strides=(2,2), padding="SAME", rngs=nnx.Rngs(keys[0]), param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(block(keys[j+(i*j)], k_in, k_out, stride=s))
        self.fc = nnx.Linear(kernels[-1]+3, dim_out, rngs=nnx.Rngs(keys[-2]), param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        # Masks
        self.create_asym_vars(keys[-1], self.pfix, self.sigma)

    def __call__(self, x, z, train=True):
        # Apply asymmetries (syre before wasym to avoid biasing the masks)
        x = x if not self.dimexp else interleave(x)
        self.apply_syre(self.sigma)
        self.apply_masks()
        # Forward pass
        x = self.conv(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        for layer in self.layers:
            x = layer(x, train=train)
        x = jnp.mean(x, axis=(1,2))
        x = self.fc(jnp.concatenate([x, z], axis=-1))
        self.apply_syre(self.sigma, application=-1)
        return x

# LeNet-5 for 36X60 images + 3 auxiliary features
class LeNet(Asymmetric):
    def __init__(self, key:jax.dtypes.prng_key, dimexp=False, pfix=1., dim_out=9, syre_sigma=0.):
        super().__init__()
        # Asymmetry params
        flat_shape = 15*27 if dimexp else 6*12
        self.dimexp = dimexp
        self.pfix = pfix
        self.sigma = syre_sigma
        # Layers
        keys = jax.random.split(key, 5)
        self.conv1 = nnx.Conv(1, 8, (4,4), rngs=nnx.Rngs(key), padding="VALID")
        self.conv2 = nnx.Conv(8, 16, (4,4), rngs=nnx.Rngs(keys[0]), padding="VALID")
        self.fc1 = nnx.Linear(flat_shape*16+3, 128, rngs=nnx.Rngs(keys[1]))
        self.fc2 = nnx.Linear(128, 64, rngs=nnx.Rngs(keys[2]))
        self.fc3 = nnx.Linear(64, dim_out, rngs=nnx.Rngs(keys[3]))
        self.create_asym_vars(keys[4], self.pfix, self.sigma)
    
    def __call__(self, x, z, train=None):
        # Apply asymmetries (syre before wasym to avoid biasing the masks)
        x = x if not self.dimexp else interleave(x)
        self.apply_syre(self.sigma)       
        self.apply_masks() 
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
        self.apply_syre(self.sigma, application=-1)
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