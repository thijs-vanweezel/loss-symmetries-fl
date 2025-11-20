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

class WAsymmetric(nnx.Module):
    def create_masks(self, key, pfix):
        # Do nothing if no asymmetry desired
        if (pfix==1.): self.asym=False; return
        self.asym=True
        # Init masks
        self.masks = {}
        self.fills = {}
        # Create mask for each layer
        for path, layer in self.iter_modules():
            # Stop if layer has no kernel
            if not hasattr(layer, "kernel"): continue
            # Note: the masks and values are deterministic, given the same key
            subkey, key = jax.random.split(key) 
            # Mask per filters if conv
            shape = layer.kernel.value.shape
            mask_shape = shape if path[-1][:-1]=="fc" else shape[-1]
            # Replace masked weights with random values
            self.masks[path] = jax.random.bernoulli(key, p=pfix, shape=mask_shape).astype(jnp.float32)
            self.fills[path] = jax.random.normal(subkey, shape)

    def apply_masks(self):
        convert_pathpart = lambda p: f".{p}" if isinstance(p, str) else f"[{p}]" if isinstance(p, int) else ""
        # Apply W-asymmetry per layer
        for path, layer in self.iter_modules():
            # Stop if no asymmetry is desired or layer has no kernel
            if (not self.asym) or (not hasattr(layer, "kernel")): continue
            # Mask layer
            mask = self.masks[path]
            layer.kernel.value = layer.kernel.value * mask# + (1-mask) * self.fills[path]
            # Re-assign masked layer TODO: anything better than eval?
            eval(f"self{''.join(convert_pathpart(p) for p in path[:-1])}").__setattr__(path[-1], layer)

# Used in resnet
class ResNetBlock(nnx.Module):
    def __init__(self, key:nnx.RngKey, in_kernels:int, out_kernels:int, stride:int=1):
        super().__init__()
        self.stride = stride
        self.conv1 = nnx.Conv(
            in_features=in_kernels,
            out_features=out_kernels,
            kernel_size=(3,3),
            strides=(stride,stride),
            padding="SAME",
            rngs=key,
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm1 = nnx.BatchNorm(out_kernels, rngs=key, param_dtype=jnp.float32, dtype=jnp.bfloat16)
        self.conv2 = nnx.Conv(
            in_features=out_kernels,
            out_features=out_kernels,
            kernel_size=(3,3),
            padding="SAME",
            rngs=key,
            param_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16
        )
        self.norm2 = nnx.BatchNorm(out_kernels, rngs=key, param_dtype=jnp.float32, dtype=jnp.bfloat16)
        if stride>1 and in_kernels!=out_kernels:
            self.id_conv = nnx.Conv(in_kernels, out_kernels, kernel_size=(1,1), strides=(stride, stride), rngs=key, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)

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
class ResNet(WAsymmetric): # TODO: 36/(2**5) is a small shape for conv
    def __init__(self, key:nnx.RngKey, block=ResNetBlock, layers=[2,2,2,2], kernels=[64,128,256,512], channels_in=1, dim_out=9, dimexp=False, pfix=1., mask_key=None, **kwargs):
        super().__init__(**kwargs)
        # Asymmetry params
        self.dimexp = dimexp
        self.mask_key = mask_key
        self.pfix = pfix
        # Layers
        self.conv = nnx.Conv(channels_in, 64, kernel_size=(7,7), strides=(2,2), padding="SAME", rngs=key, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(block(key, k_in, k_out, stride=s))
        self.fc = nnx.Linear(kernels[-1]+3, dim_out, rngs=key, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.create_masks(self.mask_key, self.pfix)

    def __call__(self, x, z, train=True):
        # Apply asymmetries
        x = x if not self.dimexp else interleave(x)
        self.apply_masks()
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
class LeNet(WAsymmetric):
    def __init__(self, key:jax.dtypes.prng_key, dimexp=False, pfix=1., mask_key=None):
        super().__init__()
        # Asymmetry params
        flat_shape = 15*27 if dimexp else 6*12
        self.dimexp = dimexp
        self.mask_key = mask_key
        self.pfix = pfix
        # Layers
        keys = jax.random.split(key, 4)
        self.conv1 = nnx.Conv(1, 8, (4,4), rngs=nnx.Rngs(key), padding="VALID")
        self.conv2 = nnx.Conv(8, 16, (4,4), rngs=nnx.Rngs(keys[0]), padding="VALID")
        self.fc1 = nnx.Linear(flat_shape*16+3, 128, rngs=nnx.Rngs(keys[1]))
        self.fc2 = nnx.Linear(128, 64, rngs=nnx.Rngs(keys[2]))
        self.fc3 = nnx.Linear(64, 9, rngs=nnx.Rngs(keys[3]))
        self.create_masks(self.mask_key, self.pfix)
    
    def __call__(self, x, z, train=None):
        # Apply asymmetries
        x = x if not self.dimexp else interleave(x)
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