from flax import nnx
from jax import numpy as jnp
from itertools import chain
import jax

# Simple model
class Simple8x8to10(nnx.Module):
    def __init__(self, key):
        super().__init__()
        self.layer1 = nnx.Linear(64, 128, rngs=key, use_bias=False)
        self.layer2 = nnx.Linear(128, 10, rngs=key, use_bias=False)
    def __call__(self, x):
        x = jnp.reshape(x, (-1, 64))
        x = self.layer1(x)
        x = nnx.relu(x)
        x = self.layer2(x)
        return x

# More complex model
class Mid8x8to10(nnx.Module):
    def __init__(self, key):
        super().__init__()
        self.layer1 = nnx.Conv(1, 32, (3,3), rngs=key)
        self.layer2 = nnx.Conv(32, 128, (3,3), rngs=key)
        self.layer3 = nnx.Conv(128, 128, (3,3), rngs=key)
        self.layer4 = nnx.Linear(128*8*8, 128, rngs=key)
        self.layer5 = nnx.Linear(128, 10, rngs=key)
    def __call__(self, x):
        x = jnp.expand_dims(x, -1)
        x = self.layer1(x)
        x = nnx.relu(x)
        x = self.layer2(x)
        x = nnx.relu(x)
        x = self.layer3(x)
        x = jnp.reshape(x, (-1, 128*8*8))
        x = nnx.relu(x)
        x = self.layer4(x)
        x = nnx.relu(x)
        x = self.layer5(x)
        return x

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
class ResNet(nnx.Module): # TODO: 36/(2**5) is a small shape for conv
    def __init__(self, key:nnx.RngKey, block=ResNetBlock, layers=[2,2,2,2], kernels=[64,128,256,512], channels_in=1, dim_out=16, **kwargs):
        super().__init__(**kwargs)
        self.conv = nnx.Conv(channels_in, 64, kernel_size=(7,7), strides=(2,2), padding="SAME", rngs=key, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(block(key, k_in, k_out, stride=s))
        self.fc = nnx.Linear(kernels[-1]+3, dim_out, rngs=key, param_dtype=jnp.bfloat16, dtype=jnp.bfloat16)

    def __call__(self, x, z, train=True):
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
    def __init__(self, key):
        super().__init__()
        self.conv1 = nnx.Conv(1, 8, (4,4), rngs=key, padding="VALID")
        self.conv2 = nnx.Conv(8, 16, (4,4), rngs=key, padding="VALID")
        self.fc1 = nnx.Linear(6*12*16+3, 128, rngs=key) # 6*12 w/o dim exp, 15*27 w/ dim exp
        self.fc2 = nnx.Linear(128, 64, rngs=key)
        self.fc3 = nnx.Linear(64, 16, rngs=key)
    def __call__(self, x, z, train=None):
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

# Dimension expansion
@jax.vmap
def interleave(img):
    img = jnp.repeat(img, 2, axis=0)
    img = jnp.repeat(img, 2, axis=1)
    img = img.at[::2].set(.5)
    img = img.at[:, ::2].set(.5)
    return img
# LeNet, adjusted for expanded images
class DimExpLeNet(nnx.Module):
    def __init__(self, key):
        super().__init__()
        self.conv1 = nnx.Conv(1, 8, (4,4), rngs=key, padding="VALID")
        self.conv2 = nnx.Conv(8, 16, (4,4), rngs=key, padding="VALID")
        self.fc1 = nnx.Linear(15*27*16+3, 128, rngs=key)
        self.fc2 = nnx.Linear(128, 64, rngs=key)
        self.fc3 = nnx.Linear(64, 16, rngs=key)
    
    def __call__(self, x, z, train=None):
        x = interleave(x)
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

# LeNet with some specifically fixed weights
class WAsymLeNet(nnx.Module):
    def __init__(self, key:jax._src.prng.PRNGKeyArray, pfix=.75):
        super().__init__()
        keys = jax.random.split(key, 5)
        _, *self.keys = jax.random.split(keys[-1], 6)
        self.pfix = pfix

        self.conv1 = nnx.Conv(1, 8, (4,4), rngs=nnx.Rngs(keys[0]), padding="VALID")
        self.conv2 = nnx.Conv(8, 16, (4,4), rngs=nnx.Rngs(keys[1]), padding="VALID")
        self.fc1 = nnx.Linear(6*12*16+3, 128, rngs=nnx.Rngs(keys[2]))
        self.fc2 = nnx.Linear(128, 64, rngs=nnx.Rngs(keys[3]))
        self.fc3 = nnx.Linear(64, 16, rngs=nnx.Rngs(keys[4]))

    def mask_kernel(self, kernel, key):
        key1, key2 = jax.random.split(key) # Note: the masks and values are deterministic
        mask = jax.random.bernoulli(key1, p=self.pfix, shape=kernel.shape).astype(jnp.float32)
        kernel = kernel * mask + (1-mask) * jax.random.normal(key2, kernel.shape)
        return kernel

    def mask(self):
        self.conv1.kernel.value = self.mask_kernel(self.conv1.kernel.value, self.keys[0])
        self.conv2.kernel.value = self.mask_kernel(self.conv2.kernel.value, self.keys[1])
        self.fc1.kernel.value = self.mask_kernel(self.fc1.kernel.value, self.keys[2])
        self.fc2.kernel.value = self.mask_kernel(self.fc2.kernel.value, self.keys[3])
        self.fc3.kernel.value = self.mask_kernel(self.fc3.kernel.value, self.keys[4])

    def __call__(self, x, z, train=None):
        self.mask()
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