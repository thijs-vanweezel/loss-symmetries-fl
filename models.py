from flax import nnx
from jax import numpy as jnp

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
    def __init__(self, key:nnx.RngKey, in_kernels:int, out_kernels:int, strides:int=1, **kwargs):
        super().__init__(**kwargs)
        self.strides = strides
        self.conv1 = nnx.Conv(
            in_features=in_kernels,
            out_features=out_kernels,
            kernel_size=1,
            strides=strides,
            padding="SAME",
            rngs=key
        )
        self.norm1 = nnx.BatchNorm(out_kernels, rngs=key)
        self.conv2 = nnx.Conv(
            in_features=out_kernels,
            out_features=out_kernels,
            kernel_size=1,
            padding="SAME",
            rngs=key
        )
        self.norm2 = nnx.BatchNorm(out_kernels, rngs=key)
        if strides>1:
            self.id_conv = nnx.Conv(out_kernels, out_kernels, kernel_size=1, strides=strides, rngs=key)

    def __call__(self, x, train=True):
        res = x if self.strides==1 else self.id_conv(x)
        x = self.conv1(x)
        x = self.norm1(x, use_running_average=not train)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.norm2(x, use_running_average=not train)
        x = res+x
        return x

# Resnet-34 for ImageNet
class ResNet(nnx.Module):
    def __init__(self, key:nnx.RngKey, block=ResNetBlock, layers=[3,4,6,3], kernels=[64,128,256,512], num_classes=1000, **kwargs):
        super().__init__(**kwargs)
        self.conv = nnx.Conv(3, 64, kernel_size=7, strides=2, padding="SAME", rngs=key)
        self.layers = []
        for j, l in enumerate(layers):
            for i in range(l):
                k_in = ([64]+kernels)[j] if i==0 else kernels[j]
                k_out = kernels[j]
                s = 2 if i==0 and j>0 else 1
                self.layers.append(block(key, k_in, k_out, strides=s))
        self.fc = nnx.Linear(kernels[-1], num_classes, rngs=key)

    def __call__(self, x, train=True):
        x = self.conv(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(3,3), strides=(2,2), padding="SAME")
        for layer in self.layers:
            x = layer(x, train=train)
        x = jnp.mean(x, axis=(1,2))
        x = self.fc(x)
        return x