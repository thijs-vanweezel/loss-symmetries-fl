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