from scipy.ndimage import rotate
from sklearn.datasets import load_digits
import jax
from jax import numpy as jnp

def create_digits(beta, batch_size=64, shuffle_per_client=False, n_clients=4, client_overlap=1.):
    # Note: n_clients must be 4 for now due implementation limitations, and shuffle_per_client is not implemented
    # Data 
    x, y, = load_digits(return_X_y=True)
    x, y = jnp.array(x).reshape(-1,8,8,1), jnp.array(y)
    x = (x-x.min()) / (x.max()-x.min())
    # Create data per client
    n_per_client = int(len(x)*client_overlap )
    shift = int(len(x)*(1-client_overlap)/n_clients)
    x = [x[i*shift:n_per_client+i*shift] for i in range(n_clients)]
    y = jnp.stack([y[i*shift:n_per_client+i*shift] for i in range(n_clients)], axis=1)
    # Create some non-IIDness 
    # Note that the outer dimension must be the num_batches dimension and dimension 1 the client dimension and dimension 2 the batch dimension
    x = jnp.stack([
        x[0], 
        beta*x[1]+(1-beta)*(1-x[1]), 
        beta*x[2]+(1-beta)*jax.random.normal(jax.random.key(42), x[2].shape), 
        rotate(x[3], axes=[1,2], angle=(1-beta)*180, reshape=False, order=1)
    ], axis=1)
    # Split into train/val/test
    x_train, x_val, x_test = x[:-384*2], x[-384*2:-384], x[-384:]
    x_val, x_test = jnp.reshape(x_val, (-1,*x_val.shape[2:]), order="F"), jnp.reshape(x_test, (-1,*x_test.shape[2:]), order="F")
    y_train, y_val, y_test = y[:-384*2], y[-384*2:-384], y[-384:]
    y_val, y_test = jnp.reshape(y_val, (-1,), order="F"), jnp.reshape(y_test, (-1,), order="F")
    # One-hot encode
    y_train = jnp.eye(10)[y_train]
    y_val = jnp.eye(10)[y_val]
    # Batch training data 
    num_batches = x_train.shape[0]//batch_size
    excess = num_batches * batch_size
    x_train = jnp.swapaxes(x_train[:excess].reshape(num_batches, batch_size, n_clients, 8, 8, 1), 1, 2)
    y_train = jnp.swapaxes(y_train[:excess].reshape(num_batches, batch_size, n_clients, -1), 1, 2)
    return x_train, y_train, x_val, y_val, x_test, y_test