from scipy.ndimage import rotate
from sklearn.datasets import load_digits
import jax, os, torchvision, torch
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp

def create_digits(beta, batch_size=64, n_clients=4, client_overlap=1.):
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

class Imagenet(Dataset):
    def __init__(self, split="train"):
        g = os.walk(f"./data/Data/CLS-LOC/{split}", topdown=True)
        self.classes = next(g)[1]
        self.paths = [os.path.join(dirname, f) for (dirname, _, filenames) in g for f in filenames]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        img = torchvision.io.read_image(fp).float() / 255.
        if img.shape[1]<128 or img.shape[2]<128 or img.shape[0]!=3:
            del self.paths[idx]
            os.remove(fp)
            return self.__getitem__(idx)
        label = self.classes.index(fp.split("/")[-2].rstrip(".JPEG"))
        label = torch.eye(1000)[label].float()
        return img, label
    
def jax_collate(batch):
    imgs, labels = zip(*batch)
    # Find minimum height and width in this batch
    min_height = min(img.shape[1] for img in imgs)
    min_width = min(img.shape[2] for img in imgs)
    # Resize images to the minimum height and width
    imgs = [torchvision.transforms.functional.resize(img, (min_height, min_width)) for img in imgs]
    # Convert and concat
    imgs = jnp.stack([jnp.swapaxes(jnp.asarray(img), 0, -1) for img in imgs])
    labels = jnp.stack([jnp.asarray(label) for label in labels])
    return imgs, labels

create_imagenet = lambda *args, **kwargs: DataLoader(Imagenet(*args, **kwargs), batch_size=64, shuffle=True, collate_fn=jax_collate)