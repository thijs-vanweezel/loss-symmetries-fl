from scipy.ndimage import rotate
from sklearn.datasets import load_digits
import jax, os, torchvision, torch, numpy as np, cv2
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp
from functools import partial

def create_digits(beta, batch_size=64, n_clients=4, client_overlap=1.):
    # Note: n_clients must be 4 for now due implementation limitations, and shuffle_per_client is not implemented
    # Note: beta=1 is IID, beta=0 is most non-IID
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

@np.vectorize(excluded=(1,2), signature="(h,w,c)->(h,w,c)")
def perspective_shift(image, angle=0., severity=0.):
    h, w = image.shape[:2]
    src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
    dx = np.cos(angle) * severity * w
    dy = np.sin(angle) * severity * h
    dst_pts = np.array([
        [0 + dx, 0],
        [w + dx, 0 + dy],
        [0 - dx, h],
        [w - dx, h - dy]
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, transform, (w, h))

class Imagenet(Dataset):
    def __init__(self, data_path):
        # Get all image paths and corresponding labels
        g = os.walk(data_path, topdown=True)
        self.classes = next(g)[1]
        self.paths = [os.path.join(dirname, f) for (dirname, _, filenames) in g for f in filenames]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        fp = self.paths[idx]
        img = torchvision.io.read_image(fp).float()/255.
        label = self.classes.index(fp.split("/")[-2].rstrip(".JPEG"))
        label = torch.eye(1000)[label]
        return img, label

def jax_collate(batch, n, key, feature_beta, label_beta, sample_overlap):
    """
    Custom collate, necessary for several reasons: 
    - Images have different sizes, and batches can only contain one shape
    - To create label skew (label_beta=0 corresponds to homogeneity)
    ...
    """
    imgs, labels = zip(*batch)
    # Find minimum height and width in this batch
    min_height = 128 #min(img.shape[1] for img in imgs)
    min_width = 128 #min(img.shape[2] for img in imgs)
    # Resize images to the minimum height and width
    imgs = [torchvision.transforms.functional.resize(img, (min_height, min_width)) for img in imgs]
    # Convert and concat
    imgs = np.swapaxes(np.stack([np.asarray(img, dtype=np.float32) for img in imgs]), 1, -1) # HWC
    labels = jnp.stack([jnp.asarray(label, dtype=jnp.float32) for label in labels])

    # Create label assignment (assuming 1000 classes)
    # Fractions derived from ( sha + n * ind = 1 ) and ( ind / (sha + ind) = label_beta )
    shared_frac = 1/(1+n*label_beta/(1-label_beta))
    indiv_frac = label_beta/(1-label_beta)*shared_frac
    # Convert to labels
    indiv_lab = [jnp.arange(i*1000*indiv_frac, (i+1)*1000*indiv_frac, dtype=jnp.int32) for i in range(n)]
    shared_lab = jnp.arange(n*1000*indiv_frac, 1000, dtype=jnp.int32)
    # Fetch corresponding idx in this batch
    clients_mask = [jnp.isin(labels.argmax(-1), jnp.concat([indiv_lab[i], shared_lab])) for i in range(n)]
    clients_imgs = [imgs[clients_mask[i]] for i in range(n)]
    clients_labels = jnp.stack([labels[clients_mask[i]] for i in range(n)], axis=0)

    # Decrease sample overlap (doing this batch-wise better reflects the overlap within the gradient)


    # Create feature skew with elastic deformation
    angles = [i*2*np.pi/n for i in range(n)]
    clients_imgs = jnp.stack([
        perspective_shift(clients_imgs[i], feature_beta, angles[i]) 
    for i in range(n)], axis=0)

    return clients_imgs, clients_labels

def create_imagenet(path="./data/Data/CLS-LOC/train", n=4, key=jax.random.key(42), feature_beta=0., label_beta=0., sample_overlap=1., batch_size=40, **kwargs):
    # TODO: batch size should be recalculated
    assert label_beta==0. or sample_overlap==1., "Sample overlap reduction not implemented for label_beta>0"
    assert label_beta==0. or batch_size%n==0, "Original batch size must be divisible by n, so that labels can be evenly distributed"
    return DataLoader(
        Imagenet(path),
        batch_size=batch_size,
        collate_fn=partial(jax_collate, n=n, key=key, feature_beta=feature_beta, label_beta=label_beta, sample_overlap=sample_overlap),
        shuffle=False,
        **kwargs
    )

class_name = lambda ds, y: {class_name: name for line in open("data/mapping.txt").read().splitlines() for class_name, name in [line.split(" ", 1)]}[ds.dataset.classes[y.argmax()]]