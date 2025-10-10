from scipy.ndimage import rotate
from scipy.io import loadmat
from sklearn.datasets import load_digits
import jax, os, torchvision, torch, numpy as np, cv2, shutil
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp
from functools import partial
from itertools import product

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
    def __init__(self, data_path, indiv_lab, shared_lab, batch_size, n):
        # Get all image paths and sort them by client/shared
        g = os.walk(data_path, topdown=True)
        self.classes = next(g)[1]
        classesbyclient = {c: [self.classes[i] for i in indiv_lab[c]] for c in range(n)}
        sharedclasses = [self.classes[i] for i in shared_lab]
        paths = [os.path.join(dirname, f) for (dirname, _, filenames) in g for f in filenames]
        paths = [x for _, x in sorted(enumerate(paths), key=lambda x: x[0]%598)] # shuffle with knowledge of fixed number of samples per class
        self.pathsbyclient = {client: [file for file in paths if file.rsplit("/")[-2] in classesbyclient[client]] for client in range(n)}
        self.sharedpaths = [file for file in paths if file.rsplit("/")[-2] in sharedclasses]
        self.n = n
        self.batch_size = batch_size
        self.frac = len(indiv_lab[0])/len(self.classes)

    def __len__(self):
        return min(len(v) for v in self.pathsbyclient.values())*self.n + len(self.sharedpaths)

    def __getitem__(self, idx):
        # If the sample index within this batch is less than the required number of individual samples
        required_indiv = self.n*self.frac*self.batch_size
        required_indiv = int(required_indiv - required_indiv%self.n)
        required_shared = self.batch_size-required_indiv
        idx_in_batch = idx-idx//self.batch_size*self.batch_size
        if idx_in_batch < required_indiv:
            # Fetch the next individual sample
            fp = self.pathsbyclient[idx%self.n][(idx-required_shared)//self.n]
        else:
            # Fetch a shared sample
            fp = self.sharedpaths[idx-int(required_indiv)]
        img = torchvision.io.read_image(fp).float()/255.
        label = self.classes.index(fp.split("/")[-2].rstrip(".JPEG"))
        label = torch.eye(len(self.classes))[label]
        return img, label

def jax_collate(batch, n, feature_beta, indiv_stop, indiv_frac):
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

    # Fetch corresponding idx in this batch
    indiv_stop = int(indiv_stop - indiv_stop%n)
    clients_idxs = [list(range(i, indiv_stop, n)) + list(range(indiv_stop, len(imgs))) for i in range(n)]
    clients_imgs = [imgs[clients_idxs[i]] for i in range(n)]
    clients_labels = jnp.stack([labels[jnp.asarray(clients_idxs[i])] for i in range(n)], axis=0)

    # Decrease sample overlap (doing this batch-wise better reflects the overlap within the gradient) NOTE: overwrites label skew
    if indiv_stop==0:
        n_per_client = int(indiv_frac * len(imgs))
        clients_idxs = [list(range(i*n_per_client, (i+1)*n_per_client)) + list(range(n*n_per_client, len(imgs))) for i in range(n)]
        clients_imgs = [imgs[clients_idxs[i]] for i in range(n)]
        clients_labels = jnp.stack([labels[jnp.asarray(clients_idxs[i])] for i in range(n)], axis=0)

    # Create feature skew with elastic deformation
    angles = [i*2*np.pi/n for i in range(n)]
    clients_imgs = jnp.stack([
        perspective_shift(clients_imgs[i], angles[i], feature_beta) 
    for i in range(n)], axis=0)

    return clients_imgs, clients_labels

def create_imagenet(path="./data/Data/CLS-LOC/train", n=4, feature_beta=0., label_beta=0., sample_overlap=1., batch_size=64, **kwargs):
    assert label_beta==0. or sample_overlap==1., "Sample overlap reduction not implemented for label_beta>0 because they are codependent"
    assert label_beta==0. or batch_size%n==0, "Original batch size must be divisible by n, so that labels can be evenly distributed"
    # Label skew fractions derived from ( sha + n * ind = 1 ) and ( ind / (sha + ind) = label_beta )
    shared_frac = 0. if np.allclose(label_beta, 1.) else 1 / (1+(1-label_beta)/label_beta*n) # numerical stability
    indiv_frac = 1. if np.allclose(label_beta, 1.) else 1 / (label_beta/(1-label_beta)+n) #label_beta/(1-label_beta)*shared_frac
    # Create label assignment (assuming 1000 classes)
    indiv_lab = [jnp.arange(i*int(1000*indiv_frac), (i+1)*int(1000*indiv_frac)) for i in range(n)]
    shared_lab = jnp.arange(n*int(1000*indiv_frac), 1000)
    # Recalculate batch size (derived from: new_batch_size * indiv_frac + new_batch_size * shared_frac = batch_size)
    if sample_overlap < 1.: # thus only applies if label_beta=0
        indiv_frac = 1 / (sample_overlap/(1-sample_overlap)+n)
        shared_frac = 1 / (1+(1-sample_overlap)/sample_overlap*n)
    batch_size = batch_size / (shared_frac + indiv_frac)
    batch_size = int(batch_size - batch_size%n) # TODO: resulting batch is not 64
    return DataLoader(
        Imagenet(path, indiv_lab, shared_lab, batch_size, n),
        batch_size=batch_size,
        collate_fn=partial(jax_collate, n=n, feature_beta=feature_beta, indiv_stop=len(indiv_lab[0])/1000*batch_size*n, indiv_frac=indiv_frac),
        shuffle=False,
        **kwargs
    )

class_name = lambda ds, y: {class_name: name for line in open("data/mapping.txt").read().splitlines() for class_name, name in [line.split(" ", 1)]}[ds.dataset.classes[y.argmax()]]

class MPIIGaze(Dataset):
    def __init__(self, path:str, n_clients:int):
        self.n_clients = n_clients
        g = os.walk(path)
        self.clients = next(g)[1][:n_clients]
        self.files = {c: [] for c in self.clients}
        for (dirname, _, filenames) in g:
            if os.path.basename(dirname) in self.clients:
                for f in filenames:
                    self.files[os.path.basename(dirname)].append(os.path.join(dirname, f))

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min(len(v) for v in self.files.values())*self.n_clients 

    def __getitem__(self, idx):
        # Notice that the data is sorted per participant, which are interleaved here
        client = self.clients[idx%self.n_clients] 
        i = idx//self.n_clients
        img, aux, label = torch.load(self.files[client][i])
        img = img if self.files[client][i].endswith("left.pt") else torch.flip(img, [1])
        label = torch.asarray([torch.arcsin(-label[1]), torch.arctan2(-label[0], -label[2])]) # polar coordinates
        regions = torch.cartesian_prod(torch.linspace(-0.3675091, 0.0831264, 4), torch.linspace(-0.31378174, 0.38604215, 4)) #+ .5 # mins and maxs of training set
        label = torch.abs(label - regions).sum(axis=1).argmin() # nearest region
        label = torch.eye(4*4)[label] # one-hot encode
        return img, aux, label

def jax_collate(batch, n_clients:int, indiv_frac:float, skew:str)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Only one type of skew can be applied at a time, with the following options:
    - "feature": where indiv_frac==1 corresponds to clients having disjoint (heterogeneous) data distributions.
    - "overlap": where indiv_frac==0 corresponds to clients having disjoint samples of homogeneous distributions.
    - "label": where indiv_frac==1 corresponds to clients having disjoint label distributions.
    """
    # Collect, convert, and concat
    imgs, auxs, labels = zip(*batch)
    auxs = jnp.stack([jnp.asarray(aux, dtype=jnp.float32) for aux in auxs])
    imgs = jnp.stack([jnp.asarray(img, dtype=jnp.float32) for img in imgs])
    labels = jnp.stack([jnp.asarray(label, dtype=jnp.float32) for label in labels])

    # Feature skew, by sharing a portion of skewed samples while retaining another portion of client-side samples
    if skew=="feature":
        indiv_stop = int(indiv_frac*len(imgs)*n_clients)
        clients_idxs = [jnp.asarray(list(range(i, indiv_stop, n_clients)) + list(range(indiv_stop, len(imgs)))) for i in range(n_clients)]
        clients_auxs = [auxs[idxs] for idxs in clients_idxs]
        clients_imgs = [imgs[idxs] for idxs in clients_idxs]
        clients_labels = [labels[idxs] for idxs in clients_idxs]
    
    # Overlap reduction, by sharing a portion of the total samples while retaining another portion of the total samples
    elif skew=="overlap":
        n_indiv = int(indiv_frac*len(imgs))
        clients_idxs = [jnp.asarray(list(range(i*n_indiv, (i+1)*n_indiv)) + list(range(n_clients*n_indiv, len(imgs)))) for i in range(n_clients)]
        clients_auxs = [auxs[idxs] for idxs in clients_idxs]
        clients_imgs = [imgs[idxs] for idxs in clients_idxs]
        clients_labels = [labels[idxs] for idxs in clients_idxs]

    # Label skew, by evenly dividing the samples into n_clients directional groups, and then sharing a portion of the total samples
    # Assumes there is no order to the labels' values
    elif skew=="label":
        # Sort the non-shared samples by label value
        indiv_stop = int(indiv_frac*len(imgs)*n_clients)
        sorted_idxs = jnp.argsort(labels[:indiv_stop].argmax(axis=1))
        # Divide into n_clients groups
        group_size = len(sorted_idxs)//n_clients
        indiv_idxs = [list(sorted_idxs[i*group_size:(i+1)*group_size]) for i in range(n_clients)]
        shared_idxs = list(range(indiv_stop, len(imgs)))
        # Select
        clients_auxs = [auxs[jnp.asarray(idxs + shared_idxs)] for idxs in indiv_idxs]
        clients_imgs = [imgs[jnp.asarray(idxs + shared_idxs)] for idxs in indiv_idxs]
        clients_labels = [labels[jnp.asarray(idxs + shared_idxs)] for idxs in indiv_idxs]

    return jnp.stack(clients_imgs, 0), jnp.stack(clients_auxs, 0), jnp.stack(clients_labels, 0)

def get_gaze(skew:str="feature", batch_size=128, n_clients=4, beta:float=0, path="MPIIGaze_preprocessed", partition="train", **kwargs)->DataLoader:
    assert skew in ["feature", "overlap", "label"], "Skew must be one of 'feature', 'overlap', or 'label'. For no skew, specify beta=0."
    assert beta>=0 and beta<=1, "Beta must be between 0 and 1"
    beta = 1-beta if skew=="overlap" else beta
    # Fractions derived
    n_indiv = int(batch_size*beta) # TODO: this doesn't account for the 1/nth in n_shared that is also an individual client's data. Then again, should it?
    n_shared = batch_size-n_indiv 
    new_batch_size = n_indiv*n_clients + n_shared
    indiv_frac = n_indiv / new_batch_size
    # Iterable batches
    return DataLoader(
        MPIIGaze(n_clients=n_clients, path=os.path.join(path, partition)),
        batch_size=new_batch_size,
        collate_fn=partial(jax_collate, n_clients=n_clients, indiv_frac=indiv_frac, skew=skew),
        shuffle=False,
        drop_last=True,
        **kwargs
    )

def preprocess(original_path="MPIIGaze/MPIIGaze/Data/Normalized"):
    """Run once."""
    # Split the MPIIGaze dataset into train/val/test sets.
    fps = [f for f in os.listdir(original_path) if f.startswith("p")]
    for person in fps:
        mats = list(filter(lambda x: x.endswith(".mat"), os.listdir(os.path.join(original_path, person))))
        os.makedirs(f"MPIIGaze_preprocessed/train/{person}", exist_ok=True)
        os.makedirs(f"MPIIGaze_preprocessed/val/{person}", exist_ok=True)
        os.makedirs(f"MPIIGaze_preprocessed/test/{person}", exist_ok=True)
        for i, mat in enumerate(mats):
            if i < .15*len(mats):
                shutil.copy(os.path.join(original_path, person, mat), f"MPIIGaze_preprocessed/test/{person}/{mat}")
            elif i < .3*len(mats):
                shutil.copy(os.path.join(original_path, person, mat), f"MPIIGaze_preprocessed/val/{person}/{mat}")
            else:
                shutil.copy(os.path.join(original_path, person, mat), f"MPIIGaze_preprocessed/train/{person}/{mat}")
    # Convert .mat files to .pt files for faster loading
    for split in ["train", "val", "test"]:
        for person in fps:
            mats = filter(lambda x: x.endswith(".mat"), os.listdir(f"MPIIGaze_preprocessed/{split}/{person}"))
            for mat in mats:
                data = loadmat(f"MPIIGaze_preprocessed/{split}/{person}/{mat}")
                for side in ["left", "right"]:
                    datum = data["data"][side].item()
                    for i in range(len(datum["image"].item())):
                        img = torch.unsqueeze(torch.tensor(datum["image"].item()[i]).float()/255., -1)
                        pose = torch.tensor(datum["pose"].item()[i]).float()
                        gaze = torch.tensor(datum["gaze"].item()[i]).float()
                        torch.save((img, pose, gaze), f"MPIIGaze_preprocessed/{split}/{person}/{mat[:-4]}_{i}_{side}.pt")
                os.remove(f"MPIIGaze_preprocessed/{split}/{person}/{mat}")