from scipy.io import loadmat
import os, torch, shutil
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp
from functools import partial

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

class MPIIGaze(Dataset):
    def __init__(self, path:str, n_clients:int, transform):
        self.transform = transform
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
        # Load
        img, aux, label = torch.load(self.files[client][i])
        # Flip if right eye
        img = img if self.files[client][i].endswith("left.pt") else torch.flip(img, [1])
        img = self.transform(img)
        # Process label into one of 16 regions
        label = torch.asarray([torch.arcsin(-label[1]), torch.arctan2(-label[0], -label[2])]) # polar coordinates
        regions = torch.concat([
            torch.cartesian_prod(torch.linspace(-0.3675091, 0.0831264, 4), torch.linspace(-0.31378174, 0.38604215, 4)[:2]),
            torch.cartesian_prod(torch.linspace(-0.3675091, 0.0831264, 4), torch.linspace(-0.31378174, 0.38604215, 4)[2:])
        ]) # mins and maxs of training set
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

    # Label skew, by evenly dividing the samples into n directional groups, and then sharing a portion of the total samples
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

def get_gaze(skew:str="feature", batch_size=128, n_clients=4, beta:float=0, path="MPIIGaze_preprocessed", partition="train", transform=lambda x:x, **kwargs)->DataLoader:
    assert skew in ["feature", "overlap", "label"], "Skew must be one of 'feature', 'overlap', or 'label'. For no skew, specify beta=0."
    assert beta>=0 and beta<=1, "Beta must be between 0 and 1"
    beta = 1-beta if skew=="overlap" else beta
    # Fractions derived
    n_indiv = int(batch_size*beta)
    n_shared = batch_size-n_indiv 
    new_batch_size = n_indiv*n_clients + n_shared
    indiv_frac = n_indiv / new_batch_size
    # Iterable batches
    return DataLoader(
        MPIIGaze(n_clients=n_clients, path=os.path.join(path, partition), transform=transform),
        batch_size=new_batch_size,
        collate_fn=partial(jax_collate, n_clients=n_clients, indiv_frac=indiv_frac, skew=skew),
        shuffle=False,
        drop_last=True,
        **kwargs
    )