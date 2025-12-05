from scipy.io import loadmat
import os, torch, shutil, torchvision, optax
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp
from functools import partial

def preprocess(original_path="MPIIGaze/MPIIGaze/Data/Normalized", new_path="MPIIGaze_preprocessed2/"):
    """Run once."""
    # Split the MPIIGaze dataset into train/val/test sets.
    shutil.copytree(original_path, "MPIIGaze_preprocessed2/", dirs_exist_ok=True)
    for person in [p for p in os.listdir("MPIIGaze_preprocessed2/") if p.startswith("p")]:
        os.makedirs(f"MPIIGaze_preprocessed2/train/{person}")
        os.makedirs(f"MPIIGaze_preprocessed2/test/{person}")
        os.makedirs(f"MPIIGaze_preprocessed2/val/{person}")
        mats = filter(lambda x: x.endswith(".mat"), os.listdir(f"MPIIGaze_preprocessed2/{person}"))
        # Convert mat files to pt files for faster loading
        for mat in mats:
            data = loadmat(f"MPIIGaze_preprocessed2/{person}/{mat}")
            for side in ["left", "right"]:
                datum = data["data"][side].item()
                for i in range(len(datum["image"].item())):
                    img = torch.unsqueeze(torch.tensor(datum["image"].item()[i]).float()/255., -1)
                    pose = torch.tensor(datum["pose"].item()[i]).float()
                    gaze = torch.tensor(datum["gaze"].item()[i]).float()
                    # Split over train/val/test with 70/15/15 ratio
                    partition = [["test", "val"], ["train", "train"]][(i%20-14)//14][i%2]
                    torch.save((img, pose, gaze), f"MPIIGaze_preprocessed2/{partition}/{person}/{mat[:-4]}_{i}_{side}.pt")
            os.remove(f"MPIIGaze_preprocessed2/{person}/{mat}")

class MPIIGaze(Dataset):
    def __init__(self, path:str, n_clients:int, discrete:bool):
        # Each client gets data from one participant
        g = os.walk(path)
        self.clients = next(g)[1][:n_clients]
        self.files = {c: [] for c in self.clients}
        for (dirname, _, filenames) in g:
            if os.path.basename(dirname) in self.clients:
                for f in filenames:
                    self.files[os.path.basename(dirname)].append(os.path.join(dirname, f))
        self.discrete = discrete
        self.n_clients = n_clients

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min(len(v) for v in self.files.values())*self.n_clients 

    def __getitem__(self, idx):
        # Notice that the data is sorted per participant, which are interleaved here
        client = self.clients[idx%self.n_clients] 
        i = idx//self.n_clients
        # Load
        img, aux, label = torch.load(self.files[client][i])
        # # Flip if right eye
        if self.files[client][i].endswith("right.pt"):
            img = torch.flip(img, [1])
            label[1] = -label[1]
        # Label to pitch and yaw
        label = torch.asarray([torch.arcsin(-label[1]), torch.arctan2(-label[0], -label[2])]) # polar coordinates
        if not self.discrete: return label, img, aux
        # Process label into one of 16 regions
        nr = 3
        min_0, max_0, min_1, max_1 = -0.36719793, 0.3623084, -0.31378174, 0.38604215 # mins and maxs taken from training set
        min_0, max_0, min_1, max_1 = min_0*(1-1/nr), max_0*(1-1/nr), min_1*(1-1/nr), max_1*(1-1/nr)
        regions = torch.concat([
            torch.cartesian_prod(torch.linspace(min_0, max_0, nr), torch.linspace(min_1, max_1, nr)[:2]),
            torch.cartesian_prod(torch.linspace(min_0, max_0, nr), torch.linspace(min_1, max_1, nr)[2:])
        ]) 
        label = torch.abs(label - regions).sum(axis=1).argmin() # nearest region
        label = torch.eye(nr*nr)[label] # one-hot encode
        return label, img, aux

class ImageNet(Dataset):
    def __init__(self, path:str="./imagenet/Data/CLS-LOC/train/", n_clients:int=4):
        # Evenly divide all labels among clients
        g = os.walk(path)
        classes = next(g)[1]
        k, m = divmod(len(classes), n_clients)
        class_splits = {classes[idx]: c for c in range(n_clients) for idx in range(c*k+min(c,m), (c+1)*k+min(c+1,m))}
        self.data = {c: [] for c in range(n_clients)}
        for label, (dirname, _, filelist) in enumerate(g):
            client = class_splits[os.path.basename(dirname)]
            filelist = [(label, dirname, f) for f in filelist]
            self.data[client].extend(filelist)
        self.n_clients = n_clients

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.data.values()])*self.n_clients
    
    def __getitem__(self, idx):
        # Load datum
        c = idx % self.n_clients
        i = idx // self.n_clients
        label, dirname, filename = self.data[c][i]
        # Load image
        img = torchvision.io.decode_image(os.path.join(dirname, filename), mode="RGB").float() / 255.
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = img.swapaxes(0,2)
        # One-hot encode labels
        label = torch.eye(1000)[label]
        return label, img

def jax_collate(batch, n_clients:int, indiv_frac:float, skew:str)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Only one type of skew can be applied at a time, with the following options:
    - "feature": where indiv_frac==1 corresponds to clients having disjoint (heterogeneous) data distributions.
    - "overlap": where indiv_frac==0 corresponds to clients having disjoint samples of homogeneous distributions.
    - "label": where indiv_frac==1 corresponds to clients having disjoint label distributions.
    """
    # Collect, convert, and concat
    aux = not len(batch[0])==2
    if aux: 
        labels, imgs, auxs = zip(*batch)
        auxs = jnp.stack([jnp.asarray(aux, dtype=jnp.float32) for aux in auxs])
    else: 
        labels, imgs = zip(*batch)
    imgs = jnp.stack([jnp.asarray(img, dtype=jnp.float32) for img in imgs])
    labels = jnp.stack([jnp.asarray(label, dtype=jnp.float32) for label in labels])

    # Feature skew, by sharing a portion of skewed samples while retaining another portion of client-side samples
    if skew=="feature":
        indiv_stop = int(indiv_frac*len(imgs)*n_clients)
        clients_idxs = [jnp.asarray(list(range(i, indiv_stop, n_clients)) + list(range(indiv_stop, len(imgs)))) for i in range(n_clients)]
        if aux: clients_auxs = [auxs[idxs] for idxs in clients_idxs]
        clients_imgs = [imgs[idxs] for idxs in clients_idxs]
        clients_labels = [labels[idxs] for idxs in clients_idxs]
    
    # Overlap reduction, by sharing a portion of the total samples while retaining another portion of the total samples
    elif skew=="overlap":
        n_indiv = int(indiv_frac*len(imgs))
        clients_idxs = [jnp.asarray(list(range(i*n_indiv, (i+1)*n_indiv)) + list(range(n_clients*n_indiv, len(imgs)))) for i in range(n_clients)]
        if aux: clients_auxs = [auxs[idxs] for idxs in clients_idxs]
        clients_imgs = [imgs[idxs] for idxs in clients_idxs]
        clients_labels = [labels[idxs] for idxs in clients_idxs]

    # Label skew, by evenly dividing the samples into n directional groups, and then sharing a portion of the total samples
    # Assumes there is no order to the labels' values
    elif skew=="label":
        # Rank the non-shared samples by quadrant angle
        indiv_stop = int(indiv_frac*len(imgs)*n_clients)
        angles = (jnp.arctan2(*labels.T) + 2*jnp.pi) % (2*jnp.pi)
        sorted_idxs = jnp.argsort(angles[:indiv_stop])
        # Divide into n_clients groups
        group_size = indiv_stop//n_clients
        indiv_idxs = [list(sorted_idxs[i*group_size:(i+1)*group_size]) for i in range(n_clients)]
        shared_idxs = list(range(indiv_stop, len(imgs)))
        # Select
        if aux: clients_auxs = [auxs[jnp.asarray(idxs + shared_idxs)] for idxs in indiv_idxs]
        clients_imgs = [imgs[jnp.asarray(idxs + shared_idxs)] for idxs in indiv_idxs]
        clients_labels = [labels[jnp.asarray(idxs + shared_idxs)] for idxs in indiv_idxs]

    if aux: return jnp.stack(clients_labels, 0), jnp.stack(clients_imgs, 0), jnp.stack(clients_auxs, 0)
    return jnp.stack(clients_labels, 0), jnp.stack(clients_imgs, 0)

def get_data(skew:str="feature", batch_size=128, n_clients=4, beta:float=0, path="MPIIGaze_preprocessed", partition="train", discrete:bool|None=False, **kwargs)->DataLoader:
    assert beta>=0 and beta<=1, "Beta must be between 0 and 1"
    beta = 1-beta if skew=="overlap" else beta
    # Fractions derived
    n_indiv = int(batch_size*beta)
    n_shared = batch_size-n_indiv 
    new_batch_size = n_indiv*n_clients + n_shared
    indiv_frac = n_indiv / new_batch_size
    # Dataset type
    if "MPIIGaze" in path:
        dataset = MPIIGaze(n_clients=n_clients, path=os.path.join(path, partition), discrete=discrete)
        assert skew in ["feature", "overlap", "label"], "Skew must be one of 'feature', 'overlap', or 'label'. For no skew, specify beta=0."
    else:
        dataset = ImageNet(path=os.path.join(path, partition), n_clients=n_clients)
        assert skew in ["overlap", "label"], "Skew must be either 'overlap' or 'label'. For no skew, specify beta=0."
    # Iterable batches
    return DataLoader(
        dataset,
        batch_size=new_batch_size,
        collate_fn=partial(jax_collate, n_clients=n_clients, indiv_frac=indiv_frac, skew=skew),
        shuffle=False,
        drop_last=True,
        **kwargs
    )