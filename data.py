from scipy.io import loadmat
import os, torch, shutil, torchvision, optax, hashlib
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
    def __init__(self, path:str, n_clients:int):
        # Each client gets data from one participant
        g = os.walk(path)
        self.clients = next(g)[1][:n_clients]
        self.files = {c: [] for c in self.clients}
        for (dirname, _, filenames) in g:
            if os.path.basename(dirname) in self.clients:
                for f in filenames:
                    self.files[os.path.basename(dirname)].append(os.path.join(dirname, f))
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
        # Flip if right eye
        if self.files[client][i].endswith("right.pt"):
            img = torch.flip(img, [1])
            label[1] = -label[1]
        # Label to pitch and yaw
        label = torch.asarray([torch.arcsin(-label[1]), torch.arctan2(-label[0], -label[2])]) # polar coordinates
        return label, img, aux

class ImageNet(Dataset):
    def __init__(self, path:str="./imagenet/Data/CLS-LOC/train/", n_clients:int=4, n_classes:int=1000):
        g = os.walk(path)
        # Get class names and limit to n_classes by skipping
        classes = next(g)[1]
        classes = classes[::len(classes)//n_classes]
        # Dedicate a client to each class
        k, m = divmod(len(classes), n_clients)
        class_splits = {classes[idx]: client for client in range(n_clients) for idx in range(client*k+min(client,m), (client+1)*k+min(client+1,m))}
        # Assign sample paths to each client
        self.data = {c: [] for c in range(n_clients)}
        label_idx = 0
        for dirname, _, filelist in g:
            classname = os.path.basename(dirname)
            # Only n_classes
            if classname in classes:
                # Assign to client
                client = class_splits[classname]
                # Insert at interleaved indices so that samples are not ordered by class (note: deterministic)
                for i, file in enumerate(filelist):
                    self.data[client].insert(
                        i*(label_idx-n_classes//n_clients*client), 
                        (label_idx, os.path.join(dirname, file))
                    )
                label_idx += 1
        # Misc attributes
        self.n_clients = n_clients
        self.n_classes = n_classes

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.data.values()])*self.n_clients
    
    def __getitem__(self, idx):
        # Load datum
        c = idx % self.n_clients
        i = idx // self.n_clients
        label, filepath = self.data[c][i]
        # Load image
        img = torchvision.io.decode_image(filepath, mode="RGB").float() / 255.
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = img.swapaxes(0,2)
        # One-hot encode labels
        label = torch.eye(self.n_classes)[label]
        return label, img

class CityScapes(Dataset):
    def __init__(self, path:str="./cityscapes/", partition:str="train", n_clients:int=4):
        g = os.walk(os.path.join(path, "leftImg8bit", partition))
        # Get location names
        cities = next(g)[1]
        # Dedicate a client to each city
        k, m = divmod(len(cities), n_clients)
        city_splits = {cities[idx]: client for client in range(n_clients) for idx in range(client*k+min(client,m), (client+1)*k+min(client+1,m))}
        # Assign sample paths to each client
        self.data = {client: [] for client in range(n_clients)}
        for dirname, _, filelist in g:
            # Assign to client
            cityname = os.path.basename(dirname)
            client = city_splits[cityname]
            filelist = [os.path.join(dirname, filename) for filename in filelist]
            self.data[client].extend(filelist)
        # Shuffle so that samples are not ordered by city (note: deterministic)
        for c in range(n_clients):
            self.data[c].sort(key=lambda x: hashlib.sha256(str(x).encode()).hexdigest())
        # Misc attributes
        self.n_clients = n_clients
        # Label mapping extracted from https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L62
        # using `i=0; torch.tensor([0 if l.ignoreInEval else (i:=i+1) for l in labels])`
        self.conversion = torch.tensor([0,0,0,0,0,0,0,1,2,0,0,3,4,5,0,0,0,6,0,7,8,9,10,11,12,13,14,15,16,0,0,17,18,19,0])

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.data.values()])*self.n_clients
    
    def __getitem__(self, idx):
        # Load datum
        c = idx % self.n_clients
        i = idx // self.n_clients
        filepath = self.data[c][i]
        # Load image
        img = torchvision.io.decode_image(filepath, mode="RGB").float() / 255.
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = torch.permute(img, (1,2,0))
        # Load label image
        labelpath = filepath.replace("leftImg8bit", "gtFine").replace(".png", "_labelIds.png")
        indices = torchvision.io.decode_image(labelpath).long()
        indices = self.conversion[indices]
        indices = torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.NEAREST)(indices)
        indices = torchvision.transforms.CenterCrop(224)(indices).squeeze()
        label = torch.zeros((*indices.shape, 20), dtype=torch.float32)
        label = label.scatter(-1, indices.unsqueeze(-1), 1.)
        return label, img

def gaze_collate(batch, n_clients:int, beta:float, skew:str)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Only one type of skew can be applied at a time, with the following options:
    - "feature": where indiv_frac==1 corresponds to clients having disjoint (heterogeneous) data distributions.
    - "overlap": where indiv_frac==0 corresponds to clients having disjoint samples of homogeneous distributions.
    - "label": where indiv_frac==1 corresponds to clients having disjoint label distributions.
    """
    # Collect, convert, and concat
    labels, imgs, auxs = zip(*batch)
    auxs = jnp.stack([jnp.asarray(aux, dtype=jnp.float32) for aux in auxs])
    imgs = jnp.stack([jnp.asarray(img, dtype=jnp.float32) for img in imgs])
    labels = jnp.stack([jnp.asarray(label, dtype=jnp.float32) for label in labels])

    # Feature skew, by drawing a portion of the samples from the same feature distribution
    # Feature groups are interleaved by the Dataset
    if skew=="feature":
        same_dist_idxs = [range(c*int((1-beta)*len(imgs)/n_clients), (c+1)*int((1-beta)*len(imgs)/n_clients)) for c in range(n_clients)]
        diff_dist_idxs = [range(int((1-beta)*len(imgs))+c, len(imgs), n_clients) for c in range(n_clients)]
        idxs = [jnp.asarray(list(same_dist_idxs[c]) + list(diff_dist_idxs[c])) for c in range(n_clients)]
        clients_auxs = [auxs[idx] for idx in idxs]
        clients_imgs = [imgs[idx] for idx in idxs]
        clients_labels = [labels[idx] for idx in idxs]
    
    # Overlap reduction, by sharing a portion of the total samples while retaining another portion of the total samples
    elif skew=="overlap":
        n_indiv = int(beta/(beta+(1-beta)*n_clients) * len(imgs))
        diff_dist_idxs = [range(c*n_indiv, (c+1)*n_indiv) for c in range(n_clients)]
        idxs = [jnp.asarray(list(diff_dist_idxs[c]) + list(range(n_clients*n_indiv, len(imgs)))) for c in range(n_clients)]
        clients_auxs = [auxs[idx] for idx in idxs]
        clients_imgs = [imgs[idx] for idx in idxs]
        clients_labels = [labels[idx] for idx in idxs]

    # Label skew, by evenly dividing the samples into n directional groups, and then drawing from the same group
    # Assumes there is no order to the labels' values
    elif skew=="label":
        # Rank the non-shared samples by quadrant angle
        num_homo = int((1-beta)*len(imgs))
        angles = (jnp.arctan2(*labels.T) + 2*jnp.pi) % (2*jnp.pi)
        sorter = jnp.argsort(angles[:num_homo])
        diff_dist_idxs = [sorter[c*num_homo//n_clients : (c+1)*num_homo//n_clients] for c in range(n_clients)]
        # Divide the remainder indifferently (i.e., simply splitting indices)
        same_dist_idxs = [range(num_homo+c*num_homo//n_clients, num_homo+(c+1)*num_homo//n_clients) for c in range(n_clients)]
        # Select
        idxs = [jnp.asarray(list(same_dist_idxs[c]) + list(diff_dist_idxs[c])) for c in range(n_clients)]
        clients_auxs = [auxs[idx] for idx in idxs]
        clients_imgs = [imgs[idx] for idx in idxs]
        clients_labels = [labels[idx] for idx in idxs]

    return jnp.stack(clients_labels, 0), jnp.stack(clients_imgs, 0), jnp.stack(clients_auxs, 0)

def imagenet_collate(batch, n_clients:int, beta:float, skew:str)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Only one type of skew can be applied at a time, with the following options:
    - "overlap": where indiv_frac==0 corresponds to clients having disjoint samples of homogeneous distributions.
    - "label": where indiv_frac==1 corresponds to clients having disjoint label distributions.
    """
    # Collect, convert, and concat
    labels, imgs = zip(*batch)
    imgs = jnp.stack([jnp.asarray(img, dtype=jnp.float32) for img in imgs])
    labels = jnp.stack([jnp.asarray(label, dtype=jnp.float32) for label in labels])

    # Label skew, by drawing a portion of the samples from the same distribution
    if skew=="label":
        same_dist_idxs = [range(c*int((1-beta)*len(imgs)/n_clients), (c+1)*int((1-beta)*len(imgs)/n_clients)) for c in range(n_clients)]
        diff_dist_idxs = [range(int((1-beta)*len(imgs))+c, len(imgs), n_clients) for c in range(n_clients)]
        idxs = [jnp.asarray(list(same_dist_idxs[c]) + list(diff_dist_idxs[c])) for c in range(n_clients)]
        clients_imgs = [imgs[idx] for idx in idxs]
        clients_labels = [labels[idx] for idx in idxs]
    
    # Overlap reduction, by sharing a portion of the total samples while retaining another portion of the total samples
    elif skew=="overlap":
        n_indiv = int(beta/(beta+(1-beta)*n_clients) * len(imgs))
        diff_dist_idxs = [range(c*n_indiv, (c+1)*n_indiv) for c in range(n_clients)]
        idxs = [jnp.asarray(list(diff_dist_idxs[c]) + list(range(n_clients*n_indiv, len(imgs)))) for c in range(n_clients)]
        clients_imgs = [imgs[idx] for idx in idxs]
        clients_labels = [labels[idx] for idx in idxs]

    return jnp.stack(clients_labels, 0), jnp.stack(clients_imgs, 0)

def fetch_data(skew:str="overlap", batch_size=128, n_clients=4, beta:float=0, dataset:int=0, partition:str="train", n_classes=1000, **kwargs)->DataLoader:
    assert beta>=0 and beta<=1, "Beta must be between 0 and 1"
    beta = 1-beta if skew=="overlap" else beta
    # Increase batch size and account for floor division
    if skew=="overlap": 
        new_batch_size = int(batch_size*beta + batch_size*(1-beta)*n_clients)
        new_beta = beta
    else: 
        new_batch_size = batch_size*n_clients
        new_beta = round(beta * new_batch_size / n_clients) * n_clients / new_batch_size
    # Dataset type
    if dataset==0:
        dataset = MPIIGaze(n_clients=n_clients, path=os.path.join("MPIIGaze_preprocessed", partition))
        assert skew in ["feature", "overlap", "label"], "Skew must be one of 'feature', 'overlap', or 'label'. For no skew, specify beta=0."
        collate = partial(gaze_collate, n_clients=n_clients, beta=new_beta, skew=skew)
    elif dataset==1:
        dataset = ImageNet(path=os.path.join("imagenet/Data/CLS-LOC", partition), n_clients=n_clients, n_classes=n_classes)
        assert skew in ["overlap", "label"], "Skew must be either 'overlap' or 'label'. For no skew, specify beta=0."
        collate = partial(imagenet_collate, n_clients=n_clients, beta=new_beta, skew=skew)
    else:
        assert n_clients<=3, "CityScapes supports a maximum of 3 clients."
        dataset = CityScapes(path="cityscapes/", partition=partition, n_clients=n_clients)
        assert skew in ["overlap", "feature"], "Skew must be either 'overlap' or 'feature'. For no skew, specify beta=0."
        skew = "label" if skew=="feature" else "overlap" # functional implementation (i.e., interleaving) is equivalent
        collate = partial(imagenet_collate, n_clients=n_clients, beta=new_beta, skew=skew)
    # Iterable batches
    return DataLoader(
        dataset,
        batch_size=new_batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=True,
        **kwargs
    )