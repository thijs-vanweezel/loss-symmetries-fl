import os, torch, shutil, torchvision, hashlib
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, default_collate
from jax import numpy as jnp
from functools import partial
from collections import defaultdict
from tqdm.auto import tqdm
torch.multiprocessing.set_sharing_strategy("file_system")

def train_aug(img:torch.Tensor, seed, mask:torch.Tensor=None):
    """Deterministic train augmentations"""
    # Flip
    if torch.randint(0, 2, (), generator=seed).item():
        img = img.flip((-1,))
        if mask is not None: mask = mask.flip((-1,))
    # Crop
    i = torch.randint(0, img.shape[1]-224+1, (), generator=seed).item()
    j = torch.randint(0, img.shape[2]-224+1, (), generator=seed).item()
    img = functional.crop(img, i, j, 224, 224)
    if mask is not None: mask = functional.crop(mask, i, j, 224, 224)
    # Color jitter
    brightness = 0.8 + torch.rand((), generator=seed).item()*0.4
    img *= brightness
    contrast = 0.8 + torch.rand((), generator=seed).item()*0.4
    mean = torch.mean(img, (1,2), keepdims=True)
    img = (img - mean) * contrast + mean
    return img, mask

class MPIIGaze(Dataset):
    def __init__(self, path:str="MPIIGaze_preprocessed", n_clients:int=4, partition:str="train", originalpath:str="MPIIGaze/MPIIGaze/Data/Normalized"):
        # Create split
        if not os.path.exists(path):
            self.preprocess(originalpath, path)
        # Each client gets data from one participant
        path = os.path.join(path, partition)
        g = os.walk(path)
        self.clients = next(g)[1][:n_clients]
        self.files = {c: [] for c in self.clients}
        for (dirname, _, filenames) in g:
            if os.path.basename(dirname) in self.clients:
                for f in filenames:
                    self.files[os.path.basename(dirname)].append(os.path.join(dirname, f))
        self.n_clients = n_clients
         
    @staticmethod
    def preprocess(original_path:str, new_path:str):
        # Split the MPIIGaze dataset into train/val/test sets.
        shutil.copytree(original_path, new_path, dirs_exist_ok=True)
        for person in tqdm([p for p in os.listdir(new_path) if p.startswith("p")], leave=False):
            os.makedirs(os.path.join(new_path, "train", person))
            os.makedirs(os.path.join(new_path, "test", person))
            os.makedirs(os.path.join(new_path, "val", person))
            mats = filter(lambda x: x.endswith(".mat"), os.listdir(os.path.join(new_path, person)))
            # Convert mat files to pt files for faster loading
            for mat in mats:
                data = loadmat(os.path.join(new_path, person, mat))
                for side in ["left", "right"]:
                    datum = data["data"][side].item()
                    for i in range(len(datum["image"].item())):
                        img = torch.unsqueeze(torch.tensor(datum["image"].item()[i]).float()/255., -1)
                        pose = torch.tensor(datum["pose"].item()[i]).float()
                        gaze = torch.tensor(datum["gaze"].item()[i]).float()
                        # Split over train/val/test with 70/15/15 ratio
                        partition = [["test", "val"], ["train", "train"]][(i%20-14)//14][i%2]
                        torch.save((img, pose, gaze), os.path.join(new_path, partition, person, f"{mat[:-4]}_{i}_{side}.pt"))
                os.remove(os.path.join(new_path, person, mat))

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
    def __init__(self, path:str="/data/bucket/traincombmodels/imnetproc", partition:str="train", n_clients:int=4, 
                 n_classes:int=1000, originalpath:str="/data/bucket/traincombmodels/imagenet", seed=None):
        # Set random seed
        if seed is not None: self.seed = seed 
        else:
            self.seed = torch.Generator()
            self.seed.manual_seed(42)
        # Augmentations
        self.val_crop = v2.CenterCrop(224)
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        # Create split
        if not os.path.exists(path):
            self.repartition(originalpath, path)
        self.partition = partition
        g = os.walk(os.path.join(path, partition))
        _ = next(g)
        # Get class names and limit to n_classes by skipping
        with open(os.path.join(path, "mapping.txt")) as f:
            classes = f.readlines()
        classes = classes[::len(classes)//n_classes]
        classes = {line.split()[0]: i for i, line in enumerate(classes)}
        # Assign sample paths to each client
        self.data = {c: [] for c in range(n_clients)}
        client = 0
        for dirname, _, filelist in tqdm(g, leave=False, total=n_classes):
            classname = os.path.basename(dirname)
            # Only n_classes
            if classname in classes.keys():
                class_idx = classes[classname]
                # Insert at interleaved indices so that samples are not ordered by class (note: deterministic)
                for i, file in enumerate(filelist):
                    self.data[client].insert(
                        i*(class_idx-n_classes//n_clients*client), 
                        (class_idx, os.path.join(dirname, file))
                    )
                client = (client+1) % n_clients
        # Misc attributes
        self.n_clients = n_clients
        self.n_classes = n_classes

    @staticmethod
    def repartition(originalpath, newpath, train_frac=0.8):
        """Divides the train partition into train/val/test"""
        os.makedirs(os.path.join(newpath, "train"))
        os.makedirs(os.path.join(newpath, "val"))
        os.makedirs(os.path.join(newpath, "test"))
        for classname in tqdm(os.listdir(os.path.join(originalpath, "Data", "CLS-LOC", "train")), leave=False):
            os.mkdir(os.path.join(newpath, "train", classname))
            os.mkdir(os.path.join(newpath, "val", classname))
            os.mkdir(os.path.join(newpath, "test", classname))
            files = os.listdir(os.path.join(originalpath, "Data", "CLS-LOC", "train", classname))
            for i, file in enumerate(files):
                partition = "train" if i<len(files)*train_frac else \
                "val" if i<len(files)*(train_frac+(1-train_frac)/2) else "test"
                shutil.copy(os.path.join(originalpath, "Data", "CLS-LOC", "train", classname, file), os.path.join(newpath, partition, classname))
    
    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.data.values()])*self.n_clients
    
    def __getitem__(self, idx):
        # Load datum
        c = idx % self.n_clients
        i = idx // self.n_clients
        label, img_path = self.data[c][i]
        with open(img_path, "rb") as f:
            bytesdata = torch.frombuffer(f.read(), dtype=torch.uint8)
        img = torchvision.io.decode_image(bytesdata, mode="RGB").half() / 255.
        # Apply augmentations
        img = functional.resize(img, 256)
        self.normalize(img)
        if self.partition=="train": 
            img, _ = train_aug(img, seed=self.seed)
            self.seed.manual_seed(torch.randint(0, int(1e6), (), generator=self.seed).item())
        else: img = self.val_crop(img)
        # Change to HWC
        img = torch.permute(img, (1,2,0))
        return label, img

class OxfordPets(Dataset):
    def __init__(self, path:str="/data/bucket/traincombmodels/oxford_pets", partition="train", seed=None, n_clients=4):
        self.n_clients = n_clients
        self.partition = partition
        if seed is not None: self.seed = seed 
        else:
            self.seed = torch.Generator()
            self.seed.manual_seed(42)
        # Load filenames and breed
        with open(os.path.join(path, "annotations", "list.txt")) as f:
            lines = f.readlines()[6:]
        # Store in dict per breed
        classes_per_client = {i+1:i%n_clients for i in range(37)}
        self.files = defaultdict(list)
        for i, line in enumerate(tqdm(lines, leave=False)):
            # Deterministic train/val/test split which likely retains each breed in each partition
            if ["train", "val", "test"][(j:=(i%20-14)//14+1)+(j*i%2)]!=partition: continue
            filename, classint, *_ = line.strip().split(" ")
            # Skip two corrupted files
            if filename in ["beagle_116", "chihuahua_121"]: continue
            client = classes_per_client[classint:=int(classint)]
            # Load in ram (move to __getitem__ if problematic)
            img = torchvision.io.decode_image(os.path.join(path, "images", filename+".jpg"), mode="RGB").float() / 255.
            img = functional.resize(img, 256)
            mask = torchvision.io.decode_image(os.path.join(path, "annotations", "trimaps", filename+".png")).long() - 1
            mask = functional.resize(mask, 256, interpolation=v2.InterpolationMode.NEAREST, antialias=False)
            self.files[client].append((img, mask))
        for client in self.files:
            self.files[client].sort(key=lambda x: hashlib.sha256(str(x).encode()).hexdigest())
        # Deterministic val augs
        self.val_aug = v2.CenterCrop(224)

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.files.values()])*self.n_clients

    def __getitem__(self, idx: int):
        client = idx % len(self.files)
        i = idx // len(self.files)
        img, mask = self.files[client][i]
        # Apply augmentations
        if self.partition=="train": 
            img, mask = train_aug(img, self.seed, mask)
            self.seed.manual_seed(torch.randint(0, int(1e6), (), generator=self.seed).item())
        else:
            img = self.val_aug(img)
            mask = self.val_aug(mask)
        # Change to HWC
        img = torch.permute(img, (1,2,0))
        mask = torch.squeeze(mask)
        return mask, img

class CelebA(Dataset):
    def __init__(self, path:str="/data/bucket/traincombmodels/celeba", partition="train", n_clients=4, seed=None):
        self.n_clients = n_clients
        self.partition = partition
        # Set random seed for augmentations
        if seed is not None: self.seed = seed 
        else:
            self.seed = torch.Generator()
            self.seed.manual_seed(42)
        # Augmentations
        self.val_crop = v2.CenterCrop(224)
        # Load filenames and labels
        with open(os.path.join(path, "identity_CelebA.txt")) as f:
            persons = f.readlines()
        with open(os.path.join(path, "list_attr_celeba.txt")) as f:
            attributes = f.readlines()[2:]
        # Store in dict per person
        persons_per_client = {}
        self.files = defaultdict(list)
        c = 0
        for i, (personline, attributeline) in enumerate(tqdm(zip(persons, attributes), leave=False, total=len(attributes))):
            filename, person = personline.strip().split()
            _filename, *attribs = attributeline.strip().split()
            assert filename==_filename, "Filenames in identity and attribute files do not match"
            # Assign to client
            client = persons_per_client.setdefault(int(person), c)
            c = (c+1) % n_clients
            # Deterministic train/val/test split which likely retains each person in each partition
            if ["train", "val", "test"][(tr:=(i%20-14)//14+1)+(tr*i%2)]!=partition: continue
            # One-hot encoded label
            label = torch.tensor([(int(attrib)+1)//2 for attrib in attribs])
            self.files[client].append((os.path.join(path, "images", filename), label))
        
    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.files.values()])*self.n_clients
        
    def __getitem__(self, idx):
        client = idx % len(self.files)
        i = idx // len(self.files)
        # Load
        impath, label = self.files[client][i]
        with open(impath, "rb") as f:
            bytesdata = torch.frombuffer(f.read(), dtype=torch.uint8)
        img = torchvision.io.decode_image(bytesdata, mode="RGB").half() / 255.
        # Augment
        img = functional.resize(img, 256)
        if self.partition=="train":
            img, _ = train_aug(img, seed=self.seed)
            self.seed.manual_seed(torch.randint(0, int(1e6), (), generator=self.seed).item())
        else: img = self.val_crop(img)
        # Change to HWC
        img = torch.permute(img, (1,2,0)).float()
        return label, img

def jax_collate(batch, n_clients:int, beta:float, skew:str)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Only one type of skew can be applied at a time, with the following options:
    - "feature": where indiv_frac==1 corresponds to clients having disjoint (heterogeneous) data distributions.
    - "overlap": where indiv_frac==0 corresponds to clients having disjoint samples of homogeneous distributions.
    - "label": where indiv_frac==1 corresponds to clients having disjoint label distributions.
    """
    # Collect, convert, and concat
    labels, imgs, *auxs = zip(*batch)
    imgs = jnp.stack([jnp.asarray(img) for img in imgs])
    labels = jnp.stack([jnp.asarray(label) for label in labels])
    if gaze:=bool(auxs): auxs = jnp.stack([jnp.asarray(aux) for aux in auxs[0]])

    # Feature skew, by drawing a portion of the samples from the same feature distribution
    # Feature groups are interleaved by the Dataset
    if skew=="feature" or (skew=="label" and not gaze):
        same_dist_idxs = [range(c*int((1-beta)*len(imgs)/n_clients), (c+1)*int((1-beta)*len(imgs)/n_clients)) for c in range(n_clients)]
        diff_dist_idxs = [range(int((1-beta)*len(imgs))+c, len(imgs), n_clients) for c in range(n_clients)]
        idxs = [jnp.asarray(list(same_dist_idxs[c]) + list(diff_dist_idxs[c]), dtype=jnp.int32) for c in range(n_clients)]
    
    # Overlap reduction, by sharing a portion of the total samples while retaining another portion of the total samples
    elif skew=="overlap":
        n_indiv = int(round(len(imgs)/(beta*n_clients + 1 - beta))*beta)
        diff_dist_idxs = [range(c*n_indiv, (c+1)*n_indiv) for c in range(n_clients)]
        idxs = [jnp.asarray(list(diff_dist_idxs[c]) + list(range(n_clients*n_indiv, len(imgs))), dtype=jnp.int32) for c in range(n_clients)]

    # Label skew, by evenly dividing the samples into n directional groups, and then drawing from the same group
    # Assumes there is no order to the labels' values
    elif skew=="label" and gaze:
        # Rank the non-shared samples by quadrant angle
        homosplit = len(imgs) - int((1-beta)*len(imgs))
        angles = (jnp.arctan2(*labels.T) + 2*jnp.pi) % (2*jnp.pi)
        sorter = jnp.argsort(angles[:homosplit])
        diff_dist_idxs = [sorter[c*homosplit//n_clients : (c+1)*homosplit//n_clients] for c in range(n_clients)]
        # Divide the remainder indifferently (i.e., simply splitting indices)
        same_dist_idxs = [range(homosplit+c*int((1-beta)*len(imgs)/n_clients), homosplit+(c+1)*int((1-beta)*len(imgs)/n_clients)) for c in range(n_clients)]
        # Consolidate
        idxs = [jnp.asarray(list(same_dist_idxs[c]) + list(diff_dist_idxs[c]), dtype=jnp.int32) for c in range(n_clients)]

    # Select and return
    clients_imgs = [imgs[idx] for idx in idxs]
    clients_labels = [labels[idx] for idx in idxs]
    if gaze: clients_auxs = [auxs[idx] for idx in idxs]

    if gaze:
        return jnp.stack(clients_labels, 0), jnp.stack(clients_imgs, 0), jnp.stack(clients_auxs, 0)
    else:
        return jnp.stack(clients_labels, 0), jnp.stack(clients_imgs, 0)

def fetch_data(skew:str="overlap", batch_size=128, n_clients=4, beta:float=0, dataset:int=0, partition:str="train", n_classes=1000, **kwargs)->DataLoader:
    assert beta>=0 and beta<=1, "Beta must be between 0 and 1"
    # Increase batch size and account for floor division
    if skew=="overlap": 
        new_beta = 1-beta
        new_batch_size = int(batch_size*new_beta)*n_clients + batch_size - int(batch_size*new_beta)
    else: 
        new_batch_size = batch_size*n_clients
        new_beta = round(beta * new_batch_size / n_clients) * n_clients / new_batch_size
    # Dataset type
    if dataset==0:
        dataset = MPIIGaze(n_clients=n_clients, partition=partition)
        assert skew in ["feature", "overlap", "label"], "Skew must be one of 'feature', 'overlap', or 'label'. For no skew, specify beta=0."
    elif dataset==1:
        dataset = ImageNet(partition=partition, n_clients=n_clients, n_classes=n_classes)
        assert skew in ["overlap", "label"], "Skew must be either 'overlap' or 'label'. For no skew, specify beta=0."
    elif dataset==2:
        dataset = OxfordPets(partition=partition, n_clients=n_clients)
        assert skew in ["overlap", "feature"], "Skew must be either 'overlap' or 'feature'. For no skew, specify beta=0."
    elif dataset==3:
        dataset = CelebA(partition=partition, n_clients=n_clients)
        assert skew in ["overlap", "label"], "Skew must be either 'overlap' or 'label'. For no skew, specify beta=0."
    collate = partial(jax_collate, n_clients=n_clients, beta=new_beta, skew=skew)
    # Iterable batches
    return DataLoader(
        dataset,
        batch_size=new_batch_size,
        collate_fn=collate,
        shuffle=False,
        drop_last=True,
        **kwargs
    )