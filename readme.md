# Eliminating Symmetries to Reduce Stochastic Client Drift in Federated Learning

## Repo Structure
This repository follows the general structure of the paper. The folder `backend` contains functions and models that are used throughout the codebase. This includes:
- `cast`: broadcasts a single global `nnx` model to `n` clients.
- `get_updates`: takes an initial/global model and clients' local models, and returns the difference.
- `aggregate`: averages updates and applies them to a global model.
- `train`: performs communication rounds, including local training and intermittent aggregation.
- `utils.py`: contains various loss functions and performance metrics, as well as convenience functions for loading and saving nnx models.
- `models.py`: models can be imported from here.
- `data.py`: paramountly contains the `fetch_data` function, which returns a specified dataset, accounting for parameters such as heterogeneity type and severity.
Moreover, the preliminary experiments of Section 3 are contained in the folder `preliminary`. The evaluation scripts of the methodology of Section 4 can be found in the folder `methodology`. Lastly, Section 5, analysis, corresponds to the folder `analysis`. For the FedMA adaptation, please see https://github.com/thijs-vanweezel/FedMA.

## Dependencies
The stable versions of the essential packages are:
- matplotlib
- torch 2.10.0+cpu
- torchvision 0.25.0+cpu
- jax 0.8.1.dev20260317
- optax 0.2.6
- flax 0.12.1
For the finetuning script, additionally install ml-collections and https://github.com/google-research/vision_transformer.
For `vissurfacedrift.ipynb` and `levelset.ipynb`, additionally install scikit-learn.
For the disentanglement script, additionally install npy-append-array.

## On Reproducibility
The preliminary experiments generally deterministic linear algebra operations. Therefore, when re-running the experiments, the results found in the paper can be reproduced exactly up to floating point precision. All scripts rely on rng keys, also reducing variability in outcomes.

## Data
The datasets used during this research can be downloaded from the following urls. To preprocess, unzip, adjust the paths in `data.py`, and run.
- MPIIGaze (`MPIIGaze.tar.gz`): https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild
- ImageNet (`ILSVRC.zip`): https://image-net.org/download-images.php
- Oxford-IIIT Pets (`annotations.tar.gz` and `images.tar.gz`): https://www.robots.ox.ac.uk/~vgg/data/pets/
- CelebA (`img_align_celeba.zip`): https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html