def main():
    import jax
    import flax
    from unetr import UNETR
    import optax
    import orbax.checkpoint as ocp
    from typing import Any
    from pathlib import Path
    import cv2
    import numpy as np
    from PIL import Image
    import time
    import orbax.checkpoint as ocp
    import optax
    from typing import Callable
    from flax import nnx
    import jax.numpy as jnp    
    from typing import Any, Callable
    import grain.python as grain    
    import albumentations as A
    import matplotlib.pyplot as plt

    class OxfordPetsDataset:
        def __init__(self, path: Path):
            assert path.exists(), path
            self.path: Path = path
            self.images = sorted((self.path / "images").glob("*.jpg"))
            self.masks = [
                self.path / "annotations" / "trimaps" / path.with_suffix(".png").name
                for path in self.images
            ]
            assert len(self.images) == len(self.masks), (len(self.images), len(self.masks))

        def __len__(self) -> int:
            return len(self.images)

        def read_image_opencv(self, path: Path):
            img = cv2.imread(str(path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                None

        def read_image_pillow(self, path: Path):
            img = Image.open(str(path))
            img = img.convert("RGB")
            return np.asarray(img)

        def read_mask(self, path: Path):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            # mask has values: 1, 2, 3
            # 1 - object mask
            # 2 - background
            # 3 - boundary
            # Define mask as 0-based int values
            mask = mask - 1
            return mask.astype("uint8")

        def __getitem__(self, index: int) -> dict[str, np.ndarray]:
            img_path, mask_path = self.images[index], self.masks[index]
            img = self.read_image_opencv(img_path)
            if img is None:
                # Fallback to Pillow if OpenCV fails to read an image
                img = self.read_image_pillow(img_path)
            mask = self.read_mask(mask_path)
            return {
                "image": img,
                "mask": mask,
            }


    class SubsetDataset:
        def __init__(self, dataset, indices: list[int]):
            # Check input indices values:
            for i in indices:
                assert 0 <= i < len(dataset)
            self.dataset = dataset
            self.indices = indices

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> Any:
            i = self.indices[index]
            return self.dataset[i]

    # %% [markdown]
    # Now, let's define the total dataset and compute data indices for training and validation splits:

    # %%
    seed = 12
    train_split = 0.7
    dataset_path = Path("oxford_pets")

    dataset = OxfordPetsDataset(dataset_path)

    rng = np.random.default_rng(seed=seed)
    le = len(dataset)
    data_indices = list(range(le))

    # Let's remove few indices corresponding to corrupted images
    # to avoid libjpeg warnings during the data loading
    corrupted_data_indices = [3017, 3425]
    for index in corrupted_data_indices:
        data_indices.remove(index)

    random_indices = rng.permutation(data_indices)

    train_val_split_index = int(train_split * le)
    train_indices = random_indices[:train_val_split_index]
    val_indices = random_indices[train_val_split_index:]

    # Ensure there is no overlapping
    assert len(set(train_indices) & set(val_indices)) == 0

    train_dataset = SubsetDataset(dataset, indices=train_indices)
    val_dataset = SubsetDataset(dataset, indices=val_indices)

    print("Training dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    # %% [markdown]
    # To verify our work so far, let's display few training and validation images and masks:

    # %%

    def display_datapoint(datapoint, label=""):
        img, mask = datapoint["image"], datapoint["mask"]
        if img.dtype in (np.float32, ):
            img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
        fig, axs = plt.subplots(1, 3, figsize=(10, 10))
        axs[0].set_title(f"Image{label}")
        axs[0].imshow(img)
        axs[1].set_title(f"Mask{label}")
        axs[1].imshow(mask)
        axs[2].set_title("Image + Mask")
        axs[2].imshow(img)
        axs[2].imshow(mask, alpha=0.5)



    display_datapoint(train_dataset[0], label=" (train set)")
    display_datapoint(val_dataset[0], label=" (val set)")

    # %% [markdown]
    # ### Data augmentations
    # 
    # Next, let's define a simple data augmentation pipeline of joined image and mask transformations using [Albumentations](https://albumentations.ai/docs/examples/example/). We apply geometric and color transformations to increase the diversity of the training data. For more details on the Albumentations transformations, we can check [Albumentations reference API](https://albumentations.ai/docs/api_reference/full_reference/).

    # %%


    img_size = 256

    train_transforms = A.Compose([
        A.Affine(rotate=(-35, 35), p=0.3),  # Random rotations -35 to 35 degrees
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),  # Crop a random part of the input and rescale it to a specified size
        A.HorizontalFlip(p=0.5),  # Horizontal random flip
        A.RandomBrightnessContrast(p=0.4),  # Randomly changes the brightness and contrast
        A.Normalize(),  # Normalize the image and cast to float
    ])


    val_transforms = A.Compose([
        A.Resize(width=img_size, height=img_size),
        A.Normalize(),  # Normalize the image and cast to float
    ])

    # %%
    output = train_transforms(**train_dataset[0])
    img, mask = output["image"], output["mask"]
    print("Image array info:", img.dtype, img.shape, img.min(), img.mean(), img.max())
    print("Mask array info:", mask.dtype, mask.shape, mask.min(), mask.max())

    # %%
    output = val_transforms(**val_dataset[0])
    img, mask = output["image"], output["mask"]
    print("Image array info:", img.dtype, img.shape, img.min(), img.mean(), img.max())
    print("Mask array info:", mask.dtype, mask.shape, mask.min(), mask.max())

    # %% [markdown]
    # ### Data loaders
    # 
    # Let's now use [`grain`](https://github.com/google/grain) to perform data loading, augmentations and batching on a single device using multiple workers. We will create a random index sampler for training and an unshuffled sampler for validation.

    # %%

    class DataAugs(grain.MapTransform):
        def __init__(self, transforms: Callable):
            self.albu_transforms = transforms

        def map(self, data):
            output = self.albu_transforms(**data)
            return output

    # %%
    train_batch_size = 16
    val_batch_size = 16


    # Create an IndexSampler with no sharding for single-device computations
    train_sampler = grain.IndexSampler(
        len(train_dataset),  # The total number of samples in the data source
        shuffle=True,            # Shuffle the data to randomize the order of samples
        seed=seed,               # Set a seed for reproducibility
        shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
        num_epochs=1,            # Iterate over the dataset for one epoch
    )

    val_sampler = grain.IndexSampler(
        len(val_dataset),  # The total number of samples in the data source
        shuffle=False,         # Do not shuffle the data
        seed=seed,             # Set a seed for reproducibility
        shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup
        num_epochs=1,          # Iterate over the dataset for one epoch
    )

    # %%
    train_loader = grain.DataLoader(
        data_source=train_dataset,
        sampler=train_sampler,                 # Sampler to determine how to access the data
        worker_count=4,                        # Number of child processes launched to parallelize the transformations among
        worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
        operations=[
            DataAugs(train_transforms),
            grain.Batch(train_batch_size, drop_remainder=True),
        ]
    )

    # Validation dataset loader
    val_loader = grain.DataLoader(
        data_source=val_dataset,
        sampler=val_sampler,                   # Sampler to determine how to access the data
        worker_count=4,                        # Number of child processes launched to parallelize the transformations among
        worker_buffer_size=2,
        operations=[
            DataAugs(val_transforms),
            grain.Batch(val_batch_size),
        ]
    )

    # Training dataset loader for evaluation (without dataaugs)
    train_eval_loader = grain.DataLoader(
        data_source=train_dataset,
        sampler=train_sampler,                 # Sampler to determine how to access the data
        worker_count=4,                        # Number of child processes launched to parallelize the transformations among
        worker_buffer_size=2,                  # Count of output batches to produce in advance per worker
        operations=[
            DataAugs(val_transforms),
            grain.Batch(val_batch_size),
        ]
    )

    # %%
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    # %%
    print("Train images batch info:", type(train_batch["image"]), train_batch["image"].shape, train_batch["image"].dtype)
    print("Train masks batch info:", type(train_batch["mask"]), train_batch["mask"].shape, train_batch["mask"].dtype)

    # %% [markdown]
    # Finally, let's display the training and validation data:

    # %%
    images, masks = train_batch["image"], train_batch["mask"]

    for img, mask in zip(images[:3], masks[:3]):
        display_datapoint({"image": img, "mask": mask}, label=" (augmented train set)")

    # %%
    images, masks = val_batch["image"], val_batch["mask"]

    for img, mask in zip(images[:3], masks[:3]):
        display_datapoint({"image": img, "mask": mask}, label=" (augmented validation set)")

    # %% [markdown]
    # ## Model for Image Segmentation
    # 
    # In this section we will implement the [UNETR](https://arxiv.org/abs/2103.10504) model from scratch using Flax NNX. The reference PyTorch implementation of this model can be found on the [MONAI Library GitHub repository](https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/unetr.py).
    # 
    # The UNETR model utilizes a transformer as the encoder to learn sequence representations of the input and to capture the global multi-scale information, while also following the “U-shaped” network design like [UNet](https://arxiv.org/abs/1505.04597) model:
    # ![image.png](https://github.com/jax-ml/jax-ai-stack/blob/main/docs/source/_static/images/unetr_architecture.png?raw=1)
    # 
    # The UNETR architecture on the image above is processing 3D inputs, but it can be easily adapted to 2D input.
    # 
    # The transformer encoder of UNETR is [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929). The feature maps returned by ViT have all the same spatial size: (H / 16, W / 16) and deconvolutions are used to upsample the feature maps. Finally, the feature maps are upsampled and concatenated up to the original image size.

    # %%



    # %%
    # We'll use a different number of heads to make a smaller model
    model = UNETR(out_channels=3, num_heads=4, key=jax.random.key(42))
    x = jnp.ones((4, 256, 256, 3))
    y = model(x)
    print(y.shape)

    # %% [markdown]
    # We can visualize and inspect the architecture on the implemented model using `nnx.display(model)`.

    # %% [markdown]
    # ## Train the model
    # 
    # In previous sections we defined training and validation dataloaders and the model. In this section we will train the model and define the loss function and the optimizer to perform the parameters optimization.
    # 
    # For the semantic segmentation task, we can define the loss function as a sum of Cross-Entropy and Jaccard loss functions. The Cross-Entropy loss function is a standard loss function for a multi-class classification tasks and the Jaccard loss function helps directly optimizing Intersection-over-Union measure for semantic segmentation.

    # %%

    num_epochs = 50
    total_steps = len(train_dataset) // train_batch_size
    learning_rate = 0.003
    momentum = 0.9

    # %%
    lr_schedule = optax.linear_schedule(learning_rate, 0.0, num_epochs * total_steps)

    iterate_subsample = np.linspace(0, num_epochs * total_steps, 100)
    plt.plot(
        np.linspace(0, num_epochs, len(iterate_subsample)),
        [lr_schedule(i) for i in iterate_subsample],
        lw=3,
    )
    plt.title("Learning rate")
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.grid()
    plt.xlim((0, num_epochs))
    plt.show()


    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule, momentum))

    # %% [markdown]
    # Let us implement Jaccard loss and the loss function combining Cross-Entropy and Jaccard losses.

    # %%
    def compute_softmax_jaccard_loss(logits, masks, reduction="mean"):
        assert reduction in ("mean", "sum")
        y_pred = nnx.softmax(logits, axis=-1)
        b, c = y_pred.shape[0], y_pred.shape[-1]
        y = nnx.one_hot(masks, num_classes=c, axis=-1)

        y_pred = y_pred.reshape((b, -1, c))
        y = y.reshape((b, -1, c))

        intersection = y_pred * y
        union = y_pred + y - intersection + 1e-8

        intersection = jnp.sum(intersection, axis=1)
        union = jnp.sum(union, axis=1)

        if reduction == "mean":
            intersection = jnp.mean(intersection)
            union = jnp.mean(union)
        elif reduction == "sum":
            intersection = jnp.sum(intersection)
            union = jnp.sum(union)

        return 1.0 - intersection / union


    def compute_losses_and_logits(model: nnx.Module, images: jax.Array, masks: jax.Array):
        logits = model(images)

        xentropy_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=masks
        ).mean()

        jacc_loss = compute_softmax_jaccard_loss(logits=logits, masks=masks)
        loss = xentropy_loss + jacc_loss
        return loss, (xentropy_loss, jacc_loss, logits)

    # %% [markdown]
    # Now, we will implement a confusion matrix metric derived from [`nnx.Metric`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/training/metrics.html#flax.nnx.metrics.Metric). A confusion matrix will help us to compute the Intersection-Over-Union (IoU) metric per class and on average. Finally, we can also compute the accuracy metric using the confusion matrix.

    # %%
    class ConfusionMatrix(nnx.Metric):
        def __init__(
            self,
            num_classes: int,
            average: str | None = None,
        ):
            assert average in (None, "samples", "recall", "precision")
            assert num_classes > 0
            self.num_classes = num_classes
            self.average = average
            self.confusion_matrix = nnx.metrics.MetricState(
                jnp.zeros((self.num_classes, self.num_classes), dtype=jnp.int32)
            )
            self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

        def reset(self):
            self.confusion_matrix.value = jnp.zeros((self.num_classes, self.num_classes), dtype=jnp.int32)
            self.count.value = jnp.array(0, dtype=jnp.int32)

        def _check_shape(self, y_pred: jax.Array, y: jax.Array):
            if y_pred.shape[-1] != self.num_classes:
                raise ValueError(f"y_pred does not have correct number of classes: {y_pred.shape[-1]} vs {self.num_classes}")

            if not (y.ndim + 1 == y_pred.ndim):
                raise ValueError(
                    f"y_pred must have shape (batch_size, num_classes (currently set to {self.num_classes}), ...) "
                    "and y must have shape of (batch_size, ...), "
                    f"but given {y.shape} vs {y_pred.shape}."
                )

        def update(self, **kwargs):
            # We assume that y.max() < self.num_classes and y.min() >= 0
            assert "y" in kwargs
            assert "y_pred" in kwargs
            y_pred = kwargs["y_pred"]
            y = kwargs["y"]
            self._check_shape(y_pred, y)
            self.count.value += y_pred.shape[0]

            y_pred = jnp.argmax(y_pred, axis=-1).ravel()
            y = y.ravel()
            indices = self.num_classes * y + y_pred
            matrix = jnp.bincount(indices, minlength=self.num_classes**2, length=self.num_classes**2)
            matrix = matrix.reshape((self.num_classes, self.num_classes))
            self.confusion_matrix.value += matrix

        def compute(self) -> jax.Array:
            if self.average:
                confusion_matrix = self.confusion_matrix.value.astype("float")
                if self.average == "samples":
                    return confusion_matrix / self.count.value
                else:
                    return self.normalize(self.confusion_matrix.value, self.average)
            return self.confusion_matrix.value

        @staticmethod
        def normalize(matrix: jax.Array, average: str) -> jax.Array:
            """Normalize given `matrix` with given `average`."""
            if average == "recall":
                return matrix / (jnp.expand_dims(matrix.sum(axis=1), axis=1) + 1e-15)
            elif average == "precision":
                return matrix / (matrix.sum(axis=0) + 1e-15)
            else:
                raise ValueError("Argument average should be one of 'samples', 'recall', 'precision'")


    def compute_iou(cm: jax.Array) -> jax.Array:
        return jnp.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - jnp.diag(cm) + 1e-15)


    def compute_mean_iou(cm: jax.Array) -> jax.Array:
        return compute_iou(cm).mean()


    def compute_accuracy(cm: jax.Array) -> jax.Array:
        return jnp.diag(cm).sum() / (cm.sum() + 1e-15)

    # %% [markdown]
    # Next, let's define training and evaluation steps:

    # %%
    @nnx.jit
    def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, np.ndarray]
    ):
        # Convert numpy arrays to jax.Array on GPU
        images = jnp.array(batch["image"])
        masks = jnp.array(batch["mask"], dtype=jnp.int32)

        grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
        (loss, (xentropy_loss, jacc_loss, logits)), grads = grad_fn(model, images, masks)

        optimizer.update(grads)  # In-place updates.

        return loss, xentropy_loss, jacc_loss

    # %%
    @nnx.jit
    def eval_step(
        model: nnx.Module, batch: dict[str, np.ndarray], eval_metrics: nnx.MultiMetric
    ):
        # Convert numpy arrays to jax.Array on GPU
        images = jnp.array(batch["image"])
        masks = jnp.array(batch["mask"], dtype=jnp.int32)
        loss, (_, _, logits) = compute_losses_and_logits(model, images, masks)

        eval_metrics.update(
            total_loss=loss,
            y_pred=logits,
            y=masks,
        )  # In-place updates.

    # %% [markdown]
    # We will also define metrics we want to compute during the evaluation phase: total loss and confusion matrix computed on training and validation datasets. Finally, we define helper objects to store the metrics history.
    # Metrics like IoU per class, mean IoU and accuracy will be computed using the confusion matrix in the evaluation code.

    # %%
    eval_metrics = nnx.MultiMetric(
        total_loss=nnx.metrics.Average('total_loss'),
        confusion_matrix=ConfusionMatrix(num_classes=3),
    )


    eval_metrics_history = {
        "train_total_loss": [],
        "train_IoU": [],
        "train_mean_IoU": [],
        "train_accuracy": [],

        "val_total_loss": [],
        "val_IoU": [],
        "val_mean_IoU": [],
        "val_accuracy": [],
    }

    # %% [markdown]
    # Let us define the training and evaluation logic. We define as well a checkpoint manager to store two best models defined by validation mean IoU metric value.

    # %%



    def train_one_epoch(epoch):
        start_time = time.time()

        model.train()  # Set model to the training mode: e.g. update batch statistics
        for step, batch in enumerate(train_loader):
            total_loss, xentropy_loss, jaccard_loss = train_step(model, optimizer, batch)

            print(
                f"\r[train] epoch: {epoch + 1}/{num_epochs}, iteration: {step}/{total_steps}, "
                f"total loss: {total_loss.item():.4f} ",
                f"xentropy loss: {xentropy_loss.item():.4f} ",
                f"jaccard loss: {jaccard_loss.item():.4f} ",
                end="")
            print("\r", end="")

        elapsed = time.time() - start_time
        print(
            f"\n[train] epoch: {epoch + 1}/{num_epochs}, elapsed time: {elapsed:.2f} seconds"
        )


    def evaluate_model(epoch):
        start_time = time.time()

        # Compute the metrics on the train and val sets after each training epoch.
        model.eval()  # Set model to evaluation model: e.g. use stored batch statistics

        for tag, eval_loader in [("train", train_eval_loader), ("val", val_loader)]:
            eval_metrics.reset()  # Reset the eval metrics
            for val_batch in eval_loader:
                eval_step(model, val_batch, eval_metrics)

            for metric, value in eval_metrics.compute().items():
                if metric == "confusion_matrix":
                    eval_metrics_history[f"{tag}_IoU"].append(
                        compute_iou(value)
                    )
                    eval_metrics_history[f"{tag}_mean_IoU"].append(
                        compute_mean_iou(value)
                    )
                    eval_metrics_history[f"{tag}_accuracy"].append(
                        compute_accuracy(value)
                    )
                else:
                    eval_metrics_history[f'{tag}_{metric}'].append(value)

            print(
                f"[{tag}] epoch: {epoch + 1}/{num_epochs} "
                f"\n - total loss: {eval_metrics_history[f'{tag}_total_loss'][-1]:0.4f} "
                f"\n - IoU per class: {eval_metrics_history[f'{tag}_IoU'][-1].tolist()} "
                f"\n - Mean IoU: {eval_metrics_history[f'{tag}_mean_IoU'][-1]:0.4f} "
                f"\n - Accuracy: {eval_metrics_history[f'{tag}_accuracy'][-1]:0.4f} "
                "\n"
            )

        elapsed = time.time() - start_time
        print(
            f"[evaluation] epoch: {epoch + 1}/{num_epochs}, elapsed time: {elapsed:.2f} seconds"
        )

        return eval_metrics_history['val_mean_IoU'][-1]


    path = ocp.test_utils.erase_and_create_empty("/tmp/output-oxford-model/")
    options = ocp.CheckpointManagerOptions(max_to_keep=2)
    mngr = ocp.CheckpointManager(path, options=options)


    def save_model(epoch):
        state = nnx.state(model)
        # We should convert PRNGKeyArray to the old format for Dropout layers
        # https://github.com/google/flax/issues/4231
        def get_key_data(x):
            if isinstance(x, jax._src.prng.PRNGKeyArray):
                if isinstance(x.dtype, jax._src.prng.KeyTy):
                    return jax.random.key_data(x)
            return x

        serializable_state = jax.tree.map(get_key_data, state)
        mngr.save(epoch, args=ocp.args.StandardSave(serializable_state))
        mngr.wait_until_finished()

    # %% [markdown]
    # Now we can start the training. It can take around 45 minutes using a single GPU and use 19GB of GPU memory.

    best_val_mean_iou = 0.0
    for epoch in range(num_epochs):
        train_one_epoch(epoch)
        if (epoch % 3 == 0) or (epoch == num_epochs - 1):
            val_mean_iou = evaluate_model(epoch)
            if val_mean_iou > best_val_mean_iou:
                save_model(epoch)
                best_val_mean_iou = val_mean_iou

    # %% [markdown]
    # We can check the saved models:

    # %% [markdown]
    # and visualize collected metrics:

    # %%
    epochs = [i for i in range(num_epochs) if (i % 3 == 0) or (i == num_epochs - 1)]

    plt.plot(epochs, eval_metrics_history["train_total_loss"], label="Loss value on training set")
    plt.plot(epochs, eval_metrics_history["val_total_loss"], label="Loss value on validation set")
    plt.legend()

    # %%
    plt.plot(epochs, eval_metrics_history["train_mean_IoU"], label="Mean IoU on training set")
    plt.plot(epochs, eval_metrics_history["val_mean_IoU"], label="Mean IoU on validation set")
    plt.legend()

    # %% [markdown]
    # Next, we will visualize model predictions on validation data:

    # %%
    model.eval()
    val_batch = next(iter(val_loader))

    # %%
    images, masks = val_batch["image"], val_batch["mask"]
    preds = model(images)
    preds = jnp.argmax(preds, axis=-1)

    # %%
    def display_image_mask_pred(img, mask, pred, label=""):
        if img.dtype in (np.float32, ):
            img = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
        fig, axs = plt.subplots(1, 5, figsize=(15, 10))
        axs[0].set_title(f"Image{label}")
        axs[0].imshow(img)
        axs[1].set_title(f"Mask{label}")
        axs[1].imshow(mask)
        axs[2].set_title("Image + Mask")
        axs[2].imshow(img)
        axs[2].imshow(mask, alpha=0.5)
        axs[3].set_title(f"Pred{label}")
        axs[3].imshow(pred)
        axs[4].set_title("Image + Pred")
        axs[4].imshow(img)
        axs[4].imshow(pred, alpha=0.5)

    # %%
    for img, mask, pred in zip(images[:4], masks[:4], preds[:4]):
        display_image_mask_pred(img, mask, pred, label=" (validation set)")

    # %% [markdown]
    # We can see that model can roughly predict the shape of the animal and the background and struggles with predicting the boundary. Carefully choosing hyperparameters we may achieve better results.


if __name__ == "__main__":
    main()