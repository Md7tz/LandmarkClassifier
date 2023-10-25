import multiprocessing
import math
import matplotlib.pyplot as plt
import torch

from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from pathlib import Path
from .helpers import get_data_location, compute_mean_and_std

INPUT_SIZE = 256


def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = -1,
    limit: int = -1,
):
    """
    Create and returns the train, validation and test data loaders.

    :param batch_size: size of mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2 means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train', 'valid' and 'test' containing respectively the train, validation and test data loaders
    """
    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        "train": T.Compose(
            [
                T.Resize(INPUT_SIZE),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                # T.RandomRotation(degrees=30),
                # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                # RandAugment has 2 main parameters: how many transformations should be
                # applied to each image, and the strength of these transformations. This
                # latter parameter should be tuned through experiments: the higher the more
                # the regularization effect
                T.RandAugment(
                    num_ops=2, magnitude=9, interpolation=T.InterpolationMode.BILINEAR
                ),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
        "valid": T.Compose(
            [
                T.Resize(INPUT_SIZE),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
        "test": T.Compose(
            [
                T.Resize(INPUT_SIZE),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        ),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train", transform=data_transforms["train"]
    )
    valid_data = datasets.ImageFolder(
        base_path / "train", transform=data_transforms["valid"]
    )
    test_data = datasets.ImageFolder(
        base_path / "test", transform=data_transforms["test"]
    )

    # Obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation batches and None for test(Default)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = None

    # Prepare data loaders
    data_loaders["train"] = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )

    data_loaders["valid"] = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = SubsetRandomSampler(indices)

    data_loaders["test"] = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            T.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names = data_loaders["train"].dataset.classes

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, dims=[0, 2, 3, 1]).clip(0, 1)

    # Plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {
        "train",
        "valid",
        "test",
    }, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, (
        "The tensors returned by your dataloaders should be 224x224. Did you "
        "forget to resize and/or crop?"
    )


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert len(labels) == 2, f"Expected a batch of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
