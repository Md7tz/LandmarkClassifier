import os
import torch
import numpy as np
import random
import urllib.request
import multiprocessing
from zipfile import ZipFile
from io import BytesIO
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. will use CPU (slow)")

    # seed random generator for repeatability
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Download data if not present already
    # TODO
    download_and_extract()
    compute_mean_and_std()

    # Make checkpoints subdir if not existing
    os.makedirs("checkpoints", exist_ok=True)

    # Make sure we can reach the installed binaries. This is needed for the workspace
    if os.path.exists("/data/DLND/C2/landmark_images"):
        os.environ["PATH"] = f"{os.environ['PATH']}:/root/.local/bin"


def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("landmark_images"):
        data_folder = "landmark_images"
    elif os.path.exists("/data/DLND/C2/landmark_images"):
        data_folder = "/data/DLND/C2/landmark_images"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    try:
        location = get_data_location()
    except IOError:
        # Dataset does not exist
        print(f"Downloading and unzipping {url}. this will take a while...")

        with urllib.request.urlopen(url) as res:
            with ZipFile(BytesIO(res.read())) as fp:
                fp.extractall(".")

        print("done")

    else:
        print(
            "Dataset already downloaded. If you need to re-download, "
            f"please delete the directory {location}"
        )
        return None


# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()

    ds = datasets.ImageFolder(folder, transform=T.Compose([T.ToTensor()]))
    dl = DataLoader(ds, batch_size=1, num_workers=multiprocessing.cpu_count())

    mean = 0.0

    for images, _ in tqdm(iterable=dl, desc="Computing mean", total=len(ds), ncols=80):
        # images.size() -> torch.Size([1, 3, 800, 533])
        # len(dl.dataset) -> 6246 = len(ds) since its one batch
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # [1, 3, 426400]
        mean += images.mean(2).sum(0)  # sum of each channel
    mean /= len(dl.dataset)

    # This code computes the standard deviation of the pixel values in the dataset
    # by processing it in batches and accumulating the squared differences between
    # the pixel values and the mean. It then scales the result to account for the
    # number of color channels and computes the final standard deviation.
    var = 0.0
    npix = 0
    for images, _ in tqdm(iterable=dl, desc="Computing std", total=len(ds), ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()
    std = torch.sqrt(var / (npix / 3))

    # cache the results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std
