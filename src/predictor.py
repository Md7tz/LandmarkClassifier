import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from .helpers import get_data_location


class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super(Predictor, self).__init__()

        self.model = model.eval()
        self.class_names = class_names

        # We use nn.Sequential and not nn.Compose because the former
        # is compatible with torch.script, while the latter isn't
        self.transforms = nn.Sequential(
            T.Resize(
                [
                    256,
                ]
            ),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # 1. apply transforms
            x = self.transforms(x)

            # 2. get the logits
            logits = self.model(x)

            # 3. apply softmax
            probabilities = F.softmax(logits, dim=1)

            return probabilities


def predictor_test(test_dataloader, model_reloaded):
    """
    Test the predictor. Since the predictor does not operate on the same tensors
    as the non-wrapped model, we need a specific test function (can't use one_epoch_test)
    """

    folder = get_data_location()
    test_data = ImageFolder(
        os.path.join(folder, "test"), transform=T.Compose([T.ToTensor()])
    )

    pred = []
    truth = []

    for x in tqdm(test_data, desc="Testing", total=len(test_data), ncols=80):
        # x[0] -> image
        # x[1] -> label
        # x[0].unsqueeze(0) -> [1, 3, 800, 533]
        # x[1] -> 0
        probabilities = model_reloaded(x[0].unsqueeze(0))

        pred.append(probabilities.argmax().item())
        truth.append(x[1])

    pred = np.array(pred)
    truth = np.array(truth)

    accuracy = (pred == truth).mean()
    print(f"Accuracy: {accuracy}")

    return truth, pred, accuracy


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(n_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    predictor = Predictor(model, class_names=["a", "b", "c"], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(), torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
