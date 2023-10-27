import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


def get_model_transfer_learning(
    model_name: str = "resnet18", n_classes: int = 50
) -> nn.Module:
    # Get the requested architecture
    if hasattr(models, model_name):
        model_transfer: nn.Module = getattr(models, model_name)(pretrained=True)

    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(
            f"Model {model_name} is not know. List of available models: "
            f"https://pytorch.org/vision/{torchvision_major_minor}/models.html"
        )

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes
    # look at the documentation of the model_transfer you loaded
    # 1. Get the number of features in the last layer
    num_ftrs = model_transfer.fc.in_features

    # 2. Create a new linear layer with the appropriate number of inputs and
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    if torch.cuda.is_available():
        model_transfer.cuda()
    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):
    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
