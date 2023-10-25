import torch
import torch.nn as nn


# define the CNN Architecture
class MyModel(nn.Module):
    def __init__(self, n_classes: int = 1000, dropout: float = 0.7) -> None:
        super(MyModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, padding=1
            ),  # match the feature map size with the input size
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2, 2),
        )

        # Fully connected layers (after 3 maxpools: 224 -> 28)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64 * 28 * 28, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        # Batch normalization
        self.batch_norm2d = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Define forward behavior
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm2d(x)
        x = self.conv3(x)

        x = x.view(-1, 64 * 28 * 28)
        x = self.classifier(x)

        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(n_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, _ = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
