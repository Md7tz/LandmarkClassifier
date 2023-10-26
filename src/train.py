import tempfile
import torch
import torch.nn as nn
import mlflow
import numpy as np
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from .helpers import after_subplot


def train_one_epoch(
    train_data_loader, model: nn.Module, optimizer: torch.optim.Optimizer, loss
) -> float:
    """
    Performs one training epoch
    """

    # Set the model to training mode
    model.train()

    if torch.cuda.is_available():
        model.cuda()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        desc="Training",
        iterable=enumerate(train_data_loader),
        total=len(train_data_loader),
        ncols=80,
    ):
        # Move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()

        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # 3. calculate the loss
        loss_value = loss(output, target)

        # 4. backward pass: compute gradient of the loss with respect to the model parameters
        loss_value.backward()

        # 5. perform a single optimization step (parameter update)
        optimizer.step()

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_data_loader, model: nn.Module, loss) -> float:
    """
    Validate at the end of one epoch
    """

    # Set the model to evaluation mode
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    valid_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        desc="Validating",
        iterable=enumerate(valid_data_loader),
        total=len(valid_data_loader),
        ncols=80,
    ):
        # Move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # 2. calculate the loss
        loss_value = loss(output, target)

        # update average training loss
        valid_loss = valid_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
        )

    return valid_loss


def optimize(
    data_loaders,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss,
    n_epochs: int,
    save_path: str,
    interactive_tracking: bool = False,
):
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # HINT: look here:
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min"
    )

    # mlflow.set_tracking_uri("http://localhost:5000")
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # with mlflow.start_run():
        # # Log the current epoch as a parameter
        # mlflow.log_param("epoch", epoch)
        # # batch size
        # mlflow.log_param("batch_size", data_loaders["train"].batch_size)
        # # optimizer name sgd/adam
        # mlflow.log_param("optimizer", optimizer.__class__.__name__)
        # # loss function name
        # mlflow.log_param("loss_function", loss.__class__.__name__)

        # # Log training and validation loss as metrics
        # mlflow.log_metric("train_loss", train_loss)
        # mlflow.log_metric("valid_loss", valid_loss)
        # mlflow.log_metric("lr", optimizer.param_groups[0]["lr"])
        # mlflow.log_metric("weight_decay", optimizer.param_groups[0]["weight_decay"])

        # Print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # if validation loss has decreased by 1% or more, save the model
        if valid_loss_min is None or valid_loss < valid_loss_min * 0.99:
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            print(
                f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}).  Saving model ..."
            )
            # mlflow.pytorch.log_model(model, "model")

        # Update learning rate, i.e. reduce it if validation loss has not improved
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_data_loader, model: nn.Module, loss) -> float:
    # Monitor test loss and accuracy
    test_loss = 0.0
    correct = 0.0
    total = 0.0

    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available():
            model.cuda()

        for batch_idx, (data, target) in tqdm(
            desc="Testing",
            iterable=enumerate(test_data_loader),
            total=len(test_data_loader),
            ncols=80,
        ):
            # Move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)

            # 2. calculate the loss
            loss_value = loss(logits, target)

            # update average training loss
            test_loss = test_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss)
            )

            # Convert logits to predicted class
            _, pred = torch.max(logits, 1)

            # Compare predictions to true label
            correct += torch.sum(
                torch.squeeze(pred.eq(target.data.view_as(pred))).cpu()
            )
            total += data.size(0)

    print("Test Loss: {:.6f}\n".format(test_loss))
    print(
        "\nTest Accuracy: %2d%% (%2d/%2d)" % (100.0 * correct / total, correct, total)
    )

    return test_loss


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss in nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, _ = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss in nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/temp.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, _ = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
