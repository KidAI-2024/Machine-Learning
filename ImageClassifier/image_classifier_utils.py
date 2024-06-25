import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import numpy as np

matplotlib.rcParams["figure.facecolor"] = "#ffffff"

batch_size = 400
random_seed = 42
torch.manual_seed(random_seed)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        # print("get batch")
        out = self(images)  # Generate predictions
        # print("get predictions")
        loss = F.cross_entropy(out, labels)  # Calculate loss
        # print("get loss")
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                result["lrs"][-1],
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # print("resnet number of classes", num_classes)
        # print("resnet in_channels", in_channels)
        # for input 320*180*3
        self.conv1 = conv_block(in_channels, 64)  # 320*180*64
        self.conv2 = conv_block(64, 128, pool=True)  # 160*90*128
        self.res1 = nn.Sequential(
            conv_block(128, 128),  # 160*90*128
            conv_block(128, 128),  # 160*90*128
        )
        # print("ResNet9 model created 1")

        self.conv3 = conv_block(128, 256, pool=True)  # 80*45*256
        self.conv4 = conv_block(256, 512, pool=True)  # 40*22*512
        self.res2 = nn.Sequential(
            conv_block(512, 512),  # 40*22*512
            conv_block(512, 512),  # 40*22*512
        )
        # print("ResNet9 model created 2")
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 10*5*512
            nn.Flatten(),  # 25600
            nn.Dropout(0.2),  # 25600
            nn.Linear(512, num_classes),
            # nn.Linear(25600, num_classes),
        )
        print("ResNet9 model created 3")

    def forward(self, xb):
        # print("xb", xb.shape)
        out = self.conv1(xb)
        # print("conv1 out", out.shape)
        out = self.conv2(out)
        # print("conv2 out", out.shape)
        out = self.res1(out) + out
        # print("res1 out", out.shape)
        out = self.conv3(out)
        # print("conv3 out", out.shape)
        out = self.conv4(out)
        # print("conv4 out", out.shape)
        out = self.res2(out) + out
        # print("res2 out", out.shape)
        out = self.classifier(out)
        # print("classifier out", out.shape)
        final_layer_dim = out.shape[1]
        # print("final_layer_dim", final_layer_dim)
        out = nn.Linear(final_layer_dim, self.num_classes)(out)
        # print("out", out)
        return out


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_cycle(
    epochs,
    max_lr,
    model,
    train_loader,
    val_loader,
    weight_decay=0,
    grad_clip=None,
    opt_func=torch.optim.SGD,
):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        # print("Starting epoch")
        for batch in train_loader:
            loss = model.training_step(batch)
            # print("training step")
            train_losses.append(loss)
            loss.backward()
            # print("backward")
            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            # print("clip grad")
            optimizer.step()
            optimizer.zero_grad()
            # print("zero grad")
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
            # print("step")
        # print("train_losses", train_losses)
        # Validation phase
        if val_loader is not None:
            result = evaluate(model, val_loader)
            result["train_loss"] = torch.stack(train_losses).mean().item()
            result["lrs"] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
    return history


def plot_accuracies(history, save_path="../../Engine/accuracy.png", save_flag=False):
    """Plot the accuracies from the history


    Args:
        history (List): List of dictionaries containing the validation accuracy
        save_path (str, optional): The path to save in the image. Defaults to "../../Engine/accuracy.png".
        save_flag (bool, optional): Flag to control saving the image or not. Defaults to False.
    """
    accuracies = [x["val_acc"] for x in history]
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. No. of epochs")
    # save the plot
    if save_flag:
        plt.savefig(save_path)


def plot_losses(history, save_path="../../Engine/losses.png", save_flag=False):
    """Plot the losses from the history

    Args:
        history (List): List of dictionaries containing the validation accuracy
        save_path (str, optional): The path to save in the image. Defaults to "../../Engine/losses.png".
        save_flag (bool, optional): Flag to control saving the image or not. Defaults to False.
    """
    train_losses = [x.get("train_loss") for x in history]
    val_losses = [x["val_loss"] for x in history]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")
    # save the plot
    if save_flag:
        plt.savefig(save_path)


def plot_lrs(history, save_path="../../Engine/lr_batches.png", save_flag=False):
    """Plot the learning rates from the history

    Args:
        history (List): List of dictionaries containing the validation accuracy
        save_path (str, optional): The path to save in the image. Defaults to "../../Engine/lr_batches.png".
        save_flag (bool, optional): Flag to control saving the image or not. Defaults to False.
    """
    lrs = np.concatenate([x.get("lrs", []) for x in history])
    plt.plot(lrs)
    plt.xlabel("Batch no.")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate vs. Batch no.")
    # save the plot
    if save_flag:
        plt.savefig(save_path)


def predict_image(img, model, device, train_ds):
    """Predict the class of the image

    Args:
        img (_type_): The image to predict
        model (_type_): The model to use for prediction
        device (_type_): The device to use for prediction(CPU or GPU)
        train_ds (_type_): The training dataset to get the classes from

    Returns:
        str: The class of the image
    """
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_ds.classes[preds[0].item()]


def calc_mean_std_of_each_channel(path, in_channels=3):
    try:
        project_path = os.path.join("..", "Engine", path)
        channels = []
        for i in range(in_channels):
            channels.append([])

        # Loop over folders in the specified path
        for folder in os.listdir(project_path):
            folder_path = os.path.join(project_path, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".png"):
                        image_path = os.path.join(folder_path, file_name)
                        image = cv2.imread(image_path)
                        # Split the image into channels
                        for i in range(in_channels):
                            channels[i].extend(image[:, :, i].flatten())

        # Calculate the mean and std of each channel
        means = []
        stds = []
        for channel in channels:
            mean = np.mean(channel)
            std = np.std(channel)
            means.append(mean)
            stds.append(std)
    except:
        # Error reading data from path (wrong path)
        print("Error reading data from path")
        return ()
    return (tuple(means), tuple(stds))
