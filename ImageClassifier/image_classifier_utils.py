import torch
import torchvision
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
import os
import numpy as np
from PIL import Image
import io
import base64
import cv2
from sklearn import svm
from sklearn import cluster
import pickle
import shutil
from sklearn.model_selection import (
    GridSearchCV,
    learning_curve,
    validation_curve,
    train_test_split,
)
import time
import json
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import local_binary_pattern

matplotlib.rcParams["figure.facecolor"] = "#ffffff"

batch_size = 400
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


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


class ImageClassificationBaseCNN(nn.Module):
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
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


class ImageClassificationBaseResnet(nn.Module):
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


class ResNet9(ImageClassificationBaseResnet):
    def __init__(self, in_channels, num_classes, img_size=256):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # print("resnet number of classes", num_classes)
        # print("resnet in_channels", in_channels)
        # for input 256*256*3
        self.conv1 = conv_block(in_channels, 64)  # 256*256*64
        self.conv2 = conv_block(64, 128, pool=True)  # 128*128*128
        self.res1 = nn.Sequential(
            conv_block(128, 128),  # 128*128*128
            conv_block(128, 128),  # 128*128*128
        )
        # print("ResNet9 model created 1")

        self.conv3 = conv_block(128, 256, pool=True)  # 64*64*256
        self.conv4 = conv_block(256, 512, pool=True)  # 32*32*512
        self.res2 = nn.Sequential(
            conv_block(512, 512),  # 32*32*512
            conv_block(512, 512),  # 32*32*512
        )
        # self.conv5 = conv_block(512, 1024, pool=True)  # 16*16*1024
        # self.conv6 = conv_block(1024, 2048, pool=True)  # 8*8*2048
        # self.res3 = nn.Sequential(
        #     conv_block(2048, 2048),  # 8*8*2048
        #     conv_block(2048, 2048),  # 8*8*2048
        # )
        # print("ResNet9 model created 2")
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),  # 4*4*512
            nn.Flatten(),  # 25600
            nn.Dropout(0.2),  # 25600
            # nn.Linear(512, num_classes),  # for 32*32 img
            # nn.Linear(32768, num_classes),  # for 256*256 img
            nn.Linear((img_size * img_size) // 2, num_classes),
        )
        # print("ResNet9 model created 3")

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
        # TODO: check on the new resnet model
        # out = self.conv5(out)
        # print("conv5 out", out.shape)
        # out = self.conv6(out)
        # print("conv6 out", out.shape)
        # out = self.res3(out) + out
        # print("res2 out", out.shape)
        out = self.classifier(out)
        # print("classifier out", out.shape)
        # final_layer_dim = out.shape[1]
        # print("final_layer_dim", final_layer_dim)
        # out = nn.Linear(final_layer_dim, self.num_classes)(out)
        # print("out", out)
        return out


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


@torch.no_grad()
def evaluate_train(model, val_loader):
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


def fit_without_decay_lr(
    epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD
):
    print("fit_without_decay_lr")
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        print(f"Epoch {epoch+1}")
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        if val_loader is not None:
            result = evaluate(model, val_loader)
            result["train_loss"] = torch.stack(train_losses).mean().item()
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


def predict_image(img, model, device):
    """Predict the class of the image

    Args:
        img (tensor): The image to predict
        model (ImageClassificationBase): The model to use for prediction
        device (str): The device to use for prediction(CPU or GPU)
        train_ds (ImageFolder): The training dataset to get the classes from

    Returns:
        str: The class of the image
    """
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # print("input shape", xb.shape)
    # Get predictions from model
    yb = model(xb)
    # print("get predictions")
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # print("get preds")
    # Retrieve the class label
    # return train_ds.classes[preds[0].item()]
    return preds[0].item()


class Cifar10CnnModel(ImageClassificationBaseCNN):

    def __init__(self, in_channels, num_classes, img_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 16 x 16  x 64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 8 x 8 x 128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output:  4 x 4 x 256
            nn.Flatten(),
            nn.Linear((img_size // 8) ** 2 * 256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        return self.network(xb)


class LogisticRegressionModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        # print(
        #     "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
        #         epoch, result["val_loss"], result["val_acc"]
        #     )
        # )
        print(
            "Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


def b64string_to_tensor(frame_bytes, width, height, in_channels=3):
    # Get the image data
    image_data = base64.b64decode(frame_bytes)
    # Convert byte data to a PIL Image
    image = Image.open(io.BytesIO(image_data))
    # Define the transformations: resize, convert to tensor, normalize
    transform = tt.Compose(
        [
            tt.Resize(
                (width, height, in_channels)
            ),  # Resize to a specific size if needed
            tt.ToTensor(),  # Convert the image to a tensor
            # tt.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),  # Normalize with ImageNet standards
        ]
    )
    # Apply the transformations
    image_tensor = transform(image)

    # Add a batch dimension (required for input to the model)
    image_tensor = image_tensor.unsqueeze(0)

    print(image_tensor.shape)  # Should print: torch.Size([1, 3, 320, 180])
    # Remove the batch dimension
    image_tensor = image_tensor.squeeze(0)
    return image_tensor