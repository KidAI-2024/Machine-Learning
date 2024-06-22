from image_classifier_utils import *
from camera_feed import CameraFeed


class ImageClassifierResNet:
    def __init__(
        self,
        num_classes=2,
        in_channels=3,
    ):
        """Constructor for the ImageClassifierReNet class

        Args:
            num_classes (int, optional): The number of classes you want predict. Defaults to 2.
            in_channels (int, optional): The number of input channels for the images. Defaults to 3.
        """
        self.device = get_default_device()
        self.model = to_device(ResNet9(in_channels, num_classes), self.device)
        self.camera = CameraFeed()

    def read_and_preprocess_train(self, path):
        """Preprocess the images
        read the images from the path and preprocess them by applying the following transformations:\n
            1.normalization by calculating the mean and standard deviation of each channel in the dataset\n
            2.data augmentation (random horizontal flip, random rotation)
        Args:
            images (_type_): _description_
        """

        # Data transforms (data augmentation)
        train_tfms = tt.Compose(
            [
                tt.RandomCrop(32, padding=4, padding_mode="reflect"),
                tt.RandomHorizontalFlip(),
                # tt.RandomRotate
                tt.RandomResizedCrop(256, scale=(0.5, 0.9), ratio=(1, 1)),
                tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tt.ToTensor(),
                # tt.Normalize(*stats, inplace=True),
            ]
        )
        # PyTorch datasets
        train_ds = ImageFolder(path + "/train", train_tfms)
        return train_ds

    def read_and_preprocess_test(self, path):
        """Preprocess the images
        read the images from the path and preprocess them by applying the following transformations:\n
            1.normalization by calculating the mean and standard deviation of each channel in the dataset\n
            2.data augmentation (random horizontal flip, random rotation)
        Args:
            images (_type_): _description_
        """
        valid_tfms = tt.Compose([tt.ToTensor()])
        valid_ds = ImageFolder(path + "/test", valid_tfms)
        return valid_ds

    def get_data_loaders(
        self, train_ds, valid_ds, batch_size=128, num_workers=4, pin_memory=True
    ):
        """Get the data loaders for the training and validation datasets

        Args:
            train_ds (Dataset): The training dataset
            valid_ds (Dataset): The validation dataset
            batch_size (int, optional): The batch size for the data loaders. Defaults to 128.
            num_workers (int, optional): The number of workers for the data loaders. Defaults to 4.
            pin_memory (bool, optional): Pin the memory for the data loaders. Defaults to True.
        """
        if train_ds is not None:
            train_dl = DataLoader(
                train_ds,
                batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            train_dl = None
        if valid_ds is not None:
            valid_dl = DataLoader(
                valid_ds, batch_size * 2, num_workers=num_workers, pin_memory=pin_memory
            )
        else:
            valid_dl = None
        return train_dl, valid_dl

    def train(
        self,
        epochs=10,
        max_lr=0.01,
        grad_clip=0.1,
        weight_decay=1e-4,
        opt_func=torch.optim.Adam,
        train_dl=None,
        valid_dl=None,
    ):
        """Train the model

        Args:
            epochs (int, optional): The number of epochs of training. Defaults to 10.
            max_lr (float, optional): The initial value of learning rate before decaying. Defaults to 0.01.
            grad_clip (float, optional): Clipping the gradient to avoid zero gradient problem. Defaults to 0.1.
            weight_decay (float, optional): The decay of the learning rate. Defaults to 1e-4.
            opt_func (torch.optim, optional): The optimizer that updates the model parameters. Defaults to torch.optim.Adam.
            train_dl (DataLoader, optional): The DataLoader for the training dataset. Defaults to None.
            valid_dl (DataLoader, optional): The DataLoader for the validation dataset. Defaults to None.
        """
        self.epochs = epochs
        self.max_lr = max_lr
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.opt_func = opt_func
        # inital accuracy without training
        self.history = [evaluate(self.model, valid_dl)]
        self.history += fit_one_cycle(
            epochs,
            max_lr,
            self.model,
            train_dl,
            valid_dl,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            opt_func=opt_func,
        )
        # plot the accuracies and save it
        plot_accuracies(
            self.history, save_path="../../Engine/renet_accuracy.png", save_flag=True
        )
        # plot losses and save it
        plot_losses(
            self.history, save_path="../../Engine/renet_loss.png", save_flag=True
        )
        # plot learning rates and save it
        plot_lrs(self.history, save_path="../../Engine/renet_lr.png", save_flag=True)

    def predict(self, img, train_ds):
        """Predict the class of the image

        Args:
            img (_type_): _description_
            train_ds (_type_): _description_

        Returns:
            ste: The class of the image
        """
        # Convert to a batch of 1
        xb = to_device(img.unsqueeze(0), self.device)
        # Get predictions from model
        yb = self.model(xb)
        # Pick index with highest probability
        _, preds = torch.max(yb, dim=1)
        # Retrieve the class label
        return train_ds.classes[preds[0].item()]

    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        # pickle.dump(self.model, open(path, "wb"))
        torch.save(self.model, path)

    def load(self, path):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        # self.model = pickle.load(open(path, "rb"))
        self.model.load_state_dict(torch.load(path))
        self.model = to_device(self.model, self.device)
        self.model.eval()

    def start_feed(self):
        """Start the camera feed"""
        return self.camera.start_feed()

    def stop_feed(self):
        """Stop the camera feed"""
        return self.camera.stop_feed()

    def get_frame(self):
        """Get a frame from the camera feed"""
        return self.camera.get_latest_frame()

    def print_model(self):
        print("Model")
