from image_classifier_utils import *


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

    def read_and_preprocess(self, path):
        """Preprocess the images
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
        valid_tfms = tt.Compose([tt.ToTensor()])
        # PyTorch datasets
        train_ds = ImageFolder(path + "/train", train_tfms)
        # valid_ds = ImageFolder(path + "/test", valid_tfms)
        return train_ds

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

    def predict(self, data):
        pass

    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        self.model = pickle.load(open(path, "rb"))
