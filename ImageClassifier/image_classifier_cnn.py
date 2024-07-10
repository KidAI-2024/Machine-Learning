from .image_classifier_utils import *
from camera_feed import CameraFeed


class ImageClassifierCNN:
    def __init__(
        self,
        num_classes=2,
        in_channels=3,
        img_size=256,
    ):
        """Constructor for the ImageClassifierReNet class

        Args:
            num_classes (int, optional): The number of classes you want predict. Defaults to 2.
            in_channels (int, optional): The number of input channels for the images. Defaults to 3.
        """
        self.device = get_default_device()
        print("Device: ", self.device)

        # self.device = torch.device("cpu")
        self.model = None
        self.num_classes = num_classes
        self.in_channels = in_channels
        # print("Device: ", self.device)
        # print("num_classes: ", num_classes)
        self.train_size = 0
        self.valid_size = 0
        self.img_size = img_size
        self.camera = CameraFeed()
        self.transform_t = tt.Compose(
            [
                tt.RandomCrop(img_size, padding=4, padding_mode="reflect"),
                tt.RandomHorizontalFlip(),
                # tt.RandomRotate
                tt.RandomResizedCrop(img_size, scale=(0.5, 0.9), ratio=(1, 1)),
                tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tt.ToTensor(),
            ]
        )

    def create_model(self):
        """Create the model"""
        self.model = to_device(
            Cifar10CnnModel(self.in_channels, self.num_classes, self.img_size),
            self.device,
        )
        print("model is created")
        return self.model

    def read_train_data(self, path, train_precentage=0.8):
        """Preprocess the images
        read the images from the path and preprocess them by applying the following transformations:\n
            1.normalization by calculating the mean and standard deviation of each channel in the dataset\n
            2.data augmentation (random horizontal flip, random rotation)
        Args:
            images (ImageFolder): The images to preprocess
        """

        # Data transforms (data augmentation)

        # PyTorch datasets
        dataset = ImageFolder(path, self.transform_t)
        self.train_size = int(train_precentage * len(dataset))
        self.valid_size = len(dataset) - self.train_size
        # train_ds = ImageFolder(path + "/train", train_tfms)
        if self.valid_size == 0:
            train_ds = dataset
            val_ds = None
        else:
            train_ds, val_ds = random_split(dataset, [self.train_size, self.valid_size])
        return train_ds, val_ds

    def read_test_data(self, path):
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
            train_ds (ImageFolder): The training dataset
            valid_ds (ImageFolder): The validation dataset
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
            train_dl = DeviceDataLoader(train_dl, self.device)
        else:
            train_dl = None
        if valid_ds is not None:
            valid_dl = DataLoader(
                valid_ds, batch_size * 2, num_workers=num_workers, pin_memory=pin_memory
            )
            valid_dl = DeviceDataLoader(valid_dl, self.device)
        else:
            valid_dl = None
        return train_dl, valid_dl

    def train(
        self,
        project_path,
        epochs=10,
        max_lr=0.01,
        grad_clip=0.1,
        weight_decay=1e-4,
        opt_func=torch.optim.Adam,
        train_dl=None,
        valid_dl=None,
        decay_lr=False,
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
        self.history = [] if valid_dl is None else [evaluate(self.model, valid_dl)]
        if decay_lr == True:
            self.history += fit_without_decay_lr(
                epochs,
                max_lr,
                self.model,
                train_dl,
                valid_dl,
                grad_clip=grad_clip,
                weight_decay=weight_decay,
                opt_func=opt_func,
            )
        else:
            self.history += fit_without_decay_lr(
                epochs,
                max_lr,
                self.model,
                train_dl,
                valid_dl,
                opt_func,
            )
        print("Training completed successfully!")
        if self.history != []:
            # plot the accuracies and save it
            plot_accuracies(
                self.history,
                save_path=f"{project_path}/epoch_accuracy.png",
                save_flag=True,
            )
            print("accuracies plot saved")
            # plot losses and save it
            plot_losses(
                self.history,
                save_path=f"{project_path}/epoch_loss.png",
                save_flag=True,
            )
            print("losses plot saved")
            if decay_lr == True:
                # plot learning rates and save it
                plot_lrs(
                    self.history,
                    save_path=f"{project_path}/epoch_lr.png",
                    save_flag=True,
                )
                print("learning rates plot saved")
                # return training and validation accuracies
                # TODO:calc the training accuracy
            return None, self.history[-1]["val_acc"]

    def predict(self, img):
        """Predict the class of the image

        Args:
            img (numpy.ndarray): The image to predict
            train_ds (ImageFolder): The training dataset

        Returns:
            ste: The class of the image
        """
        # convert the numpy image to a tensor
        # img = torch.tensor(img)
        # print("img shape: ", img.shape)
        # call predict_image function
        return predict_image(img, self.model, self.device)

    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path, img_size=256):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        self.create_model(img_size=img_size)
        print("start loading model")
        self.model.load_state_dict(torch.load(path))
        self.model = to_device(self.model, self.device)
        self.model.eval()
        print("model loaded")

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
