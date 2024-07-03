from .image_classifier_utils import *
from camera_feed import CameraFeed


class ImageClassifierSVM:
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
        self.camera = CameraFeed()
        self.filename1 = "/kmeans_model.sav"
        self.filename2 = "/svm_model.sav"
        self.sift = cv2.SIFT_create()
        self.descriptors = []
        self.feature_set_train = np.array([])
        self.feature_set_valid = np.array([])
        self.bag_of_words_train = []
        self.y_train = []
        self.bag_of_words_valid = []
        self.y_valid = []
        self.device = get_default_device()
        self.model = None
        self.num_classes = num_classes
        self.in_channels = in_channels
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

    def read_train_data(self, path, train_precentage=0.8):
        """Preprocess the images
        read the images from the path and preprocess them by applying the following transformations:\n
            1.normalization by calculating the mean and standard deviation of each channel in the dataset\n
            2.data augmentation (random horizontal flip, random rotation)
        Args:
            images (ImageFolder): The images to preprocess
        """
        dataset = ImageFolder(path, self.transform_t)
        self.train_size = int(train_precentage * len(dataset))
        self.valid_size = len(dataset) - self.train_size
        # train_ds = ImageFolder(path + "/train", train_tfms)
        if self.valid_size == 0:
            train_ds = dataset
            val_ds = None
        else:
            train_ds, val_ds = random_split(dataset, [self.train_size, self.valid_size])
        # preprocess the training set
        for img, label in train_ds:
            # convert tensor to numpy array
            img = img.numpy()
            # Keypoints, descriptors
            kp, descriptor = self.sift.detectAndCompute(img, None)
            # Each keypoint has a descriptor with length 128
            if descriptor is None:
                continue
            else:
                self.descriptors.append(np.array(descriptor))
                self.feature_set_train = np.concatenate(
                    (self.feature_set_train, descriptor), axis=0
                )
                self.y_train.append(label)
        # preprocess the validation set
        if val_ds is not None:
            for img, label in val_ds:
                # convert tensor to numpy array
                img = img.numpy()
                # Keypoints, descriptors
                kp, descriptor = self.sift.detectAndCompute(img, None)
                # Each keypoint has a descriptor with length 128
                if descriptor is None:
                    continue
                else:
                    self.feature_set_valid = np.concatenate(
                        (self.feature_set_valid, descriptor), axis=0
                    )
                    self.y_valid.append(label)
        return train_ds, val_ds

    def create_model(self, n_clusters=1600):
        """Create the model"""
        # Kmeans clustering on all training set
        print("Success")
        print("Running kmeans...")
        self.n_clusters = n_clusters
        self.k_means = cluster.KMeans(
            n_clusters=self.n_clusters, init="k-means++", n_init="auto", max_iter=1000
        )

        self.svm = svm.SVC(
            decision_function_shape="ovo", random_state=42, max_iter=1000
        )
        print("Success svm")

    def read_test_data(self, path):
        """Preprocess the images
        read the images from the path and preprocess them by applying the following transformations:\n
            1.normalization by calculating the mean and standard deviation of each channel in the dataset\n
            2.data augmentation (random horizontal flip, random rotation)
        Args:
            images (_type_): _description_
        """
        test_tfms = tt.Compose([tt.ToTensor()])
        test_ds = ImageFolder(path + "/test", test_tfms)
        return test_ds

    def train(
        self,
        project_path,
    ):
        """Train the model

        Args:
            project_path (str): The path to the project
        Returns:
            float: The accuracy of the model
        """
        self.k_means.fit(self.feature_set_train)
        # Produce "bag of words" histogram for each image
        print("Success")
        print("Generating bag of words...")
        for descriptor in self.descriptors:
            vq = [0] * self.n_clusters
            descriptor = self.k_means.predict(descriptor)
            for feature in descriptor:
                vq[feature] = vq[feature] + 1
            self.bag_of_words_train.append(vq)

        # Train the SVM multiclass classification model
        print(f"Success")
        print(f"Training SVM model...")
        # self.model.fit(self.bag_of_words_train, self.y_train)
        # evaluate the model
        print("Evaluating the model...")
        # Define the parameter grid to search over
        param_grid = {
            "C": [0.1, 1, 10],
            "gamma": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
        }
        grid_search = GridSearchCV(
            self.svm,
            param_grid,
            cv=10,
            scoring="f1_weighted",
            return_train_score=True,
        )

        # Fit the grid search object to the dataset to find the best hyperparameters
        self.model = grid_search.fit(self.bag_of_words_train, self.y_train)
        # validate the model
        print("Validating the model...")
        for descriptor in self.feature_set_valid:
            vq = [0] * self.n_clusters
            descriptor = self.k_means.predict(descriptor)
            for feature in descriptor:
                vq[feature] = vq[feature] + 1
            self.bag_of_words_valid.append(vq)
        # Predict the labels of the validation set
        model_accuracy = self.model.score(self.bag_of_words_valid, self.y_valid)
        return model_accuracy

    def predict(self, img):
        """Predict the class of the image

        Args:
            img (numpy.ndarray): The image to predict
        Returns:
            str: The class of the image
        """
        # Keypoints, descriptors
        kp, descriptor = self.sift.detectAndCompute(img, None)
        # Each keypoint has a descriptor with length 128
        if descriptor is None:
            return "No keypoints"
        else:
            vq = [0] * self.n_clusters
            descriptor = self.k_means.predict(descriptor)
            for feature in descriptor:
                vq[feature] = vq[feature] + 1
            return self.model.predict([vq])[0]

    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        pickle.dump(self.k_means, open(path + self.filename1, "wb"))
        # # save the SVM model to disk
        pickle.dump(self.model, open(path + self.filename2, "wb"))

    def load(self, path, img_size=256):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        self.k_means = pickle.load(open(path + self.filename1, "rb"))
        self.model = pickle.load(open(path + self.filename2, "rb"))

    def start_feed(self):
        """Start the camera feed"""
        return self.camera.start_feed()

    def stop_feed(self):
        """Stop the camera feed"""
        return self.camera.stop_feed()

    def get_frame(self):
        """Get a frame from the camera feed"""
        return self.camera.get_latest_frame()
