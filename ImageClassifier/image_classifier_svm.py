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
        self.n_clusters = num_classes
        self.filename1 = "kmeans_model.sav"
        self.filename2 = "svm_model.sav"
        self.sift = cv2.SIFT_create()
        self.descriptors_t = []
        self.descriptors_v = []
        self.feature_set_train = None
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
            # print("img shape", img.shape)
            # convert tensor to numpy array
            img = np.array(img)
            # convert the shape to [32,32,3] instead of [3,32,32]
            img = np.transpose(img, (1, 2, 0))
            # print("img shape", img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print("img shape", img.shape)
            gray = np.uint8(
                255
                * (gray - np.min(gray))
                / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
            )

            # Keypoints, descriptors
            kp, descriptor = self.sift.detectAndCompute(gray, None)
            # Each keypoint has a descriptor with length 128
            if descriptor is None:
                continue
            else:
                self.descriptors_t.append(np.array(descriptor))
                if self.feature_set_train is None:
                    self.feature_set_train = np.copy(descriptor)
                else:
                    self.feature_set_train = np.concatenate(
                        (self.feature_set_train, descriptor), axis=0
                    )
                self.y_train.append(label)
        # preprocess the validation set
        if val_ds is not None:
            for img, label in val_ds:
                # convert tensor to numpy array
                img = np.array(img)
                # convert the shape to [32,32,3] instead of [3,32,32]
                img = np.transpose(img, (1, 2, 0))
                # print("img shape", img.shape)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print("img shape", img.shape)
                gray = np.uint8(
                    255
                    * (gray - np.min(gray))
                    / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
                )
                # Keypoints, descriptors
                kp, descriptor = self.sift.detectAndCompute(gray, None)
                # Each keypoint has a descriptor with length 128
                if descriptor is None:
                    continue
                else:
                    self.descriptors_v.append(np.array(descriptor))
                    self.y_valid.append(label)

        print("length of y_train", len(self.y_train))
        print("length of y_valid", len(self.y_valid))
        print("length of feature_set_train", len(self.feature_set_train))

        return train_ds, val_ds

    def create_model(self):
        """Create the model"""
        # Kmeans clustering on all training set
        print("creating kmeans...")
        self.n_clusters = np.floor(np.sqrt(len(self.feature_set_train) / 2)).astype(int)
        # self.n_clusters = np.floor(5 * np.sqrt(len(self.feature_set_train))).astype(int)
        self.k_means = cluster.KMeans(
            n_clusters=self.n_clusters, init="k-means++", n_init="auto", max_iter=1000
        )

        self.svm = svm.SVC(
            decision_function_shape="ovo", random_state=42, max_iter=1000
        )

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
        for descriptor in self.descriptors_t:
            vq = [0] * self.n_clusters
            descriptor = self.k_means.predict(descriptor)
            for feature in descriptor:
                vq[feature] = vq[feature] + 1
            self.bag_of_words_train.append(vq)

        # Train the SVM multiclass classification model
        print(f"Training SVM model...")
        # self.model.fit(self.bag_of_words_train, self.y_train)
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
        # print the training accuracy
        training_score = grid_search.best_score_
        print("Training accyracy: ", training_score)
        # validate the model
        print("Validating the model...")
        for descriptor in self.descriptors_v:
            vq = [0] * self.n_clusters
            descriptor = self.k_means.predict(descriptor)
            for feature in descriptor:
                vq[feature] = vq[feature] + 1
            self.bag_of_words_valid.append(vq)
        # Predict the labels of the validation set
        print("Predicting the labels of the validation set...")
        valid_accuracy = self.model.score(
            np.array(self.bag_of_words_valid), np.array(self.y_valid)
        )
        return valid_accuracy

    def predict(self, img):
        """Predict the class of the image

        Args:
            img (numpy.ndarray): The image to predict
        Returns:
            str: The class of the image
        """
        # convert tensor to numpy array
        # img = np.array(img)
        # convert the shape to [32,32,3] instead of [3,32,32]
        # img = np.transpose(img, (1, 2, 0))
        # print("img shape", img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print("img shape", img.shape)
        gray = np.uint8(
            255
            * (gray - np.min(gray))
            / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
        )
        # Keypoints, descriptors
        kp, descriptor = self.sift.detectAndCompute(gray, None)
        # Each keypoint has a descriptor with length 128
        if descriptor is None:
            return -1
        else:
            # print("kmeans prediction...")
            vq = [0] * self.n_clusters
            descriptor = self.k_means.predict(descriptor)
            # print("len descriptor", len(descriptor))
            for feature in descriptor:
                vq[feature] = vq[feature] + 1
            # print("svm prediction...")
            return self.model.predict([vq])[0]

    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        # save the number of clusters in json file
        with open(os.path.join(path, "n_clusters.json"), "w") as f:
            json.dump({"n_clusters": str(self.n_clusters)}, f)
        pickle.dump(self.k_means, open(os.path.join(path, self.filename1), "wb"))
        # # save the SVM model to disk
        pickle.dump(self.model, open(os.path.join(path, self.filename2), "wb"))

    def load(self, path, img_size=256):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        try:
            self.k_means = pickle.load(open(os.path.join(path, self.filename1), "rb"))
            self.model = pickle.load(open(os.path.join(path, self.filename2), "rb"))
            # read the number of clusters from the json file
            with open(os.path.join(path, "n_clusters.json"), "r") as f:
                data = json.load(f)
                self.n_clusters = int(data["n_clusters"])
        except Exception as e:
            print(f"Error in load: {e}")
            # self.create_model()

    def start_feed(self):
        """Start the camera feed"""
        return self.camera.start_feed()

    def stop_feed(self):
        """Stop the camera feed"""
        return self.camera.stop_feed()

    def get_frame(self):
        """Get a frame from the camera feed"""
        return self.camera.get_latest_frame()
