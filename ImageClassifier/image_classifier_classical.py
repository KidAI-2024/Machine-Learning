from .image_classifier_utils import *
from camera_feed import CameraFeed


class ImageClassifierClassical:
    def __init__(
        self,
        num_classes=2,
        in_channels=3,
        img_size=256,
        feature_extraction_type=0,
        model_type=0,
    ):
        """Constructor for the ImageClassifierReNet class

        Args:
            num_classes (int, optional): The number of classes you want predict. Defaults to 2.
            in_channels (int, optional): The number of input channels for the images. Defaults to 3.
        """
        self.camera = CameraFeed()
        self.n_clusters = num_classes
        self.filename1 = "kmeans_model.sav"
        self.filename2 = "clf_model.sav"
        self.sift = cv2.SIFT_create()
        self.descriptors_t = []
        self.descriptors_v = []
        self.feature_set_train = None
        self.bag_of_words_train = []
        self.y_train = []
        self.bag_of_words_valid = []
        self.y_valid = []
        self.hog_train = []
        self.hog_valid = []
        self.model = None
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.transform_t = tt.Compose(
            [
                # tt.RandomCrop(img_size, padding=4, padding_mode="reflect"),
                tt.Resize((img_size, img_size)),
                tt.RandomHorizontalFlip(),
                # tt.RandomRotate
                tt.RandomResizedCrop(img_size, scale=(0.5, 0.9), ratio=(1, 1)),
                tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tt.ToTensor(),
            ]
        )
        self.feature_extraction_type = (
            feature_extraction_type  # 0 for SIFT, 1 for HOG, 2 for LBP
        )
        self.model_type = (
            model_type  # 0 for SVM, 1 for Logistic Regression, 2 for Random Forest
        )
        # Parameters for LBP
        self.radius = 3  # Radius of circle
        self.n_points = (
            8 * self.radius
        )  # Number of points to consider around the circle
        self.lbp_train = []
        self.lbp_valid = []

    def read_train_data(self, path, train_precentage=0.8):
        """Preprocess the images
        read the images from the path and preprocess them by applying the following transformations:\n
            1.normalization by calculating the mean and standard deviation of each channel in the dataset\n
            2.data augmentation (random horizontal flip, random rotation)
        Args:
            images (ImageFolder): The images to preprocess
        """
        test_dir = os.path.join(path, "test")
        if os.path.exists(test_dir):
            print("test directory exists")
            # delete the test directory
            shutil.rmtree(test_dir)
        dataset = ImageFolder(path, self.transform_t)
        self.train_size = int(train_precentage * len(dataset))
        self.valid_size = len(dataset) - self.train_size
        # train_ds = ImageFolder(path + "/train", train_tfms)
        if self.valid_size == 0:
            train_ds = dataset
            val_ds = None
        else:
            train_ds, val_ds = random_split(dataset, [self.train_size, self.valid_size])
            # save val_ds to teh disk
            val_dir = os.path.join(path, "test")
            os.makedirs(val_dir, exist_ok=True)
            for idx in val_ds.indices:
                img_path, label = dataset.imgs[idx]
                class_name = dataset.classes[label]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy(
                    img_path, os.path.join(class_dir, os.path.basename(img_path))
                )
        # preprocess the training set
        if self.feature_extraction_type == 0:
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
        elif self.feature_extraction_type == 1:
            for img, label in train_ds:
                # print("img shape", img.shape)
                # convert tensor to numpy array
                img = np.array(img)
                # convert the shape to [32,32,3] instead of [3,32,32]
                img = np.transpose(img, (1, 2, 0))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print("img shape", img.shape)
                gray = np.uint8(
                    255
                    * (gray - np.min(gray))
                    / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
                )
                # HOG features
                self.hog_train.append(
                    hog(
                        gray,
                        pixels_per_cell=(8, 8),
                        transform_sqrt=True,
                        feature_vector=True,
                    ),
                )
                self.y_train.append(label)
        else:  # self.feature_extraction_type == 2 => LBP
            print("LBP")
            for img, label in train_ds:
                # print("img shape", img.shape)
                # convert tensor to numpy array
                img = np.array(img)
                # convert the shape to [32,32,3] instead of [3,32,32]
                img = np.transpose(img, (1, 2, 0))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # print("img shape", img.shape)
                gray = np.uint8(
                    255
                    * (gray - np.min(gray))
                    / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
                )
                # LBP features
                # Calculate LBP
                lbp_image = local_binary_pattern(
                    gray, self.n_points, self.radius, method="uniform"
                )
                # Compute the histogram of the LBP
                n_bins = int(lbp_image.max() + 1)
                # print("n_bins:", n_bins)
                hist, _ = np.histogram(
                    lbp_image, bins=n_bins, range=(0, n_bins), density=True
                )
                self.lbp_train.append(hist)
                self.y_train.append(label)

        # preprocess the validation set
        if val_ds is not None:
            if self.feature_extraction_type == 0:
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
            elif self.feature_extraction_type == 1:
                for img, label in val_ds:
                    # convert tensor to numpy array
                    img = np.array(img)
                    # convert the shape to [32,32,3] instead of [3,32,32]
                    img = np.transpose(img, (1, 2, 0))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # print("img shape", img.shape)
                    gray = np.uint8(
                        255
                        * (gray - np.min(gray))
                        / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
                    )
                    # HOG features
                    self.hog_valid.append(
                        hog(
                            gray,
                            pixels_per_cell=(8, 8),
                            transform_sqrt=True,
                            feature_vector=True,
                        ),
                    )
                    self.y_valid.append(label)
            else:  # self.feature_extraction_type == 2 => LBP
                for img, label in val_ds:
                    # print("img shape", img.shape)
                    # convert tensor to numpy array
                    img = np.array(img)
                    # convert the shape to [32,32,3] instead of [3,32,32]
                    img = np.transpose(img, (1, 2, 0))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # print("img shape", img.shape)
                    gray = np.uint8(
                        255
                        * (gray - np.min(gray))
                        / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
                    )
                    # LBP features
                    # Calculate LBP
                    lbp_image = local_binary_pattern(
                        gray, self.n_points, self.radius, method="uniform"
                    )
                    # Compute the histogram of the LBP
                    n_bins = int(lbp_image.max() + 1)
                    hist, _ = np.histogram(
                        lbp_image, bins=n_bins, range=(0, n_bins), density=True
                    )
                    self.lbp_valid.append(hist)
                    self.y_valid.append(label)
        return train_ds, val_ds

    def create_model(self, img_size=256):
        """Create the model"""
        # Kmeans clustering on all training set
        # print("creating kmeans...")
        if self.feature_extraction_type == 0:  # sift
            self.n_clusters = np.floor(np.sqrt(len(self.feature_set_train) / 2)).astype(
                int
            )
            # self.n_clusters = np.floor(5 * np.sqrt(len(self.feature_set_train))).astype(int)
            self.k_means = cluster.KMeans(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init="auto",
                max_iter=1000,
            )
        if self.model_type == 0:
            print("Creating SVM model...")
            self.clf = svm.SVC(
                decision_function_shape="ovo",
                random_state=42,
                max_iter=1000,
            )
        elif self.model_type == 1:
            print("Creating Logistic Regression model...")
            self.clf = LogisticRegression(
                random_state=42,
                max_iter=1000,
            )
        elif self.model_type == 2:
            print("Creating Random Forest model...")
            self.clf = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
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
        if self.feature_extraction_type == 0:
            self.k_means.fit(self.feature_set_train)
            # Produce "bag of words" histogram for each image
            print("Success")
            print("Generating bag of words...")
            for i, descriptor in enumerate(self.descriptors_t):
                vq = [0] * self.n_clusters
                descriptor = self.k_means.predict(descriptor)
                for feature in descriptor:
                    vq[feature] = vq[feature] + 1
                self.bag_of_words_train.append(vq)
        elif self.feature_extraction_type == 1:
            self.bag_of_words_train = self.hog_train
        else:
            self.bag_of_words_train = self.lbp_train
        # Train the multiclass classification model
        print(f"Training model...")
        # Define the parameter grid to search over
        if self.model_type == 0:  # svm
            param_grid = {
                "C": [0.1, 1, 10],
                "gamma": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
            }
        elif self.model_type == 1:  # logistic regression
            param_grid = {
                "C": [0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear", "saga"],
            }
        elif self.model_type == 2:  # random forest
            param_grid = {
                "n_estimators": [20, 50, 100],
                "max_depth": [10, 20, 50],
                # "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
        print("Grid search started")
        grid_search = GridSearchCV(
            self.clf,
            param_grid,
            cv=10,
            scoring="f1_weighted",
            return_train_score=True,
        )
        # Fit the grid search object to the dataset to find the best hyperparameters
        self.model = grid_search.fit(self.bag_of_words_train, self.y_train)
        # print the training accuracy
        training_accuracy = grid_search.best_score_
        print("Training accyracy: ", training_accuracy)
        # validate the model
        print("Validating the model...")
        if self.feature_extraction_type == 0:  # sift
            for descriptor in self.descriptors_v:
                vq = [0] * self.n_clusters
                descriptor = self.k_means.predict(descriptor)
                for feature in descriptor:
                    vq[feature] = vq[feature] + 1
                self.bag_of_words_valid.append(vq)
        elif self.feature_extraction_type == 1:  # hog
            self.bag_of_words_valid = self.hog_valid
        else:
            self.bag_of_words_valid = self.lbp_valid
        # Predict the labels of the validation set
        valid_accuracy = 0
        print("Predicting the labels of the validation set...")
        if len(self.y_valid) != 0:
            valid_accuracy = self.model.score(
                np.array(self.bag_of_words_valid), np.array(self.y_valid)
            )
        return training_accuracy, valid_accuracy

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
        print("img shape", img.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("img shape", img.shape)
        gray = np.uint8(
            255
            * (gray - np.min(gray))
            / (np.max(gray) - np.min(gray) + np.finfo(float).eps)
        )
        # Keypoints, descriptors
        if self.feature_extraction_type == 0:  # sift
            print("predict sift")
            kp, descriptor = self.sift.detectAndCompute(gray, None)
            # Each keypoint has a descriptor with length 128
            if descriptor is None:
                return -1
            else:
                print("kmeans prediction...")
                vq = [0] * self.n_clusters
                descriptor = self.k_means.predict(descriptor)
                print("len descriptor", len(descriptor))
                for feature in descriptor:
                    vq[feature] = vq[feature] + 1
                if self.model is None:
                    print("Model is None")
                    return -1
                pred = self.model.predict([vq])[0]
        elif self.feature_extraction_type == 1:  # hog
            print("predict hog")
            # hog features
            hog_features = hog(
                gray,
                pixels_per_cell=(8, 8),
                transform_sqrt=True,
                feature_vector=True,
            )
            print("hog_features shape", hog_features.shape)
            if self.model is None:
                print("Model is None")
                return -1
            pred = self.model.predict([hog_features])[0]
        else:  # self.feature_extraction_type == 2 => LBP
            print("predict lbp")
            # Calculate LBP
            lbp_image = local_binary_pattern(
                gray, self.n_points, self.radius, method="uniform"
            )
            # Compute the histogram of the LBP
            n_bins = int(lbp_image.max() + 1)
            hist, _ = np.histogram(
                lbp_image, bins=n_bins, range=(0, n_bins), density=True
            )
            print("hist shape", hist.shape)
            if self.model is None:
                print("Model is None")
                return -1
            pred = self.model.predict([hist])[0]
        return pred

    def save(self, path):
        """Save the model to disk
        Args:
            path (str): The path to save the model to
        """
        if self.feature_extraction_type == 0:
            # save the number of clusters in json file
            with open(os.path.join(path, "n_clusters.json"), "w") as f:
                json.dump({"n_clusters": str(self.n_clusters)}, f)

            file_path = os.path.join(path, "n_clusters.txt")
            write_integer_to_file(file_path, self.n_clusters)
            # save the kmeans model to disk
            pickle.dump(self.k_means, open(os.path.join(path, self.filename1), "wb"))
        # # save the SVM model to disk
        pickle.dump(self.model, open(os.path.join(path, self.filename2), "wb"))

    def load(self, path, img_size=256):
        """Load the model from disk
        Args:
            path (str): The path to load the model from
        """
        if self.feature_extraction_type == 0:
            self.k_means = pickle.load(open(os.path.join(path, self.filename1), "rb"))
            # read the number of clusters from the json file
            try:
                with open(os.path.join(path, "n_clusters.json"), "r") as f:
                    data = json.load(f)
                    self.n_clusters = int(data["n_clusters"])
            except Exception as e:
                print(f"Error in load json: {e}")
            # read the number of clusters from the txt file
            file_path = os.path.join(path, "n_clusters.txt")
            self.n_clusters = read_integer_from_file(file_path)
        self.model = pickle.load(open(os.path.join(path, self.filename2), "rb"))

    def start_feed(self):
        """Start the camera feed"""
        return self.camera.start_feed()

    def stop_feed(self):
        """Stop the camera feed"""
        return self.camera.stop_feed()

    def get_frame(self):
        """Get a frame from the camera feed"""
        return self.camera.get_latest_frame()
