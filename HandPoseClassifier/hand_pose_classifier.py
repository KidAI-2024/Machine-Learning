import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# import sklearn
from .hand_pose_utils import HandPoseUtils
from camera_feed import CameraFeed


class HandPoseClassifier:
    def __init__(self):
        self.model = None
        self.hand_pose_utils = HandPoseUtils()
        self.camera = CameraFeed()
        self.selected_features_list = None
        self.training_accuracy = None

    def set_model(self, model_name: str):
        if model_name == "SVM":
            self.model = SVC(kernel="linear")
        elif model_name == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=2, random_state=0
            )
        elif model_name == "GradientBoosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
            )
        elif model_name == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=5)

        else:
            self.model = SVC(kernel="linear")
            print(f"Invalid model name: {model_name}, setting model to default SVM")

    def preprocess(self, data):
        """Preprocess the data to extract features from the dictionary of images"""
        features_map = self.hand_pose_utils.get_training_features(
            data, self.selected_features_list
        )
        # print(f"selected_features_list: {len(self.selected_features_list)}")

        return features_map

    def preprocess_draw_landmarks(self, image):
        try:
            landmarks = self.hand_pose_utils.get_hand_landmarks(image)
            return self.hand_pose_utils.draw_hand_landmarks(image, landmarks)
        except Exception as e:
            print(f"Error in preprocess_draw_landmarks: {e}")
            return image

    def train(self, features_map):
        """Train the model using the extracted features"""
        # Concatenate features and labels
        X_train = np.concatenate([features_map[key] for key in features_map])
        y_train = np.concatenate(
            [[key] * len(features_map[key]) for key in features_map]
        )

        # Reshape X_train
        X_train = X_train.reshape(X_train.shape[0], -1)

        # Train the model
        self.model.fit(X_train, y_train)
        self.training_accuracy = self.model.score(X_train, y_train)

    def predict(self, image):
        """Predict the class of an image"""
        # ignore the image it was all black
        if np.all(image == 0):
            return -1

        features = self.hand_pose_utils.extract_features(
            image, self.selected_features_list
        )
        # features = features[0]
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        # print(f"Predicted class: {prediction[0]}")
        return prediction[0]

    def start_feed(self):
        """Start the camera feed"""
        return self.camera.start_feed()

    def stop_feed(self):
        """Stop the camera feed"""
        return self.camera.stop_feed()

    def get_frame(self):
        """Get a frame from the camera feed"""
        return self.camera.get_latest_frame()

    def save(self, path):
        """Save the model to disk"""
        pickle.dump(self.model, open(path, "wb"))

    def load(self, path):
        """Load the model from disk"""
        self.model = pickle.load(open(path, "rb"))

    def feature_importance_graph(self):
        """Returns the feature importance graph image based on the linear SVM coefficients"""
        if hasattr(self.model, "coef_"):
            feature_importance = np.abs(
                self.model.coef_[0]
            )  # Take the absolute value of coefficients
            # Sort the feature importances
            sorted_idx = np.argsort(feature_importance)
            sorted_feature_importance = feature_importance[sorted_idx]
            sorted_feature_names = np.array(self.selected_features_list)[sorted_idx]

            plt.figure(figsize=(10, 6))
            plt.bar(
                range(len(feature_importance)),
                sorted_feature_importance,
                align="center",
            )
            plt.xticks(
                range(len(feature_importance)), sorted_feature_names, rotation=90
            )
            plt.xlabel("Feature Names")
            plt.ylabel("Absolute Feature Importance")
            plt.title("Feature Importance in Linear SVM")
            plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
            # plt.show()
            # Convert plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=50)
            buf.seek(0)
            plt.close()  # Close the figure to avoid memory leaks

            # Encode bytes as base64 to be returned as string
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            # print size of image
            # print(f"Size of image: {len(image_base64)}")
            return image_base64

        else:
            print(
                "Model does not have coef_ attribute. Ensure that the model is a linear SVM."
            )
            return None
