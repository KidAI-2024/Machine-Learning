import numpy as np

from sklearn.svm import SVC
import pickle

# import sklearn
from .hand_pose_utils import HandPoseUtils
from camera_feed import CameraFeed


class HandPoseClassifier:
    def __init__(self):
        self.model = SVC(kernel="linear")
        self.hand_pose_utils = HandPoseUtils()
        self.camera = CameraFeed()
        self.selected_features_list = None

    def preprocess(self, data):
        """Preprocess the data to extract features from the dictionary of images"""
        features_map = self.hand_pose_utils.get_training_features(
            data, self.selected_features_list
        )
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
