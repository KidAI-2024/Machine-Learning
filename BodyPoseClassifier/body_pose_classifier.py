import numpy as np

from sklearn.svm import SVC
import pickle

# import sklearn
from .body_pose_utils import BodyPoseUtils
from camera_feed import CameraFeed

class BodyPoseClassifier:
    def __init__(self):
        self.model = SVC(kernel="linear")
        self.body_pose_utils = BodyPoseUtils()
        self.camera = CameraFeed()

   
    def preprocess(self, data):
        """Preprocess the data to extract features from the dictionary of images"""
        features_map = self.body_pose_utils.get_training_features(data)
        return features_map

    def preprocess_draw_landmarks(self, image):
        try:
            landmarks = self.body_pose_utils.get_body_landmarks(image)
            return self.body_pose_utils.draw_body_landmarks(image, landmarks)
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

        features = self.body_pose_utils.extract_features(image)
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
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        """Load the model from disk"""
        self.model = pickle.load(open(path, "rb"))