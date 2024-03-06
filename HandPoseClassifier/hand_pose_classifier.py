import numpy as np

from sklearn.svm import SVC

# import sklearn
from .hand_pose_utils import HandPoseUtils


class HandPoseClassifier:
    def __init__(self):
        self.model = SVC(kernel="linear")
        self.hand_pose_utils = HandPoseUtils()

    def preprocess(self, data):
        features_map = self.hand_pose_utils.get_training_features(data)
        return features_map

    def train(self, features_map):
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
        # ignore the image it was all black
        if np.all(image == 0):
            return -1

        features = self.hand_pose_utils.extract_features(image)
        # features = features[0]
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        print(f"Predicted class: {prediction[0]}")
        return prediction[0]
