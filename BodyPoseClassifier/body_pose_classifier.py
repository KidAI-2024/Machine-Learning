from . import media_pipe
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import cv2

RELATIVE_POSTIONS = "relative_positions"
RELATIVE_ANGLES = "relative_angles"

class BodyPoseClassifier:
    def __init__(self):
        self.model = None
        self.feature_type = None
    
    def preprocess(self, data):
        preprocessed_data = {}
        landmarks_map = {}
        # loop over each class in the data
        for key in data:
            preprocessed_images = []
            landmarks = []
            preprocessed_images, landmarks = media_pipe.get_landmarks_from_images(data[key])
            # print(landmarks)
            preprocessed_data[key] = preprocessed_images
            landmarks_map[key] = landmarks
            # print(f"Preprocessed {len(preprocessed_images)} images for class {key}")
        return preprocessed_data, landmarks_map

    def get_training_features(self, landmarks_map, feature_type = RELATIVE_POSTIONS):
        self.feature_type = feature_type
        features_map = {}
        for key in landmarks_map:
            features_map[key] = self.extract_features(landmarks_map[key], feature_type)
        return features_map

    def extract_features(self, landmarks, feature_type):
        features = []
        if feature_type == RELATIVE_POSTIONS:
            for landmark in landmarks:
                feat = media_pipe.relative_positions_features(landmark)
                features.append([feat])
                np.array(features)
        elif feature_type == RELATIVE_ANGLES:
            pass
        return features


    def train(self, features_map):
        # Concatenate features and labels
        X_train = np.concatenate([features_map[key] for key in features_map])
        y_train = np.concatenate([[key] * len(features_map[key]) for key in features_map])

        # Reshape X_train
        X_train = X_train.reshape(X_train.shape[0], -1)

        # Train the model
        self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)

    def predict(self, image):
        # ignore the image it was all black
        if np.all(image == 0):
            return "99"
        
        preprocessed_image, landmark = media_pipe.get_landmarks(image)
        features = self.extract_features([landmark], feature_type = self.feature_type)
        features = features[0]
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        print(f"Predicted class: {prediction[0]}")
        return prediction[0]
    
    # save and load model still not working
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        
