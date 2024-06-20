import numpy as np

from sklearn.svm import SVC
import pickle

# import sklearn
from .body_pose_utils import BodyPoseUtils
from camera_feed import CameraFeed
import io
import base64
import matplotlib.pyplot as plt

class BodyPoseClassifier:
    def __init__(self):
        self.model = SVC(kernel="linear")
        self.feature_extraction_type = "mediapipe"
        self.selected_features = []
        self.body_pose_utils = BodyPoseUtils()
        self.camera = CameraFeed()

    def SelectModel(self, model):
        if model == "SVC":
            self.model = SVC(kernel="linear")
        elif model == "NeuralNetwork":
            self.model = None
   
    def preprocess(self, data):
        """Preprocess the data to extract features from the dictionary of images"""
        features_map = self.body_pose_utils.get_training_features(data, self.selected_features)
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

    def feature_importance_graph(self):
        """Returns the feature importance graph image based on the linear SVM coefficients"""
        if hasattr(self.model, 'coef_'):
            feature_importance = np.abs(self.model.coef_[0])  # Take the absolute value of coefficients
            # Sort the feature importances
            sorted_idx = np.argsort(feature_importance)
            sorted_feature_importance = feature_importance[sorted_idx]
            sorted_feature_names = np.array(self.selected_features)[sorted_idx]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(feature_importance)), sorted_feature_importance, align='center')
            plt.xticks(range(len(feature_importance)), sorted_feature_names, rotation=90)
            plt.xlabel('Feature Names')
            plt.ylabel('Absolute Feature Importance')
            plt.title('Feature Importance in Linear SVM')
            plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
            plt.show()
            # Convert plot to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()  # Close the figure to avoid memory leaks

            # Encode bytes as base64 to be returned as string
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return image_base64

        else:
            print("Model does not have coef_ attribute. Ensure that the model is a linear SVM.")

    def convert_plot_to_image(self, plot):
        """Converts a plot to an image"""
        
    

    def predict(self, image):
        """Predict the class of an image"""
        # ignore the image it was all black
        if np.all(image == 0):
            return -1

        features = self.body_pose_utils.extract_features(image, self.selected_features)
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