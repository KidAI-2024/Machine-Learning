from . import media_pipe
import numpy as np

RELATIVE_POSTIONS = "relative_positions"
RELATIVE_ANGLES = "relative_angles"

class BodyPoseClassifier:
    def __init__(self):
        pass
    
    def preprocess(self, data):
        preprocessed_data = {}
        # loop over each class in the data
        for key in data:
            preprocessed_images = []
            landmarks = []
            preprocessed_images, landmarks = media_pipe.get_landmarks_from_images(data[key])
            preprocessed_data[key] = preprocessed_images
            print(f"Preprocessed {len(preprocessed_images)} images for class {key}")
        return preprocessed_data

    def extract_features(self, data, feature_type = RELATIVE_POSTIONS):
        if feature_type == RELATIVE_POSTIONS:
            features_map = {}
            for key in data:
                features = []
                for image in data[key]:
                    center_x, center_y, dist_shoulder_to_elbow, angle_elbow = media_pipe.relative_postions_features(image)
                    features.append([center_x, center_y, dist_shoulder_to_elbow, angle_elbow])
                    np.array(features)
                features_map[key] = features
            return np.array(features)
        

        elif feature_type == RELATIVE_ANGLES:
            pass
        else:
            return None


    def train(self, data):
        pass

    def predict(self, data):
        pass