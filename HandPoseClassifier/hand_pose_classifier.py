from .hand_pose_utils import HandPoseUtils


class HandPoseClassifier:
    def __init__(self):
        self.hand_pose_utils = HandPoseUtils()

    def preprocess(self, data):
        land_marks = self.hand_pose_utils.get_hand_landmarks(data)
        return land_marks

    def train(self, data):
        pass

    def predict(self, data):
        pass
