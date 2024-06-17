from ImageClassifier.image_classifier import ImageClassifier

import utils
from decorators import event
from server_utils import Req, Res

image_classifier = ImageClassifier()


@event("train_hand_pose")
def train_hand_pose(req: Req, res: Res) -> int:
    path = req.msg["path"]
    # training_data is map {"Class Number(first character in the folder name)" : [images]}
    print("Reading data...")
    training_data = utils.read_data(path)
    print("Extracting features...")
    try:
        features_map = image_classifier.preprocess(training_data)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return -1
    print("Training...")
    try:
        image_classifier.train(features_map)
    except Exception as e:
        print(f"Error in train: {e}")
        return -1
    print("Saving model...")
    model_path = "./hand_pose_model.pkl"
    image_classifier.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    return None