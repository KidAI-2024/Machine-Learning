from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
import utils
from decorators import event
from server_utils import Req, Res
import os

hand_pose_classifier = HandPoseClassifier()


@event("start_feed_hand_pose")
def start_feed_hand_pose(req: Req, res: Res):
    # start camera feed
    status = hand_pose_classifier.start_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("stop_feed_hand_pose")
def stop_feed_hand_pose(req: Req, res: Res):
    # stop camera feed
    status = hand_pose_classifier.stop_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("get_feed_frame_handpose")
def get_feed_frame_hand_pose(req: Req, res: Res):
    frame = hand_pose_classifier.get_frame()
    if frame is not None:
        # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
        preprocess_image = hand_pose_classifier.preprocess_draw_landmarks(frame)
        preprocess_image_str = utils.image_to_b64string(preprocess_image)
        res_msg = {"frame": preprocess_image_str}
        return res.build(req.event, res_msg)
    else:
        res_msg = {"frame": None}
        return res.build(req.event, res_msg)


@event("preprocess_hand_pose")
def preprocess_hand_pose(req: Req, res: Res):
    frame_bytes = req.msg["frame"]
    width_str = req.msg["width"]
    height_str = req.msg["height"]
    try:
        width = int(width_str)
    except ValueError:
        width = 320
        print(f"Invalid width: {width_str}")
    try:
        height = int(height_str)
    except ValueError:
        height = 180
        print(f"Invalid height: {height_str}")
    # Convert the bytes to an image
    image = utils.b64string_to_image(frame_bytes, (height, width, 3))
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
    preprocess_image = hand_pose_classifier.preprocess_draw_landmarks(image)
    preprocess_image_str = utils.image_to_b64string(preprocess_image)
    res_msg = {"preprocessed_image": preprocess_image_str}
    return res.build(req.event, res_msg)
    # res.send(req.event, res_msg)


@event("train_hand_pose")
def train_hand_pose(req: Req, res: Res) -> int:
    path = req.msg["path"]
    model = req.msg["model"]
    print(f"Training hand pose model using {model} model")
    feature_extraction_type = req.msg["feature_extraction_type"]
    print(f"Feature extraction type: {feature_extraction_type}")
    selected_features = req.msg["features"].split(",")
    hand_pose_classifier.selected_features_list = selected_features
    print(f"Selected features: {selected_features}")
    if feature_extraction_type == "mediapipe" and selected_features == [""]:
        res_msg = {
            "status": "failed",
            "error": "Select some features to train the model",
        }
        return res.build(req.event, res_msg)
    # training_data is map {"Class Number(first character in the folder name)" : [images]}
    print("Reading data...")
    training_data = utils.read_data(path)
    print("Extracting features...")
    try:
        features_map = hand_pose_classifier.preprocess(training_data)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        res_msg = {
            "status": "failed",
            "error": "Training data contains invalid images",
        }
        return res.build(req.event, res_msg)
    print("Training...")
    try:
        hand_pose_classifier.train(features_map)
    except Exception as e:
        print(f"Error in train: {e}")
        res_msg = {"status": "failed"}
        return res.build(req.event, res_msg)
    print("Saving model...")
    project_name = path.split("/")[-1]
    saved_model_name = "hand_pose_model.pkl"
    model_path = os.path.join(
        path, project_name, saved_model_name
    )  # Currect directory is Machine-Learning
    hand_pose_classifier.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    res_msg = {"status": "success", "saved_model_name": saved_model_name}
    return res.build(req.event, res_msg)


@event("predict_hand_pose")
def predict_hand_pose(req: Req, res: Res):
    frame_bytes = req.msg["frame"]
    width_str = req.msg["width"]
    height_str = req.msg["height"]
    try:
        width = int(width_str)
    except ValueError:
        width = 320
        print(f"Invalid width: {width_str}")
    try:
        height = int(height_str)
    except ValueError:
        height = 180
        print(f"Invalid height: {height_str}")
    # Convert the bytes to an image
    image = utils.b64string_to_image(frame_bytes, (height, width, 3))
    # preprocessed_img = hand_pose_classifier.preprocess(image)
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", preprocessed_img)
    # hand_pose_classifier.load("./hand_pose_model.pkl")
    try:
        pred = hand_pose_classifier.predict(image)
    except Exception as e:
        print(f"Error in predict: {e}")
        pred = -1
    print(f"Predicted class: {pred}")

    res_msg = {"prediction": pred}
    return res.build(req.event, res_msg)


@event("load_hand_pose_model")
def load_hand_pose_model(req: Req, res: Res) -> int:
    path = req.msg["path"]
    saved_model_name = req.msg["saved_model_name"]
    model_path = os.path.join(path, saved_model_name)
    model = req.msg["model"]
    feature_extraction_type = req.msg["feature_extraction_type"]
    hand_pose_classifier.selected_features_list = req.msg["features"].split(",")
    model_path = os.path.join(path, saved_model_name)
    try:
        hand_pose_classifier.load(model_path)
        print(f"Model loaded from {model_path}")
        res_msg = {"status": "success"}
    except Exception as e:
        res_msg = {"status": "Model file not found"}

    return res.build(req.event, res_msg)
