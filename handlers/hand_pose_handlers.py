from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
import utils
from decorators import event
from server_utils import Req, Res

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
    # training_data is map {"Class Number(first character in the folder name)" : [images]}
    print("Reading data...")
    training_data = utils.read_data(path)
    print("Extracting features...")
    try:
        features_map = hand_pose_classifier.preprocess(training_data)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return -1
    print("Training...")
    try:
        hand_pose_classifier.train(features_map)
    except Exception as e:
        print(f"Error in train: {e}")
        return -1
    print("Saving model...")
    model_path = "./hand_pose_model.pkl"
    hand_pose_classifier.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    return None


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
        return -1
    print(f"Predicted class: {pred}")

    res_msg = {"prediction": pred}
    return res.build(req.event, res_msg)
