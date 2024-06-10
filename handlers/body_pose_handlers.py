from BodyPoseClassifier.body_pose_classifier import BodyPoseClassifier
import utils
from decorators import event
from server_utils import Req, Res


body_pose_classifier = BodyPoseClassifier()
@event("start_feed_body_pose")
def start_feed_body_pose(req: Req, res: Res):
    # start camera feed
    status = body_pose_classifier.start_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("stop_feed_body_pose")
def stop_feed_body_pose(req: Req, res: Res):
    # stop camera feed
    status = body_pose_classifier.stop_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("get_feed_frame_bodypose")
def get_feed_frame_body_pose(req: Req, res: Res):
    frame = body_pose_classifier.get_frame()
    if frame is not None:
        # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
        preprocess_image = body_pose_classifier.preprocess_draw_landmarks(frame)
        preprocess_image_str = utils.image_to_b64string(preprocess_image)
        res_msg = {"frame": preprocess_image_str}
        return res.build(req.event, res_msg)
    else:
        res_msg = {"frame": None}
        return res.build(req.event, res_msg)


@event("preprocess_body_pose")
def preprocess_body_pose(req: Req, res: Res):
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
    print(f"frame: {frame_bytes}")
    # Convert the bytes to an image
    # image = utils.b64string_to_image(frame_bytes, (height, width, 3))
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
    # preprocess_image = body_pose_classifier.preprocess_draw_landmarks(image)
    # preprocess_image_str = utils.image_to_b64string(preprocess_image)
    res_msg = {"preprocessed_image": "gamed yala"}
    return res.build(req.event, res_msg)


@event("train_body_pose")
def train_body_pose(req: Req, res: Res) -> int:
    path = req.msg["path"]
    # training_data is map {"Class Number(first character in the folder name)" : [images]}
    print("Reading data...")
    training_data = utils.read_data(path)
    print("Extracting features...")
    try:
        features_map = body_pose_classifier.preprocess(training_data)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return -1
    print("Training...")
    try:
        body_pose_classifier.train(features_map)
    except Exception as e:
        print(f"Error in train: {e}")
        return -1
    print("Saving model...")
    model_path = "./body_pose_model.pkl"
    body_pose_classifier.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    return None

@event("predict_body_pose")
def predict_body_pose(req: Req, res: Res):
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
    # preprocessed_img = body_pose_classifier.preprocess(image)
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", preprocessed_img)
    # body_pose_classifier.load("./body_pose_model.pkl")
    try:
        pred = body_pose_classifier.predict(image)
    except Exception as e:
        print(f"Error in predict: {e}")
        return -1
    print(f"Predicted class: {pred}")

    res_msg = {"prediction": pred}
    return res.build(req.event, res_msg)