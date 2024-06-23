# from ImageClassifier.image_classifier_cnn import ImageClassifierCNN
from ImageClassifier.image_classifier_resnet import ImageClassifierResNet
from utils import *
from decorators import event
from server_utils import Req, Res

# image_classifier_cnn = ImageClassifierCNN()
image_classifier_resnet = None | ImageClassifierResNet

NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 400

train_ds = None
valid_ds = None


@event("start_feed_image_classifier")
def start_feed_image_classifier(req: Req, res: Res):
    # start camera feed
    status = image_classifier_resnet.start_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("stop_feed_image_classifier")
def stop_feed_image_classifier(req: Req, res: Res):
    # stop camera feed
    status = image_classifier_resnet.stop_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("get_feed_frame_image_classifier")
def get_feed_frame_image_classifier(req: Req, res: Res):
    frame = image_classifier_resnet.get_frame()
    if frame is not None:
        # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
        preprocess_image = image_classifier_resnet.preprocess_draw_landmarks(frame)
        preprocess_image_str = image_to_b64string(preprocess_image)
        res_msg = {"frame": preprocess_image_str}
        return res.build(req.event, res_msg)
    else:
        res_msg = {"frame": None}
        return res.build(req.event, res_msg)


# @event("preprocess_image_classifier")
# def preprocess_image_classifier(req: Req, res: Res):
#     frame_bytes = req.msg["frame"]
#     width_str = req.msg["width"]
#     height_str = req.msg["height"]
#     try:
#         width = int(width_str)
#     except ValueError:
#         width = 320
#         print(f"Invalid width: {width_str}")
#     try:
#         height = int(height_str)
#     except ValueError:
#         height = 180
#         print(f"Invalid height: {height_str}")
#     # Convert the bytes to an image
#     image = b64string_to_image(frame_bytes, (height, width, 3))
#     # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
#     preprocess_image = image_classifier_resnet.preprocess_draw_landmarks(image)
#     preprocess_image_str = image_to_b64string(preprocess_image)
#     res_msg = {"preprocessed_image": preprocess_image_str}
#     return res.build(req.event, res_msg)


@event("start_image_classifier_train")
def train_image_classifier(req: Req, res: Res) -> int:
    path = req.msg["path"]  # the path to the training data
    num_classes = req.msg["num_classes"]  # the number of classes
    epochs = req.msg["epochs"]  # the number of epochs
    max_lr = req.msg["max_lr"]  # the maximum learning rate
    # grad_clip = req.msg["grad_clip"]
    # weight_decay = req.msg["weight_decay"]
    # opt_func = req.msg["opt_func"]
    # TODO: model trial
    image_classifier_resnet = ImageClassifierResNet(num_classes)
    try:
        print("Reading data...")
        # training_data = read_data(path)
        train_ds = image_classifier_resnet.read_and_preprocess_train(path)
        valid_ds = image_classifier_resnet.read_and_preprocess_test(path)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return -1
    print("Training...")
    try:
        # PyTorch data loaders
        train_dl, valid_dl = image_classifier_resnet.get_data_loaders(
            train_ds, valid_ds, BATCH_SIZE, NUM_WORKERS, True
        )
        image_classifier_resnet.train(
            epochs=epochs, max_lr=max_lr, train_dl=train_dl, valid_dl=valid_dl
        )
    except Exception as e:
        print(f"Error in train: {e}")
        return -1
    print("Saving model...")
    project_name = path.split("/")[-1]
    saved_model_name = "image_classifier_model.pkl"
    model_path = os.path.join(
        "..", "Engine", "Projects", project_name, saved_model_name
    )  # Currect directory is Machine-Learning
    image_classifier_resnet.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    res_msg = {"status": "success", "saved_model_name": saved_model_name}
    return res.build(req.event, res_msg)


@event("load_image_classifier_model")
def load_image_classifier_model(req: Req, res: Res) -> int:
    project_name = req.msg["project_name"]
    saved_model_name = req.msg["saved_model_name"]
    model_path = os.path.join(
        "..", "Engine", "Projects", project_name, saved_model_name
    )
    try:
        image_classifier_resnet.load(model_path)
        print(f"Model loaded from {model_path}")
        res_msg = {"status": "success"}
    except Exception as e:
        res_msg = {"status": "Model file not found"}

    return res.build(req.event, res_msg)


@event("predict_image_classifier")
def predict_image_classifier(req: Req, res: Res):
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
    image = b64string_to_image(frame_bytes, (height, width, 3))
    # preprocessed_img = image_classifier_classifier.preprocess(image)
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", preprocessed_img)
    # image_classifier_classifier.load("./image_classifier_model.pkl")
    try:
        pred = image_classifier_resnet.predict(image, train_ds)
    except Exception as e:
        print(f"Error in predict: {e}")
        return -1
    # print(f"Predicted class: {pred}")

    res_msg = {"prediction": pred}
    return res.build(req.event, res_msg)
