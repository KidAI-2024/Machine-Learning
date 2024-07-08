from ImageClassifier.image_classifier import ImageClassifier

from utils import *
from decorators import event
from server_utils import Req, Res
import time

NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 64

train_ds = None
valid_ds = None
IMG_SIZE = 32

# image_classifier = ImageClassifierCNN(img_size=IMG_SIZE)
# image_classifier = ImageClassifierResNet(img_size=IMG_SIZE)

image_classifier = ImageClassifier(img_size=IMG_SIZE)


@event("start_feed_hand_pose")
def start_feed_hand_pose(req: Req, res: Res):
    # start camera feed
    status = image_classifier.start_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("stop_feed_hand_pose")
def stop_feed_hand_pose(req: Req, res: Res):
    # stop camera feed
    status = image_classifier.stop_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("get_feed_frame_image_classifier")
def get_feed_frame_image_classifier(req: Req, res: Res):
    frame = image_classifier.get_frame()
    if frame is not None:
        # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
        preprocess_image = image_classifier.preprocess_draw_landmarks(frame)
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
#     preprocess_image = image_classifier.preprocess_draw_landmarks(image)
#     preprocess_image_str = image_to_b64string(preprocess_image)
#     res_msg = {"preprocessed_image": preprocess_image_str}
#     return res.build(req.event, res_msg)


@event("start_image_classifier_train")
def train_image_classifier(req: Req, res: Res) -> int:
    path = req.msg["path"]  # the path to the training data
    print(f"Training image classifier with data in {path}")
    num_classes = int(req.msg["num_classes"])  # the number of classes
    print(f"Training image classifier with {num_classes} classes")
    epochs = int(req.msg["epochs"])  # the number of epochs
    print(f"Training image classifier for {epochs} epochs")
    max_lr = float(req.msg["max_lr"])  # the maximum learning rate
    print(f"Training image classifier with max learning rate {max_lr}")
    model_category = int(req.msg["model_category"])  # the model category
    print(f"Training image classifier with model category {model_category}")
    classical_model_type = int(
        req.msg["classical_model_type"]
    )  # the classical model type
    print(f"Training image classifier with classical model type {classical_model_type}")
    feature_extraction_type_img = int(
        req.msg["feature_extraction_type_img"]
    )  # the feature extraction type for images
    print(
        f"Training image classifier with feature extraction type for images {feature_extraction_type_img}"
    )
    image_classifier.num_classes = num_classes
    image_classifier.model_category = model_category
    image_classifier.classical_model_type = classical_model_type
    image_classifier.feature_extraction_type_img = feature_extraction_type_img
    image_classifier.epochs = epochs
    image_classifier.max_lr = max_lr
    try:
        image_classifier.set_model_category()
        print("Reading data...")
        train_ds, valid_ds = image_classifier.read_train_data(path, 0.9)
        print("Creating model...")
        image_classifier.create_model()
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return -1
    try:
        print("Creating data loaders...")
        # PyTorch data loaders
        train_dl, valid_dl = image_classifier.get_data_loaders(
            train_ds, valid_ds, BATCH_SIZE, NUM_WORKERS, True
        )
        print("Training...")
        training_accuracy, valid_accuracy = image_classifier.train(
            path,
            epochs=epochs,
            max_lr=max_lr,
            train_dl=train_dl,
            valid_dl=valid_dl,
        )
        if training_accuracy is not None and valid_accuracy is not None:
            print(f"Training accuracy: {training_accuracy}")
            print(f"Validation accuracy: {valid_accuracy}")
    except Exception as e:
        print(f"Error in train: {e}")
        return -1
    print("Saving model...")
    print("Path:", path)
    project_name = path.split("/")[-1]
    saved_model_name = "image_classifier_model.pkl"
    project_path = os.path.join(path, project_name)
    print("project_path", project_path)
    model_path = os.path.join(path, project_name, saved_model_name)
    image_classifier.save(project_path, model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    res_msg = {
        "status": "success",
        "saved_model_name": saved_model_name,
        # "training_accuracy": training_accuracy,
        "valid_accuracy": valid_accuracy,
    }
    return res.build(req.event, res_msg)


@event("load_image_classifier_model")
def load_image_classifier_model(req: Req, res: Res) -> int:
    path = req.msg["path"]
    saved_model_name = req.msg["saved_model_name"]
    num_classes = int(req.msg["num_classes"])  # the number of classes
    model_category = int(req.msg["model_category"])  # the model category
    classical_model_type = int(req.msg["classical_model_type"])
    feature_extraction_type_img = int(req.msg["feature_extraction_type_img"])
    model_path = os.path.join(path, saved_model_name)
    print(f"Loading model from {model_path}")
    print(f"Number of classes: {num_classes}")
    try:
        image_classifier.model_category = model_category
        image_classifier.classical_model_type = classical_model_type
        image_classifier.feature_extraction_type_img = feature_extraction_type_img
        image_classifier.num_classes = num_classes
        image_classifier.load(path, model_path, img_size=IMG_SIZE)
        print(f"Model loaded from {model_path}")
        res_msg = {"status": "success"}
    except Exception as e:
        print(f"Error in load: {e}")
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
    image = b64string_to_image_float(frame_bytes, (height, width, 3))
    # ignore the image it was all black
    if np.all(image == 0):
        pred = -1
    else:
        print("Predicting...")
        print(f"Image shape: {image.shape}")
        # convert the image to tensor
        img_tensor = torch.tensor(image)
        # print("img shape: ", img_tensor.shape)
        # Convert to shape [3, 320, 180]
        converted_tensor = img_tensor.permute(2, 1, 0)
        # print("converted_tensor shape: ", converted_tensor.shape)
        # transformed_tensor = image_classifier.transform_v(converted_tensor)
        transformed_tensor = torch.nn.functional.interpolate(
            converted_tensor.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            # size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        transformed_tensor = transformed_tensor.squeeze(0)
        try:
            if image_classifier.model_category == 0:
                pred = image_classifier.predict(image)  # classical model
            else:
                pred = image_classifier.predict(
                    transformed_tensor
                )  # deep learning model
        except Exception as e:
            print(f"Error in predict: {e}")
            return -1

    res_msg = {"prediction": str(pred)}
    print(f"Predicted class: {pred}")
    return res.build(req.event, res_msg)
