# from ImageClassifier.image_classifier_cnn import ImageClassifierCNN
from ImageClassifier.image_classifier_resnet import ImageClassifierResNet
from utils import *
from decorators import event
from server_utils import Req, Res
import time

NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 32

train_ds = None
valid_ds = None
IMG_SIZE = 64

# image_classifier_cnn = ImageClassifierCNN()
image_classifier_resnet = ImageClassifierResNet(img_size=IMG_SIZE)


@event("start_feed_hand_pose")
def start_feed_hand_pose(req: Req, res: Res):
    # start camera feed
    status = image_classifier_resnet.start_feed()
    res_msg = {"message": "success" if status == 0 else "failed"}
    return res.build(req.event, res_msg)


@event("stop_feed_hand_pose")
def stop_feed_hand_pose(req: Req, res: Res):
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
    print(f"Training image classifier with data in {path}")
    num_classes = int(req.msg["num_classes"])  # the number of classes
    print(f"Training image classifier with {num_classes} classes")
    epochs = int(req.msg["epochs"])  # the number of epochs
    print(f"Training image classifier for {epochs} epochs")
    max_lr = float(req.msg["max_lr"])  # the maximum learning rate
    print(f"Training image classifier with max learning rate {max_lr}")
    # grad_clip = req.msg["grad_clip"]
    # weight_decay = req.msg["weight_decay"]
    # opt_func = req.msg["opt_func"]
    image_classifier_resnet.num_classes = num_classes
    print(
        f"Training image classifier with {image_classifier_resnet.num_classes} classes"
    )
    try:
        print("Creating model...")
        image_classifier_resnet.create_model(img_size=IMG_SIZE)
        print("Reading data...")
        # training_data = read_data(path)
        train_ds = image_classifier_resnet.read_and_preprocess_train(path)
        # valid_ds = image_classifier_resnet.read_and_preprocess_test(path)
    except Exception as e:
        print(f"Error in preprocess: {e}")
        return -1
    try:
        print("Creating data loaders...")
        # PyTorch data loaders
        train_dl, valid_dl = image_classifier_resnet.get_data_loaders(
            train_ds, valid_ds, BATCH_SIZE, NUM_WORKERS, True
        )
        print("Training...")
        image_classifier_resnet.train(
            path,
            epochs=epochs,
            max_lr=max_lr,
            train_dl=train_dl,
            valid_dl=valid_dl,
        )
    except Exception as e:
        print(f"Error in train: {e}")
        return -1
    print("Saving model...")
    project_name = path.split("/")[-1]
    saved_model_name = "image_classifier_model.pkl"
    model_path = os.path.join(path, project_name, saved_model_name)  # Currect directory is Machine-Learning
    image_classifier_resnet.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
    res_msg = {"status": "success", "saved_model_name": saved_model_name}
    return res.build(req.event, res_msg)


@event("load_image_classifier_model")
def load_image_classifier_model(req: Req, res: Res) -> int:
    path = req.msg["path"]
    saved_model_name = req.msg["saved_model_name"]
    num_classes = req.msg["num_classes"]
    model_path = os.path.join(path, saved_model_name)
    # project_path = os.path.join("..", "Engine", "Projects", project_name)
    print(f"Loading model from {model_path}")
    print(f"Number of classes: {num_classes}")
    try:
        image_classifier_resnet.num_classes = int(num_classes)
        image_classifier_resnet.load(model_path, img_size=IMG_SIZE)
        print(f"Model loaded from {model_path}")
        # train_ds = image_classifier_resnet.read_and_preprocess_train(project_path)
        print("Creating data loaders...")
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
        # print("Image is all black")
        pred = -1
    else:
        # convert the image to tensor
        img_tensor = torch.tensor(image)
        # print("img shape: ", img_tensor.shape)
        # Convert to shape [3, 320, 180]
        converted_tensor = img_tensor.permute(2, 1, 0)
        # print("converted_tensor shape: ", converted_tensor.shape)
        # transformed_tensor = image_classifier_resnet.transform_v(converted_tensor)
        transformed_tensor = torch.nn.functional.interpolate(
            converted_tensor.unsqueeze(0),
            size=((IMG_SIZE), IMG_SIZE),
            # size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )
        transformed_tensor = transformed_tensor.squeeze(0)
        # print("transformed_tensor shape: ", transformed_tensor.shape)
        # img_tensor = ImageClassifierResNet.b64string_to_tensor(
        #     frame_bytes, height, width, 3
        # )
        # image = b64string_to_image_float(frame_bytes, (height, width, 3))
        # cv2.imwrite(f"./frame_{time.time()}.png", image)
        try:
            pred = image_classifier_resnet.predict(transformed_tensor)
        except Exception as e:
            print(f"Error in predict: {e}")
            return -1
        print(f"Predicted class: {pred}")

    res_msg = {"prediction": pred}
    return res.build(req.event, res_msg)
