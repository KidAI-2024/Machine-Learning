from ImageClassifier.image_classifier_cnn import ImageClassifierCNN
from ImageClassifier.image_classifier_classical import ImageClassifierClassical
from ImageClassifier.image_classifier_resnet import ImageClassifierResNet
from .image_classifier_utils import *


class ImageClassifier:

    def __init__(
        self,
        model_category=0,
        classical_model_type=0,
        feature_extraction_type_img=0,
        num_classes=2,
        in_channels=3,
        img_size=32,
    ):
        self.model_category = model_category
        self.classical_model_type = classical_model_type
        self.feature_extraction_type_img = feature_extraction_type_img
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.epochs = 5
        self.max_lr = 0.01
        self.image_classifier = None

    def set_model_category(self):
        if self.model_category == 0:
            print("Classical model")
            self.image_classifier = ImageClassifierClassical(
                img_size=self.img_size,
                num_classes=self.num_classes,
                in_channels=self.in_channels,
                model_type=self.classical_model_type,
                feature_extraction_type=self.feature_extraction_type_img,
            )
        elif self.model_category == 1:
            print("ResNet model")
            self.image_classifier = ImageClassifierResNet(
                img_size=self.img_size,
                num_classes=self.num_classes,
                in_channels=self.in_channels,
            )
        else:  # model_category == 2
            print("CNN model")
            self.image_classifier = ImageClassifierCNN(
                img_size=self.img_size,
                num_classes=self.num_classes,
                in_channels=self.in_channels,
            )

    def create_model(self):
        return self.image_classifier.create_model()

    def read_train_data(self, path, train_precentage=0.8):
        return self.image_classifier.read_train_data(path, train_precentage)

    def get_data_loaders(
        self, train_ds, valid_ds, batch_size=128, num_workers=4, pin_memory=True
    ):
        if self.model_category == 0:
            return None, None
        return self.image_classifier.get_data_loaders(
            train_ds, valid_ds, batch_size, num_workers, pin_memory
        )

    def train(
        self,
        project_path,
        epochs=10,
        max_lr=0.01,
        grad_clip=0.1,
        weight_decay=1e-4,
        opt_func=torch.optim.Adam,
        train_dl=None,
        valid_dl=None,
        decay_lr=False,
    ):
        if self.model_category == 0:
            return self.image_classifier.train(project_path)

        return self.image_classifier.train(
            project_path,
            epochs,
            max_lr,
            grad_clip,
            weight_decay,
            opt_func,
            train_dl,
            valid_dl,
            decay_lr,
        )

    def predict(self, img):
        return self.image_classifier.predict(img)

    def save(self, project_path, model_path):
        if self.model_category == 0:
            return self.image_classifier.save(project_path)
        return self.image_classifier.save(model_path)

    def load(self, path, model_path, img_size=256):
        if self.model_category == 0:
            return self.image_classifier.load(path, img_size)
        return self.image_classifier.load(model_path, img_size)

    def start_feed(self):
        return self.image_classifier.start_feed()

    def stop_feed(self):
        return self.image_classifier.stop_feed()

    def get_frame(self):
        return self.image_classifier.get_frame()

    def b64string_to_tensor(self, frame_bytes, width, height, in_channels=3):
        if self.model_category == 1:
            return ImageClassifierResNet.b64string_to_tensor(
                frame_bytes, width, height, in_channels
            )
        elif self.model_category == 2:
            return ImageClassifierCNN.b64string_to_tensor(
                frame_bytes, width, height, in_channels
            )
