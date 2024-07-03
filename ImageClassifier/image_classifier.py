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
        if model_category == 0:
            self.image_classifier = ImageClassifierClassical(
                img_size=img_size,
                num_classes=num_classes,
                in_channels=in_channels,
                model_type=classical_model_type,
                feature_extraction_type=feature_extraction_type_img,
            )
        elif model_category == 1:
            self.image_classifier = ImageClassifierResNet(
                img_size=img_size,
                num_classes=num_classes,
                in_channels=in_channels,
            )
        else:
            self.image_classifier = ImageClassifierCNN(
                img_size=img_size,
                num_classes=num_classes,
                in_channels=in_channels,
            )
