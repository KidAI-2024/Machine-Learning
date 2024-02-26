import time
import numpy as np
import cv2
import base64

from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
from ImageClassifier.image_classifier import ImageClassifier
from BodyPoseClassifier.body_pose_classifier import BodyPoseClassifier
from AudioClassifier.audio_classifier import AudioClassifier


# Model instance
hand_pose_classifier = HandPoseClassifier()
image_classifier = ImageClassifier()
body_pose_classifier = BodyPoseClassifier()
audio_classifier = AudioClassifier()
