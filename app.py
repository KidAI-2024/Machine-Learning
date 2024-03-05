import logging
import argparse

from socket_server import SocketServer
from event_handlers import EventHandlers

from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
from ImageClassifier.image_classifier import ImageClassifier
from BodyPoseClassifier.body_pose_classifier import BodyPoseClassifier
from AudioClassifier.audio_classifier import AudioClassifier

# Model instance
hand_pose_classifier = HandPoseClassifier()
image_classifier = ImageClassifier()
body_pose_classifier = BodyPoseClassifier()
audio_classifier = AudioClassifier()

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Argument parsing
parser = argparse.ArgumentParser(
    description="A UDP Socket Server that receives and processes images from Unity Client"
)
parser.add_argument("--port", type=int, default = 5065, help="The port to listen on")
args = parser.parse_args()


if __name__ == "__main__":
    event_handlers = EventHandlers(
        hand_pose_classifier, image_classifier, body_pose_classifier, audio_classifier
    )
    socket_server = SocketServer("localhost", args.port, event_handlers)
    # socket_server.trainBodyPose()
    socket_server.start()
