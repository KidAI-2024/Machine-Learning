import time
import numpy as np
import cv2
import base64
import time
import socket
import logging
import argparse
import os
import json

from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
from ImageClassifier.image_classifier import ImageClassifier
from BodyPoseClassifier.body_pose_classifier import BodyPoseClassifier
from AudioClassifier.audio_classifier import AudioClassifier


# Model instance
hand_pose_classifier = HandPoseClassifier()
image_classifier = ImageClassifier()
body_pose_classifier = BodyPoseClassifier()
audio_classifier = AudioClassifier()


# Constants
CHUNK_SIZE = 60000
# IMAGE_SHAPE = (180, 320, 3)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Argument parsing
parser = argparse.ArgumentParser(
    description="A UDP Socket Server that receives and processes images from Unity Client"
)
parser.add_argument("--port", type=int, default=5065, help="The port to listen on")
args = parser.parse_args()


# ---- Event handlers ----
def predict_frame(image):
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", image)
    pass


# Map event names to handlers
EVENTS = {"predict_frame": predict_frame}


#  ---- Socket functions ----
def receive_message(sock):
    """Receives a message from a UDP socket and returns it as a dictionary and the address of the sender."""
    # Initialize the image data
    message_bytes = b""
    # Receive the chunks until the last one
    while True:
        chunk, addr = sock.recvfrom(CHUNK_SIZE)
        message_bytes += chunk
        if len(chunk) < CHUNK_SIZE:
            break
    # Convert bytes to json
    message_string = message_bytes.decode("utf-8")
    message_obj = json.loads(message_string)
    # Return the image and the address
    return message_obj, addr


def bytes_to_image(frame_bytes, shape):
    # Get the image data
    image_data = base64.b64decode(frame_bytes)
    color_array = np.frombuffer(image_data, dtype=np.uint8)
    image = color_array.reshape(shape)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flip the image vertically
    image = cv2.flip(image, 0)
    return image


def main():
    # Create a UDP socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # Bind the socket to the port
        sock.bind(("localhost", args.port))
        logging.info(f"Listening on port {args.port}")
        # Initialize the FPS counter
        frame_count = 0
        previous_time = time.time()
        while True:
            # Receive an image from the socket
            message_obj, addr = receive_message(sock)

            # Get the frame bytes
            frame_bytes = message_obj["frame"]
            width = int(message_obj["width"])
            height = int(message_obj["height"])
            event = message_obj["event"]

            # Convert the bytes to an image
            image = bytes_to_image(frame_bytes, (height, width, 3))

            # Call the event handler
            if event in EVENTS:
                EVENTS[event](image)
            else:
                logging.error(f"Event '{event}' not found")

            # Count the FPS
            frame_count += 1
            if time.time() - previous_time > 1:
                logging.info(f"FPS: {frame_count}")
                previous_time = time.time()
                frame_count = 0

            # Send the processed image back to the client
            # -- Not tested yet --
            # _, jpeg = cv2.imencode('.jpg', processed_image)
            # sock.sendto(jpeg.tobytes(), addr)

            # Save the image
            # cv2.imwrite(
            #     os.path.join("frames_test", f"frame_{time.time()}.png"), processed_image
            # )
            # Send a message back to the client
            sock.sendto(b"Image received", addr)


if __name__ == "__main__":
    main()
