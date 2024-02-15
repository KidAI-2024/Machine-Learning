from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
import time
import numpy as np
import cv2
import base64

from HandPoseClassifier.hand_pose_classifier import HandPoseClassifier
from ImageClassifier.image_classifier import ImageClassifier
from BodyPoseClassifier.body_pose_classifier import BodyPoseClassifier
from AudioClassifier.audio_classifier import AudioClassifier

# init the Flask app and the SocketIO
app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")
# Model instance
hand_pose_classifier = HandPoseClassifier()
image_classifier = ImageClassifier()
body_pose_classifier = BodyPoseClassifier()
audio_classifier = AudioClassifier()


# ----------------- SocketIO Events -----------------
# ------ Connection Events ------
@socketio.on("connect")
def connect():
    # Get the token from the handshake query
    token = request.args.get("token")

    # Check if the token is valid
    if token != "UNITY":
        # If the token is not valid, disconnect the client
        disconnect()
    else:
        print("A user connected")


@socketio.on("disconnect")
def disconnect():
    print("User disconnected")


# ------ Training Events ------
@socketio.on("hand_pose_train")
def hand_pose_train(data):
    print(f"Training successful! {time.time()}")
    emit("hand_pose_train_result", f"{time.time()} Training successful!")


@socketio.on("body_pose_train")
def body_pose_train(data):
    print(f"Training successful! {time.time()}")
    emit("body_pose_train_result", f"{time.time()} Training successful!")


@socketio.on("image_train")
def image_train(data):
    print(f"Training successful! {time.time()}")
    emit("image_train_result", f"{time.time()} Training successful!")


@socketio.on("audio_train")
def audio_train(data):
    print(f"Training successful! {time.time()}")
    emit("audio_train_result", f"{time.time()} Training successful!")


# ------ Prediction Events ------
@socketio.on("hand_pose_frame")
def hand_pose_frame(data):
    height = data["height"]
    width = data["width"]
    frame = data["frame"]
    frame = restore_frame(frame, width, height)

    preprocessed_frame = hand_pose_classifier.preprocess(frame)
    prediction = hand_pose_classifier.predict(preprocessed_frame)
    # Save the image locally, name it with timestamp
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", img)

    print(f"Frame processed successfully! {time.time()}")
    emit("hand_pose_result", f"{time.time()} Frame Processed Successfully!")


@socketio.on("body_pose_frame")
def body_pose_frame(data):
    height = data["height"]
    width = data["width"]
    frame = data["frame"]
    frame = restore_frame(frame, width, height)

    preprocessed_frame = body_pose_classifier.preprocess(frame)
    prediction = body_pose_classifier.predict(preprocessed_frame)
    # Save the image locally, name it with timestamp
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", img)

    print(f"Frame processed successfully! {time.time()}")
    emit("body_pose_result", f"{time.time()} Frame Processed Successfully!")


@socketio.on("image_frame")
def image_frame(data):
    height = data["height"]
    width = data["width"]
    frame = data["frame"]
    frame = restore_frame(frame, width, height)

    preprocessed_frame = image_classifier.preprocess(frame)
    prediction = image_classifier.predict(preprocessed_frame)
    # Save the image locally, name it with timestamp
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", img)

    print(f"Frame processed successfully! {time.time()}")
    emit("image_result", f"{time.time()} Frame Processed Successfully!")


@socketio.on("audio_frame")
def audio_frame(data):
    height = data["height"]
    width = data["width"]
    frame = data["frame"]
    frame = restore_frame(frame, width, height)

    preprocessed_frame = audio_classifier.preprocess(frame)
    prediction = audio_classifier.predict(preprocessed_frame)
    # Save the image locally, name it with timestamp
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", img)

    print(f"Frame processed successfully! {time.time()}")
    emit("audio_result", f"{time.time()} Frame Processed Successfully!")


# ----------------- Utility Functions -----------------
def restore_frame(frame, width, height):
    # Decode the Base64 string back to a byte array
    frame = base64.b64decode(frame)
    color_array = np.frombuffer(frame, dtype=np.uint8)

    # Reshape the color array back to an image
    # You need to replace width and height with the actual values
    img = color_array.reshape((height, width, 3))

    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flip the image vertically
    img = cv2.flip(img, 0)
    return img


if __name__ == "__main__":
    socketio.run(app, host="localhost", port=5000)
