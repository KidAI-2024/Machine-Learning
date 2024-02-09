from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
import time
import numpy as np
import cv2
import base64

app = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")


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


@socketio.on("frame")
def frame(data):
    height = data["height"]
    width = data["width"]
    frame = data["frame"]
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

    # Save the image locally, name it with timestamp
    # cv2.imwrite(f"./frames_test/frame_{time.time()}.png", img)

    print(f"Frame processed successfully! {time.time()}")
    emit("result", f"{time.time()} Frame Processed Successfully!")


@socketio.on("disconnect")
def disconnect():
    print("User disconnected")


if __name__ == "__main__":
    socketio.run(app, host="localhost", port=5000)
