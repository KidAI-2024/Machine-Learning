import time
import socketio
import eventlet
import numpy as np
import cv2
import base64

sio = socketio.Server()
# Calculate frame rate
# frame_count = 0
# start_time = time.time()


@sio.on("connect")
def connect(sid, environ):
    print("A user connected")


@sio.on("frame")
def frame(sid, data):
    # global frame_count, start_time

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

    print("Frame processed successfully!")

    # frame_count += 1
    # elapsed_time = time.time() - start_time

    # if elapsed_time > 0:  # Avoid division by zero
    #     frame_rate = frame_count / elapsed_time
    #     print(f"Frame rate: {frame_rate} frames per second")

    # # Reset the frame count every second to get a running frame rate
    # if elapsed_time > 1.0:
    #     frame_count = 0
    #     start_time = time.time()

    # sio.emit("result", [])
    sio.emit("result", f"{time.time()} Frame Processed Successfully!")


@sio.on("disconnect")
def disconnect(sid):
    print("User disconnected")


if __name__ == "__main__":
    app = socketio.WSGIApp(sio)
    port = 5000
    eventlet.wsgi.server(eventlet.listen(("localhost", port)), app)
