import socketio
import eventlet
import numpy as np
import cv2

sio = socketio.Server()


@sio.on("connect")
def connect(sid, environ):
    print("A user connected")


@sio.on("frame")
def frame(sid, data):
    # print(f"Frame received, {data}")
    # Convert byte array back to color array
    # print(data)
    height = data["height"]
    width = data["width"]
    frame = data["frame"]
    color_array = np.frombuffer(frame, dtype=np.uint8)

    # Reshape the color array back to an image
    # You need to replace width and height with the actual values
    img = color_array.reshape((height, width, 3))

    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Flip the image vertically
    img = cv2.flip(img, 0)

    # Save the image locally
    cv2.imwrite("frame.png", img)

    # sio.emit("result", [])
    sio.emit("result", "Frame Processed Successfully!")


@sio.on("disconnect")
def disconnect(sid):
    print("User disconnected")


if __name__ == "__main__":
    app = socketio.WSGIApp(sio)
    port = 5000
    eventlet.wsgi.server(eventlet.listen(("localhost", port)), app)
