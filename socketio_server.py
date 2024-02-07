import socketio
import eventlet

sio = socketio.Server()


@sio.on("connect")
def connect(sid, environ):
    print("A user connected")


@sio.on("frame")
def frame(sid, data):
    print(f"Frame received, {data}")
    sio.emit("result", "Frame Processed Successfully!")


@sio.on("disconnect")
def disconnect(sid):
    print("User disconnected")


if __name__ == "__main__":
    app = socketio.WSGIApp(sio)
    port = 5000
    eventlet.wsgi.server(eventlet.listen(("localhost", port)), app)
