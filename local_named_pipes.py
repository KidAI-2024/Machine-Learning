import win32pipe
import win32file
import cv2
import numpy as np


def process_frame(frame_bytes):
    # Decode the image bytes
    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Process the frame (e.g., apply computer vision algorithms)
    # ...

    # Encode the processed frame back to bytes
    processed_bytes = cv2.imencode(".png", frame)[1].tobytes()
    return processed_bytes


def pipe_server():
    pipe_name = r"\\.\pipe\my_named_pipe"  # Same pipe name as in Unity
    pipe = win32pipe.CreateNamedPipe(
        pipe_name,
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT,
        1,
        65536,
        65536,
        0,
        None,
    )

    try:
        print("Waiting for client connection...")
        win32pipe.ConnectNamedPipe(pipe, None)
        print("Client connected.")

        while True:
            frame_bytes = win32file.ReadFile(pipe, 64 * 1024)[1]
            processed_bytes = process_frame(frame_bytes)

            # Send the processed frame back to Unity
            win32file.WriteFile(pipe, processed_bytes)

    finally:
        win32file.CloseHandle(pipe)


if __name__ == "__main__":
    pipe_server()
