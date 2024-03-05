import time
import numpy as np
import cv2
import base64
import socket
import logging
import json


class SocketServer:
    def __init__(self, host, port, event_handlers):
        self.CHUNK_SIZE = 60000
        self.host = host
        self.port = port
        self.event_handlers = event_handlers
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        logging.info(f"Listening on {self.host} port {self.port}")
        # self.event_handlers = event_handlers

    def receive_message(self):
        """Receives a message from a UDP socket and returns it as a dictionary and the address of the sender."""
        # Initialize the image data
        message_bytes = b""
        # Receive the chunks until the last one
        while True:
            chunk, addr = self.socket.recvfrom(self.CHUNK_SIZE)
            message_bytes += chunk
            if len(chunk) < self.CHUNK_SIZE:
                break
        # Convert bytes to json
        message_string = message_bytes.decode("utf-8")
        message_obj = json.loads(message_string)
        # Return the image and the address
        return message_obj, addr

    def start(self):
        # Initialize the FPS counter
        frame_count = 0
        previous_time = time.time()
        while True:
            # Receive message from the socket
            message_obj, addr = self.receive_message()

            event = message_obj["event"]

            # Call the event handler
            res = self.event_handlers.handle_event(event, message_obj)

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
            binary = f'{res}'.encode('utf-8')
            self.socket.sendto(binary, addr)

    def trainBodyPose(self):
        self.event_handlers.start_body_pose_train("Projects/test")
        # image = cv2.imread("../Engine/Projects/test/0_t-pose/0_t-pose_4.png")
        # res = self.event_handlers.predict_frame(image)