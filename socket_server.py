import time
from typing import Dict
import numpy as np
import cv2
import base64
import socket
import json


class SocketServer:
    def __init__(self, host, port, event_handlers):
        self.CHUNK_SIZE = 60000
        self.host = host
        self.port = port
        self.event_handlers = event_handlers
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        print(f"Listening on {self.host} port {self.port}")
        # Initialize the FPS counter
        self.FPS_count = 0
        self.FPS = 0
        self.FPS_previous_time = time.time()
        #
        # Get the current receive buffer size
        # recv_buffer_size = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        # print(f"Current receive buffer size: {recv_buffer_size} bytes")
        # set udp socket buffer size to 2MB
        self.receive_buffer_size = 2 * 1024 * 1024
        # print(f"Setting buffer size to: {self.receive_buffer_size} bytes")
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_size
        )
        recv_buffer_size = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(f"Current receive buffer size: {recv_buffer_size} bytes")
        # self.event_handlers = event_handlers

    def receive_message(self):
        """Receives a message from a UDP socket and returns it as a dictionary and the address of the sender."""
        # Initialize the image data
        message_bytes, addr = self._receive_message_in_chunks()
        # Convert bytes to json
        message_string = message_bytes.decode("utf-8")
        message_obj = json.loads(message_string)
        # Return the image and the address
        return message_obj, addr

    def _receive_message_in_chunks(self):
        message_bytes = b""
        while True:
            chunk, addr = self.socket.recvfrom(self.CHUNK_SIZE)
            message_bytes += chunk
            if len(chunk) < self.CHUNK_SIZE:
                break
        return message_bytes, addr

    def _receive_complete_message(self):
        message_bytes, addr = self.socket.recvfrom(self.receive_buffer_size)
        return message_bytes, addr

    def start(self):

        while True:
            # Receive message from the socket
            message_obj, addr = self.receive_message()
            # Count the FPS
            self._count_FPS()

            event = message_obj["event"]

            # Call the event handler
            handler_res = self.event_handlers.handle_event(event, message_obj)
            # Save the image
            # cv2.imwrite(
            #     os.path.join("frames_test", f"frame_{time.time()}.png"), processed_image
            # )
            # Send a message back to the client

            res_message = self.event_handlers.build_response_message(
                event,
                str(self.FPS),
                handler_res,
            )
            self.respond(res_message, addr)

    # def _send_camera_feed_hand_pose(self, addr):
    #     # Get a frame from the camera feed
    #     frame = self.event_handlers.handle_event("get_feed_frame_handpose", {})
    #     print("inside _send_camera_feed_hand_pose")
    #     if frame is not None:
    #         # Convert the image to bytes
    #         # Send the frame to the client
    #         response = self.build_response_message("get_feed_frame_handpose", frame)
    #         self._count_FPS()
    #         self.respond(response, addr)

    def _count_FPS(self):
        self.FPS_count += 1
        if time.time() - self.FPS_previous_time > 1:
            print(f"FPS: {self.FPS_count}")
            self.FPS_previous_time = time.time()
            self.FPS = self.FPS_count
            self.FPS_count = 0

    def respond(self, response: Dict[str, str], addr):
        try:
            response_str = json.dumps(response)
            response_bytes = response_str.encode("utf-8")
            self._respond_complete_message(response_bytes, addr)
        except Exception as e:
            print(f"Error in socket_server::respond: {e}")
            return

    def _respond_in_chunks(self, response_bytes, addr):
        # split the response into chunks of size self.CHUNK_SIZE
        if len(response_bytes) < self.CHUNK_SIZE:
            self.socket.sendto(response_bytes, addr)
        else:
            for i in range(0, len(response_bytes), self.CHUNK_SIZE):
                chunk = response_bytes[i : i + self.CHUNK_SIZE]
                self.socket.sendto(chunk, addr)
            # Handle the last chunk
            if len(response_bytes) % self.CHUNK_SIZE != 0:
                last_chunk = response_bytes[
                    len(response_bytes) - (len(response_bytes) % self.CHUNK_SIZE) :
                ]
                self.socket.sendto(last_chunk, addr)

    def _respond_complete_message(self, response_bytes, addr):
        try:
            self.socket.sendto(response_bytes, addr)
        except:
            print(f"Error sending response in socket_server::_respond_complete_message")
            return

    def close(self):
        self.socket.close()
        print("Socket closed")
