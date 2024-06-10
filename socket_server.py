import time
from typing import Dict
import socket
import json
from event_handlers import EventHandlers
import threading
import errno


class SocketServer:
    def __init__(
        self,
        host: str,
        port: int,
    ):
        self.CHUNK_SIZE = 60 * 1024
        self.host = host
        self.port = port
        self.event_handlers = EventHandlers(self.respond)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # make the socket non blocking
        self.socket.setblocking(False)

        self.receive_buffer_size = 2 * 1024 * 1024
        self.send_buffer_size = 2 * 1024 * 1024
        # print(f"Setting buffer size to: {self.receive_buffer_size} bytes")
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_size
        )
        recv_buffer_size = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(f"Current receive buffer size: {recv_buffer_size} bytes")
        # set send buffer size to 2MB
        self.socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer_size
        )
        snd_buffer_size = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        print(f"Current send buffer size: {snd_buffer_size} bytes")

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

        # self.event_handlers = event_handlers
        self.send_queue = []
        self.receive_buffer = b""
        # Managing threads
        self.receive_lock = threading.Lock()
        self.send_lock = threading.Lock()
        self.receive_threads = []
        self.send_threads = []

    def start(self):
        # =====================================
        # ===========Single-threaded===========
        # =====================================
        while True:
            # ===============================================
            # ================Receive message================
            # ===============================================
            self.receive_and_handle_message()
            # ===============================================
            # ==================Send message=================
            # ===============================================
            self.send_one_message_from_queue()
        # =====================================
        # ===========Multi-threaded===========
        # =====================================
        # num_threads = 1
        # Start the receiving threads
        # self._start_receive_threads(num_threads)
        # Start the sending threads
        # self._start_send_threads(num_threads)
        # Start receiving
        # self._start_receiving()
        # Start sending
        # self._start_sending()
        # while True:
        #     pass

    # ============================================================================
    # ============================================================================
    # ===================================RECEIVE==================================
    # ============================================================================
    # ============================================================================
    def receive_and_handle_message(self):
        message_obj, addr = self.receive_message()
        if message_obj is not None:
            event = message_obj["event"]
            # Call the event handler
            handler_res = self.event_handlers.handle_event(event, message_obj, addr)
            # Send a message back to the client
            if handler_res is not None:
                handler_res["FPS"] = str(self.FPS)
                self.send_queue.append((handler_res, addr))
                # self.respond(handler_res, addr)
                # Count the FPS
                self._count_FPS()

    def receive_message(self):
        """Receives a message from a UDP socket and returns it as a dictionary and the address of the sender."""
        # Initialize the image data
        message_bytes, addr = self._receive_message_in_chunks()
        # message_bytes, addr = self._receive_complete_message()
        if message_bytes is None:
            return None, None
        # Convert bytes to json
        message_string = message_bytes.decode("utf-8")
        message_obj = json.loads(message_string)
        # Return the image and the address
        return message_obj, addr

    def _receive_message_in_chunks(self):
        try:
            chunk, addr = self.socket.recvfrom(self.CHUNK_SIZE)
        except socket.error as e:
            # If the error is due to the socket being non-blocking, return None
            if e.errno == errno.EWOULDBLOCK:
                return None, None
            else:
                print(f"Error in socket_server::_receive_message_in_chunks: {e}")
                return None, None
        except Exception as e:
            print(f"Error in socket_server::_receive_message_in_chunks: {e}")
            return None, None
        # if the socket received a chunk of data successfully, add it to the receive buffer
        self.receive_buffer += chunk
        # if the length of the chunk is less than the chunk size, then the message is complete
        if len(chunk) < self.CHUNK_SIZE:
            message_bytes = self.receive_buffer
            # reset the receive buffer
            self.receive_buffer = b""
            # return the message and the address
            return message_bytes, addr
        # if the length of the chunk is equal to the chunk size, then the message is incomplete
        return None, None

    def _receive_complete_message(self):
        try:
            message_bytes, addr = self.socket.recvfrom(self.receive_buffer_size)
        except socket.error as e:
            if e.errno == errno.EWOULDBLOCK:
                return None, None
            else:
                print(f"Error in socket_server::_receive_complete_message: {e}")
                return None, None
        return message_bytes, addr

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

    # ============================================================================
    # ============================================================================
    # ===================================RESPOND==================================
    # ============================================================================
    # ============================================================================
    def send_one_message_from_queue(self):
        if len(self.send_queue) > 0:
            message = self.send_queue.pop(0)
            self.respond(message[0], message[1])

    def respond(self, response: Dict[str, str], addr):
        try:
            response_str = json.dumps(response)
            response_bytes = response_str.encode("utf-8")
            self._respond_complete_message(response_bytes, addr)
        except socket.error as e:
            return
        except Exception as e:
            return

    def _respond_in_chunks(self, response_bytes: bytes, addr):
        # split the response into chunks of size self.CHUNK_SIZE
        if len(response_bytes) < self.CHUNK_SIZE:
            try:
                self.socket.sendto(response_bytes, addr)
            except socket.error as e:
                print(
                    f"Error sending response in socket_server::_respond_in_chunks: {e}"
                )
                raise e
            except Exception as e:
                print(
                    f"Error sending response in socket_server::_respond_in_chunks: {e}"
                )
                raise e
        else:
            for i in range(0, len(response_bytes), self.CHUNK_SIZE):
                try:
                    chunk = response_bytes[i : i + self.CHUNK_SIZE]
                    self.socket.sendto(chunk, addr)
                except socket.error as e:
                    print(
                        f"Error sending response in socket_server::_respond_in_chunks: {e}"
                    )
                    raise e
                except Exception as e:
                    print(
                        f"Error sending response in socket_server::_respond_in_chunks: {e}"
                    )
                    raise e

            # Handle the last chunk
            if len(response_bytes) % self.CHUNK_SIZE != 0:
                try:
                    last_chunk = response_bytes[
                        len(response_bytes) - (len(response_bytes) % self.CHUNK_SIZE) :
                    ]
                    self.socket.sendto(last_chunk, addr)
                except socket.error as e:
                    print(
                        f"Error sending response in socket_server::_respond_in_chunks: {e}"
                    )
                    raise e
                except Exception as e:
                    print(
                        f"Error sending response in socket_server::_respond_in_chunks: {e}"
                    )
                    raise e

    def _respond_complete_message(self, response_bytes: bytes, addr):
        try:
            self.socket.sendto(response_bytes, addr)
        except socket.error as e:
            print(
                f"Error sending response in socket_server::_respond_complete_message: {e}"
            )
            raise e
        except Exception as e:
            print(
                f"Error sending response in socket_server::_respond_complete_message: {e}"
            )
            raise e

    # ===========================================================================
    # ===========================================================================
    # ===========================MULTI-THREADING=================================
    # ===========================================================================
    # ===========================================================================
    def _start_sending(self):
        while True:
            with self.send_lock:
                # pop the message from send_queue
                self.send_one_message_from_queue()

    def _start_receiving(self):
        while True:
            # Receive message from the socket
            self.receive_and_handle_message()

    def _start_send_threads(self, num_threads: int):
        for _ in range(num_threads):
            send_thread = threading.Thread(target=self._start_sending)
            self.send_threads.append(send_thread)
            send_thread.start()

    def _start_receive_threads(self, num_threads: int):
        for _ in range(num_threads):
            receive_thread = threading.Thread(target=self._start_receiving)
            self.receive_threads.append(receive_thread)
            receive_thread.start()

    # ============================================================================
    # ============================================================================
    # =================================UTILS======================================
    # ============================================================================
    # ============================================================================
    def _count_FPS(self):
        self.FPS_count += 1
        if time.time() - self.FPS_previous_time > 1:
            print(f"FPS: {self.FPS_count}")
            self.FPS_previous_time = time.time()
            self.FPS = self.FPS_count
            self.FPS_count = 0

    def close(self):
        self.socket.close()
        print("Socket closed")
