import pytest
import socket
from server_utils import Req, Res
from decorators import event
from socket_server import SocketServer
import time
import threading


@event("event1")
def event1(req: Req, res: Res):
    return {"result": "success"}


class TestIntegrationSocketserverHandlers:
    def test_handle_received_event(self, mocker):
        # Start the server in a separate thread
        addr = ("localhost", 12345)
        socket_server = SocketServer(addr[0], addr[1])

        server_thread = threading.Thread(
            target=self.start_socket_server, args=(socket_server,)
        )
        server_thread.start()

        # Create a socket client
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Send a message with event1
        client_socket.sendto(b'{"event":"event1"}', addr)

        # Receive the response
        time.sleep(1)  # Add a delay to ensure the response is received

        response, _ = client_socket.recvfrom(60 * 1024)

        # Assert that it receives result success
        assert response.decode() == '{"result": "success", "FPS": "0"}'

        client_socket.close()
        # Stop the thread immediately
        socket_server.close()
        print("Stopping the server")
        server_thread.join(0)

    def start_socket_server(self, socket_server):
        socket_server.start()
