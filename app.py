import argparse

from socket_server import SocketServer
from event_handlers import EventHandlers

# Argument parsing
parser = argparse.ArgumentParser(
    description="A UDP Socket Server that receives and processes messages from Unity Client"
)
parser.add_argument("--port", type=int, default=5065, help="The port to listen on")
args = parser.parse_args()


if __name__ == "__main__":
    print("Starting the Socket Server")
    socket_server = SocketServer("localhost", args.port)
    socket_server.start()
