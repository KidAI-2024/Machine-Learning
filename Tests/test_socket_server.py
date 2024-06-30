import pytest
from socket_server import SocketServer
import socket
import errno
import json


@pytest.fixture
def socket_server_mock(mocker):
    mock_addr = ("localhost", 0)  # Use 0 to bind to a free port dynamically
    server = SocketServer(mock_addr[0], mock_addr[1])
    yield server
    server.socket.close()


class TestSockerServer:
    # ============================================
    # ====Testing the `receive_message` method====
    # ============================================
    def test_receive_message_success(self, mocker, socket_server_mock):
        # Mocking the `_receive_message_in_chunks` method
        mock_message = json.dumps({"key": "value"}).encode("utf-8")
        mock_addr = ("localhost", 12345)
        mocker.patch.object(
            socket_server_mock,
            "_receive_message_in_chunks",
            return_value=(mock_message, mock_addr),
        )

        result, addr = socket_server_mock.receive_message()

        # Assertions
        assert result == {"key": "value"}
        assert addr == mock_addr

    def test_receive_message_no_data(self, mocker, socket_server_mock):
        # Mocking the `_receive_message_in_chunks` method to return None
        mocker.patch.object(
            socket_server_mock, "_receive_message_in_chunks", return_value=(None, None)
        )
        mock_addr = ("localhost", 12345)

        # Creating an instance of the target class

        # Calling the method
        result, addr = socket_server_mock.receive_message()

        # Assertions
        assert result is None
        assert addr is None

    # ======================================================
    # ====Testing the `_receive_complete_message` method====
    # ======================================================
    def test_receive_complete_message_success(self, mocker, socket_server_mock):
        mock_message = b"test message"
        mock_addr = ("localhost", 12345)
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.return_value = (mock_message, mock_addr)

        result, addr = socket_server_mock._receive_complete_message()

        assert result == mock_message
        assert addr == mock_addr

    def test_receive_complete_message_ewouldblock(self, mocker, socket_server_mock):
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.side_effect = socket.error(
            errno.EWOULDBLOCK, "Would block"
        )

        result, addr = socket_server_mock._receive_complete_message()

        assert result is None
        assert addr is None

    def test_receive_complete_message_other_error(self, mocker, socket_server_mock):
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.side_effect = socket.error(
            errno.ECONNRESET, "Connection reset by peer"
        )

        result, addr = socket_server_mock._receive_complete_message()

        assert result is None
        assert addr is None

    # =======================================================
    # ====Testing the `_receive_message_in_chunks` method====
    # =======================================================
    def test_receive_message_in_chunks_success(self, mocker, socket_server_mock):
        mock_chunk = b"test chunk"
        mock_addr = ("localhost", 12345)
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.return_value = (mock_chunk, mock_addr)

        result, addr = socket_server_mock._receive_message_in_chunks()

        assert result == mock_chunk
        assert addr == mock_addr
        assert socket_server_mock.receive_buffer == b""

    def test_receive_message_in_chunks_ewouldblock(self, mocker, socket_server_mock):
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.side_effect = socket.error(
            errno.EWOULDBLOCK, "Would block"
        )

        result, addr = socket_server_mock._receive_message_in_chunks()

        assert result is None
        assert addr is None

    def test_receive_message_in_chunks_other_error(self, mocker, socket_server_mock):
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.side_effect = socket.error(
            errno.ECONNRESET, "Connection reset by peer"
        )

        result, addr = socket_server_mock._receive_message_in_chunks()

        assert result is None
        assert addr is None

    def test_receive_message_in_chunks_generic_exception(
        self, mocker, socket_server_mock
    ):
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.side_effect = Exception("Generic exception")

        result, addr = socket_server_mock._receive_message_in_chunks()

        assert result is None
        assert addr is None

    def test_receive_message_in_chunks_incomplete(self, mocker, socket_server_mock):
        mock_addr = ("localhost", 12345)
        mock_chunk = b"x" * socket_server_mock.CHUNK_SIZE
        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.recvfrom.return_value = (mock_chunk, mock_addr)

        result, addr = socket_server_mock._receive_message_in_chunks()

        assert result is None
        assert addr is None
        assert socket_server_mock.receive_buffer == mock_chunk

    # =======================================================
    # ====Testing the `send_one_message_from_queue` method====
    # =======================================================
    def test_send_one_message_from_queue(self, mocker, socket_server_mock):
        mock_response = {"key": "value"}
        mock_addr = ("localhost", 12345)
        socket_server_mock.send_queue.append((mock_response, mock_addr))

        mock_respond = mocker.patch.object(socket_server_mock, "respond")

        socket_server_mock.send_one_message_from_queue()

        assert len(socket_server_mock.send_queue) == 0
        mock_respond.assert_called_once_with(mock_response, mock_addr)

    # =======================================================
    # ============Testing the `respond` method===============
    # =======================================================
    def test_respond_success(self, mocker, socket_server_mock):
        mock_response = {"key": "value"}
        mock_addr = ("localhost", 12345)
        mock_response_bytes = json.dumps(mock_response).encode("utf-8")

        mock_respond_complete_message = mocker.patch.object(
            socket_server_mock, "_respond_complete_message"
        )

        socket_server_mock.respond(mock_response, mock_addr)

        mock_respond_complete_message.assert_called_once_with(
            mock_response_bytes, mock_addr
        )

    def test_respond_socket_error(self, mocker, socket_server_mock):
        mock_response = {"key": "value"}
        mock_addr = ("localhost", 12345)

        mocker.patch.object(
            socket_server_mock, "_respond_complete_message", side_effect=socket.error
        )

        try:
            socket_server_mock.respond(mock_response, mock_addr)
        except socket.error:
            pytest.fail("Socket error was not handled")

    def test_respond_generic_exception(self, mocker, socket_server_mock):
        mock_response = {"key": "value"}
        mock_addr = ("localhost", 12345)

        mocker.patch.object(
            socket_server_mock, "_respond_complete_message", side_effect=Exception
        )

        try:
            socket_server_mock.respond(mock_response, mock_addr)
        except Exception:
            pytest.fail("Generic exception was not handled")

    # =======================================================
    # ========Testing the `_respond_in_chunks` method========
    # =======================================================

    def test_respond_in_chunks_single_chunk(self, mocker, socket_server_mock):
        mock_response_bytes = b"x" * (socket_server_mock.CHUNK_SIZE - 1)
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")

        socket_server_mock._respond_in_chunks(mock_response_bytes, mock_addr)

        mock_socket.sendto.assert_called_once_with(mock_response_bytes, mock_addr)

    def test_respond_in_chunks_multiple_chunks(self, mocker, socket_server_mock):
        mock_response_bytes = b"x" * (socket_server_mock.CHUNK_SIZE * 3)
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")

        socket_server_mock._respond_in_chunks(mock_response_bytes, mock_addr)

        assert mock_socket.sendto.call_count == 3

    def test_respond_in_chunks_socket_error(self, mocker, socket_server_mock):
        mock_response_bytes = b"x" * (socket_server_mock.CHUNK_SIZE - 1)
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.sendto.side_effect = socket.error

        with pytest.raises(socket.error):
            socket_server_mock._respond_in_chunks(mock_response_bytes, mock_addr)

    def test_respond_in_chunks_generic_exception(self, mocker, socket_server_mock):
        mock_response_bytes = b"x" * (socket_server_mock.CHUNK_SIZE - 1)
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.sendto.side_effect = Exception

        with pytest.raises(Exception):
            socket_server_mock._respond_in_chunks(mock_response_bytes, mock_addr)

    # ==============================================================
    # ========Testing the `_respond_complete_message` method========
    # ==============================================================

    def test_respond_complete_message_success(self, mocker, socket_server_mock):
        mock_response_bytes = b"test response"
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")

        socket_server_mock._respond_complete_message(mock_response_bytes, mock_addr)

        mock_socket.sendto.assert_called_once_with(mock_response_bytes, mock_addr)

    def test_respond_complete_message_socket_error(self, mocker, socket_server_mock):
        mock_response_bytes = b"test response"
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.sendto.side_effect = socket.error

        with pytest.raises(socket.error):
            socket_server_mock._respond_complete_message(mock_response_bytes, mock_addr)

    def test_respond_complete_message_generic_exception(
        self, mocker, socket_server_mock
    ):
        mock_response_bytes = b"test response"
        mock_addr = ("localhost", 12345)

        mock_socket = mocker.patch.object(socket_server_mock, "socket")
        mock_socket.sendto.side_effect = Exception

        with pytest.raises(Exception):
            socket_server_mock._respond_complete_message(mock_response_bytes, mock_addr)

    # ==============================================================
    # ================Testing the `_count_FPS` method===============
    # ==============================================================

    def test_count_fps_increment(self, mocker, socket_server_mock):
        obj = socket_server_mock
        mocker.patch("time.time", return_value=0)  # Mock time to simulate initial time
        obj.FPS_previous_time = 0  # Set initial FPS_previous_time to 0

        # Call _count_FPS once
        obj._count_FPS()

        # Assert that FPS_count is incremented to 1
        assert obj.FPS_count == 1

    def test_count_fps_update(self, mocker, socket_server_mock):
        obj = socket_server_mock
        mocker.patch(
            "time.time", return_value=2
        )  # Mock time to simulate time greater than 1 second
        obj.FPS_previous_time = 0  # Set initial FPS_previous_time to 0
        obj.FPS_count = 5  # Set FPS_count to 5

        # Call _count_FPS once
        obj._count_FPS()

        # Assert that FPS_count is reset to 0 and FPS is updated to 6
        assert obj.FPS_count == 0
        assert obj.FPS == 6

    def test_count_fps_no_update(self, mocker, socket_server_mock):
        obj = socket_server_mock
        mocker.patch("time.time", return_value=0)  # Mock time to simulate initial time
        obj.FPS_previous_time = 0  # Set initial FPS_previous_time to 0
        obj.FPS_count = 5  # Set FPS_count to 5

        # Call _count_FPS once without exceeding 1 second
        obj._count_FPS()

        # Assert that FPS_count is incremented to 6 and FPS is unchanged
        assert obj.FPS_count == 6
        assert obj.FPS == 0  # FPS should remain 0 until time exceeds 1 second

    def test_close_method(self, mocker, socket_server_mock):
        # Mock the socket object
        mock_socket = mocker.patch.object(socket, "socket")

        socket_server_mock.socket = mock_socket

        # Patch the close method of the socket object
        mocker.patch.object(mock_socket, "close")

        # Call the close method of SocketServer
        socket_server_mock.close()

        # Assert that SocketServer calls close() on the socket object
        mock_socket.close.assert_called_once()
