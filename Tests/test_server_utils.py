import pytest
from server_utils import Res


class TestServerUtils:
    @pytest.fixture
    def res(self):
        return Res()

    def test_build_response_message(self, res):
        event = "test_event"
        message = {"key": "value"}
        expected_response = {"event": event, "key": "value"}
        assert res._build_response_message(event, message) == expected_response

    def test_send(self, res, mocker):
        event = "test_event"
        message = {"key": "value"}
        expected_response = {"event": event, "key": "value"}
        res._respond_func = mocker.Mock()
        res._send(event, message)
        # Assert that the response message is built correctly and sent
        assert res._respond_func.call_args[0][0] == expected_response
        assert res._respond_func.call_args[0][1] == res.addr
