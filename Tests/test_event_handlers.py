import pytest
from server_utils import Req, Res
from decorators import event


@event("event1")
def event1(req: Req, res: Res):
    return {"result": "success"}


@event("event2")
def event2(req: Req, res: Res):
    return {"result": "failure"}


from event_handlers import EventHandlers


@pytest.fixture
def event_handlers_mock(mocker):
    return EventHandlers(respond_func=lambda x, y: None)  # Mock respond_func as needed


class TestEventHandlers:
    def test_handle_event_found(self, mocker, event_handlers_mock):
        result = event_handlers_mock.handle_event(
            "event1", {"key": "value"}, ("localhost", 12345)
        )

        # Assert that the result matches the expected response
        assert result == {"result": "success"}

    def test_handle_event_not_found(self, mocker, event_handlers_mock):

        # Call handle_event for a non-existent event
        result = event_handlers_mock.handle_event(
            "non_existent_event", {}, ("localhost", 12345)
        )

        # Assert that the result is None for non-existent event
        assert result is None
