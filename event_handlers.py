from typing import Any, Dict, Union
from handlers.hand_pose_handlers import *
from handlers.general_handlers import *
from handlers.body_pose_handlers import *
from decorators import events
from server_utils import Req, Res


class EventHandlers:

    def handle_event(
        self, event: str, message_obj: Dict[str, Any]
    ) -> Union[Dict[str, str], None]:
        res = None
        req_obj = Req()
        req_obj.event = event
        req_obj.msg = message_obj
        res_obj = Res()
        if event in events:
            res = events[event](req_obj, res_obj)
        else:
            print(f"Event '{event}' not found")
        return res
