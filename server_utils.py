from typing import Dict


class Req:
    def __init__(self):
        self.msg: Dict[str, str]
        self.event: str
        self.FPS: str


class Res:
    def __init__(self, respond_func=None, addr=None):
        self.build = self._build_response_message
        self._respond_func = respond_func
        self.send = self._send
        self.addr = addr

    def _build_response_message(
        self,
        event: str,
        message: Dict[str, str],
    ) -> Dict[str, str]:
        response = {"event": event}
        response.update(message)
        return response

    def _send(
        self,
        event: str,
        message: Dict[str, str],
    ):
        res_msg = self.build(event, message)
        self._respond_func(res_msg, self.addr)
