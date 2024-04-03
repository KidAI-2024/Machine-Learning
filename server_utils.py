from typing import Dict


class Req:
    def __init__(self):
        self.msg: Dict[str, str]
        self.event: str
        self.FPS: str


class Res:
    def __init__(self):
        self.build = self._build_response_message

    def _build_response_message(
        self,
        event: str,
        message: Dict[str, str],
    ) -> Dict[str, str]:
        response = {"event": event}
        response.update(message)
        return response
