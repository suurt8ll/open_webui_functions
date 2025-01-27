"""
title: Smallthinker Filter
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.1.0
"""

from pydantic import BaseModel
import json


class Filter:
    class Valves(BaseModel):
        ENABLE_UPPERCASE: bool = False

    def __init__(self):
        self.valves = self.Valves()

    def outlet(self, body: dict, **kwargs) -> dict:
        print("`body` object before filtering:")
        print(json.dumps(body, indent=4))
        if self.valves.ENABLE_UPPERCASE:
            for message in body["messages"]:
                if message["role"] == "assistant":
                    message["content"] = message["content"].upper()
        print("`body` object after filtering:")
        print(json.dumps(body, indent=4))
        return body
