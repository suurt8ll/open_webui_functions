"""
title: This Is The Function Title
id: pipe_function_template
description: Good starting point for creating new pipe functions for Open WebUI.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

from typing import AsyncGenerator, Generator, Iterator, Union
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.requests import Request
import json

# You can import any module from the Open WebUI backend itself.
from open_webui.models.files import Files


class Pipe:
    class Valves(BaseModel):
        EXAMPLE_STRING: str = Field(default="")

    class UserValves(BaseModel):
        EXAMPLE_STRING_USER: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        # FIXME Figure out how to type hint the event emitter and event call.
        __event_emitter__,
        __event_call__,
        __task__: str,
        __task_body__: dict,
        __files__: list,
        __metadata__: dict,
        __tools__: list,
    ) -> Union[str, dict, StreamingResponse, Iterator, AsyncGenerator, Generator]:

        string_from_valve = self.valves.EXAMPLE_STRING
        string_from_user_valve = __user__["valves"].EXAMPLE_STRING_USER

        print("[pipe] String from valve: ", string_from_valve)
        print("[pipe] String from user valve: ", string_from_user_valve)

        stored_files = Files.get_files()
        print("[pipe] Stored files: ", stored_files)

        all_params = {
            "body": body,
            "__user__": __user__,
            "__request__": __request__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__task__": __task__,
            "__task_body__": __task_body__,
            "__files__": __files__,
            "__metadata__": __metadata__,
            "__tools__": __tools__,
        }

        all_params_json = json.dumps(all_params, indent=2, default=str)

        print("[pipe] Returning all parameters as JSON:")
        print(all_params_json)

        return "Hello from pipe function!"
