"""
title: Lean Pipe Function Skeleton
id: pipe_function_skeleton
description: Good starting point for creating new pipe functions for Open WebUI.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

import asyncio
from typing import (
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterator,
    Callable,
    Any,
    Literal,
    TypedDict,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.requests import Request
import json
import traceback
import inspect
from open_webui.models.files import Files

COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "RESET": "\033[0m",
}


def print_colored(message: str, level: str = "INFO") -> None:
    color_map = {
        "INFO": COLORS["GREEN"],
        "WARNING": COLORS["YELLOW"],
        "ERROR": COLORS["RED"],
        "DEBUG": COLORS["BLUE"],
    }
    color = color_map.get(level, COLORS["WHITE"])
    frame = inspect.currentframe()
    if frame:
        frame = frame.f_back
    method_name = frame.f_code.co_name if frame else "<unknown>"
    print(
        f"{color}[{level}][pipe_function_skeleton][{method_name}]{COLORS['RESET']} {message}"
    )


class StatusEventData(TypedDict):
    description: str
    done: bool
    hidden: bool


class ChatEventData(TypedDict):
    type: Literal["status"]
    data: StatusEventData


class Pipe:
    class Valves(BaseModel):
        EXAMPLE_STRING: str = Field(
            default="", title="Admin String", description="String configurable by admin"
        )

    class UserValves(BaseModel):
        EXAMPLE_STRING_USER: str = Field(
            default="", title="User String", description="String configurable by user"
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        __event_emitter__: Callable[[ChatEventData], Awaitable[None]],
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]],
        __task__: str,
        __task_body__: dict[str, Any],
        __files__: list[dict[str, Any]],
        __metadata__: dict[str, Any],
        __tools__: list[Any],
    ) -> (
        str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator
    ):
        try:
            if __task__ == "title_generation":
                print_colored("Detected title generation task!", "INFO")
                return '{"title": "Example Title"}'
            if __task__ == "tags_generation":
                print_colored("Detected tag generation task!", "INFO")
                return '{"tags": ["tag1", "tag2", "tag3"]}'

            async def countdown():
                for i in range(5, 0, -1):
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Time remaining: {i}s",
                                "done": False,
                                "hidden": False,
                            },
                        }
                    )
                    await asyncio.sleep(1)
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Process complete!",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )

            asyncio.create_task(countdown())

            string_from_valve = self.valves.EXAMPLE_STRING
            string_from_user_valve = __user__["valves"].EXAMPLE_STRING_USER

            print_colored(f"String from valve: {string_from_valve}", "INFO")
            print_colored(f"String from user valve: {string_from_user_valve}", "INFO")

            # stored_files = Files.get_files()
            # print_colored(f"Stored files: {stored_files}", "DEBUG")

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
            print_colored("Returning all parameters as JSON:", "DEBUG")
            print(all_params_json)

            if __files__ and len(__files__) > 0:
                print_colored(
                    f'Detected a file upload! {__files__[0]["file"]["path"]}', "INFO"
                )

            return "Instant response sent!"

        except Exception as e:
            error_msg = f"Pipe function error: {str(e)}\n{traceback.format_exc()}"
            print_colored(error_msg, "ERROR")
            return error_msg
