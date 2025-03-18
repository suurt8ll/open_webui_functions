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
    Literal,
    TypedDict,
    Any,
    NotRequired,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
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


class StatusEventData(TypedDict):
    description: str
    done: bool
    hidden: bool


class ChatEventData(TypedDict):
    type: Literal["status"]
    data: StatusEventData


class UserData(TypedDict):
    id: str
    email: str
    name: str
    role: Literal["admin", "user", "pending"]
    valves: NotRequired[Any]  # object of type UserValves


class Pipe:
    class Valves(BaseModel):
        EXAMPLE_STRING: str = Field(
            default="", title="Admin String", description="String configurable by admin"
        )
        LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    class UserValves(BaseModel):
        EXAMPLE_STRING_USER: str = Field(
            default="", title="User String", description="String configurable by user"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._print_colored("Function has been initialized!", "INFO")

    def pipes(self) -> list[dict]:
        self._print_colored("Registering models.", "INFO")
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: UserData,
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
                self._print_colored("Detected title generation task!", "INFO")
                return '{"title": "Example Title"}'

            if __task__ == "tags_generation":
                self._print_colored("Detected tag generation task!", "INFO")
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
            string_from_user_valve = getattr(
                __user__.get("valves"), "EXAMPLE_STRING_USER", None
            )

            self._print_colored(f"String from valve: {string_from_valve}", "INFO")
            self._print_colored(
                f"String from user valve: {string_from_user_valve}", "INFO"
            )

            # stored_files = Files.get_files()
            # self._print_colored(f"Stored files: {stored_files}", "DEBUG")

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
            self._print_colored("Returning all parameters as JSON:", "DEBUG")
            print(all_params_json)

            if __files__ and len(__files__) > 0:
                self._print_colored(
                    f'Detected a file upload! {__files__[0]["file"]["path"]}', "INFO"
                )

            return "Instant response sent!"

        except Exception as e:
            error_msg = f"Pipe function error: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return error_msg

    """Helper functions inside the Pipe class."""

    def _print_colored(self, message: str, level: str = "INFO") -> None:
        """
        Prints a colored log message to the console, respecting the configured log level.
        """
        if not hasattr(self, "valves"):
            return

        # Define log level hierarchy
        level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }

        # Only print if message level is >= configured level
        if level_priority.get(level, 0) >= level_priority.get(self.valves.LOG_LEVEL, 0):
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
