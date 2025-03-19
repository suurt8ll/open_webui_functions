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
import sys
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
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
import json
import traceback
from open_webui.utils.logger import stdout_format
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler


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


log = logger.bind(auditable=False)


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
        print("[pipe_function_template_no_comments] Function has been initialized!")

    def pipes(self) -> list[dict]:
        self._add_log_handler()
        log.info("Registering models.")
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
                log.info("Detected title generation task!")
                return '{"title": "Example Title"}'

            if __task__ == "tags_generation":
                log.info("Detected tag generation task!")
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

            log.debug("String from valve:", data=string_from_valve)
            log.debug("String from user valve:", data=string_from_user_valve)

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

            log.debug(
                "Returning all parameters as JSON:",
                data=str(all_params),
            )

            return "Hello World!"

        except Exception as e:
            error_msg = f"Pipe function error: {str(e)}\n{traceback.format_exc()}"
            log.error(error_msg)
            return error_msg

    """Helper functions inside the Pipe class."""

    def _add_log_handler(self):
        """Adds handler to the root loguru instance for this plugin if one does not exist already."""

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__  # Filter by module name

        # Access the internal state of the logger
        handlers: dict[int, "Handler"] = logger._core.handlers  # type: ignore
        for key, handler in handlers.items():
            existing_filter = handler._filter
            if (
                hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            ):
                log.debug("Handler for this plugin is already present!")
                return

        logger.add(
            sys.stdout,
            level=self.valves.LOG_LEVEL,
            format=stdout_format,
            filter=plugin_filter,
        )
        log.info(
            f"Added new handler to loguru with level {self.valves.LOG_LEVEL} and filter {__name__}."
        )
