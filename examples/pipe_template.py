"""
title: Pipe Function Template
id: pipe_template
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
    Any,
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterator,
    Callable,
    Literal,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
from open_webui.utils.logger import stdout_format
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
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

    async def pipes(self) -> list["ModelData"]:
        self._add_log_handler()
        log.info("Registering models.")
        try:
            return [
                {"id": "model_id_1", "name": "model_1"},
                {"id": "model_id_2", "name": "model_2"},
            ]
        except Exception as e:
            error_msg = f"Error during registering models: {str(e)}"
            return [_return_error_model(error_msg)]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]],
        __task__: str,
        __task_body__: dict[str, Any],
        __files__: list[dict[str, Any]],
        __metadata__: dict[str, Any],
        __tools__: list[Any],
    ) -> (
        str
        | dict[str, Any]
        | StreamingResponse
        | Iterator
        | AsyncGenerator
        | Generator
        | None
    ):

        self.__event_emitter__ = __event_emitter__

        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrival phase: {str(__metadata__["model"])}'
            log.exception(error_msg)
            await self._emit_error(error_msg)
            return

        if __task__ == "title_generation":
            log.info("Detected title generation task!")
            return '{"title": "Example Title"}'

        if __task__ == "tags_generation":
            log.info("Detected tag generation task!")
            return '{"tags": ["tag1", "tag2", "tag3"]}'

        # FIXME: Move the countdown example to a separate status emission example file.
        async def countdown():
            for i in range(5, 0, -1):
                status_count: StatusEvent = {
                    "type": "status",
                    "data": {
                        "description": f"Time remaining: {i}s",
                        "done": False,
                        "hidden": False,
                    },
                }
                await __event_emitter__(status_count)
                await asyncio.sleep(1)

            status_finish: StatusEvent = {
                "type": "status",
                "data": {
                    "description": "Process complete!",
                    "done": True,
                    "hidden": False,
                },
            }
            await __event_emitter__(status_finish)

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

        # FIXME: Move error handling examples to a separate function.
        try:
            raise Exception("NameError, this is a test.")
        except Exception as e:
            error_msg = f"Error happened inside the pipe function: {str(e)}"
            await self._emit_error(error_msg)
            return

        return "Hello World!"

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

    async def _emit_error(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""
        error: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {
                "done": True,
                "error": {"detail": "\n" + error_msg},
            },
        }
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        await self.__event_emitter__(error)


def _return_error_model(
    error_msg: str, warning: bool = False, exception: bool = True
) -> "ModelData":
    """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
    if warning:
        log.opt(depth=1, exception=False).warning(error_msg)
    else:
        log.opt(depth=1, exception=exception).error(error_msg)
    return {
        "id": "error",
        "name": "[pipe_template] " + error_msg,
    }
