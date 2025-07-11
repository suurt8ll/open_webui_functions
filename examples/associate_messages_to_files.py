"""
title: Associate Messages to Files
id: associate_messages_to_files
description: Test function to understand how to associate file to a message that uploaded it.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

import json
import sys
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    cast,
    TYPE_CHECKING,
)

from pydantic import BaseModel, Field
from open_webui.models.chats import Chats
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
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        print("[associate_messages_to_files] Initialization done!")

    async def pipes(self) -> list["ModelData"]:
        self._add_log_handler()
        log.info("Registering the pipe model.")
        return [
            {"id": "associate_messages_to_files", "name": "Associate Messages to Files"}
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __user__: "UserData",
        __metadata__: dict[str, Any],
    ) -> str | None:

        chat_id = __metadata__.get("chat_id")
        if not chat_id:
            error_msg = "Chat ID not found in request body or metadata."
            await self._emit_error(
                error_msg, event_emitter=__event_emitter__, exception=False
            )
            return None

        # Get the message history directly from the backend.
        chat = Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=__user__["id"])

        if chat:
            print(json.dumps(chat.model_dump(), indent=2, default=str))
        else:
            error_msg = f"Chat with ID {chat_id} not found."
            await self._emit_error(
                error_msg, event_emitter=__event_emitter__, exception=False
            )
            return None
        return "Hello World!"

    """
    ---------- Helper methods inside the Pipe class. ----------
    """

    @staticmethod
    async def _emit_completion(
        event_emitter: Callable[["Event"], Awaitable[None]],
        content: str | None = None,
        done: bool = False,
        error: str | None = None,
        sources: list["Source"] | None = None,
    ):
        """Constructs and emits completion event."""
        emission: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {"done": done},
        }
        if content:
            emission["data"]["content"] = content
        if error:
            emission["data"]["error"] = {"detail": error}
        if sources:
            emission["data"]["sources"] = sources
        await event_emitter(emission)

    async def _emit_error(
        self,
        error_msg: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        warning: bool = False,
        exception: bool = True,
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        await self._emit_completion(
            error=f"\n{error_msg}", event_emitter=event_emitter, done=True
        )

    async def _process_chat_messages(
        self, chat: "ChatObjectDataTD"
    ) -> list[dict[str, Any]]:
        """
        Turns the Open WebUI's ChatModel object into more lean dict object that contains only the messages.
        """
        messages = chat.get("messages", [])
        result = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            files = []
            if role == "user":
                message = cast("UserMessage", message)
                files_for_message = message.get("files")
                if files_for_message:
                    files = [file_data.get("name") for file_data in files_for_message]
            elif role == "assistant":
                message = cast("AssistantMessage", message)
                if not hasattr(message, "done"):
                    continue
            result.append(
                {
                    "role": role,
                    "content": content,
                    "files": files,
                }
            )
        return result

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
