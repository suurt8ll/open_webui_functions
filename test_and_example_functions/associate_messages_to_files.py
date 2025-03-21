"""
title: Associate Messages to Files
id: associate_messages_to_files
description: Test function to understand how to associate file to a message that uploaded it.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version:
requirements:
"""

import json
from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Any,
    Literal,
    Optional,
)
import sys

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict
from uuid import UUID
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.requests import Request
import traceback
import inspect
from open_webui.models.chats import Chats, ChatModel

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


class FileInfo(TypedDict):
    type: str
    file: dict[str, Any]
    id: str
    url: str
    name: str
    status: str
    size: int
    error: str
    itemId: str


class Message(BaseModel):
    id: UUID
    parentId: Optional[UUID] = None
    childrenIds: list[UUID] = Field(default_factory=list)
    role: Literal["user", "assistant"]
    content: str
    files: Optional[list[FileInfo]]
    timestamp: int
    models: list[str] = Field(default_factory=list)
    model: Optional[str] = None  # Only for assistant role
    modelName: Optional[str] = None  # Only for assistant role
    modelIdx: Optional[int] = None  # Only for assistant role
    userContext: Optional[Any] = None  # Only for assistant role
    sources: Optional[list[dict[str, Any]]]  # Only for assistant role
    done: Optional[bool]  # Only for assistant role


class Pipe:
    class Valves(BaseModel):
        LOG_LEVEL: Literal["INFO", "WARNING", "ERROR", "DEBUG", "OFF"] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self._print_colored("Function has been initialized!", "INFO")

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __metadata__: dict[str, Any],
    ) -> (
        str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator
    ):
        try:
            chat_id = __metadata__.get("chat_id")
            if not chat_id:
                self._print_colored(
                    "Error: Chat ID not found in request body or metadata.", "ERROR"
                )
                return "Error: Chat ID not found."

            # Get the message history directly from the backend.
            chat = Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=__user__["id"])

            if chat:
                result = await self._process_chat_messages(chat)
                self._print_colored(
                    f"Printing the processed messages:\n{json.dumps(result, indent=2)}",
                    "DEBUG",
                )
            else:
                self._print_colored(f"Chat with ID {chat_id} not found.", "WARNING")
                return f"Chat with ID {chat_id} not found."
            return "Hello World!"

        except Exception as e:
            error_msg = f"Pipe function error: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return error_msg

    """Helper functions inside pipe method."""

    async def _process_chat_messages(self, chat: ChatModel) -> list[dict[str, Any]]:
        """Turns the Open WebUI's ChatModel object into more lean dict object that contains only the messages."""
        self._print_colored(
            f"Printing the raw ChatModel object:\n{json.dumps(chat.model_dump(), indent=2)}",
            "DEBUG",
        )
        messages: list[Message] = chat.chat.get("messages", [])
        result = []
        for message in messages:
            role = message.role
            content = message.content
            files = []
            if role == "user":
                files_for_message = message.files
                if files_for_message:
                    files = [
                        file_data.get("name", "") for file_data in files_for_message
                    ]
            elif role == "assistant":
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

    def _print_colored(self, message: str, level: str = "INFO") -> None:
        if not hasattr(self, "valves") or self.valves.LOG_LEVEL == "OFF":
            return

        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
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
                f"{color}[{level}][associate_messages_to_files][{method_name}]{COLORS['RESET']} {message}"
            )
