"""
title: Inspect File Bytes
id: inspect_file_bytes
description: Logs all pipe parameters, the chat DB object, and detailed info (including raw byte size) for every file in the prompt.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

import json
import datetime
import inspect
import re
import base64
import os
from collections.abc import Callable, Awaitable
from typing import Any, TYPE_CHECKING
import pydantic_core
from fastapi import Request

# Open WebUI backend imports for direct data access
from open_webui.models.chats import Chats
from open_webui.models.files import Files
from open_webui.storage.provider import Storage

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types


class LeanLogger:
    """
    A simple, dependency-free logger for clean and truncated console output.
    """

    MAX_VALUE_LENGTH = 128
    TRUNCATION_SUFFIX = "..."

    def _recursively_truncate(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                key: self._recursively_truncate(value) for key, value in data.items()
            }
        if isinstance(data, list):
            return [self._recursively_truncate(item) for item in data]
        if isinstance(data, str) and len(data) > self.MAX_VALUE_LENGTH:
            return data[: self.MAX_VALUE_LENGTH] + self.TRUNCATION_SUFFIX
        if isinstance(data, (bytes, bytearray)):
            return f"<binary data: {len(data)} bytes>"
        return data

    def log(self, message: str, data: Any | None = None, level: str = "INFO"):
        timestamp = datetime.datetime.now().isoformat()
        caller_name = inspect.stack()[2].function
        print(f"[{timestamp}] [{level}] [{__name__}.{caller_name}] {message}")
        if data:
            serializable_data = pydantic_core.to_jsonable_python(
                data, serialize_unknown=True
            )
            sanitized_data = self._recursively_truncate(serializable_data)
            pretty_data = json.dumps(sanitized_data, indent=2, default=str)
            indented_data = "\n".join(
                [f"  {line}" for line in pretty_data.splitlines()]
            )
            print(indented_data)


_logger_instance = LeanLogger()


def _log(message: str, data: Any | None = None, level: str = "INFO"):
    _logger_instance.log(message, data, level)


class Pipe:
    def __init__(self):
        _log(f"{self.__class__.__name__} instance initialized.", data=self.__dict__)

    async def pipe(
        self,
        body: "Body",
        __user__: "UserData",
        __request__: Request,
        __metadata__: "Metadata",
        __tools__: dict[str, dict],
        __event_emitter__: Callable[["Event"], Awaitable[None]] | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __task__: str | None,
        __task_body__: dict[str, Any] | None,
        __files__: list[dict[str, Any]] | None,
        __message_id__: str | None,
        __chat_id__: str | None,
        __session_id__: str | None,
    ) -> str:

        # 1. Log all parameters passed to the pipe method.
        _log("Logging all local variables passed to pipe():", data=locals())

        # 2. Get and log the chat object directly from the backend database.
        chat_id = __metadata__.get("chat_id", "")
        chat = Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=__user__["id"])

        if not chat:
            _log(f"Chat with ID {chat_id} not found.", level="ERROR")
            return "Error: Could not find the chat object in the database."

        _log(f"Chat with ID {chat_id} found in DB.", data=chat)
        chat_data: "ChatObjectDataTD" = chat.chat  # type: ignore

        # 3. Find the latest user message to get the correctly ordered file list.
        latest_user_message = next(
            (msg for msg in reversed(chat_data["messages"]) if msg["role"] == "user"),
            None,
        )

        if latest_user_message and latest_user_message.get("files"):
            # 4. Process all files and log detailed information.
            file_details = self._get_raw_file_bytes(latest_user_message["files"])  # type: ignore
            _log("File inspection process complete.", data={"summary": file_details})
        else:
            _log("No files found in the latest user message.")

        return (
            f"Hello! I'm {__name__}. Inspection complete. See server logs for details."
        )

    def _get_raw_file_bytes(
        self, chat_history_files: list["FileAttachmentTD"]
    ) -> list[dict[str, Any]]:
        """
        Processes the 'files' list from a chat history message, retrieves the
        raw bytes for each file, and returns detailed information.
        """
        _log("Starting file inspection process...")
        processed_files_details = []

        for idx, file_info in enumerate(chat_history_files):
            file_name = file_info.get("name", f"untitled_file_{idx}")
            file_type = file_info.get("type")
            file_bytes: bytes | None = None
            detail = ""

            try:
                if file_type == "image":
                    base64_string = re.sub(
                        r"^data:image/.+;base64,", "", file_info["url"]
                    )
                    file_bytes = base64.b64decode(base64_string)
                    detail = "Decoded from base64 data URI."

                elif file_type == "file":
                    file_id = file_info.get("id")
                    if not file_id:
                        raise ValueError("RAG file entry is missing an 'id'.")

                    # Step 1: Get the DB record using the file_id
                    file_record = Files.get_file_by_id(id=file_id)
                    if not file_record or not file_record.path:
                        raise FileNotFoundError(
                            f"DB record or path not found for file ID: {file_id}"
                        )

                    db_path = file_record.path
                    detail = f"DB path is '{db_path}'. "

                    # Step 2: Get a guaranteed local path from the Storage provider
                    local_path = Storage.get_file(file_path=db_path)
                    if not os.path.exists(local_path):
                        raise FileNotFoundError(
                            f"Storage provider failed to produce local file at: {local_path}"
                        )

                    detail += f"Local path is '{local_path}'. "

                    # Step 3: Read the raw bytes from the local file
                    with open(local_path, "rb") as f:
                        file_bytes = f.read()

                if file_bytes is not None:
                    result = {
                        "name": file_name,
                        "type": file_type,
                        "status": "Success",
                        "bytes_size": len(file_bytes),
                        "detail": detail,
                    }
                    processed_files_details.append(result)
                    _log(
                        f"FILE_INSPECTION_SUCCESS: {file_name}",
                        data=result,
                        level="SUCCESS",
                    )
                else:
                    raise ValueError(
                        f"Unsupported file type '{file_type}' or processing failed."
                    )

            except Exception as e:
                result = {
                    "name": file_name,
                    "type": file_type,
                    "status": "Error",
                    "error_message": str(e),
                }
                processed_files_details.append(result)
                _log(f"FILE_INSPECTION_ERROR: {file_name}", data=result, level="ERROR")

        return processed_files_details
