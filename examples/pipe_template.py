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

import json
import datetime
import inspect
from collections.abc import Iterator, AsyncGenerator, Generator, Callable, Awaitable
from typing import (
    Any,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
import pydantic_core
from starlette.responses import StreamingResponse
from fastapi import Request

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


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
        _log(f"{self.__class__.__name__} instance initialized.", data=self.__dict__)

    async def pipes(self) -> list["ModelData"]:
        models: list["ModelData"] = [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]
        _log(f"Registering models:", data=models)
        return models

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: "UserData",
        __request__: Request,
        __metadata__: dict[str, Any],
        __tools__: dict[str, dict],
        __event_emitter__: Callable[["Event"], Awaitable[None]] | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        __task__: str | None,
        __task_body__: dict[str, Any] | None,
        __files__: list[dict[str, Any]] | None,
        __message_id__: str | None,
        __chat_id__: str | None,
        __session_id__: str | None,
    ) -> (
        str
        | dict[str, Any]
        | BaseModel
        | StreamingResponse
        | Iterator
        | AsyncGenerator
        | Generator
    ):
        _log("Returning all local variables as JSON:", data=locals())
        return f"Hello! I'm {__name__}."
