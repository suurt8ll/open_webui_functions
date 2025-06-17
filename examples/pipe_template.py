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
from starlette.responses import StreamingResponse
from fastapi import Request

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


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
        self._log("Function has been initialized!")

    def _log(self, message: str):
        timestamp = datetime.datetime.now().isoformat()
        caller_name = inspect.stack()[1].function
        print(f"[{timestamp}] [{__name__}.{caller_name}] {message}")

    async def pipes(self) -> list["ModelData"]:
        models: list["ModelData"] = [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]
        self._log(f"Registering models: {models}")
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
        message_content = (
            "Returning all local variables as JSON:\n"
            f"{json.dumps(locals(), indent=2, default=str)}"
        )
        self._log(message_content)
        return f"Hello! I'm {__name__}."
