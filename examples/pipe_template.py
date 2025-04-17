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
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterator,
    Callable,
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
        print(f"{[__name__]} Function has been initialized!")

    async def pipes(self) -> list["ModelData"]:
        models: list["ModelData"] = [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]
        print(f"{[__name__]} Registering models: {models}")
        return models

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

        string_from_valve = self.valves.EXAMPLE_STRING
        string_from_user_valve = getattr(
            __user__.get("valves"), "EXAMPLE_STRING_USER", None
        )

        print(f"{[__name__]} String from valve: {string_from_valve}")
        print(f"{[__name__]} String from user valve: {string_from_user_valve}")

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

        print(
            f"{[__name__]} Returning all parameters as JSON:\n{json.dumps(all_params, indent=2, default=str)}"
        )

        return f"Message from: {[__name__]}."
