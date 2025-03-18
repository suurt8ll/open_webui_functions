"""
title: Log Testing
id: log_testing
description: Test function for figuring out how logging works.
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
    Generator,
    Iterator,
    Literal,
    NotRequired,
    TypedDict,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


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

    class UserValves(BaseModel):
        EXAMPLE_STRING_USER: str = Field(
            default="", title="User String", description="String configurable by user"
        )

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        print("Registering models.")
        return [
            {"id": "log_testing_1", "name": "Log Testing 1"},
            {"id": "log_testing_2", "name": "Log Testing 2"},
        ]

    async def pipe(
        self,
        body: dict,
        __user__: UserData,
    ) -> str | dict | StreamingResponse | Iterator | AsyncGenerator | Generator | None:
        string_from_valve = self.valves.EXAMPLE_STRING

        print(f"String from valve: {string_from_valve}")

        all_params = {
            "body": body,
            "__user__": __user__,
        }

        all_params_json = json.dumps(all_params, indent=2, default=str)
        print("Returning all parameters as JSON:", "DEBUG")
        print(all_params_json)

        return "Hello World!"
