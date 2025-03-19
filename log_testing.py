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
import sys
import inspect
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
from loguru import logger
from loguru._handler import Handler
from open_webui.utils.logger import stdout_format


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
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    class UserValves(BaseModel):
        EXAMPLE_STRING_USER: str = Field(
            default="", title="User String", description="String configurable by user"
        )

    def __init__(self):
        # The actual values for self.valves get applied later for some reason, they are not corrent in __init__.
        # Correct values do exists in pipes and pipe methods.
        self.valves = self.Valves()
        print("Initialization done!")

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

        # FIXME Avoid duplicating handlers.
        logger.add(
            sys.stdout,
            level="TRACE",
            format=stdout_format,
            filter=__name__,
        )

        logger.trace("DEBUG message!")
        # Access the internal state of the logger
        handlers: dict[int, Handler] = logger._core.handlers  # type: ignore
        for key, value in handlers.items():
            print("Key: ", key)
            print(json.dumps(value.__dict__, indent=2, default=str))
            try:
                # Returns the original str filter, can be used for duplicate detection I think.
                print(inspect.signature(value._filter).parameters["parent"].default)
            except Exception:
                print("fail")

        string_from_valve = self.valves.EXAMPLE_STRING

        print(f"String from valve: {string_from_valve}")

        all_params = {
            "body": body,
            "__user__": __user__,
        }

        all_params_json = json.dumps(all_params, indent=2, default=str)
        print("Returning all parameters as JSON:")
        print(all_params_json)

        return "Hello World!"
