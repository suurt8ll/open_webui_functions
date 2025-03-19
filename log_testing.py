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


# Setting auditable avoids duplicate output for log levels that would be printed out by the main logger too.
log = logger.bind(auditable=False)


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
        self.valves = self.Valves()
        print("[log_testing] Initialization done!")

    def pipes(self) -> list[dict]:
        self._add_log_handler()
        log.info("Registering models.")
        return [
            {"id": "log_testing_1", "name": "Log Testing 1"},
            {"id": "log_testing_2", "name": "Log Testing 2"},
        ]

    async def pipe(
        self,
        body: dict,
        __user__: UserData,
    ) -> str | dict | StreamingResponse | Iterator | AsyncGenerator | Generator | None:

        log.trace("TRACE message!")
        log.debug("DEBUG message!")
        log.info("INFO message!")
        log.warning("WARNING message!")
        log.error("ERROR message!")
        log.critical("CRITICAL message!")

        string_from_valve = self.valves.EXAMPLE_STRING

        log.debug(f"String from valve: {string_from_valve}")

        all_params = {
            "body": body,
            "__user__": __user__,
        }

        all_params_json = json.dumps(all_params, indent=2, default=str)
        log.debug("Returning all parameters as JSON:")
        print(all_params_json)

        return "Hello World!"

    def _add_log_handler(self):
        """Adds handler to the root loguru instance for this plugin if one does not exist already."""
        # Access the internal state of the logger
        handlers: dict[int, Handler] = logger._core.handlers  # type: ignore
        for key, value in handlers.items():
            try:
                # Returns the original str filter, can be used for duplicate detection.
                handler_filter = (
                    inspect.signature(value._filter).parameters["parent"].default[:-1]
                )
                if handler_filter == __name__:
                    log.debug("Handler for this plugin is already present!")
                    return
            except Exception as e:
                continue
        logger.add(
            sys.stdout,
            level=self.valves.LOG_LEVEL,
            format=stdout_format,
            filter=__name__,
        )
        log.info(
            f"Added new handler to loguru with level {self.valves.LOG_LEVEL} and filter {__name__}."
        )
