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

import sys
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Iterator,
    Literal,
    NotRequired,
    TypedDict,
    TYPE_CHECKING,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from open_webui.utils.logger import stdout_format
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from utils.manifold_types import (
        UserData,
    )  # My personal types in a separate file for more robustness.


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
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

    async def pipes(self) -> list[dict]:
        self._add_log_handler()
        log.info("Registering models.")
        return [
            {"id": "log_testing_1", "name": "Log Testing 1"},
            {"id": "log_testing_2", "name": "Log Testing 2"},
        ]

    async def pipe(
        self,
        body: dict,
        __user__: "UserData",
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

        log.debug("Returning all parameters as JSON:", data=str(all_params))

        return "Hello World!"

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
