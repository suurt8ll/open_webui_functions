"""
title: Automatic Function Update Example
id: automatic_updates
description: Example function that demostrates how to update the function itself automatically.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.1
requirements:
"""

import sys
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterator,
    Callable,
    Literal,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
from open_webui.utils.logger import stdout_format
from loguru import logger

# --- Imports for self-update ---
from open_webui.routers.functions import update_function_by_id
from open_webui.models.functions import FunctionForm, FunctionMeta
from open_webui.models.users import Users
from open_webui.utils.plugin import extract_frontmatter
import inspect
import httpx
from packaging.version import Version

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from manifold_types import *  # My personal types in a separate file for more robustness.
    from open_webui.models.users import UserModel


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


class Pipe:
    class Valves(BaseModel):
        LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
            Field(
                default="INFO",
                description="Select logging level. Use `docker logs -f open-webui` to view logs.",
            )
        )

    def __init__(self):
        self.valves = self.Valves()
        self.current_version = self._get_current_version()
        print(
            f"[automatic_updates] Function initialized. Version: {self.current_version}"
        )

    async def pipes(self) -> list["ModelData"]:
        self._add_log_handler()
        return [{"id": "automatic_updates", "name": "Automatic Updates"}]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __event_call__: Callable[["Event"], Awaitable[Any]],
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

        self.__event_emitter__ = __event_emitter__

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

        log.debug(
            "Returning all parameters as JSON:",
            data=str(all_params),
        )

        # --- Version Check and Update Notification ---
        github_code = await self._fetch_github_code()
        if github_code:
            github_version_str = self._extract_version_from_code(github_code)
            if github_version_str and self._is_new_version_available(
                github_version_str
            ):
                event_data: ConfirmationEvent = {
                    "type": "confirmation",
                    "data": {
                        "title": "Update Available",
                        "message": f"A new version ({self.current_version} -> {github_version_str}) is available. Update?",
                    },
                }
                response = await __event_call__(event_data)

                if response is not None and response is True:  # Check for confirmation
                    await self._self_update(__request__, __user__)

        return "Hello World!"

    """
    ---------- Helper functions inside the Pipe class. ----------
    """

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

    async def _emit_error(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""
        error: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {
                "done": True,
                "error": {"detail": "\n" + error_msg},
            },
        }
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        await self.__event_emitter__(error)

    async def _self_update(self, __request__: Request, __user__: "UserData"):
        """Attempts to update the function itself."""
        log.info("Attempting self-update...")

        # 1. Get the admin user
        try:
            admin_user: Optional["UserModel"] = Users.get_user_by_id(id=__user__["id"])
            if admin_user is None:
                await self._emit_error("Could not retrieve admin user for self-update.")
                return
        except Exception as e:
            await self._emit_error(f"Error getting admin user: {e}")
            return

        # 2. Get the function ID (from the module name, in this example)
        function_id = __name__.split(".")[-1]  # Extract ID from module name
        log.debug(f"Function ID for self-update: {function_id}")

        # 3. Create FunctionMeta (example values)
        function_meta = FunctionMeta(
            description="Updated description via self-update",
            manifest={"updated": True},
        )
        log.debug(f"FunctionMeta for self-update: {function_meta}")

        # 4. Create FunctionForm (example new content)
        new_content = (
            "def updated_function():\n    return 'This function has been updated!'"
        )
        function_form = FunctionForm(
            id=function_id,
            name="Automatic Updates (Updated)",  # Updated name
            content=new_content,
            meta=function_meta,
        )
        log.debug(f"FunctionForm for self-update: {function_form}")

        # 5. Call update_function_by_id (PLACEHOLDER)
        log.info(
            "PLACEHOLDER: Would call update_function_by_id here with:"
            f" request={__request__}, id={function_id}, form_data={function_form}, user={admin_user}"
        )
        # await update_function_by_id(
        #     request=__request__,
        #     id=function_id,
        #     form_data=function_form,
        #     user=admin_user,
        # )
        await self._emit_error("Self-update is disabled for safety.", warning=True)

    def _get_current_version(self):
        """Gets the current version from the function's frontmatter."""
        try:
            module = inspect.getmodule(self.__class__)
            if module:
                try:
                    source_code = inspect.getsource(module)
                except OSError as e:
                    log.error(f"Error getting source code for module: {e}")
                    source_code = None
            else:
                log.error("Could not determine module for class.")
                source_code = None

            if source_code:
                frontmatter = extract_frontmatter(source_code)
                return frontmatter.get("version", "0.0.0")  # Default if not found
            else:
                return "0.0.0"  # Return default if source code is None
        except Exception as e:
            log.error(f"Error getting current version: {e}")
            return "0.0.0"

    async def _fetch_github_code(self):
        """Fetches the code from the raw GitHub URL."""
        raw_url = "https://raw.githubusercontent.com/suurt8ll/open_webui_functions/refs/heads/feat/automatic-function-version-updates/test_and_example_functions/automatic_updates.py"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(raw_url)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                return response.text
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error fetching GitHub code: {e}")
            return None
        except httpx.RequestError as e:
            log.error(f"Request error fetching GitHub code: {e}")
            return None

    def _extract_version_from_code(self, code):
        """Extracts the version from the code using extract_frontmatter."""

        frontmatter = extract_frontmatter(code)
        return frontmatter.get("version")

    def _is_new_version_available(self, github_version_str):
        """Compares the GitHub version with the current version."""
        try:
            current_version = Version(self.current_version)
            github_version = Version(github_version_str)
            return github_version > current_version
        except Exception as e:
            log.error(f"Error comparing versions: {e}")
            return False
