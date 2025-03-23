"""
title: Automatic Function Update Example
id: automatic_updates
description: Example function that demostrates how to update the function itself automatically.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

import sys
import inspect
import httpx
from typing import (
    Any,
    Optional,
    Awaitable,
    Callable,
    Literal,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.utils.logger import stdout_format
from loguru import logger
from packaging.version import Version

# --- Imports for self-update ---
from open_webui.routers.functions import update_function_by_id
from open_webui.models.functions import FunctionForm, FunctionMeta
from open_webui.models.users import Users
from open_webui.utils.plugin import extract_frontmatter

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from manifold_types import *  # My personal types in a separate file for more robustness.


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
        GITHUB_RAW_URL: str = Field(
            default="https://raw.githubusercontent.com/suurt8ll/open_webui_functions/refs/heads/feat/automatic-function-version-updates/test_and_example_functions/automatic_updates.py",
            description="GitHub raw url to use for update checks.",
        )

    def __init__(self):
        self.valves = self.Valves()

        # Put the frontmatter values into self object.
        frontmatter = self._extract_frontmatter()
        self.title: Optional[str] = frontmatter.get("title")
        self.id: Optional[str] = frontmatter.get("id")
        self.description: Optional[str] = frontmatter.get("description")
        self.author: Optional[str] = frontmatter.get("author")
        self.author_url: Optional[str] = frontmatter.get("author_url")
        self.funding_url: Optional[str] = frontmatter.get("funding_url")
        self.license: Optional[str] = frontmatter.get("license")
        self.requirements: Optional[str] = frontmatter.get("requirements")

        self.version = None
        try:
            v = frontmatter.get("version")
            self.version = Version(v) if v else None
        except ValueError:
            # Fails if version string does not follow PEP 440.
            pass

        log.info(f"[{self.id}] Function initialized.", self=str(self.__dict__))

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
    ) -> Optional[str]:

        self.__event_emitter__ = __event_emitter__

        all_params = {
            "body": body,
            "__user__": __user__,
            "__request__": __request__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
        }

        log.debug(
            "Returning all parameters as JSON:",
            data=str(all_params),
        )

        # --- Version Check and Update Notification ---
        # TODO: Branch to it completely independent task that does not block event this plugin itself.
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
                        "message": f"A new version ({self.version} -> {github_version_str}) is available. Update?",
                    },
                }
                response = await __event_call__(event_data)

                if response is not None and response is True:  # Check for confirmation
                    await self._self_update(__request__, __user__)

        # TODO: Use event emission to send a notification toast to the front-end telling how the update went and infroming the user to reload the browser.

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

        # TODO: Implement Throttling
        # FIXME: If requirments: is persent then backend restarts? Could destory the logic here.

        try:
            # 1. Get the admin user
            admin_user = Users.get_user_by_id(id=__user__["id"])
            if admin_user is None:
                raise ValueError("Could not retrieve admin user for self-update.")

            # 2. Fetch the latest code from GitHub
            github_code = await self._fetch_github_code()
            if not github_code:
                raise ValueError("Failed to fetch the latest code from GitHub.")

            # 3. Extract metadata from the GitHub code
            github_frontmatter = extract_frontmatter(github_code)
            function_meta = FunctionMeta(
                description=github_frontmatter.get("description", ""),
                manifest=github_frontmatter,
            )
            log.debug(f"FunctionMeta for self-update: {function_meta}")

            # 4. Get the function ID
            function_id = github_frontmatter.get("id", __name__.split(".")[-1])
            log.debug(f"Function ID for self-update: {function_id}")

            # 5. Create FunctionForm
            function_form = FunctionForm(
                id=function_id,
                name=github_frontmatter.get("title", "Automatic Updates"),
                content=github_code,
                meta=function_meta,
            )
            log.debug(f"FunctionForm for self-update: {function_form}")

            # 6. Call update_function_by_id
            updated_function = await update_function_by_id(
                request=__request__,
                id=function_id,
                form_data=function_form,
                user=admin_user,
            )

            if updated_function:
                log.info(f"Function '{function_id}' successfully updated.")
            else:
                raise Exception(
                    f"update_function_by_id returned None: Function '{function_id}' update failed."
                )
            # TODO: Update last update check timestamp in metadata.

        except Exception as e:
            log.error(f"Error updating function: {e}")
            await self._emit_error(f"Error updating function: {e}")

    def _extract_frontmatter(self):
        """Extracts the frontmatter from the function's source code."""
        try:
            module = inspect.getmodule(self.__class__)
            if not module:
                raise ValueError("Could not determine module for class.")

            source_code = inspect.getsource(module)
            frontmatter = extract_frontmatter(source_code)
            return frontmatter
        except OSError as e:
            log.error(f"Error getting source code: {e}")
            return {}
        except Exception as e:
            log.error(f"Error getting frontmatter: {e}")
            return {}

    async def _fetch_github_code(self):
        """Fetches the code from the raw GitHub URL."""
        raw_url = self.valves.GITHUB_RAW_URL
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(raw_url)
                response.raise_for_status()
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
            if current_version := self.version:
                github_version = Version(github_version_str)
                return github_version > current_version
            else:
                raise ValueError("Current version is missing.")
        except Exception as e:
            log.error(f"Error comparing versions: {e}")
            return False
