"""
title: Advanced Prompt Injector
description: Filter that will detect and inject prompt configurations from user input using a <details> block.
id: system_prompt_injector
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.6.0
"""

# The injection must follow this format (without triple quotes).
"""
<details type="prompt">
<summary>{{prompt_title}}</summary>
```json
{
    system: "{{system_prompt}}",
    temperature: {{temperature}}
}
```
</details>
{{content}}
"""

# IMPORTANT: Disable "Rich Text Input for Chat" in Open WebUI settings for this plugin to work correctly.

import json
import re
import copy
import sys
from loguru import logger
from pydantic import BaseModel, Field
from typing import Any, Awaitable, Callable, Literal, cast, TYPE_CHECKING
from open_webui.models.functions import Functions
from open_webui.utils.logger import stdout_format

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.
    from loguru import Record
    from loguru._handler import Handler

# --- Constants ---
# Regex to find the entire details block and capture its components
# - Group 1: Full <details> block
# - Group 2: Content inside <summary> tag (prompt title)
# - Group 3: Content between </summary> and </details> (parameters)
DETAILS_BLOCK_REGEX = re.compile(
    r'(<details type="prompt">\s*<summary>(.*?)</summary>(.*?)^\s*</details>)',
    re.DOTALL | re.MULTILINE,
)


class PluginLogger:
    """
    A wrapper around Loguru's logger that provides enhanced data structure logging.
    """

    def __init__(
        self,
        log_level: Literal[
            "TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ] = "INFO",
        truncate: bool = True,
        max_length: int = 64,
    ):
        # Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
        self.logger = logger.bind(auditable=False)
        self.log_level = log_level
        print(f"Got log level: {log_level}")
        self.truncate = truncate
        self.max_length = max_length
        self._add_handler()

    def _add_handler(self):
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
                self.logger.debug("Handler for this plugin is already present!")
                return

        logger.add(
            sys.stdout,
            level=self.log_level,
            format=stdout_format,
            filter=plugin_filter,
        )
        self.logger.info(
            f"Added new handler to loguru with level {self.log_level} and filter {__name__}."
        )

    def _log_with_data(
        self,
        message: str,
        level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        exception: Exception | None = None,
        data: Any = None,
    ):
        self.logger.opt(depth=2, exception=exception).log(level, message)
        if not data:
            return
        if self.logger.level(level) >= self.logger.level(self.log_level):
            if self.truncate is True:
                self.logger.opt(depth=2, exception=exception, raw=True).log(
                    level, self._truncate_long_strings(data) + "\n"
                )
            else:
                self.logger.opt(depth=2, exception=exception, raw=True).log(
                    level, json.dumps(data, indent=2, default=str) + "\n"
                )

    def trace(self, message: str, data: Any = None):
        self._log_with_data(message, level="TRACE", data=data)

    def debug(self, message: str, data: Any = None):
        self._log_with_data(message, level="DEBUG", data=data)

    def info(self, message: str, data: Any = None):
        self._log_with_data(message, level="INFO", data=data)

    def warning(self, message: str, data: Any = None):
        self._log_with_data(message, level="WARNING", data=data)

    def error(self, message: str, data: Any = None):
        self._log_with_data(message, level="ERROR", data=data)

    def exception(self, message: str, exception: Exception, data: Any = None):
        self._log_with_data(message, level="ERROR", exception=exception, data=data)

    def critical(self, message: str, data: Any = None):
        self._log_with_data(message, level="CRITICAL", data=data)

    def _truncate_long_strings(self, data: Any, max_length: int = 64) -> str:

        def process_data(data: Any, max_length: int = self.max_length) -> Any:
            if isinstance(data, BaseModel):
                data_dict = data.model_dump()
                return process_data(data_dict, max_length)
            elif isinstance(data, dict):
                for key, value in list(data.items()):
                    data[key] = process_data(value, max_length)
                return data
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    data[idx] = process_data(item, max_length)
                return data
            elif isinstance(data, str):
                if len(data) > max_length:
                    truncated_length = len(data) - max_length
                    return f"{data[:max_length]}[{truncated_length} chars truncated]"
                return data
            elif isinstance(data, bytes):
                hex_str = data.hex()
                if len(hex_str) > max_length:
                    truncated_length = len(hex_str) - max_length
                    return f"{hex_str[:max_length]}[{truncated_length} chars truncated]"
                else:
                    return hex_str
            else:
                return data

        copied_data = copy.deepcopy(data)
        processed = process_data(copied_data, max_length)
        return json.dumps(processed, indent=2, default=str)


class Filter:

    class Valves(BaseModel):
        LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
            Field(
                default="INFO",
                description="Select logging level. Use `docker logs -f open-webui` to view logs.",
            )
        )

    def __init__(self):
        # Valves are not used in this version but kept for potential future config
        valves = Functions.get_function_valves_by_id("gemini_manifold_google_genai")
        self.valves = self.Valves(**(valves if valves else {}))
        # Store the prompt title extracted during inlet for use in outlet
        self.prompt_title: str | None = None
        # FIXME: If log level is changes inside the valve, it does not have effect until the module reloads.
        self.log = PluginLogger(self.valves.LOG_LEVEL, truncate=True, max_length=128)
        self.log.info("Function has been initialized.")

    def inlet(self, body: "Body") -> "Body":

        self.log.debug(f"\n--- Inlet Filter ---")
        self.log.debug("Original Request Body:", body)

        messages: list["Message"] = body.get("messages", [])
        if not messages:
            self.log.warning("No messages found in the body.")
            return body

        latest_system_prompt: str | None = None
        latest_options: "Options | None" = None
        prompt_title: str | None = None
        self.prompt_title = None  # Reset stored title

        for message in messages:
            if message.get("role") == "user":
                message = cast("UserMessage", message)
                content = message.get("content", "")

                if isinstance(content, list):  # Handle messages with images and text
                    # Separate non-text (images) and text parts
                    non_text_content = [
                        item for item in content if item.get("type") != "text"
                    ]
                    text_segments = [
                        cast("TextContent", item)["text"]
                        for item in content
                        if item.get("type") == "text"
                    ]

                    # Process each text segment for injections
                    new_text_segments: list[str] = []
                    system_prompt_found, options_found, title_found = None, None, None
                    for ts in text_segments:
                        sp, title, mod_ts, opt = self._extract_injection_params(ts)
                        new_text_segments.append(mod_ts)
                        if title is not None:
                            system_prompt_found = sp
                            options_found = opt
                            title_found = title

                    # Combine text into a single part (as per your assumption)
                    combined_text = (
                        "".join(new_text_segments) if new_text_segments else None
                    )  # Use appropriate join method if needed

                    # Rebuild content: all images first, then single text part
                    new_content = non_text_content
                    if combined_text:
                        new_content.append({"type": "text", "text": combined_text})
                    message["content"] = new_content

                    # Track parameters from this message
                    if title_found is not None:
                        latest_system_prompt = system_prompt_found
                        latest_options = options_found
                        prompt_title = title_found

                else:  # Handle traditional text-only messages
                    system_prompt, title, modified_content, options = (
                        self._extract_injection_params(content)
                    )
                    message["content"] = modified_content

                    if title is not None:
                        latest_system_prompt = system_prompt
                        latest_options = options
                        prompt_title = title

        # Apply latest parameters if found
        if latest_system_prompt is not None and latest_system_prompt.strip():
            self._apply_system_prompt(body, latest_system_prompt)
        else:
            self.log.debug("No valid system prompt found.")

        if latest_options:
            for k, v in latest_options.items():
                self._update_options(body, k, v)
                self.log.debug(f"Set {k} to: {v}")
        self.prompt_title = prompt_title

        self.log.debug("Modified Request Body (before sending to LLM):", data=body)
        return body

    async def outlet(
        self, body: "Body", __event_emitter__: Callable[["Event"], Awaitable[None]]
    ) -> "Body":

        self.log.debug(f"\n--- Outlet Filter ---")
        self.log.debug(
            "Original Response Body:",
            data=body,
        )

        # Only add header if a prompt title was set during inlet
        if self.prompt_title:
            status_event: "StatusEvent" = {
                "type": "status",
                "data": {"description": self.prompt_title},
            }
            await __event_emitter__(status_event)

        self.log.debug(
            "Modified Response Body (after adding header):",
            data=body,
        )

        return body

    # region Helper methods inside the Pipe class

    def _update_options(self, body: "Body", key: str, value: Any):
        """Safely updates the 'options' dictionary in the request body."""
        if "options" not in body:
            body["options"] = {}
        body["options"][key] = value
        self.log.debug(f"Updated options: set '{key}' to {value}")

    def _extract_injection_params(
        self, user_message_content: str
    ) -> tuple[str | None, str | None, str, "Options | None"]:

        system_prompt = None
        prompt_title = None
        modified_content = user_message_content
        json_data: "Options | None" = None
        injection_block = None

        match = DETAILS_BLOCK_REGEX.search(user_message_content)
        if match:
            injection_block = match.group(1)
            prompt_title = match.group(2).strip()
            params_block = match.group(3).strip()

            # Remove injection block from content
            modified_content = user_message_content.replace(injection_block, "").strip()

            # Extract JSON content from code block
            json_match = re.search(r"```json\s*(.*?)\s*```", params_block, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    json_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    self.log.exception("JSON Parse Error.", exception=e)
                if not json_data:
                    return (None, prompt_title, modified_content, None)

                system_prompt = json_data.get("system")
            else:
                self.log.warning("No JSON block found in parameters section")

        return (system_prompt, prompt_title, modified_content, json_data)

    def _apply_system_prompt(self, body: "Body", system_prompt_content: str):
        """
        Applies a *non-empty* system prompt to the request body.
        Updates the existing system message or inserts a new one at the beginning.
        Also updates the 'system' key in the 'options' dictionary if present.
        """
        # This function should only be called with non-empty system_prompt_content
        # due to the check in the inlet method.
        if not system_prompt_content:
            self.log.warning(
                "_apply_system_prompt called with empty content. This should not happen."
            )
            return  # Do nothing if called with empty string despite the check

        messages: list["Message"] = body.get("messages", [])
        system_message_found = False

        # Iterate through messages to find and update the system message
        for message in messages:
            if message.get("role") == "system":
                message["content"] = system_prompt_content
                system_message_found = True
                self.log.debug("Updated existing system message.")
                break

        # If no system message exists, insert one at the beginning
        if not system_message_found:
            messages.insert(0, {"role": "system", "content": system_prompt_content})
            body["messages"] = messages  # Ensure the body reflects the change
            self.log.debug("Inserted new system message at the beginning.")

        # Also update the 'system' option if it exists (some backends might use this)
        # We update this even if the message was pre-existing, to ensure consistency
        self._update_options(body, "system", system_prompt_content)

    # endregion
