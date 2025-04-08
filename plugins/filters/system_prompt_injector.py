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
    system_prompt: "{{system_prompt}}",
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
    from loguru import Record, Logger
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

    def log_data(
        self,
        level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        message: str,
        data: Any | None = None,
    ):
        log_method = getattr(
            self.logger, level.lower()
        )  # Get the appropriate log method
        log_method(message)  # Log the main message

        if data:
            if self.logger.level(level) >= self.logger.level(self.log_level):
                if self.truncate is True:
                    print(self._truncate_long_strings(data))
                else:
                    print(json.dumps(data, indent=2, default=str))

    def trace(self, message: str, data: Any | None = None):
        self.log_data("TRACE", message, data)

    def debug(self, message: str, data: Any | None = None):
        self.log_data("DEBUG", message, data)

    def info(self, message: str, data: Any | None = None):
        self.log_data("INFO", message, data)

    def warning(self, message: str, data: Any | None = None):
        self.log_data("WARNING", message, data)

    def error(self, message: str, data: Any | None = None):
        self.log_data("ERROR", message, data)

    def critical(self, message: str, data: Any | None = None):
        self.log_data("CRITICAL", message, data)

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
        self.log = PluginLogger(self.valves.LOG_LEVEL, truncate=True, max_length=128)
        self.log.debug(
            "Function has been initialized:",
            data=[json.dumps(self.__dict__, indent=2, default=str)],
        )

    def inlet(self, body: "Body") -> "Body":
        """
        Processes incoming request body.
        Detects and extracts prompt injection details from all user messages.
        Applies extracted system prompt (if not empty) and temperature.
        Removes the injection block from all user messages.
        """
        self.log.debug(f"\n--- Inlet Filter ---")
        self.log.debug("Original Request Body:", body)

        messages: list["Message"] = body.get("messages", [])
        if not messages:
            self.log.warning("No messages found in the body.")
            return body

        latest_system_prompt: str | None = None
        latest_temperature: float | None = None
        prompt_title: str | None = None
        self.prompt_title = None  # Reset stored title

        # Process all user messages to remove injections and track latest parameters
        for message in messages:
            if message.get("role") == "user":
                message = cast("UserMessage", message)
                content = message.get("content", "")
                if isinstance(content, list):  # Handle messages with images and text
                    text_segments = []
                    non_text_content = []
                    # Separate text and non-text content
                    for item in content:
                        if item.get("type") == "text":
                            item = cast("TextContent", item)
                            text_segments.append(item["text"])
                        else:
                            non_text_content.append(item)

                    system_prompt_found = None
                    temperature_found = None
                    title_found = None
                    new_text_segments = []

                    # Process each text segment for injections
                    for ts in text_segments:
                        sp, temp, title, mod_ts, _ = self._extract_injection_params(ts)
                        if title is not None:  # Injection found in this segment
                            system_prompt_found = sp
                            temperature_found = temp
                            title_found = title
                        new_text_segments.append(mod_ts)

                    # Rebuild the content with processed text and original non-text parts
                    new_content = []
                    text_idx = 0
                    non_text_idx = 0
                    for item in content:
                        if item.get("type") == "text":
                            new_content.append(
                                {"type": "text", "text": new_text_segments[text_idx]}
                            )
                            text_idx += 1
                        else:
                            new_content.append(non_text_content[non_text_idx])
                            non_text_idx += 1

                    # Update message content
                    message["content"] = new_content

                    # Track parameters from this message
                    system_prompt = system_prompt_found
                    temperature = temperature_found
                    prompt_title = title_found

                else:  # Handle traditional text-only messages
                    system_prompt, temperature, title, modified_content, _ = (
                        self._extract_injection_params(content)
                    )
                    message["content"] = modified_content

                # Update latest parameters if injection was found in this message
                if prompt_title is not None:
                    latest_system_prompt = system_prompt
                    latest_temperature = temperature
                    self.prompt_title = prompt_title

        # Apply the latest parameters if any were found
        if latest_system_prompt is not None:
            if latest_system_prompt:  # Check if it's a non-empty string
                self._apply_system_prompt(body, latest_system_prompt)
            else:
                self.log.debug(
                    "Extracted empty system prompt; will not set system prompt."
                )

        if latest_temperature is not None:
            self._update_options(body, "temperature", latest_temperature)
            self.log.debug(f"Set temperature to: {latest_temperature}")

        self.log.debug(
            "Modified Request Body (before sending to LLM):",
            data=body,
        )

        return body

    async def outlet(
        self, body: "Body", __event_emitter__: Callable[["Event"], Awaitable[None]]
    ) -> "Body":
        """
        Processes the response body from the LLM.
        Adds the stored prompt title as a header to the latest assistant message.
        """
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
            # Reset prompt title after adding it to the response
            # self.prompt_title = None # Optional: Reset state if desired after use

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
    ) -> tuple[str | None, float | None, str | None, str, str | None]:

        system_prompt = None
        temperature = None
        prompt_title = None
        modified_content = user_message_content
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
                    # FIXME: Use log.exeption
                    self.log.error(f"JSON Parse Error: {e}")
                    return (None, None, prompt_title, modified_content, injection_block)

                system_prompt = json_data.get("system_prompt")
                temp_value = json_data.get("temperature")
                if isinstance(temp_value, (int, float)):
                    temperature = float(temp_value)
                else:
                    self.log.warning(f"Invalid temperature type: {type(temp_value)}")
            else:
                self.log.warning("No JSON block found in parameters section")

        return (
            system_prompt,
            temperature,
            prompt_title,
            modified_content,
            injection_block,
        )

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
