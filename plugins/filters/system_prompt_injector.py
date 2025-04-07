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
> system_prompt:{{system_prompt}}
> temperature:{{temperature}}
</details>
{{content}}
"""

# IMPORTANT: Disable "Rich Text Input for Chat" in Open WebUI settings for this plugin to work correctly.
# TODO: Keep the latest prompt template settings if no setting were given in the last message.

from pydantic import BaseModel
from typing import Any, Awaitable, Callable, TYPE_CHECKING
import json
import re

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.

# --- Constants ---
# Regex to find the entire details block and capture its components
# - Group 1: Full <details> block
# - Group 2: Content inside <summary> tag (prompt title)
# - Group 3: Content between </summary> and </details> (parameters)
DETAILS_BLOCK_REGEX = re.compile(
    r'(<details type="prompt">\s*<summary>(.*?)</summary>(.*?)^\s*</details>)',
    re.DOTALL | re.MULTILINE,
)
# Regex to find key-value pairs within the parameters block
PARAM_REGEX = re.compile(r"^\s*>\s*(\w+):\s*(.*)", re.MULTILINE)


class Filter:
    class Valves(BaseModel):
        # Pass configuration options here if needed in the future
        # Example: default_system_prompt: str = "Default prompt"
        pass

    def __init__(self):
        # Valves are not used in this version but kept for potential future config
        self.valves = self.Valves()
        # Store the prompt title extracted during inlet for use in outlet
        self.prompt_title: str | None = None
        self.debug = True  # Enable or disable debug prints

    def inlet(
        self, body: dict[str, Any], __user__: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Processes incoming request body.
        Detects and extracts prompt injection details from the latest user message.
        Applies extracted system prompt (if not empty) and temperature.
        Removes the injection block from the user message.
        """
        if self.debug:
            print(f"\n--- Inlet Filter ---")
            print(f"Original Request Body:\n{json.dumps(body, indent=2)}")

        latest_system_prompt: str | None = None
        latest_temperature: float | None = None
        injection_found_in_last_message = False

        messages: list[dict[str, Any]] = body.get("messages", [])
        if not messages:
            print(f"Warning: No messages found in the body.")
            return body

        # Process only the *last* message if it's from the user for injection
        last_message = messages[-1]
        if last_message.get("role") == "user":
            content = last_message.get("content", "")
            (
                extracted_system_prompt,
                extracted_temperature,
                extracted_title,
                modified_content,
                injection_block_content,  # Keep track of the raw block for debugging
            ) = self._extract_injection_params(content)

            if extracted_title is not None:  # Check if extraction was successful
                injection_found_in_last_message = True
                self.prompt_title = extracted_title  # Store for outlet
                last_message["content"] = modified_content  # Update message content

                if self.debug:
                    print(f"Injection block detected in the last user message.")
                    print(f"Extracted Title: {extracted_title}")
                    print(f"Raw Injection Block:\n{injection_block_content}")

                # Store extracted system prompt, including empty string
                if extracted_system_prompt is not None:
                    latest_system_prompt = extracted_system_prompt
                    if self.debug:
                        if latest_system_prompt == "":
                            print(
                                f"Extracted empty System Prompt. System prompt will NOT be set."
                            )
                        else:
                            print(f"Extracted System Prompt: {latest_system_prompt}")

                if extracted_temperature is not None:
                    latest_temperature = extracted_temperature
                    if self.debug:
                        print(f"Extracted Temperature: {latest_temperature}")

                if self.debug:
                    print(f"User message content updated (injection block removed).")
            else:
                if self.debug:
                    print(f"Injection block not detected in the last user message.")
                # Reset prompt title if no injection found in the current turn
                self.prompt_title = None
        else:
            # Reset prompt title if the last message isn't from the user
            self.prompt_title = None

        # Apply the extracted parameters to the request body
        if injection_found_in_last_message:
            # Only apply system prompt if it's not None AND not an empty string
            if (
                latest_system_prompt
            ):  # This evaluates to False if latest_system_prompt is None or ""
                self._apply_system_prompt(body, latest_system_prompt)
            elif latest_system_prompt == "" and self.debug:
                print(
                    "Skipping system prompt application because extracted value was empty."
                )

            if latest_temperature is not None:
                # FIXME: Groq models might break if temperature is set like this.
                # This is the standard way to set options for OpenAI-compatible APIs.
                # If Groq fails, it might require a different option key or not support it.
                self._update_options(body, "temperature", latest_temperature)
                if self.debug:
                    print(f"Set temperature to: {latest_temperature}")

        if self.debug:
            print(
                f"\nModified Request Body (before sending to LLM):\n{json.dumps(body, indent=2)}"
            )

        return body

    async def outlet(
        self,
        body: dict[str, Any],
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Processes the response body from the LLM.
        Adds the stored prompt title as a header to the latest assistant message.
        """
        if self.debug:
            print(f"\n--- Outlet Filter ---")
            print(f"Original Response Body:\n{json.dumps(body, indent=2)}")

        # Only add header if a prompt title was set during inlet
        if self.prompt_title:
            status_event: "StatusEvent" = {
                "type": "status",
                "data": {"description": self.prompt_title},
            }
            await __event_emitter__(status_event)
            # Reset prompt title after adding it to the response
            # self.prompt_title = None # Optional: Reset state if desired after use

        if self.debug:
            print(
                f"\nModified Response Body (after adding header):\n{json.dumps(body, indent=2)}"
            )

        return body

    """
    ---------- Helper methods inside the Pipe class ----------
    """

    def _update_options(self, body: dict[str, Any], key: str, value: Any):
        """Safely updates the 'options' dictionary in the request body."""
        if "options" not in body:
            body["options"] = {}
        body["options"][key] = value
        if self.debug:
            print(f"Updated options: set '{key}' to {value}")

    def _extract_injection_params(
        self, user_message_content: str
    ) -> tuple[str | None, float | None, str | None, str, str | None]:
        """
        Extracts parameters (system prompt, temperature, title) from the injection block
        using regex and removes the block from the content.

        Returns:
            - system_prompt (str | None): Extracted system prompt. None if not found. Can be empty string.
            - temperature (float | None): Extracted temperature. None if not found or invalid.
            - prompt_title (str | None): Extracted prompt title. None if block not found.
            - modified_content (str): Original content with the injection block removed.
            - injection_block (str | None): The raw content of the matched injection block for debugging.
        """
        system_prompt: str | None = None
        temperature: float | None = None
        prompt_title: str | None = None
        modified_content: str = user_message_content
        injection_block: str | None = None

        match = DETAILS_BLOCK_REGEX.search(user_message_content)

        if match:
            injection_block = match.group(
                1
            )  # Full matched block <details>...</details>
            prompt_title = match.group(2).strip()  # Content of <summary>
            params_block = match.group(3)  # Content between </summary> and </details>

            # Remove the injection block from the original content
            # Be careful with potential leading/trailing whitespace after removal
            if injection_block:  # Check if injection_block is not None
                modified_content = user_message_content.replace(
                    injection_block, ""
                ).strip()
            else:
                # This case should ideally not happen if match is successful, but added for safety
                print(f"Warning: injection_block is None after regex match.")

            # Extract key-value pairs from the parameters block
            for param_match in PARAM_REGEX.finditer(params_block):
                key = param_match.group(1).strip().lower()
                value = param_match.group(2).strip()

                if key == "system_prompt":
                    # Allow empty string as a valid system prompt value
                    system_prompt = value
                elif key == "temperature":
                    try:
                        temperature = float(value)
                    except ValueError:
                        print(
                            f"Warning: Invalid temperature value '{value}' found in injection block. Ignoring."
                        )

        return (
            system_prompt,
            temperature,
            prompt_title,
            modified_content,
            injection_block,
        )

    def _apply_system_prompt(self, body: dict[str, Any], system_prompt_content: str):
        """
        Applies a *non-empty* system prompt to the request body.
        Updates the existing system message or inserts a new one at the beginning.
        Also updates the 'system' key in the 'options' dictionary if present.
        """
        # This function should only be called with non-empty system_prompt_content
        # due to the check in the inlet method.
        if not system_prompt_content:
            if self.debug:
                print(
                    "Internal Warning: _apply_system_prompt called with empty content. This should not happen."
                )
            return  # Do nothing if called with empty string despite the check

        messages: list[dict[str, Any]] = body.get("messages", [])
        system_message_found = False

        # Iterate through messages to find and update the system message
        for message in messages:
            if message.get("role") == "system":
                message["content"] = system_prompt_content
                system_message_found = True
                if self.debug:
                    print(f"Updated existing system message.")
                break

        # If no system message exists, insert one at the beginning
        if not system_message_found:
            messages.insert(0, {"role": "system", "content": system_prompt_content})
            body["messages"] = messages  # Ensure the body reflects the change
            if self.debug:
                print(f"Inserted new system message at the beginning.")

        # Also update the 'system' option if it exists (some backends might use this)
        # We update this even if the message was pre-existing, to ensure consistency
        self._update_options(body, "system", system_prompt_content)
