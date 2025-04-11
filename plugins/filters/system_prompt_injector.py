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
# Temperature is not the only adjustable parameter, many others can be changed.
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
# See https://github.com/open-webui/open-webui/issues/9759 for more.

import json
import re
from pydantic import BaseModel
from typing import Any, Awaitable, Callable, cast, TYPE_CHECKING

from open_webui.models.functions import Functions


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


class Filter:

    class Valves(BaseModel):
        pass

    def __init__(self):
        valves = Functions.get_function_valves_by_id("gemini_manifold_google_genai")
        self.valves = self.Valves(**(valves if valves else {}))
        # Store the prompt title extracted during inlet for use in outlet
        self.prompt_title: str | None = None
        print("Function has been initialized.")

    def inlet(self, body: "Body") -> "Body":
        print(f"\n--- Inlet Filter ---")
        print("Original Request Body:")
        # Use json.dumps for potentially large/complex body structure
        try:
            print(json.dumps(body, indent=2, default=str))
        except Exception:
            print(body)  # Fallback if json serialization fails

        messages: list["Message"] = body.get("messages", [])
        if not messages:
            print("Warning: No messages found in the body.")
            return body

        latest_system_prompt, latest_options, prompt_title = None, None, None

        # Process each user message to extract parameters
        for idx, message in enumerate(messages):
            if message.get("role") == "user":
                message = cast("UserMessage", message)
                processed_message, (sp, opt, title) = self._process_user_message(
                    message
                )
                messages[idx] = processed_message  # Update message in-place

                # Track parameters only when title is found
                if title is not None:
                    latest_system_prompt = sp
                    latest_options = opt
                    prompt_title = title
        print(f"\n{latest_system_prompt=}\n{latest_options=}\n{prompt_title=}\n")

        # Apply extracted parameters
        self._handle_system_prompt(body, latest_system_prompt)
        if latest_options:
            self._handle_options(body, latest_options)  # type: ignore
        # Display title only when params where changed.
        if latest_options or latest_system_prompt:
            self.prompt_title = prompt_title  # Store title for later use
        else:
            self.prompt_title = None

        print("Modified Request Body (before sending to LLM):")
        try:
            print(json.dumps(body, indent=2, default=str))
        except Exception:
            print(body)  # Fallback
        return body

    async def outlet(
        self, body: "Body", __event_emitter__: Callable[["Event"], Awaitable[None]]
    ) -> "Body":

        print(f"\n--- Outlet Filter ---")
        print("Response Body:")
        try:
            print(json.dumps(body, indent=2, default=str))
        except Exception:
            print(body)  # Fallback

        # Only add header if a prompt title was set during inlet
        if self.prompt_title:
            status_event: "StatusEvent" = {
                "type": "status",
                "data": {"description": self.prompt_title},
            }
            await __event_emitter__(status_event)

        return body

    # region ----- Helper methods inside the Pipe class -----

    def _process_user_message(
        self, user_message: "UserMessage"
    ) -> tuple["UserMessage", tuple[str | None, dict[str, Any] | None, str | None]]:
        content = user_message.get("content", "")

        if isinstance(content, list):  # Handle mixed content (images + text)
            non_text_content = [item for item in content if item.get("type") != "text"]
            text_segments = [
                cast("TextContent", item)["text"]
                for item in content
                if item.get("type") == "text"
            ]

            new_text_segments = []
            system_prompt, title, options = None, None, None
            for ts in text_segments:
                sp, ti, mod_ts, opt = self._extract_injection_params(ts)
                new_text_segments.append(mod_ts)
                if ti is not None:
                    system_prompt = sp
                    options = opt
                    title = ti

            # Rebuild content structure
            combined_text = "".join(new_text_segments) if new_text_segments else None
            new_content = non_text_content.copy()
            if combined_text:
                new_content.append({"type": "text", "text": combined_text})
            user_message["content"] = new_content

        else:  # Handle traditional text-only content
            system_prompt, title, modified_content, options = (
                self._extract_injection_params(content)
            )
            user_message["content"] = modified_content

        return user_message, (system_prompt, options, title)

    def _extract_injection_params(
        self, user_message_content: str
    ) -> tuple[str | None, str | None, str, dict[str, Any] | None]:

        system_prompt = None
        prompt_title = None
        modified_content = user_message_content
        json_data: dict[str, Any] | None = None
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
                    print(f"JSON Parse Error: {e}")  # Print exception directly
                if not json_data:
                    return (None, prompt_title, modified_content, None)

                system_prompt = json_data.pop("system", None)
            else:
                print("Warning: No JSON block found in parameters section")

        # Remove keys that have value None.
        if json_data:
            for k, v in json_data.copy().items():
                if v is None:
                    json_data.pop(k)

        return (system_prompt, prompt_title, modified_content, json_data)

    def _is_ollama_model(self, body: "Body") -> bool:
        """Checks if the model specified in the body is owned by Ollama."""
        return body.get("metadata", {}).get("model", {}).get("owned_by") == "ollama"

    def _handle_options(self, body: "Body", options: dict[str, Any]):
        """
        Applies options to the request body.

        Updates top-level keys in the body. If the model is owned by Ollama,
        it also updates the nested 'options' dictionary within the body.
        Falsy values in the input 'options' lead to key removal.
        """
        is_ollama = self._is_ollama_model(body)
        body_options: dict[str, Any] | None = None

        if is_ollama:
            body_options = body.setdefault("options", {})

        for key, value in options.items():
            if value:
                body[key] = value
                if is_ollama and body_options is not None:
                    body_options[key] = value
                print(f"Set option '{key}' to: {value}")
            else:
                body.pop(key, None)
                if is_ollama and body_options is not None:
                    body_options.pop(key, None)
                print(f"Removed option '{key}' due to falsy value.")

    def _handle_system_prompt(self, body: "Body", system_prompt: str | None) -> None:
        """
        Adds, updates, or removes the system prompt in the body's 'messages' list.

        - If system_prompt is a non-empty string: Adds or updates the system message.
        - If system_prompt is an empty string (""): Removes the system message.
        - If system_prompt is None: Does nothing to the messages list.

        If the model is owned by Ollama, it also updates/removes the 'system' key
        in the nested 'options' dictionary accordingly.
        """
        is_ollama = self._is_ollama_model(body)
        messages: list[Message] = body.setdefault("messages", [])

        system_message_index = -1
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                system_message_index = i
                break

        body_options: dict[str, Any] | None = None
        if is_ollama:
            if system_prompt is not None:
                body_options = body.setdefault("options", {})

        if system_prompt is None:
            print("No system prompt provided; leaving existing messages unchanged.")

        elif system_prompt == "":
            print("Empty system prompt provided; removing existing system message.")
            if system_message_index != -1:
                messages.pop(system_message_index)
                print("Removed system message from 'messages' list.")
            if is_ollama and body_options is not None:
                body_options.pop("system", None)
                print("Removed 'system' key from Ollama options.")

        else:
            print(f"Applying system prompt: '{system_prompt[:50]}...'")
            system_message: Message = {"role": "system", "content": system_prompt}

            if system_message_index != -1:
                messages[system_message_index] = system_message
                print("Updated existing system message in 'messages' list.")
            else:
                messages.insert(0, system_message)
                print(
                    "Inserted new system message at the beginning of 'messages' list."
                )

            if is_ollama and body_options is not None:
                body_options["system"] = system_prompt
                print("Set 'system' key in Ollama options.")

    # endregion
