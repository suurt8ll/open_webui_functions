"""
title: Advanced Prompt Injector
description: Filter that will detect and inject prompt configurations from user input.
id: system_prompt_injector
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.6.0
"""

# The injection must follow this format (without triple quotes).
"""
**###INJECT_START###**

prompt_title:{{prompt_title}}

system_prompt:{{system_prompt}}

temperature:{{temperature}}

**###INJECT_END###**

***

{{content}}
"""

from pydantic import BaseModel
import json

DEBUG = True


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.prompt_title = None

    def _update_options(self, body: dict, key: str, value):
        if "options" in body:
            body["options"][key] = value
        else:
            body["options"] = {key: value}

    def _remove_prompt_title_header(self, message: dict) -> dict:
        if message["role"] == "assistant" and self.prompt_title:
            header = f"*{self.prompt_title}*\n\n***\n\n"
            message["content"] = message["content"].replace(
                header, ""
            )  # replace all occurrences
        return message

    def _extract_injection_params(
        self, user_message_content: str
    ) -> tuple[str | None, float | None, str | None, str]:
        inject_start_tag = "**###INJECT_START###**\n"
        inject_end_tag = "**###INJECT_END###**\n* * *\n"
        system_prompt = None
        temperature = None
        injection_block = None
        modified_user_message_content = user_message_content

        if (
            inject_start_tag in user_message_content
            and inject_end_tag in user_message_content
        ):
            start_index = user_message_content.find(inject_start_tag) + len(
                inject_start_tag
            )
            end_index = user_message_content.find(inject_end_tag)
            injection_block = user_message_content[start_index:end_index].strip()

            for line in injection_block.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if key == "prompt_title":
                        self.prompt_title = value
                    elif key == "system_prompt":
                        system_prompt = value
                    elif key == "temperature":
                        try:
                            temperature = float(value)
                        except ValueError:
                            print(
                                f"Warning: Invalid temperature value: {value}"
                            )  # Keep warning but avoid print in core logic for modularity

            modified_user_message_content = (
                user_message_content[
                    : start_index - len(inject_start_tag)
                ].strip()  # Correctly remove tags
                + user_message_content[
                    end_index + len(inject_end_tag) :
                ].strip()  # Correctly remove tags
            )

        return (
            system_prompt,
            temperature,
            injection_block,
            modified_user_message_content,
        )

    def _apply_system_prompt(self, body: dict, latest_system_prompt: str | None):
        if latest_system_prompt:
            system_message_exists = False
            for message in body["messages"]:
                if message["role"] == "system":
                    message["content"] = latest_system_prompt
                    system_message_exists = True
                    break
            if not system_message_exists:
                body["messages"].insert(
                    0, {"role": "system", "content": latest_system_prompt}
                )
            self._update_options(body, "system", latest_system_prompt)

    def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        if DEBUG:
            print("\n--- Inlet Filter ---")
            print("Original User Input Body:")
            print(json.dumps(body, indent=4))

        latest_system_prompt = None
        latest_temperature = None
        modified_messages = []

        # Loop throught messages to extract injection tags and remove them.
        # Also removes the prompt title headers from assistant messages.
        for message in body["messages"]:
            if message["role"] == "user":
                system_prompt, temperature, injection_block, modified_content = (
                    self._extract_injection_params(message["content"])
                )
                message["content"] = modified_content  # Update user message content

                if injection_block:  # Only update if injection was found
                    if DEBUG:
                        print("\nInjection Tags Detected in User Message.")
                        print("Injection Block Content:\n", injection_block)
                    if system_prompt:
                        latest_system_prompt = system_prompt
                    if temperature is not None:
                        latest_temperature = temperature
                    if DEBUG:
                        print(
                            "User Message Body Updated (Tags and Injection Block Removed)."
                        )
                else:
                    if DEBUG:
                        print("\nInjection Tags Not Detected in User Message.")

            modified_messages.append(self._remove_prompt_title_header(message))

        body["messages"] = modified_messages

        # TODO If the system prompt exists but is empty string, then start using the default system prompt that is set in the frontend.
        self._apply_system_prompt(body, latest_system_prompt)
        # FIXME Groq models break if temperature is set like this.
        if latest_temperature is not None:
            self._update_options(body, "temperature", latest_temperature)
        if not latest_system_prompt and latest_temperature is None:
            self.prompt_title = None

        if DEBUG:
            print("\nModified User Input Body:")
            print(json.dumps(body, indent=4))

        return body

    def outlet(self, body: dict, __user__: dict | None = None) -> dict:

        # TODO Reasoning model support, put the prompt title before the collapsible reasoning content.

        # Add prompt title to the latest assistant message, indicating which preset was used in the response.
        for message in reversed(body["messages"]):
            if message["role"] == "assistant" and self.prompt_title:
                message["content"] = (
                    f"*{self.prompt_title}*\n\n***\n\n{message['content']}"
                )
                break
        return body
