"""
title: Advanced Prompt Injector
description: Filter that will detect and inject prompt configurations from user input.
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
from typing import Optional
import json


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.prompt_title = None

    def _update_options(self, body: dict, key: str, value):
        if "options" in body:
            body["options"][key] = value
            print(f"Options - {key.capitalize()} Updated/Added.")
        else:
            body["options"] = {key: value}
            print(f"Options field created and {key.capitalize()} Added.")

    # Helper function that will return message object without the prompt_title header (f"*{self.prompt_title}*\n\n***\n\n") if it exists.
    def _remove_prompt_title_header(self, message: dict) -> dict:
        if message["role"] == "assistant" and self.prompt_title:
            message["content"] = message["content"].replace(
                f"*{self.prompt_title}*\n\n***\n\n", ""
            )
        return message

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # This is where you manipulate user inputs.
        print("\n--- Inlet Filter ---")
        print("Original User Input Body:")
        print(json.dumps(body, indent=4))
        inject_start_tag = "**###INJECT_START###**\n"
        inject_end_tag = "**###INJECT_END###**\n* * *\n"
        latest_system_prompt = None
        latest_temperature = None
        modified_messages = []
        for message in body["messages"]:
            if message["role"] == "user":
                user_message_content = message["content"]
                if (
                    inject_start_tag in user_message_content
                    and inject_end_tag in user_message_content
                ):
                    print("\nInjection Tags Detected in User Message.")
                    start_index = user_message_content.find(inject_start_tag) + len(
                        inject_start_tag
                    )
                    end_index = user_message_content.find(inject_end_tag)
                    injection_block = user_message_content[
                        start_index:end_index
                    ].strip()
                    print("Injection Block Content:\n", injection_block)
                    system_prompt = None
                    temperature = None
                    for line in injection_block.strip().split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            key = key.strip().lower()
                            value = value.strip()
                            if key == "prompt_title":
                                self.prompt_title = value
                                print("Extracted Prompt Title:", self.prompt_title)
                            elif key == "system_prompt":
                                system_prompt = value
                                print("Extracted System Prompt:", system_prompt)
                            elif key == "temperature":
                                try:
                                    temperature = float(value)
                                    print("Extracted Temperature:", temperature)
                                except ValueError:
                                    print("Warning: Invalid temperature value:", value)
                    if system_prompt:
                        latest_system_prompt = system_prompt
                    if temperature is not None:
                        latest_temperature = temperature
                    # Remove tags and injection block from user message content
                    modified_user_message_content = (
                        user_message_content[
                            : user_message_content.find(inject_start_tag)
                        ].strip()
                        + user_message_content[
                            user_message_content.find(inject_end_tag)
                            + len(inject_end_tag) :
                        ].strip()
                    )
                    message["content"] = modified_user_message_content
                    print(
                        "User Message Body Updated (Tags and Injection Block Removed)."
                    )
                else:
                    print("\nInjection Tags Not Detected in User Message.")
                modified_messages.append(message)
            else:
                modified_messages.append(self._remove_prompt_title_header(message))
        body["messages"] = modified_messages
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
                print("System Prompt Message Added to 'messages'.")
            else:
                print("System Prompt Message Updated in 'messages'.")
            self._update_options(body, "system", latest_system_prompt)
        if latest_temperature is not None:
            self._update_options(body, "temperature", latest_temperature)
        # Reset the prompt_title value if latest_system_prompt and latest_temperature do not exist.
        if not latest_system_prompt and latest_temperature is None:
            self.prompt_title = None
        print("\nModified User Input Body:")
        print(json.dumps(body, indent=4))
        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:

        # TODO Reasoning model support

        print("\n--- Outlet Filter ---")
        print("Original User Output Body:")
        print(json.dumps(body, indent=4))

        for message in reversed(body["messages"]):
            if message["role"] == "assistant" and self.prompt_title:
                message["content"] = (
                    f"*{self.prompt_title}*\n\n***\n\n{message['content']}"
                )
                break

        print("\nModified User Output Body:")
        print(json.dumps(body, indent=4))

        return body
