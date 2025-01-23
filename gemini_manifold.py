"""
title: Gemini Manifold (google-genai)
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.1.0
"""

import re
from typing import Iterator, List, Union, Dict, Tuple, Optional
from google import genai
from google.genai import types, _api_client
from pydantic import BaseModel, Field

DEBUG = True


class Pipe:

    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="")
        MODEL_WHITELIST: str = Field(
            default="", description="Comma-separated list of allowed model names"
        )

    def __init__(self):
        try:
            self.valves = self.Valves()
            if DEBUG:
                print("[INIT] Initialized Pipe with Valves configuration.")
        except Exception as e:
            if DEBUG:
                print(f"[INIT] Error during initialization: {e}")
        finally:
            if DEBUG:
                print("[INIT] Initialization complete.")

    def __get_google_models(self):
        """Retrieve Google models with prefix stripping."""
        try:
            whitelist = (
                self.valves.MODEL_WHITELIST.split(",")
                if self.valves.MODEL_WHITELIST
                else []
            )
            models = self.client.models.list(config={"query_base": True})
            if DEBUG:
                print(
                    f"[get_google_models] Retrieved {len(models)} models from Gemini Developer API."
                )
            return [
                {
                    "id": self.__strip_prefix(model.name),
                    "name": model.display_name,
                }
                for model in models
                if not whitelist or model.name in [f"models/{w}" for w in whitelist]
                if model.supported_actions
                and "generateContent" in model.supported_actions
                if model.name and model.name.startswith("models/")
            ]
        except Exception as e:
            if DEBUG:
                print(f"[get_google_models] Error retrieving models: {e}")
            return [
                {
                    "id": "error",
                    "name": "Error retrieving models. Please check the logs.",
                }
            ]

    def __strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        This makes the method generic and adaptable to varying prefixes.
        """
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            if DEBUG:
                print(
                    f"[strip_prefix] Stripped prefix: '{stripped}' from '{model_name}'"
                )
            return stripped
        except Exception as e:
            if DEBUG:
                print(f"[strip_prefix] Error stripping prefix: {e}")
            return model_name  # Return original if stripping fails
        finally:
            if DEBUG:
                print("[strip_prefix] Completed prefix stripping.")

    def pipes(self) -> List[dict]:
        """Register all available Google models."""
        try:
            if not self.valves.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")
            # GEMINI_API_KEY is not available inside __init__ for whatever reason so we initialize the client here
            self.client = genai.Client(
                api_key=self.valves.GEMINI_API_KEY,
                http_options=_api_client.HttpOptions(api_version="v1alpha"),
            )
            models = self.__get_google_models()
            if DEBUG:
                print(f"[pipes] Registered models: {models}")
            return models
        except Exception as e:
            if DEBUG:
                print(f"[pipes] Error in pipes method: {e}")
            return []
        finally:
            if DEBUG:
                print("[pipes] Completed pipes method.")

    async def pipe(self, body: dict) -> Union[str, Iterator[str]]:

        def pop_system_prompt(messages: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
            """
            Extracts and removes the system message from a list of messages.

            Args:
                messages: A list of message dictionaries, where each dictionary has a 'role' and 'content' key.

            Returns:
                A tuple containing:
                    - The content of the system message (str) if found, otherwise None.
                    - The list of messages (List[Dict]) with the system message removed (if it existed).
            """
            system_message_content = None
            messages_without_system = []
            for message in messages:
                if message.get("role") == "system":
                    system_message_content = message.get("content")
                else:
                    messages_without_system.append(message)
            return system_message_content, messages_without_system

        def transform_messages_to_contents(messages: List[Dict]) -> List[types.Content]:
            """
            Transforms a list of messages into the 'contents' parameter structure for google-genai, for text-only conversations.

            Args:
                messages: A list of message dictionaries, where each dictionary has a 'role' and 'content' key.

            Returns:
                A list of types.Content objects, suitable for the 'contents' parameter in google-genai.
            """
            contents: List[types.Content] = []
            for message in messages:
                role = message.get("role")
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(item.get("text", "")))
                        # TODO: Add image handling logic here in the future
                        # elif item.get("type") == "image_url":
                        #     # Image handling logic
                    contents.append(types.Content(role=role, parts=parts))
                else:
                    contents.append(
                        types.Content(role=role, parts=[types.Part.from_text(content)])
                    )
            return contents

        """Main pipe method to process incoming requests."""

        messages = body.get("messages", [])

        # Assuming 'messages' is your original list of messages from body.get("messages", [])

        # 1. Extract the system prompt
        system_prompt, remaining_messages = pop_system_prompt(messages)

        # 2. Transform the remaining messages into 'contents'
        contents = transform_messages_to_contents(remaining_messages)

        # Now 'contents' is ready to be used with client.models.generate_content()

        last_user_message = next(
            (
                message["content"]
                for message in reversed(messages)
                if message["role"] == "user"
            ),
            None,
        )
        if DEBUG:
            print(f"[pipe] Received request: {body}")
            print(f"[pipe] Received user message: {last_user_message}")
            print(f"[pipe] System prompt: {system_prompt}")
            print(f"[pipe] Contents: {contents}")

        if last_user_message is None:
            return "Hello World!"

        return last_user_message
