"""
title: Gemini Manifold (google-genai)
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.5.0
"""

# NB! This is currently work in progress and not yet fully functional.

import json
import re
from typing import AsyncGenerator, AsyncIterator, List, Union, Dict, Tuple, Optional
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
            # FIXME We need better way to ensure that the client is initialized at all times.
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

    async def pipe(self, body: dict) -> Union[str, AsyncGenerator[str, None]]:

        def pop_system_prompt(messages: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
            """Extracts system message from messages."""
            system_message_content = None
            messages_without_system = []
            for message in messages:
                if message.get("role") == "system":
                    system_message_content = message.get("content")
                else:
                    messages_without_system.append(message)
            return system_message_content, messages_without_system

        def transform_messages_to_contents(
            messages: List[Dict],
        ) -> List[types.ContentUnion]:
            """Transforms messages to google-genai contents."""
            contents: List[types.ContentUnion] = []
            for message in messages:
                role = (
                    "model"
                    if message.get("role") == "assistant"
                    else message.get("role")
                )
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if item.get("type") == "text":
                            parts.append(types.Part.from_text(item.get("text", "")))
                    contents.append(types.Content(role=role, parts=parts))
                else:
                    contents.append(
                        types.Content(role=role, parts=[types.Part.from_text(content)])
                    )
            return contents

        async def generate_content_stream_text(
            self, model, contents, config
        ) -> AsyncGenerator[str, None]:
            """Wraps generate_content_stream for text only, wrapping thoughts in <think> tags.
            Defaults to the first candidate if multiple candidates are present and prints a warning in DEBUG mode.
            """
            response_stream: AsyncIterator[types.GenerateContentResponse] = (
                self.client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
            )
            is_thinking = False
            async for response in response_stream:
                if response.candidates:
                    if len(response.candidates) > 1:
                        if DEBUG:
                            print(
                                "WARNING: Multiple candidates found in response, defaulting to the first candidate."
                            )
                    candidate = response.candidates[0]  # Default to the first candidate
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            text_part = getattr(
                                part, "text", None
                            )  # use getattr to avoid AttributeError if 'text' does not exist
                            thought_part = getattr(
                                part, "thought", False
                            )  # use getattr to avoid AttributeError if 'thought' does not exist, default to False

                            if thought_part:
                                if not is_thinking:
                                    yield "<think>"
                                    yield "\n"
                                    is_thinking = True
                                if text_part is not None:
                                    yield text_part
                            else:
                                if is_thinking:
                                    # Interesting note: yielding "</think>\n" all at once will mess up the formatting.
                                    yield "</think>"
                                    yield "\n"
                                    is_thinking = False
                                if text_part is not None:
                                    yield text_part
            if (
                is_thinking
            ):  # in case the stream ends while still in a thinking block, close tag
                yield "</think>\n"

        def extract_thoughts(response: types.GenerateContentResponse) -> str:
            """Extracts and concatenates thought parts from response."""
            if (
                not response.candidates
                or not response.candidates[0].content
                or not response.candidates[0].content.parts
            ):
                return ""
            if len(response.candidates) > 1:
                print(f"More than one candidate found, using thoughts from the first.")
            thoughts = ""
            for part in response.candidates[0].content.parts:
                if (
                    isinstance(part.text, str)
                    and isinstance(part.thought, bool)
                    and part.thought
                ):
                    thoughts += part.text
            return thoughts

        """Main pipe method."""

        # TODO When stream_options: { include_usage: true } is enabled, the response will contain usage information.
        # TODO Image support.

        messages = body.get("messages", [])
        system_prompt, remaining_messages = pop_system_prompt(messages)
        contents = transform_messages_to_contents(remaining_messages)

        if DEBUG:
            print(f"[pipe] Received request:")
            print(json.dumps(body, indent=4))
            print(f"[pipe] System prompt: {system_prompt}")
            print(f"[pipe] Contents: [")
            for content in enumerate(contents):
                print(f"    {content},")
            print("]")

        model_name = self.__strip_prefix(body.get("model", ""))
        is_thinking_model = "thinking" in model_name.lower()

        config_params = {
            "system_instruction": system_prompt,
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 40),
            "max_output_tokens": body.get("max_tokens", 8192),
            "stop_sequences": body.get("stop", []),
        }
        if is_thinking_model:
            config_params["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True
            )
        config = types.GenerateContentConfig(**config_params)

        gen_content_args = {
            "model": model_name,
            "contents": contents,
            "config": config,
        }

        try:
            if body.get("stream", False):
                if DEBUG:
                    print("[pipe] Streaming enabled.")
                response_generator = generate_content_stream_text(
                    self, **gen_content_args
                )
                return response_generator
            else:  # streaming is disabled
                if DEBUG:
                    print("[pipe] Streaming disabled.")
                response = self.client.models.generate_content(**gen_content_args)
                response_text = response.text if response.text else "No response text."

                if is_thinking_model:
                    # TODO Handle formatting of thoughts in non-streaming mode.
                    thoughts = extract_thoughts(response)
                    tagged_thoughts = (
                        f"<think>\n{thoughts}</think>\n" if thoughts else ""
                    )
                    final_response = tagged_thoughts + response_text
                    if DEBUG:
                        print(
                            f"[pipe] Currently formatting does not work when streaming is off. See https://github.com/open-webui/open-webui/discussions/8936 for more."
                        )
                    return final_response
                else:  # handle regular response without thoughts.
                    return response_text

        except Exception as e:
            error_message = f"Content generation error: {str(e)}"
            if DEBUG:
                print(f"[pipe] {error_message}")
            return error_message
        finally:
            if DEBUG:
                print("[pipe] Content generation completed.")
