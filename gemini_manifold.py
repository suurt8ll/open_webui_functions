"""
title: Gemini Manifold google_genai
description: Manifold function for Gemini Developer API. Uses google-genai, supports thinking models and streaming.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API. Complete with thinking support.
# Open WebUI v0.5.5 or greater is required for this function to work properly.
# NB! google-genai==0.7.0 must be installed in the Open WebUI environment. Currently it's not by default.
# Be sure to check out my GitHub repository for more information! Feel free to contribute and post questions.


import base64
import json
import re
import fnmatch
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    List,
    Union,
    Dict,
    Tuple,
    Optional,
)
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

DEBUG = False  # Set to True to enable debug output. Use `docker logs -f open-webui` to view logs.


class Pipe:

    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="")
        MODEL_WHITELIST: str = Field(
            default="",
            description="Comma-separated list of allowed model names. Supports wildcards (*).",
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
                else ["*"]
            )
            models = self.client.models.list(config={"query_base": True})
            if DEBUG:
                print(
                    f"[get_google_models] Retrieved {len(models)} models from Gemini Developer API."
                )
            model_list = [
                {
                    "id": self.__strip_prefix(model.name),
                    "name": model.display_name,
                }
                for model in models
                if model.name
                and any(fnmatch.fnmatch(model.name, f"models/{w}") for w in whitelist)
                if model.supported_actions
                and "generateContent" in model.supported_actions
                if model.name and model.name.startswith("models/")
            ]
            if not model_list:
                if DEBUG:
                    print("[get_google_models] No models found matching whitelist.")
                return [
                    {
                        "id": "no_models_found",
                        "name": "No models found matching whitelist.",
                    }
                ]
            return model_list
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
            # Version Check
            import google.genai

            installed_version = google.genai.__version__
            required_version = "0.7.0"

            if required_version:
                if installed_version != required_version:
                    error_message = f"google-genai version mismatch. Required: {required_version}, Installed: {installed_version}"
                    if DEBUG:
                        print(f"[pipes] {error_message}")
                    return [{"id": "version_error", "name": error_message}]

            # GEMINI_API_KEY is not available inside **init** for whatever reason so we initialize the client here
            # FIXME We need better way to ensure that the client is initialized at all times.
            self.client = genai.Client(
                api_key=self.valves.GEMINI_API_KEY,
                http_options=types.HttpOptions(api_version="v1alpha"),
            )
            models = self.__get_google_models()
            if models and models[0].get("id") == "error":
                return models  # Propagate error model from __get_google_models
            if models and models[0].get("id") == "no_models_found":
                return models  # Propagate no_models_found model
            if DEBUG:
                print(f"[pipes] Registered models: {models}")
            return (
                models
                if models
                else [{"id": "no_models", "name": "No models available"}]
            )
        except Exception as e:
            if DEBUG:
                print(f"[pipes] Error in pipes method: {e}")
            return [{"id": "error", "name": f"Error initializing models: {e}"}]
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

        def get_mime_type(file_uri: str) -> str:
            """
            Utility function to determine MIME type based on file extension.
            Extend this function to support more MIME types as needed.
            """
            if file_uri.endswith(".jpg") or file_uri.endswith(".jpeg"):
                return "image/jpeg"
            elif file_uri.endswith(".png"):
                return "image/png"
            elif file_uri.endswith(".gif"):
                return "image/gif"
            elif file_uri.endswith(".bmp"):
                return "image/bmp"
            elif file_uri.endswith(".webp"):
                return "image/webp"
            else:
                # Default MIME type if unknown
                return "application/octet-stream"

        def transform_messages_to_contents(
            messages: List[Dict],
        ) -> List[types.Content]:
            """Transforms messages to google-genai contents, supporting both text and images."""
            contents: List[types.Content] = []
            for message in messages:
                # Determine the role: "model" if assistant, else the provided role
                role = (
                    "model"
                    if message.get("role") == "assistant"
                    else message.get("role")
                )
                content = message.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        item_type = item.get("type")
                        if item_type == "text":
                            # Handle text parts
                            text = item.get("text", "")
                            part = types.Part.from_text(text=text)
                            parts.append(part)
                        elif item_type == "image_url":
                            # Handle image parts
                            image_url_dict = item.get("image_url", {})
                            image_url = image_url_dict.get(
                                "url", ""
                            )  # Safely get url, default to empty string if not found or image_url is not a dict

                            if isinstance(
                                image_url, str
                            ):  # Ensure image_url is a string before calling startswith
                                if image_url.startswith("gs://"):
                                    # Image is stored in Google Cloud Storage
                                    part = types.Part.from_uri(
                                        file_uri=image_url,
                                        mime_type=get_mime_type(image_url),
                                    )
                                    parts.append(part)
                                elif image_url.startswith("data:image"):
                                    # Image is embedded as a data URI (Base64)
                                    try:
                                        # Extract the Base64 data and MIME type using regex
                                        match = re.match(
                                            r"data:(image/\w+);base64,(.+)", image_url
                                        )
                                        if match:
                                            mime_type = match.group(1)
                                            base64_data = match.group(2)
                                            part = types.Part.from_bytes(
                                                data=base64.b64decode(base64_data),
                                                mime_type=mime_type,
                                            )
                                            parts.append(part)
                                        else:
                                            # Invalid data URI format
                                            raise ValueError(
                                                "Invalid data URI for image."
                                            )
                                    except Exception as e:
                                        # Handle invalid image data
                                        raise ValueError(
                                            f"Error processing image data: {e}"
                                        )
                                else:
                                    # Assume it's a standard URL (HTTP/HTTPS)
                                    part = types.Part.from_uri(
                                        file_uri=image_url,
                                        mime_type=get_mime_type(image_url),
                                    )
                                    parts.append(part)
                            else:
                                print(
                                    f"Warning: Unexpected image_url format: {image_url_dict}. Skipping image part."
                                )  # Handle case where image_url is not a string
                                continue  # Skip this image part and continue to the next item
                        else:
                            # Handle other types if necessary
                            # For now, ignore unsupported types
                            continue
                    contents.append(types.Content(role=role, parts=parts))
                else:
                    # Handle content that is a simple string (text only)
                    contents.append(
                        types.Content(
                            role=role, parts=[types.Part.from_text(text=content)]
                        )
                    )
            return contents

        async def generate_content_stream_text(
            self, model, contents, config
        ) -> AsyncGenerator[str, None]:
            """Wraps generate_content_stream for text only, wrapping thoughts in <think> tags.
            Defaults to the first candidate if multiple candidates are present and prints a warning in DEBUG mode.
            """
            response_stream: Awaitable[AsyncIterator[types.GenerateContentResponse]] = (
                self.client.aio.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
            )
            is_thinking = False
            async for response in await response_stream:
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
        if model_name in ["no_models_found", "error", "version_error", "no_models"]:
            return f"Error: {model_name.replace('_', ' ')}"
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
