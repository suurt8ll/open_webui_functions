"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Supports native image generation, grounding with Google Search and streaming. Uses google-genai.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.4.0
requirements: google-genai==1.6.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Be sure to check out my GitHub repository for more information! Feel free to contribute and post questions/suggestions.

# Supported features:
#   - Native image generation (image output)
#   - Image input
#   - Streaming
#   - Grounding with Google Search

# Features that are supported by API but not yet implemented in the manifold:
#   TODO Audio input support.
#   TODO Video input support.
#   TODO PDF (other documents?) input support, __files__ param that is passed to the pipe() func can be used for this.
#   TODO Display usage statistics (token counts)
#   TODO Display citations in the front-end.

# Other things to do:
#   TODO Better type checking.
#   TODO Return errors as correctly formatted error types for the frontend to handle (red text in the front-end).
#   TODO Refactor, make this mess more readable lol.

import base64
import inspect
import json
import re
import fnmatch
import traceback
from pydantic import BaseModel, Field
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
    Literal,
    Tuple,
    Optional,
)
from starlette.responses import StreamingResponse
from google import genai
from google.genai import types

COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "RESET": "\033[0m",
}

# according to https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini
ALLOWED_GROUNDING_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
]


class Pipe:
    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="")
        MODEL_WHITELIST: str = Field(
            default="",
            description="Comma-separated list of allowed model names. Supports wildcards (*).",
        )
        USE_GROUNDING_SEARCH: bool = Field(
            default=False,
            description="Whether to use Grounding with Google Search. For more info: https://ai.google.dev/gemini-api/docs/grounding",
        )
        # FIXME Show this only when USE_GROUNDING_SEARCH is True
        GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD: float = Field(
            default=0.3,
            description="See Google AI docs for more information. Only supported for 1.0 and 1.5 models",
        )
        LOG_LEVEL: Literal["INFO", "WARNING", "ERROR", "DEBUG", "OFF"] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )
        IMAGE_OUTPUT: bool = Field(
            default=False,
            description="Enable image output for gemini-2.0-flash-exp.  If False, only text is returned.",
        )

    def __init__(self):
        try:
            self.valves = self.Valves()
            self._print_colored(f"Valves: {self.valves}", "DEBUG")
            self.models = []
            self.client = None
        except Exception as e:
            self._print_colored(f"Error during initialization: {e}", "ERROR")
        finally:
            self._print_colored("Initialization complete.", "INFO")

    # FIXME Make it async
    def pipes(self) -> list[dict]:
        """Register all available Google models."""
        try:
            if not self.valves.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")

            # GEMINI_API_KEY is not available inside __init__ for whatever reason so we initialize the client here.
            # TODO Allow user to choose if they want to fetch models only during function initialization or every time pipes is called.
            if not self.client:
                self.client = genai.Client(
                    api_key=self.valves.GEMINI_API_KEY,
                )
            else:
                self._print_colored("Client already initialized.", "INFO")

            # Return existing models if already initialized
            if self.models:
                self._print_colored("Models already initialized.", "INFO")
                return self.models

            # Get and process new models
            models = self._get_google_models()

            # Handle error cases
            if models and models[0].get("id") in ["error", "no_models_found"]:
                return models

            self._print_colored(f"Registered models: {models}", "DEBUG")
            self.models = models
            return (
                models
                if models
                else [{"id": "no_models", "name": "No models available"}]
            )

        except Exception as e:
            error_msg = f"Error in pipes method: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return [{"id": "error", "name": f"Error initializing models: {e}"}]
        finally:
            self._print_colored("Completed pipes method.", "DEBUG")

    async def pipe(
        self, body: dict
    ) -> (
        str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator
    ):
        """Helper functions inside the pipe() method"""

        def _pop_system_prompt(
            messages: list[dict],
        ) -> Tuple[Optional[str], list[dict]]:
            """Extracts system message from messages."""
            system_message_content = None
            messages_without_system = []
            for message in messages:
                if message.get("role") == "system":
                    system_message_content = message.get("content")
                else:
                    messages_without_system.append(message)
            return system_message_content, messages_without_system

        def _get_mime_type(file_uri: str) -> str:
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

        def _extract_markdown_image(text: str) -> list[types.Part]:
            """Extracts and converts markdown images to parts, preserving text order."""
            parts = []
            last_pos = 0
            for match in re.finditer(
                r"\n*\s*!\[([^\]]*)\]\((data:image/([^;]+);base64,([^)]+))\)\s*\n*",
                text,
            ):
                # Add text before the image
                text_segment = text[last_pos : match.start()]
                if text_segment.strip():
                    parts.append(types.Part.from_text(text=text_segment))

                # Add image part
                try:
                    image_part = types.Part.from_bytes(
                        data=base64.b64decode(match.group(4)),
                        mime_type="image/" + match.group(3),
                    )
                    parts.append(image_part)
                except Exception as e:
                    print(f"Error decoding base64 image: {e}")
                    # Optionally, add alt text as a text part or handle the error

                last_pos = match.end()

            # Add remaining text
            remaining_text = text[last_pos:]
            if remaining_text.strip():
                parts.append(types.Part.from_text(text=remaining_text))
            return parts

        def _process_image_url(image_url: str) -> Optional[types.Part]:
            """Processes an image URL, handling GCS, data URIs, and standard URLs."""
            try:
                if image_url.startswith("gs://"):
                    return types.Part.from_uri(
                        file_uri=image_url, mime_type=_get_mime_type(image_url)
                    )
                elif image_url.startswith("data:image"):
                    match = re.match(r"data:(image/\w+);base64,(.+)", image_url)
                    if match:
                        return types.Part.from_bytes(
                            data=base64.b64decode(match.group(2)),
                            mime_type=match.group(1),
                        )
                    else:
                        raise ValueError("Invalid data URI for image.")
                else:  # Assume standard URL
                    return types.Part.from_uri(
                        file_uri=image_url, mime_type=_get_mime_type(image_url)
                    )
            except Exception as e:
                print(f"Error processing image URL '{image_url}': {e}")
                return None  # Return None to indicate failure

        def _process_content_item(item: dict) -> list[types.Part]:
            """Processes a single content item, handling text and image_url types."""
            item_type = item.get("type")
            parts = []

            if item_type == "text":
                text = item.get("text", "")
                parts.extend(
                    _extract_markdown_image(text)
                )  # Always process for markdown images
            elif item_type == "image_url":
                image_url_dict = item.get("image_url", {})
                image_url = image_url_dict.get("url", "")
                if isinstance(image_url, str):
                    part = _process_image_url(image_url)
                    if part:
                        parts.append(part)
                else:
                    print(
                        f"Warning: Unexpected image_url format: {image_url_dict}. Skipping."
                    )
            else:
                print(f"Warning: Unsupported item type: {item_type}. Skipping.")

            return parts

        def _transform_messages_to_contents(
            messages: list[dict],
        ) -> list[types.Content]:
            """Transforms messages to google-genai contents, supporting text and images."""
            if not genai or not types:
                raise ValueError(
                    "google-genai is not installed. Please install it to proceed."
                )

            contents: list[types.Content] = []

            for message in messages:
                role = (
                    "model"
                    if message.get("role") == "assistant"
                    else message.get("role")
                )
                content = message.get("content", "")
                parts = []

                if isinstance(content, list):
                    for item in content:
                        parts.extend(_process_content_item(item))
                else:  # Treat as plain text
                    parts.extend(
                        _extract_markdown_image(content)
                    )  # process markdown images
                    if not parts:  # if there were no markdown images, add the text
                        parts.append(types.Part.from_text(text=content))

                contents.append(types.Content(role=role, parts=parts))

            return contents

        async def _process_stream(
            self, gen_content_args: dict[str, Any]
        ) -> AsyncGenerator[str, None]:
            """Helper function to process the stream and yield text chunks.

            Args:
                gen_content_args: The arguments to pass to generate_content_stream.

            Yields:
                str: Text chunks from the response.
            """
            response_stream: AsyncIterator[types.GenerateContentResponse] = (
                await self.client.aio.models.generate_content_stream(**gen_content_args)
            )

            async for chunk in response_stream:
                if self.valves.LOG_LEVEL == "DEBUG":
                    print(chunk.text, end="")
                if chunk.candidates:
                    if len(chunk.candidates) > 1:
                        self._print_colored(
                            "Multiple candidates found in response, defaulting to the first candidate.",
                            "WARNING",
                        )
                    candidate = chunk.candidates[0]  # Default to the first candidate
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            text_part = getattr(
                                part, "text", None
                            )  # Safely get the text part
                            if text_part is not None:
                                yield text_part
                                continue  # Skip to the next part if it's text
                            # --- Image Handling Logic ---
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data is not None:
                                mime_type = inline_data.mime_type
                                image_data = base64.b64encode(inline_data.data).decode(
                                    "utf-8"
                                )
                                # TODO Maybe use Open WebUI's Files API here? Injecting base64 strainght into the chat history could make things slow.
                                markdown_image = f"\n\n![image](data:{mime_type};base64,{image_data})\n\n"
                                yield markdown_image

        """Main pipe method."""

        messages = body.get("messages", [])
        system_prompt, remaining_messages = _pop_system_prompt(messages)
        contents = _transform_messages_to_contents(remaining_messages)

        max_len = 50
        self._print_colored("Received request:", "DEBUG")
        if self.valves.LOG_LEVEL == "DEBUG":
            truncated_body = self.truncate_long_strings(body.copy(), max_len)
            print(json.dumps(truncated_body, indent=2))
        self._print_colored(f"System prompt: {system_prompt}", "DEBUG")
        self._print_colored("Contents:", "DEBUG")
        if self.valves.LOG_LEVEL == "DEBUG":
            for content in contents:
                truncated_content = self.truncate_long_strings(
                    content.model_dump().copy(), max_len
                )
                print(json.dumps(truncated_content, indent=2))

        model_name = self._strip_prefix(body.get("model", ""))
        if model_name in [
            "no_models_found",
            "error",
            "version_error",
            "no_models",
            "import_error",
        ]:
            return f"Error: {model_name.replace('_', ' ')}"
        self._print_colored(f"Model name: {model_name}", "DEBUG")

        config_params = {
            "system_instruction": system_prompt,
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 40),
            "max_output_tokens": body.get("max_tokens", 8192),
            "stop_sequences": body.get("stop", []),
        }

        if "gemini-2.0-flash-exp" in model_name and self.valves.IMAGE_OUTPUT:
            config_params["response_modalities"] = ["Text", "Image"]
        else:
            config_params["response_modalities"] = ["Text"]

        if self.valves.USE_GROUNDING_SEARCH:
            if model_name in ALLOWED_GROUNDING_MODELS:
                print("[pipe] Using grounding search.")
                gs = None
                # Dynamic retrieval only supported for 1.0 and 1.5 models
                if "1.0" in model_name or "1.5" in model_name:
                    gs = types.GoogleSearchRetrieval(
                        dynamic_retrieval_config=types.DynamicRetrievalConfig(
                            dynamic_threshold=self.valves.GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD
                        )
                    )
                    config_params["tools"] = [types.Tool(google_search_retrieval=gs)]
                else:
                    gs = types.GoogleSearchRetrieval()
                    config_params["tools"] = [
                        types.Tool(google_search=types.GoogleSearch())
                    ]
            else:
                print(f"[pipe] model {model_name} doesn't support grounding search")

        config = types.GenerateContentConfig(**config_params)

        gen_content_args = {
            "model": model_name,
            "contents": contents,
            "config": config,
        }

        try:
            self._print_colored(f'Streaming is {body.get("stream", False)}', "DEBUG")
            if body.get("stream", False):
                return _process_stream(self, gen_content_args)
            else:  # streaming is disabled
                if not self.client:
                    return "Error: Client not initialized."
                # FIXME Make it async
                response = self.client.models.generate_content(**gen_content_args)
                response_text = response.text if response.text else "No response text."
                return response_text

        except Exception as e:
            error_msg = f"Content generation error: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return error_msg
        finally:
            self._print_colored("Content generation completed.", "INFO")

    """Helper functions inside the Pipe class."""

    def _print_colored(self, message: str, level: str = "INFO") -> None:
        """
        Prints a colored log message to the console, respecting the configured log level.
        """
        if not hasattr(self, "valves") or self.valves.LOG_LEVEL == "OFF":
            return

        # Define log level hierarchy
        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

        # Only print if message level is >= configured level
        if level_priority.get(level, 0) >= level_priority.get(self.valves.LOG_LEVEL, 0):
            color_map = {
                "INFO": COLORS["GREEN"],
                "WARNING": COLORS["YELLOW"],
                "ERROR": COLORS["RED"],
                "DEBUG": COLORS["BLUE"],
            }
            color = color_map.get(level, COLORS["WHITE"])
            frame = inspect.currentframe()
            if frame:
                frame = frame.f_back
            method_name = frame.f_code.co_name if frame else "<unknown>"
            print(
                f"{color}[{level}][gemini_manifold][{method_name}]{COLORS['RESET']} {message}"
            )

    def _get_google_models(self):
        """Retrieve Google models with prefix stripping."""

        # Check if client is initialized and return error if not.
        if not self.client:
            self._print_colored("Client not initialized.", "ERROR")
            return [
                {
                    "id": "error",
                    "name": "Client not initialized. Please check the logs.",
                }
            ]

        try:
            whitelist = (
                self.valves.MODEL_WHITELIST.split(",")
                if self.valves.MODEL_WHITELIST
                else ["*"]
            )
            models = self.client.models.list(config={"query_base": True})
            self._print_colored(
                f"[get_google_models] Retrieved {len(models)} models from Gemini Developer API.",
                "INFO",
            )
            model_list = [
                {
                    "id": self._strip_prefix(model.name),
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
                self._print_colored("No models found matching whitelist.", "WARNING")
                return [
                    {
                        "id": "no_models_found",
                        "name": "No models found matching whitelist.",
                    }
                ]
            return model_list
        except Exception as e:
            error_msg = f"Error retrieving models: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return [
                {
                    "id": "error",
                    "name": "Error retrieving models. Please check the logs.",
                }
            ]

    def _strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        This makes the method generic and adaptable to varying prefixes.
        """
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            return stripped
        except Exception as e:
            error_msg = f"Error stripping prefix: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "WARNING")
            # FIXME OR should it error out??
            return model_name  # Return original if stripping fails
        finally:
            self._print_colored("Completed prefix stripping.", "DEBUG")

    def truncate_long_strings(self, data: dict, max_length: int = 50) -> dict:
        """
        Recursively truncates all string and bytes fields within a dictionary that exceed
        the specified maximum length. Bytes are converted to strings before truncation.

        Args:
            data: A dictionary representing the JSON data.
            max_length: The maximum length of strings before truncation.

        Returns:
            The modified dictionary with long strings truncated.
        """
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_length:
                data[key] = value[:max_length] + "..."  # Truncate and add ellipsis
            elif isinstance(value, bytes) and len(value) > max_length:
                data[key] = (
                    value.hex()[:max_length] + "..."
                )  # Convert to hex, truncate, and add ellipsis
            elif isinstance(value, dict):
                self.truncate_long_strings(
                    value, max_length
                )  # Recursive call for nested dictionaries
            elif isinstance(value, list):
                # Iterate through the list and process each element if it's a dictionary
                for item in value:
                    if isinstance(item, dict):
                        self.truncate_long_strings(item, max_length)

        return data
