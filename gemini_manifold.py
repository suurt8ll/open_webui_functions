"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Uses google-genai, supports streaming and grounding with Google Search.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.1.0
requirements: google-genai==1.2.0
"""

# TODO Add a list of supported features here and also exiting features that can be coded in theory.
# TODO Audio input support.
# TODO Video input support.
# TODO PDF (other documents?) input support, __files__ param that is passed to the pipe() func can be used for this.
# TODO Better debug output (colors, more info)
# TODO Better type checking.

# ^ Open WebUI front-end throws error when trying to upload videos or audios,
# but the file still gets uploaded to database and is passed to the pipe function.
# We can use this fact to implement audio and video input support.

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Open WebUI v0.5.5 or greater is required for this function to work properly.
# Be sure to check out my GitHub repository for more information! Feel free to contribute and post questions.

import base64
import json
import re
import fnmatch
from pydantic import BaseModel, Field
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
    List,
    Tuple,
    Optional,
)
from starlette.responses import StreamingResponse
from google import genai
from google.genai import types


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
        DEBUG: bool = Field(
            default=False,
            description="Enable debug output. Use `docker logs -f open-webui` to view logs.",
        )

    def __init__(self):
        try:
            self.valves = self.Valves()
            self.models = []
            self.client = None
            if self.valves.DEBUG:
                print("[INIT] Initialized Pipe with Valves configuration.")
                print(f"[INIT] Valves: {self.valves}")
        except Exception as e:
            if self.valves.DEBUG:
                print(f"[INIT] Error during initialization: {e}")
        finally:
            if self.valves.DEBUG:
                print("[INIT] Initialization complete.")

    def _get_google_models(self):
        """Retrieve Google models with prefix stripping."""

        # Check if client is initialized and return error if not.
        if not self.client:
            if self.valves.DEBUG:
                print("[get_google_models] Client not initialized.")
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
            if self.valves.DEBUG:
                print(
                    f"[get_google_models] Retrieved {len(models)} models from Gemini Developer API."
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
                if self.valves.DEBUG:
                    print("[get_google_models] No models found matching whitelist.")
                return [
                    {
                        "id": "no_models_found",
                        "name": "No models found matching whitelist.",
                    }
                ]
            return model_list
        except Exception as e:
            if self.valves.DEBUG:
                print(f"[get_google_models] Error retrieving models: {e}")
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
            if self.valves.DEBUG:
                print(f"[strip_prefix] Error stripping prefix: {e}")
            return model_name  # Return original if stripping fails
        finally:
            if self.valves.DEBUG:
                print("[strip_prefix] Completed prefix stripping.")

    def pipes(self) -> List[dict]:
        """Register all available Google models."""
        try:
            if not self.valves.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")

            # GEMINI_API_KEY is not available inside __init__ for whatever reason so we initialize the client here.
            if not self.client:
                self.client = genai.Client(
                    api_key=self.valves.GEMINI_API_KEY,
                    http_options=types.HttpOptions(api_version="v1alpha"),
                )
            else:
                if self.valves.DEBUG:
                    print("[pipes] Client already initialized.")
            # TODO Allow user to choose if they want to fetch models only during function initialization or every time pipes is called.

            if not self.models:
                models = self._get_google_models()
                if models and models[0].get("id") == "error":
                    return models  # Propagate error model from _get_google_models
                if models and models[0].get("id") == "no_models_found":
                    return models  # Propagate no_models_found model
                if self.valves.DEBUG:
                    print(f"[pipes] Registered models: {models}")
                self.models = models
                return (
                    models
                    if models
                    else [{"id": "no_models", "name": "No models available"}]
                )
            else:
                if self.valves.DEBUG:
                    print("[pipes] Models already initialized.")
                return self.models
        except Exception as e:
            if self.valves.DEBUG:
                print(f"[pipes] Error in pipes method: {e}")
            return [{"id": "error", "name": f"Error initializing models: {e}"}]
        finally:
            if self.valves.DEBUG:
                print("[pipes] Completed pipes method.")

    async def pipe(
        self, body: dict
    ) -> (
        str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator
    ):
        # TODO Return errors as correctly formatted error types for the frontend to handle (red text in the front-end).

        if not genai or not types:
            return "Error: google-genai is not installed. Please install it to proceed."

        def _pop_system_prompt(
            messages: List[dict],
        ) -> Tuple[Optional[str], List[dict]]:
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

        def _transform_messages_to_contents(
            messages: List[dict],
        ) -> List[types.Content]:
            """Transforms messages to google-genai contents, supporting both text and images."""
            if not genai or not types:
                raise ValueError(
                    "google-genai is not installed. Please install it to proceed."
                )
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
                                        mime_type=_get_mime_type(image_url),
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
                                        mime_type=_get_mime_type(image_url),
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
                if self.valves.DEBUG:
                    print(f"[_process_stream] Response: {chunk.text}", end="")
                if chunk.candidates:
                    if len(chunk.candidates) > 1:
                        if self.valves.DEBUG:
                            print(
                                "WARNING: Multiple candidates found in response, defaulting to the first candidate."
                            )
                    candidate = chunk.candidates[0]  # Default to the first candidate
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            text_part = getattr(
                                part, "text", None
                            )  # Safely get the text part
                            if text_part is not None:
                                yield text_part

        """Main pipe method."""

        # TODO When stream_options: { include_usage: true } is enabled, the response will contain usage information.

        messages = body.get("messages", [])
        system_prompt, remaining_messages = _pop_system_prompt(messages)
        contents = _transform_messages_to_contents(remaining_messages)

        if self.valves.DEBUG:
            print(f"[pipe] Received request:")
            print(json.dumps(body, indent=2))
            print(f"[pipe] System prompt: {system_prompt}")
            print(f"[pipe] Contents: [")
            for content in enumerate(contents):
                print(f"    {content},")
            print("]")

        model_name = self._strip_prefix(body.get("model", ""))
        if model_name in [
            "no_models_found",
            "error",
            "version_error",
            "no_models",
            "import_error",
        ]:
            return f"Error: {model_name.replace('_', ' ')}"
        if self.valves.DEBUG:
            print(f"[pipe] Model name: {model_name}")

        config_params = {
            "system_instruction": system_prompt,
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 40),
            "max_output_tokens": body.get("max_tokens", 8192),
            "stop_sequences": body.get("stop", []),
        }

        # TODO Display the citations in the frontend response somehow.
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
            if body.get("stream", False):
                if self.valves.DEBUG:
                    print("[pipe] Streaming enabled.")
                return _process_stream(self, gen_content_args)
            else:  # streaming is disabled
                if self.valves.DEBUG:
                    print("[pipe] Streaming disabled.")
                if not self.client:
                    return "Error: Client not initialized."
                # FIXME Make it async?
                response = self.client.models.generate_content(**gen_content_args)
                response_text = response.text if response.text else "No response text."
                return response_text

        except Exception as e:
            error_message = f"Content generation error: {str(e)}"
            if self.valves.DEBUG:
                print(f"[pipe] {error_message}")
            return error_message
        finally:
            if self.valves.DEBUG:
                print("[pipe] Content generation completed.")
