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
# TODO Audio input support (currently in development). https://ai.google.dev/gemini-api/docs/audio?lang=python
# TODO Video input support.
# TODO PDF (other documents?) input support, __files__ param that is passed to the pipe() func can be used for this.
# TODO Better type checking.
# TODO Return errors as correctly formatted error types for the frontend to handle (red text in the front-end).

# ^ Open WebUI front-end throws error when trying to upload videos or audios,
# but the file still gets uploaded to database and is passed to the pipe function.
# We can use this fact to implement audio and video input support.

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Open WebUI v0.5.5 or greater is required for this function to work properly.
# Be sure to check out my GitHub repository for more information! Feel free to contribute and post questions.

import base64
import inspect
import json
import re
import fnmatch
import time
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

FILES_HEADER_TEMPLATE = """<details>
<summary>Files in the context window</summary>
PLACEHOLDER
</details>
"""


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

    def pipes(self) -> list[dict[str, str]]:
        """Register all available Google models."""
        try:
            if not self.valves.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")

            # GEMINI_API_KEY is not available inside __init__ for whatever reason so we initialize the client here.
            # TODO Allow user to choose if they want to fetch models only during function initialization or every time pipes is called.
            if not self.client:
                self.client = genai.Client(
                    api_key=self.valves.GEMINI_API_KEY,
                    http_options=types.HttpOptions(api_version="v1alpha"),
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
        self,
        body: dict[str, Any],
        __files__: list[dict[str, Any]],
    ) -> (
        str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator
    ):

        pipe_start_time = time.time()
        self._print_colored(f"I have started! Time is {pipe_start_time}.", "DEBUG")

        """Helper functions inside the pipe() method"""

        def _process_messages_object(
            messages: list[dict[str, str]],
        ) -> Tuple[Optional[str], Optional[str], list[dict[str, str]]]:
            """Extracts system message and files context from messages."""

            system_message_content = None
            processed_messages = []
            user_message_index = -1
            assistant_message_index = -1
            latest_files = None

            for message in messages:
                match message.get("role"):
                    case "system":
                        system_message_content = message.get("content")
                    case "user":
                        user_message_index += 1
                        self._print_colored(
                            f"Processing user message {user_message_index}...", "DEBUG"
                        )
                        processed_messages.append(message)
                    case "assistant":
                        assistant_message_index += 1
                        content = message.get("content", "")

                        # Check if content starts with our template format
                        template_start = FILES_HEADER_TEMPLATE.split("PLACEHOLDER")[0]
                        template_end = FILES_HEADER_TEMPLATE.split("PLACEHOLDER")[1]

                        if (
                            content.startswith(template_start)
                            and template_end in content
                        ):
                            # Extract everything between the template parts
                            start_idx = len(template_start)
                            end_idx = content.find(template_end)
                            latest_files = content[start_idx:end_idx]

                            # Remove the header from the message
                            clean_content = content[
                                end_idx + len(template_end) :
                            ].strip()
                            message["content"] = clean_content

                        processed_messages.append(message)
                    case _:
                        self._print_colored(
                            f'Ran into weird role while processing messages: {message.get("role")}',
                            "WARNING",
                        )
                        processed_messages.append(message)

            return system_message_content, latest_files, processed_messages

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
            messages: list[dict],
        ) -> list[types.Content]:
            """Transforms messages to google-genai contents, supporting both text and images."""
            if not genai or not types:
                raise ValueError(
                    "google-genai is not installed. Please install it to proceed."
                )
            contents: list[types.Content] = []
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
            gen_content_args: dict[str, Any], first_yield: Optional[str] = None
        ) -> AsyncGenerator[str, None]:
            """Helper function to process the stream and yield text chunks."""

            if first_yield:
                # TODO Keep track of uploaded files here.
                yield first_yield

            # FIXME Figure out why type checker freaks out here.
            response_stream: AsyncIterator[types.GenerateContentResponse] = (
                await self.client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
            )
            try:
                async for chunk in response_stream:
                    if chunk.candidates:
                        if len(chunk.candidates) > 1:
                            self._print_colored(
                                "Multiple candidates found in response, defaulting to the first candidate.",
                                "WARNING",
                            )
                        candidate = chunk.candidates[
                            0
                        ]  # Default to the first candidate
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                text_part = getattr(
                                    part, "text", None
                                )  # Safely get the text part
                                if text_part is not None:
                                    yield text_part
            finally:
                pipe_end_time = time.time()
                self._print_colored(
                    f"Content generation completed. Time is {pipe_end_time}, took {pipe_end_time - pipe_start_time} seconds.",
                    "DEBUG",
                )

        """Main pipe method."""

        # TODO When stream_options: { include_usage: true } is enabled, the response will contain usage information.

        messages = body.get("messages", [])
        self._print_colored("Received request:", "DEBUG")
        if self.valves.LOG_LEVEL == "DEBUG":
            print(json.dumps(body, indent=2))

        # Split the message object into different parts that are easier to work with.
        system_prompt, file_header, remaining_messages = _process_messages_object(
            messages
        )
        self._print_colored(f"Original system prompt:\n{system_prompt}", "DEBUG")
        if system_prompt:
            # TODO This assumes the RAG prompt ends with this tag, always. Splitting is needed to keep the user's own system prompt.
            if "</user_query>" in system_prompt:
                system_prompt = system_prompt.split("</user_query>", 1)[1].strip()
                if not system_prompt:
                    system_prompt = None
                self._print_colored(
                    "Removed everything before </user_query> from system prompt.",
                    "DEBUG",
                )

        # Convert the sanitized message object into list of type.Content objects that google-genai understands.
        contents = _transform_messages_to_contents(remaining_messages)
        self._print_colored("Contents: [", "DEBUG")
        if self.valves.LOG_LEVEL == "DEBUG":
            for content in enumerate(contents):
                print(f"    {content},")
            print("]")

        # Get the current model name from the body object.
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

        self._print_colored(
            "Creating GenerateContentConfig object with following values:"
        )
        if self.valves.LOG_LEVEL == "DEBUG":
            print(json.dumps(config_params, indent=2, default=str))
        config = types.GenerateContentConfig(**config_params)

        gen_content_args = {
            "model": model_name,
            "contents": contents,
            "config": config,
        }

        try:
            if not self.client:
                return "Error: Client not initialized."

            self._print_colored(f'Streaming is {body.get("stream", False)}', "DEBUG")

            files_context = None
            if __files__:
                # TODO If this list contains a file that is not in the files context (between <details> tags), then we can assume that user just uploaded a new file.
                # We can infer from the growing files context, when the file was uploaded in the chat history.
                self._print_colored("You have uploaded files to this chat:", "DEBUG")
                if self.valves.LOG_LEVEL == "DEBUG":
                    print(json.dumps(__files__, indent=2))
                # TODO This can also be a list I think, we do not need the position value.
                files_str = ""
                for f in __files__:
                    files_str += f'{f.get("name", "missing")}\n'
                self._print_colored("Lean file list:", "DEBUG")
                if self.valves.LOG_LEVEL == "DEBUG":
                    print(files_str)
                # TODO Add "> " in the begging of each line inside the collapsible.
                files_context = FILES_HEADER_TEMPLATE.replace("PLACEHOLDER", files_str)
                files_str = ""

            # Backend is able to handle AsyncGenerator object if streming is set to False.
            return _process_stream(
                gen_content_args,
                files_context,
            )
        except Exception as e:
            error_msg = f"Content generation error: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return error_msg

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

    def _get_google_models(self) -> list[dict[str, str]]:
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
