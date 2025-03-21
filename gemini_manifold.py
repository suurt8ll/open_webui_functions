"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Supports native image generation, grounding with Google Search and streaming. Uses google-genai.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.6.0
requirements: google-genai==1.7.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

# Supported features:
#   - Native image generation (image output), use "gemini-2.0-flash-exp-image-generation"
#   - Image input
#   - Streaming
#   - Grounding with Google Search
#   - Safety settings

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

from google import genai
from google.genai import types
import base64
import re
import fnmatch
import sys
from fastapi import Request
from pydantic import BaseModel, Field
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Literal,
    NotRequired,
    Tuple,
    Optional,
    TypedDict,
    TYPE_CHECKING,
)
from starlette.responses import StreamingResponse
from open_webui.routers.images import upload_image
from open_webui.models.files import Files
from open_webui.models.users import Users
from open_webui.utils.logger import stdout_format
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler

# FIXME What about other 2.0 models?
# according to https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini
ALLOWED_GROUNDING_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
]

# To avoid conflict name in the future, here use suffix not in gemini naming pattern.
SEARCH_MODEL_SUFFIX = "++SEARCH"


class UserData(TypedDict):
    """This is how `__user__` `dict` looks like."""

    id: str
    email: str
    name: str
    role: Literal["admin", "user", "pending"]
    valves: NotRequired[Any]  # object of type UserValves


class ModelData(TypedDict):
    """This is how the `pipes` function expects the `dict` to look like."""

    id: str
    name: str


class ErrorData(TypedDict):
    detail: str


class ChatCompletionEventData(TypedDict):
    content: Optional[str]
    done: bool
    error: NotRequired[ErrorData]


class ChatCompletionEvent(TypedDict):
    type: Literal["chat:completion"]
    data: ChatCompletionEventData


Event = ChatCompletionEvent

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


class Pipe:
    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="")
        GEMINI_API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com",
            description="The base URL for calling the Gemini API",
        )
        MODEL_WHITELIST: str = Field(
            default="",
            description="Comma-separated list of allowed model names. Supports wildcards (*).",
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description="Whether to request models only on first load and when whitelist changes.",
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
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=False, description="Whether to request relaxed safety filtering"
        )
        LOG_LEVEL: Literal["INFO", "WARNING", "ERROR", "DEBUG", "OFF"] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )
        USE_FILES_API: bool = Field(
            title="Use Files API",
            default=True,
            description="Save the image files using Open WebUI's API for files.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.last_whitelist = self.valves.MODEL_WHITELIST
        self.models = []
        self.client = None
        print("[gemini_manifold] Function has been initialized!")

    # FIXME Make it async
    def pipes(self) -> list[ModelData]:
        """Register all available Google models."""

        self._add_log_handler()

        # Return existing models if all conditions are met
        if (
            self.models
            and self.valves.CACHE_MODELS
            and self.last_whitelist == self.valves.MODEL_WHITELIST
        ):
            log.info("Models are already initialized. Returning the cached list.")
            return self.models

        if not self.valves.GEMINI_API_KEY:
            error_msg = "GEMINI_API_KEY is not set."
            return [_return_error_model(error_msg, exception=False)]

        # GEMINI_API_KEY is not available inside __init__ for whatever reason so we initialize the client here.
        if not self.client:
            http_options = types.HttpOptions(base_url=self.valves.GEMINI_API_BASE_URL)
            try:
                self.client = genai.Client(
                    api_key=self.valves.GEMINI_API_KEY,
                    http_options=http_options,
                )
            except Exception as e:
                error_msg = f"genai client initalization failed: {str(e)}"
                return [_return_error_model(error_msg)]
        else:
            log.info("Client already initialized.")

        self.last_whitelist = self.valves.MODEL_WHITELIST

        # Get and process new models, errors are handler inside the method.
        models = self._get_google_models()
        log.debug("Registered models:", data=models)

        self.models = models
        return models

    async def pipe(
        self,
        body: dict,
        __user__: UserData,
        __request__: Request,
        __event_emitter__: Callable[[Event], Awaitable[None]],
        __metadata__: dict[str, Any],
    ) -> (
        str
        | dict[str, Any]
        | StreamingResponse
        | Iterator
        | AsyncGenerator
        | Generator
        | None
    ):
        """Helper functions inside the pipe() method"""
        # FIXME: Move all of these into Pipe class for cleaner look.

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
                r"!\[.*?\]\((data:(image/[^;]+);base64,([^)]+)|/api/v1/files/([a-f0-9\-]+)/content)\)",
                text,
            ):
                # Add text before the image
                text_segment = text[last_pos : match.start()]
                if text_segment.strip():
                    parts.append(types.Part.from_text(text=text_segment))

                # Determine if it's base64 or a file URL
                if match.group(2):  # Base64 encoded image
                    try:
                        mime_type = match.group(2)
                        base64_data = match.group(3)
                        log.debug(
                            "Found base64 image link!",
                            mime_type=mime_type,
                            base64_data=match.group(3)[:50] + "...",
                        )
                        image_part = types.Part.from_bytes(
                            data=base64.b64decode(base64_data),
                            mime_type=mime_type,
                        )
                        parts.append(image_part)
                    except Exception:
                        log.exception("Error decoding base64 image:")

                elif match.group(4):  # File URL
                    log.debug("Found API image link!", id=match.group(4))
                    file_id = match.group(4)
                    file_model = Files.get_file_by_id(file_id)

                    if file_model is None:
                        log.warning("File with this ID not found.", id=file_id)
                        #  Could add placeholder text here if desired
                        continue  # Skip to the next match

                    try:
                        # "continue" above ensures that file_model is not None
                        content_type = file_model.meta.get("content_type")  # type: ignore
                        if content_type is None:
                            log.warning(
                                "Content type not found for this file ID.",
                                id=file_id,
                            )
                            continue

                        with open(file_model.path, "rb") as file:  # type: ignore
                            image_data = file.read()

                        image_part = types.Part.from_bytes(
                            data=image_data, mime_type=content_type
                        )
                        parts.append(image_part)

                    except FileNotFoundError:
                        log.exception(
                            "File not found on disk for this ID.",
                            id=file_id,
                            path=file_model.path,
                        )
                    except KeyError:
                        log.exception(
                            "Metadata error for this file ID: 'content_type' missing.",
                            id=file_id,
                        )
                    except Exception:
                        log.exception("Error processing file with this ID", id=file_id)

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
            gen_content_args: dict, __request__: Request, __user__: UserData
        ) -> AsyncGenerator[str, None]:
            """Helper function to process the stream and yield text chunks.

            Args:
                gen_content_args: The arguments to pass to generate_content_stream.

            Yields:
                str: Text chunks from the response.
            """
            # FIXME Get type checking working in here.
            response_stream: AsyncIterator[types.GenerateContentResponse] = (
                await self.client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
            )

            async for chunk in response_stream:
                if chunk.candidates:
                    if len(chunk.candidates) > 1:
                        log.warning(
                            "Multiple candidates found in response, defaulting to the first candidate."
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
                                image_data = inline_data.data
                                if self.valves.USE_FILES_API:
                                    image_url = self._upload_image(
                                        image_data,
                                        mime_type,
                                        gen_content_args.get("model", ""),
                                        "TAKE IT FROM gen_content_args contents",
                                        __user__,
                                        __request__,
                                    )
                                    markdown_image = f"![Generated Image]({image_url})"
                                    yield markdown_image
                                else:
                                    image_data_decoded = base64.b64encode(
                                        image_data
                                    ).decode()
                                    markdown_image = f"![Generated Image](data:{mime_type};base64,{image_data_decoded})"
                                    yield markdown_image

        """Main pipe method."""

        self.__event_emitter__ = __event_emitter__

        if not self.client:
            error_msg = "genai client is not initialized."
            await self._emit_error(error_msg, exception=False)
            return

        # TODO Contruct a type for `__metadata__`.
        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrival phase: {str(__metadata__["model"])}'
            await self._emit_error(error_msg, exception=False)
            return

        messages = body.get("messages", [])
        system_prompt, remaining_messages = _pop_system_prompt(messages)
        contents = _transform_messages_to_contents(remaining_messages)

        max_len = 50
        log.debug(
            "Received body:", body=str(self.truncate_long_strings(body.copy(), max_len))
        )
        log.debug(f"System prompt: {system_prompt}")
        turn_content_dict_list: list[dict] = []
        for content in contents:
            truncated_content = self.truncate_long_strings(
                content.model_dump().copy(), max_len
            )
            turn_content_dict_list.append(truncated_content)
        log.debug(
            "list[google.genai.types.Content] object that will be given to the Gemini API:",
            content_list=str(turn_content_dict_list),
        )

        model_name = self._strip_prefix(body.get("model", ""))
        log.debug(f"Model name: {model_name}")

        config_params = {
            "system_instruction": system_prompt,
            "temperature": body.get("temperature", 0.7),
            "top_p": body.get("top_p", 0.9),
            "top_k": body.get("top_k", 40),
            "max_output_tokens": body.get("max_tokens", 8192),
            "stop_sequences": body.get("stop", []),
            "safety_settings": self._get_safety_settings(model_name),
        }

        # FIXME: refac
        if "gemini-2.0-flash-exp-image-generation" in model_name:
            config_params["response_modalities"] = ["Text", "Image"]
            # Image Generation model does not support the system prompt message
            if config_params.get("system_instruction"):
                log.warning(
                    "Image Generation model does not support the system prompt message! Removing the system prompt."
                )
                del config_params["system_instruction"]
        else:
            config_params["response_modalities"] = ["Text"]

        if self.valves.USE_GROUNDING_SEARCH:
            if model_name.endswith(SEARCH_MODEL_SUFFIX):
                log.info("Using grounding search.")
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
                log.debug(f"Model {model_name} doesn't support grounding search.")

        config = types.GenerateContentConfig(**config_params)

        gen_content_args = {
            "model": model_name.replace(SEARCH_MODEL_SUFFIX, ""),
            "contents": contents,
            "config": config,
        }

        try:
            # TODO: Handle errors related to Google Safety Settings feature.
            if body.get("stream", False):
                return _process_stream(gen_content_args, __request__, __user__)
            else:
                if "gemini-2.0-flash-exp-image-generation" in model_name:
                    warn_msg = "Non-streaming responses with native image gen are not currently supported! Stay tuned! Please enable streaming."
                    await self._emit_error(warn_msg, warning=True)
                    return None
                # FIXME: Make it async.
                # FIXME: Support native image gen here too.
                response = self.client.models.generate_content(**gen_content_args)
                if not response.text:
                    warn_msg = "Non-stremaing response did not have any text inside it."
                    await self._emit_error(warn_msg, warning=True)
                    return None
                return response.text

        except Exception as e:
            error_msg = f"Content generation error: {str(e)}"
            await self._emit_error(error_msg)
            return None

    """Helper functions inside the Pipe class."""

    def _add_log_handler(self):
        """Adds handler to the root loguru instance for this plugin if one does not exist already."""

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__  # Filter by module name

        # Access the internal state of the logger
        handlers: dict[int, "Handler"] = logger._core.handlers  # type: ignore
        for key, handler in handlers.items():
            existing_filter = handler._filter
            if (
                hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            ):
                log.debug("Handler for this plugin is already present!")
                return

        logger.add(
            sys.stdout,
            level=self.valves.LOG_LEVEL,
            format=stdout_format,
            filter=plugin_filter,
        )
        log.info(
            f"Added new handler to loguru with level {self.valves.LOG_LEVEL} and filter {__name__}."
        )

    def _get_google_models(self) -> list[ModelData]:
        """Retrieve Google models with prefix stripping."""

        if not self.client:
            log.error("Client is not initialized.")
            return []

        whitelist = (
            self.valves.MODEL_WHITELIST.replace(" ", "").split(",")
            if self.valves.MODEL_WHITELIST
            else ["*"]
        )

        try:
            models = self.client.models.list(config={"query_base": True})
        except Exception as e:
            error_msg = f"Error retrieving models: {str(e)}"
            return [_return_error_model(error_msg)]
        log.info(f"Retrieved {len(models)} models from Gemini Developer API.")

        model_list = [
            ModelData(
                id=self._strip_prefix(model.name),
                name=model.display_name,
            )
            for model in models
            if (
                model.name is not None  # Ensure name is present
                and model.display_name is not None  # Ensure display_name is present
                and any(fnmatch.fnmatch(model.name, f"models/{w}") for w in whitelist)
                and model.supported_actions
                and "generateContent" in model.supported_actions
                and model.name.startswith("models/")
            )
        ]

        if not model_list:
            log.warning("No models found matching whitelist.")
            return []

        # Add synthesis model id which support search if grounding search is enabled.
        # TODO Add this logic into the previous `model_list` construction logic? We are already looping over models there.
        if not self.valves.USE_GROUNDING_SEARCH:
            return model_list
        for original_model in model_list:
            if original_model["id"] in ALLOWED_GROUNDING_MODELS:
                model_list.append(
                    {
                        "id": original_model["id"] + SEARCH_MODEL_SUFFIX,
                        "name": original_model["name"] + " with Search",
                    }
                )

        return model_list

    def _get_safety_settings(self, model_name: str):
        """Get safety settings based on model name and permissive setting."""

        if not self.valves.USE_PERMISSIVE_SAFETY:
            return []

        # Settings supported by most models
        category_threshold_map = {
            types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.OFF,
            types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: types.HarmBlockThreshold.BLOCK_NONE,
        }

        # Older models use BLOCK_NONE
        if model_name in [
            "gemini-1.5-pro-001",
            "gemini-1.5-flash-001",
            "gemini-1.5-flash-8b-exp-0827",
            "gemini-1.5-flash-8b-exp-0924",
            "gemini-pro",
            "gemini-1.0-pro",
            "gemini-1.0-pro-001",
        ]:
            for category in category_threshold_map:
                category_threshold_map[category] = types.HarmBlockThreshold.BLOCK_NONE

        # Gemini 2.0 Flash supports CIVIC_INTEGRITY OFF
        if model_name in [
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-exp",
        ]:
            category_threshold_map[types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = (
                types.HarmBlockThreshold.OFF
            )

        log.debug(
            f"Safety settings: {str({k.value: v.value for k, v in category_threshold_map.items()})}"
        )

        safety_settings = [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in category_threshold_map.items()
        ]
        return safety_settings

    def _strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        This makes the method generic and adaptable to varying prefixes.
        """
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            return stripped
        except Exception:
            error_msg = "Error stripping prefix:"
            log.exception(error_msg)
            # FIXME OR should it error out??
            return model_name  # Return original if stripping fails

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

    def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        model: str,
        prompt: str,
        __user__: UserData,
        __request__: Request,
    ) -> str:

        # FIXME: Handle potental errors.

        # Create metadata for the image
        image_metadata = {
            "model": model,
            "prompt": prompt,
        }

        # Get the *full* user object from the database
        user = Users.get_user_by_id(__user__["id"])
        if user is None:
            return "Error: User not found"

        # Upload the image using the imported function
        image_url: str = upload_image(
            request=__request__,
            image_metadata=image_metadata,
            image_data=image_data,
            content_type=mime_type,
            user=user,
        )
        log.info("Image uploaded.", image_url=image_url)
        return image_url

    async def _emit_error(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""
        error = ChatCompletionEvent(
            type="chat:completion",
            data=ChatCompletionEventData(
                content=None,
                done=True,
                error=ErrorData(detail="\n" + error_msg),
            ),
        )
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        await self.__event_emitter__(error)


def _return_error_model(
    error_msg: str, warning: bool = False, exception: bool = True
) -> ModelData:
    """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
    if warning:
        log.opt(depth=1, exception=False).warning(error_msg)
    else:
        log.opt(depth=1, exception=exception).error(error_msg)
    return {
        "id": "error",
        "name": "[gemini_manifold] " + error_msg,
    }
