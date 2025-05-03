"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Supports native image generation, grounding with Google Search and streaming. Uses google-genai.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.16.0
requirements: google-genai==1.11.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

# Supported features:
#   - Native image generation (image output), use "gemini-2.0-flash-exp-image-generation"
#   - Display citations in the front-end.
#   - Image input
#   - YouTube video input (automatically detects youtube.com and youtu.be URLs in messages)
#   - Streaming
#   - Grounding with Google Search (this requires installing "Gemini Manifold Companion" >= 1.2.0 filter, see GitHub README)
#   - Safety settings
#   - Each user can decide to use their own API key.
#   - Token usage data
#   - Code execution tool. (Gemini Manifold Companion >= 1.1.0 required)

# Features that are supported by API but not yet implemented in the manifold:
#   TODO Audio input support.
#   TODO Video input support (other than YouTube URLs).
#   TODO PDF (other documents?) input support, __files__ param that is passed to the pipe() func can be used for this.

import asyncio
import copy
import json
import time
from fastapi.datastructures import State
from google import genai
from google.genai import types
import io
import mimetypes
import os
import uuid
import base64
import re
import fnmatch
import sys
from fastapi import Request
from pydantic import BaseModel, Field
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import (
    Any,
    AsyncGenerator,
    Literal,
    TYPE_CHECKING,
    cast,
)

from open_webui.models.chats import Chats
from open_webui.models.files import FileForm, Files
from open_webui.models.functions import Functions
from open_webui.storage.provider import Storage
from open_webui.utils.logger import stdout_format, start_logger
from loguru import logger as global_logger

if TYPE_CHECKING:
    from loguru import Logger
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Pipe:
    class Valves(BaseModel):
        GEMINI_API_KEY: str | None = Field(default=None)
        REQUIRE_USER_API_KEY: bool = Field(
            default=False,
            description="""Whether to require user's own API key (applies to admins too).
            User can give their own key through UserValves.
            Default value is False.""",
        )
        GEMINI_API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com",
            description="The base URL for calling the Gemini API",
        )
        MODEL_WHITELIST: str = Field(
            default="*",
            description="""Comma-separated list of allowed model names.
            Supports `fnmatch` patterns: *, ?, [seq], [!seq].
            Default value is * (all models allowed).""",
        )
        MODEL_BLACKLIST: str | None = Field(
            default=None,
            description="""Comma-separated list of blacklisted model names.
            Supports `fnmatch` patterns: *, ?, [seq], [!seq].
            Default value is None (no blacklist).""",
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description="Whether to request models only on first load and when white- or blacklist changes.",
        )
        THINKING_BUDGET: int | None = Field(
            default=None,
            description="Indicates the thinking budget in tokens. Default value is None.",
        )
        USE_PERMISSIVE_SAFETY: bool = Field(
            default=False, description="Whether to request relaxed safety filtering."
        )
        USE_FILES_API: bool = Field(
            title="Use Files API",
            default=True,
            description="Save the image files using Open WebUI's API for files.",
        )
        THINKING_MODEL_PATTERN: str = Field(
            default=r"thinking|gemini-2.5",
            description="Regex pattern to identify thinking models.",
        )
        EMIT_INTERVAL: int = Field(
            default=1,
            description="Interval in seconds between status updates during thinking.",
        )
        EMIT_STATUS_UPDATES: bool = Field(
            default=False,
            description="Whether to emit status updates during model thinking.",
        )
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    class UserValves(BaseModel):
        GEMINI_API_KEY: str | None = Field(default=None)
        GEMINI_API_BASE_URL: str = Field(
            default="https://generativelanguage.googleapis.com",
            description="The base URL for calling the Gemini API",
        )
        THINKING_BUDGET: int | None = Field(
            default=None,
            description="Indicates the thinking budget in tokens. Default value is None.",
        )
        # TODO: Add more options that can be changed by the user.

    def __init__(self):

        # This hack makes the valves values available to the `__init__` method.
        # TODO: Get the id from the frontmatter instead of hardcoding it.
        valves = Functions.get_function_valves_by_id("gemini_manifold_google_genai")
        self.valves = self.Valves(**(valves if valves else {}))
        self.log_level = self.valves.LOG_LEVEL
        self._setup_plugin_logger()
        # Initialize the genai client with default API given in Valves.
        self.clients = {"default": self._get_genai_client()}
        self.models: list["ModelData"] = []
        self.last_whitelist: str = self.valves.MODEL_WHITELIST
        self.last_blacklist = self.valves.MODEL_BLACKLIST

        self.log.success("Function has been initialized.")

    async def pipes(self) -> list["ModelData"]:
        """Register all available Google models."""
        # Detect log level change inside self.valves
        if self.log_level != self.valves.LOG_LEVEL:
            self.log.info(
                f"Detected log level change: {self.log_level=} and {self.valves.LOG_LEVEL=}. "
                "Running the logging setup again."
            )
            self._setup_plugin_logger()
        # TODO: SUCCESS log in here.
        # Return existing models if all conditions are met and no error models are present
        if (
            self.models
            and self.valves.CACHE_MODELS
            and self.last_whitelist == self.valves.MODEL_WHITELIST
            and self.last_blacklist == self.valves.MODEL_BLACKLIST
            and not any(model["id"] == "error" for model in self.models)
        ):
            self.log.info(
                f"Models are already initialized. Returning the cached list ({len(self.models)} models)."
            )
            return self.models

        # Filter the model list based on white- and blacklist.
        self.models = self._filter_models(await self._get_genai_models())
        self.log.info(f"Returning {len(self.models)} models to Open WebUI.")
        self._log_with_data("DEBUG", "Model list:", self.models, truncate=False)

        return self.models

    async def pipe(
        self,
        body: "Body",
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __metadata__: dict[str, Any],
    ) -> AsyncGenerator | str | None:

        # Obtain Genai client
        if not (client := self._get_user_client(__user__)):
            error_msg = "There are no usable genai clients, check the logs."
            raise ValueError(error_msg)
        if self.clients.get("default") == client:
            if self.valves.REQUIRE_USER_API_KEY:
                error_msg = "You have not defined your own API key in UserValves. You need to define in to continue."
                raise ValueError(error_msg)
            else:
                self.log.info("Using genai client with the default API key.")
        else:
            self.log.info(
                f'Using genai client with user {__user__.get("email")} API key.'
            )

        # Check if user is chatting with an error model for some reason.
        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrival phase: {str(__metadata__["model"])}'
            raise ValueError(error_msg)

        # Get the message history directly from the backend.
        # This allows us to see data about sources and files data.
        chat_id = __metadata__.get("chat_id", "")
        if chat := Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=__user__["id"]):
            chat_content: ChatChatModel = chat.chat  # type: ignore
            # Last message is the upcoming assistant response, at this point in the logic it's empty.
            messages_db = chat_content.get("messages")[:-1]
        else:
            warn_msg = f"Chat with ID - {chat_id} - not found. Can't filter out the citation marks."
            self.log.warning(warn_msg)
            messages_db = None
        self.log.info(
            "Converting Open WebUI's `body` dict into list of `Content` objects that `google-genai` understands."
        )
        contents, system_prompt = self._genai_contents_from_messages(
            body.get("messages"), messages_db
        )
        model_name = self._strip_prefix(body.get("model", ""))
        # API does not stream thoughts sadly. See https://github.com/googleapis/python-genai/issues/226#issuecomment-2631657100
        thinking_conf = None
        if self.is_thinking_model(model_name):
            self.log.info(f"Model ID '{model_name}' is a thinking model.")
            thinking_conf = types.ThinkingConfig(
                thinking_budget=self.valves.THINKING_BUDGET, include_thoughts=None
            )
        # TODO: Take defaults from the general front-end config.
        gen_content_conf = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            max_output_tokens=body.get("max_tokens"),
            stop_sequences=body.get("stop"),
            safety_settings=self._get_safety_settings(model_name),
            thinking_config=thinking_conf,
        )

        gen_content_conf.response_modalities = ["Text"]
        if (
            "gemini-2.0-flash-exp-image-generation" in model_name
            or "gemma" in model_name
        ):
            if "gemini-2.0-flash-exp-image-generation" in model_name:
                gen_content_conf.response_modalities.append("Image")
            # TODO: append to user message instead.
            if gen_content_conf.system_instruction:
                gen_content_conf.system_instruction = None
                self.log.warning(
                    "Image Generation model does not support the system prompt message! Removing the system prompt."
                )

        features = __metadata__.get("features", {}) or {}
        gen_content_conf.tools = []

        if features.get("google_search_tool"):
            self.log.info("Using grounding with Google Search as a Tool.")
            gen_content_conf.tools.append(
                types.Tool(google_search=types.GoogleSearch())
            )
        elif features.get("google_search_retrieval"):
            self.log.info("Using grounding with Google Search Retrieval.")
            gs = types.GoogleSearchRetrieval(
                dynamic_retrieval_config=types.DynamicRetrievalConfig(
                    dynamic_threshold=features.get("google_search_retrieval_threshold")
                )
            )
            gen_content_conf.tools.append(types.Tool(google_search_retrieval=gs))

        # NB: It is not possible to use both Search and Code execution at the same time,
        # however, it can be changed later, so let's just handle it as a common error
        if features.get("google_code_execution"):
            self.log.info("Using code execution on Google side.")
            gen_content_conf.tools.append(
                types.Tool(code_execution=types.ToolCodeExecution())
            )

        gen_content_args = {
            "model": model_name,
            "contents": contents,
            "config": gen_content_conf,
        }
        self._log_with_data(
            "DEBUG", "Passing these args to the Google API:", gen_content_args
        )

        if body.get("stream", False):
            # Streaming response
            response_stream: AsyncIterator[types.GenerateContentResponse] = (
                await client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
            )
            self.log.info("Streaming enabled. Returning AsyncGenerator.")
            return self._stream_response_generator(
                response_stream,
                __request__,
                __user__,
                gen_content_args,
                __event_emitter__,
                __metadata__,
            )
        else:
            # Non-streaming response.
            if "gemini-2.0-flash-exp-image-generation" in model_name:
                warn_msg = "Non-streaming responses with native image gen are not currently supported! Stay tuned! Please enable streaming."
                raise NotImplementedError(warn_msg)
            # TODO: Support native image gen here too.
            # TODO: Support code execution here too.
            res = await client.aio.models.generate_content(**gen_content_args)
            if raw_text := res.text:
                await self._do_post_processing(
                    res, __event_emitter__, __metadata__, __request__
                )
                self.log.info("Streaming disabled. Returning full response as str.")
                self.log.success("Pipe.pipe method has finished it's run!")
                return raw_text
            else:
                warn_msg = "Non-streaming response did not have any text inside it."
                raise ValueError(warn_msg)

    # region Helper methods inside the Pipe class

    # region Event emission and error logging
    async def _emit_completion(
        self,
        event_emitter: Callable[["Event"], Awaitable[None]],
        content: str | None = None,
        done: bool = False,
        error: str | None = None,
        sources: list["Source"] | None = None,
    ):
        """Constructs and emits completion event."""
        emission: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {"done": done},
        }
        if content:
            emission["data"]["content"] = content
        if error:
            emission["data"]["error"] = {"detail": error}
        if sources:
            emission["data"]["sources"] = sources
        await event_emitter(emission)

    def is_thinking_model(self, model_id: str) -> bool:
        """Check if the model is a thinking model based on the valve pattern."""
        try:
            result = bool(
                re.search(self.valves.THINKING_MODEL_PATTERN, model_id, re.IGNORECASE)
            )
            return result
        except Exception:
            self.log.exception("Error checking if model is a thinking model")
            return False

    async def _emit_status(
        self,
        message: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        done: bool = False,
    ) -> None:
        """Emit status updates asynchronously."""
        try:
            if self.valves.EMIT_STATUS_UPDATES:
                status_event: "StatusEvent" = {
                    "type": "status",
                    "data": {"description": message, "done": done},
                }
                await event_emitter(status_event)
                self.log.debug(f"Emitted status: '{message}', {done=}")
            else:
                self.log.debug(
                    f"EMIT_STATUS_UPDATES is disabled. Skipping status: '{message}'"
                )
        except Exception:
            self.log.exception("Error emitting status")

    async def thinking_timer(
        self, event_emitter: Callable[["Event"], Awaitable[None]]
    ) -> None:
        """Asynchronous task to emit periodic status updates."""
        elapsed = 0
        try:
            self.log.info("Thinking timer started.")
            while True:
                await asyncio.sleep(self.valves.EMIT_INTERVAL)
                elapsed += self.valves.EMIT_INTERVAL
                # Format elapsed time
                if elapsed < 60:
                    time_str = f"{elapsed}s"
                else:
                    minutes, seconds = divmod(elapsed, 60)
                    time_str = f"{minutes}m {seconds}s"
                status_message = f"Thinking... ({time_str} elapsed)"
                await self._emit_status(
                    status_message, event_emitter=event_emitter, done=False
                )
        except asyncio.CancelledError:
            self.log.debug("Timer task cancelled.")
        except Exception:
            self.log.exception("Error in timer task")

    async def _emit_error(
        self,
        error_msg: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        warning: bool = False,
        exception: bool = True,
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""

        if warning:
            self.log.opt(depth=1, exception=False).warning(error_msg)
        else:
            self.log.opt(depth=1, exception=exception).error(error_msg)
        await self._emit_completion(
            error=f"\n{error_msg}", event_emitter=event_emitter, done=True
        )

    async def _emit_toast(
        self,
        msg: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        toastType: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        # TODO: Use this method in more places, even for info toasts.
        event: NotificationEvent = {
            "type": "notification",
            "data": {"type": toastType, "content": msg},
        }
        await event_emitter(event)

    def _return_error_model(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> "ModelData":
        """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
        if warning:
            self.log.opt(depth=1, exception=False).warning(error_msg)
        else:
            self.log.opt(depth=1, exception=exception).error(error_msg)
        return {
            "id": "error",
            "name": "[gemini_manifold] " + error_msg,
            "description": error_msg,
        }

    # endregion

    # region ChatModel.chat.messages -> list[genai.types.Content] conversion
    def _genai_contents_from_messages(
        self, messages_body: list["Message"], messages_db: list["MessageModel"] | None
    ) -> tuple[list[types.Content], str | None]:
        """Transforms `body.messages` list into list of `genai.types.Content` objects"""

        def process_user_message(message: "UserMessage") -> list[types.Part]:
            user_parts = []
            user_content = message.get("content")
            if isinstance(user_content, str):
                # Check for YouTube URLs in text content
                youtube_urls = self._extract_youtube_urls(user_content)
                if youtube_urls:
                    for url in youtube_urls:
                        user_parts.append(
                            types.Part(file_data=types.FileData(file_uri=url))
                        )

                # Add text content as usual
                user_parts.extend(self._genai_parts_from_text(user_content))
            elif isinstance(user_content, list):
                for c in user_content:
                    c_type = c.get("type")
                    if c_type == "text":
                        c = cast("TextContent", c)
                        # Don't process empty strings.
                        if c_text := c.get("text"):
                            # Check for YouTube URLs in text content
                            youtube_urls = self._extract_youtube_urls(c_text)
                            if youtube_urls:
                                for url in youtube_urls:
                                    user_parts.append(
                                        types.Part(
                                            file_data=types.FileData(file_uri=url)
                                        )
                                    )

                            user_parts.extend(self._genai_parts_from_text(c_text))
                    elif c_type == "image_url":
                        c = cast("ImageContent", c)
                        if img_part := self._genai_part_from_image_url(
                            c.get("image_url").get("url")
                        ):
                            user_parts.append(img_part)
            return user_parts

        def process_assistant_message(
            message: "AssistantMessage", sources: list["Source"] | None
        ) -> list[types.Part]:
            assistant_text = message.get("content")
            if sources:
                assistant_text = self._remove_citation_markers(assistant_text, sources)
            return self._genai_parts_from_text(assistant_text)

        system_prompt = None
        contents = []
        parts = []
        for i, message in enumerate(messages_body):
            role = message.get("role")
            if role == "user":
                message = cast("UserMessage", message)
                parts = process_user_message(message)
            elif role == "assistant":
                message = cast("AssistantMessage", message)
                # Google API's assistant role is "model"
                role = "model"
                # Offset to correct location if system prompt was inside the body's messages list.
                if system_prompt:
                    i -= 1
                sources = None
                if messages_db:
                    message_db = cast("AssistantMessageModel", messages_db[i])
                    sources = message_db.get("sources")
                parts = process_assistant_message(message, sources)
            elif role == "system":
                message = cast("SystemMessage", message)
                system_prompt = message.get("content")
                continue
            else:
                self.log.warning(
                    f"Role {role} is not valid, skipping to the next message."
                )
                continue
            contents.append(types.Content(role=role, parts=parts))
        return contents, system_prompt

    def _genai_part_from_image_url(self, image_url: str) -> types.Part | None:
        """
        Processes an image URL and returns a genai.types.Part object from it
        Handles GCS, data URIs, and standard URLs.
        """
        try:
            if image_url.startswith("gs://"):
                # FIXME: mime type helper would error out here, it only handles filenames.
                return types.Part.from_uri(
                    file_uri=image_url, mime_type=self._get_mime_type(image_url)
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
                # FIXME: mime type helper would error out here too, it only handles filenames.
                return types.Part.from_uri(
                    file_uri=image_url, mime_type=self._get_mime_type(image_url)
                )
        except Exception:
            # TODO: Send warnin toast to user in front-end.
            self.log.exception(f"Error processing image URL: {image_url[:64]}[...]")
            return None

    def _genai_parts_from_text(self, text: str) -> list[types.Part]:
        """
        Turns raw text into list of genai.types.Parts objects.
        Extracts and converts markdown images to parts, preserving text order.
        """
        parts: list[types.Part] = []
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
                    self.log.debug(
                        "Found base64 image link!",
                        mime_type=mime_type,
                        base64_data=match.group(3)[:64] + "[...]",
                    )
                    image_part = types.Part.from_bytes(
                        data=base64.b64decode(base64_data),
                        mime_type=mime_type,
                    )
                    parts.append(image_part)
                except Exception:
                    # TODO: Emit toast
                    self.log.exception("Error decoding base64 image:")

            elif match.group(4):  # File URL
                self.log.debug("Found API image link!", id=match.group(4))
                file_id = match.group(4)
                file_model = Files.get_file_by_id(file_id)

                if file_model is None:
                    # TODO: Emit toast
                    self.log.warning("File with this ID not found.", id=file_id)
                    #  Could add placeholder text here if desired
                    continue  # Skip to the next match

                try:
                    # "continue" above ensures that file_model is not None
                    content_type = file_model.meta.get("content_type")  # type: ignore
                    if content_type is None:
                        # TODO: Emit toast
                        self.log.warning(
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
                    # TODO: Emit toast
                    self.log.exception(
                        "File not found on disk for this ID.",
                        id=file_id,
                        path=file_model.path,
                    )
                except KeyError:
                    # TODO: Emit toast
                    self.log.exception(
                        "Metadata error for this file ID: 'content_type' missing.",
                        id=file_id,
                    )
                except Exception:
                    # TODO: Emit toast
                    self.log.exception("Error processing file with this ID", id=file_id)

            last_pos = match.end()

        # Add remaining text
        remaining_text = text[last_pos:]
        if remaining_text.strip():
            parts.append(types.Part.from_text(text=remaining_text))

        return parts

    # endregion

    # region Model response streaming
    async def _stream_response_generator(
        self,
        response_stream: AsyncIterator[types.GenerateContentResponse],
        __request__: Request,
        __user__: "UserData",
        gen_content_args: dict,
        event_emitter: Callable[["Event"], Awaitable[None]],
        metadata: dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """
        Yields text chunks from the stream and spawns metadata processing task on completion.
        """
        final_response_chunk: types.GenerateContentResponse | None = None
        error_occurred = False

        # Initialize timer variables
        thinking_timer_task = None
        start_time = None
        model_name = gen_content_args.get("model", "")
        # FIXME: This is not needed?
        first_chunk_received = False

        # Check if this is a thinking model and initialize timer if needed
        # TODO: refac
        if self.is_thinking_model(model_name):
            # Emit initial 'Thinking' status
            if self.valves.EMIT_STATUS_UPDATES:
                await self._emit_status(
                    "Thinking...", event_emitter=event_emitter, done=False
                )

            # Record the start time
            start_time = time.time()

            # Start the thinking timer
            # NOTE: It's important to note that the model could not be actually thinking
            # when the status message starts. API could be just slow or the chat data
            # payload could still be uploading.
            if self.valves.EMIT_STATUS_UPDATES:
                thinking_timer_task = asyncio.create_task(
                    self.thinking_timer(event_emitter)
                )

        try:
            async for chunk in response_stream:
                final_response_chunk = chunk

                # Stop the timer when we receive the first chunk
                if not first_chunk_received and thinking_timer_task and start_time:
                    first_chunk_received = True

                    # Cancel the timer task
                    thinking_timer_task.cancel()
                    try:
                        await thinking_timer_task
                    except asyncio.CancelledError:
                        self.log.success(
                            "Thinking timer task successfully cancelled on first chunk."
                        )
                    except Exception:
                        self.log.exception(
                            "Error cancelling thinking timer task on first chunk"
                        )

                    # Calculate elapsed time and emit final status message
                    if self.valves.EMIT_STATUS_UPDATES:
                        total_elapsed = int(time.time() - start_time)
                        if total_elapsed < 60:
                            total_time_str = f"{total_elapsed}s"
                        else:
                            minutes, seconds = divmod(total_elapsed, 60)
                            total_time_str = f"{minutes}m {seconds}s"

                        final_status = f"Thinking completed in {total_time_str}."
                        await self._emit_status(
                            final_status, event_emitter=event_emitter, done=True
                        )

                    # Set timer task to None to avoid duplicate cancellation in finally block
                    thinking_timer_task = None

                if not (candidate := self._get_first_candidate(chunk.candidates)):
                    self.log.warning("Stream chunk has no candidates, skipping.")
                    continue
                if not (parts := candidate.content and candidate.content.parts):
                    self.log.warning(
                        "candidate does not contain content or content.parts, skipping."
                    )
                    continue
                # Process parts and yield text
                for part in parts:
                    # To my knowledge it's not possible for a part to have multiple fields below at the same time.
                    if part.text:
                        yield part.text
                    elif part.inline_data:
                        # _process_image_part returns a Markdown URL.
                        yield (
                            self._process_image_part(
                                part.inline_data,
                                gen_content_args,
                                __user__,
                                __request__,
                            )
                            or ""
                        )
                    elif part.executable_code:
                        yield (
                            self._process_executable_code_part(part.executable_code)
                            or ""
                        )
                    elif part.code_execution_result:
                        yield (
                            self._process_code_execution_result_part(
                                part.code_execution_result
                            )
                            or ""
                        )
        except Exception as e:
            error_occurred = True
            error_msg = f"Stream ended with error: {e}"
            await self._emit_error(error_msg, event_emitter)
        finally:
            # Cancel the timer task if it exists and wasn't already cancelled
            # TODO: Can this possibly happen?
            if thinking_timer_task:
                thinking_timer_task.cancel()
                try:
                    await thinking_timer_task
                except asyncio.CancelledError:
                    self.log.success(
                        "Thinking timer task successfully cancelled in finally block."
                    )
                except Exception:
                    self.log.exception(
                        "Error cancelling thinking timer task in finally block"
                    )

            if not error_occurred:
                self.log.success(f"Stream finished.")
            try:
                # Catch and emit any errors that might happen here as a toast message.
                await self._do_post_processing(
                    # Metadata about the model response is always in the final chunk of the stream.
                    final_response_chunk,
                    event_emitter,
                    metadata,
                    __request__,
                    error_occurred,
                )
            except Exception as e:
                error_msg = f"Post-processing failed with error:\n\n{e}"
                # Using toast here in order to keep the inital AI response untouched.
                await self._emit_toast(error_msg, event_emitter, "error")
                self.log.exception(error_msg)
            self.log.debug("AsyncGenerator finished.")

    async def _do_post_processing(
        self,
        model_response: types.GenerateContentResponse | None,
        event_emitter: Callable[["Event"], Awaitable[None]],
        metadata: dict[str, Any],
        request: Request,
        stream_error_happened: bool = False,
    ):
        """Handles emitting usage, grounding, and sources after the main response/stream is done."""
        self.log.info("Post-processing the model response.")
        if stream_error_happened:
            self.log.warning(
                "An error occured during the stream, cannot do post-processing."
            )
            # All the needed metadata is always in the last chunk, so if error happened then we cannot do anything.
            return
        if not model_response:
            self.log.warning("model_response is empty, cannot do post-processing.")
            return
        if not (candidate := self._get_first_candidate(model_response.candidates)):
            self.log.warning(
                "Response does not contain any canditates. Cannot do post-processing."
            )
            return

        finish_reason = candidate.finish_reason
        if finish_reason not in (
            types.FinishReason.STOP,
            types.FinishReason.MAX_TOKENS,
        ):
            # MAX_TOKENS is often acceptable, but others might indicate issues.
            error_msg = f"Stream finished with sus reason:\n\n{finish_reason}."
            await self._emit_toast(error_msg, event_emitter, "error")
            self.log.error(error_msg)
            return
        else:
            self.log.debug(f"Response has correct finish reason: {finish_reason}.")

        # Emit token usage data.
        if usage_event := self._get_usage_data_event(model_response):
            self._log_with_data("DEBUG", "Emitting usage data:", usage_event)
            # TODO: catch potential errors?
            await event_emitter(usage_event)
        self._add_grounding_data_to_state(model_response, metadata, request)

    def _add_grounding_data_to_state(
        self,
        response: types.GenerateContentResponse,
        chat_metadata: dict[str, Any],
        request: Request,
    ):
        candidate = self._get_first_candidate(response.candidates)
        grounding_metadata_obj = candidate.grounding_metadata if candidate else None

        chat_id: str = chat_metadata.get("chat_id", "")
        message_id: str = chat_metadata.get("message_id", "")
        storage_key = f"grounding_{chat_id}_{message_id}"

        if grounding_metadata_obj:
            self.log.debug(
                f"Found grounding metadata. Storing in in request's app state using key {storage_key}."
            )
            # Using shared `request.app.state` to pass grounding metadata to Filter.outlet.
            # This is necessary because the Pipe finishes during the initial `/api/completion` request,
            # while Filter.outlet is invoked by a separate, later `/api/chat/completed` request.
            # `request.state` does not persist across these distinct request lifecycles.
            app_state: State = request.app.state
            app_state._state[storage_key] = grounding_metadata_obj
        else:
            self.log.debug(f"Response {message_id} does not have grounding metadata.")

    def _get_first_candidate(
        self, candidates: list[types.Candidate] | None
    ) -> types.Candidate | None:
        """Selects the first candidate, logging a warning if multiple exist."""
        if not candidates:
            # Logging warnings is handled downstream.
            return None
        if len(candidates) > 1:
            self.log.warning(
                "Multiple candidates found, defaulting to first candidate."
            )
        return candidates[0]

    def _process_image_part(
        self, inline_data, gen_content_args: dict, user: "UserData", request: Request
    ) -> str | None:
        """Handles image data conversion to markdown."""
        mime_type = inline_data.mime_type
        image_data = inline_data.data

        if self.valves.USE_FILES_API:
            image_url = self._upload_image(
                image_data,
                mime_type,
                gen_content_args.get("model", ""),
                "Not implemented yet. TAKE IT FROM gen_content_args contents",
                user["id"],
                request,
            )
            return f"![Generated Image]({image_url})" if image_url else None
        else:
            encoded = base64.b64encode(image_data).decode()
            return f"![Generated Image](data:{mime_type};base64,{encoded})"

    def _process_executable_code_part(
        self, executable_code_part: types.ExecutableCode | None
    ) -> str | None:
        """
        Processes an executable code part and returns the formatted string representation.
        """

        if not executable_code_part:
            return None

        lang_name = "python"  # Default language
        if executable_code_part_lang_enum := executable_code_part.language:
            if lang_name := executable_code_part_lang_enum.name:
                lang_name = executable_code_part_lang_enum.name.lower()
            else:
                self.log.warning(
                    f"Could not extract language name from {executable_code_part_lang_enum}. Default to python."
                )
        else:
            self.log.warning("Language Enum is None, defaulting to python.")

        if executable_code_part_code := executable_code_part.code:
            return f"```{lang_name}\n{executable_code_part_code.rstrip()}\n```\n\n"
        return ""

    def _process_code_execution_result_part(
        self, code_execution_result_part: types.CodeExecutionResult | None
    ) -> str | None:
        """
        Processes a code execution result part and returns the formatted string representation.
        """

        if not code_execution_result_part:
            return None

        if code_execution_result_part_output := code_execution_result_part.output:
            return f"**Output:**\n\n```\n{code_execution_result_part_output.rstrip()}\n```\n\n"
        else:
            return None

    def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        model: str,
        prompt: str,
        user_id: str,
        __request__: Request,
    ) -> str | None:
        """
        Helper method that uploads the generated image to a storage provider configured inside Open WebUI settings.
        Returns the url to uploaded image.
        """
        image_format = mimetypes.guess_extension(mime_type)
        id = str(uuid.uuid4())
        # TODO: Better filename? Prompt as the filename?
        name = os.path.basename(f"generated-image{image_format}")
        imagename = f"{id}_{name}"
        image = io.BytesIO(image_data)
        image_metadata = {
            "model": model,
            "prompt": prompt,
        }

        # Upload the image to user configured storage provider.
        self.log.info("Uploading the model generated image to Open WebUI backend.")
        self.log.debug("Uploading to the configured storage provider.")
        try:
            contents, image_path = Storage.upload_file(image, imagename)
        except Exception:
            error_msg = f"Error occurred during upload to the storage provider."
            # TODO: emit toast
            self.log.exception(error_msg)
            return None
        # Add the image file to files database.
        self.log.debug("Adding the image file to Open WebUI files database.")
        file_item = Files.insert_new_file(
            user_id,
            FileForm(
                id=id,
                filename=name,
                path=image_path,
                meta={
                    "name": name,
                    "content_type": mime_type,
                    "size": len(contents),
                    "data": image_metadata,
                },
            ),
        )
        if not file_item:
            self.log.warning(
                "Files.insert_new_file did not return anything. Image upload to Open WebUI database likely failed."
            )
            return None
        # Get the image url.
        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        self.log.success("Image upload finished!")
        return image_url

    # endregion

    # region Client initialization and model retrival from Google API
    def _get_genai_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> genai.Client | None:
        client = None
        api_key = api_key if api_key else self.valves.GEMINI_API_KEY
        base_url = base_url if base_url else self.valves.GEMINI_API_BASE_URL
        if api_key:
            http_options = types.HttpOptions(base_url=base_url)
            try:
                client = genai.Client(
                    api_key=api_key,
                    http_options=http_options,
                )
                self.log.success("Genai client successfully initialized!")
            except Exception:
                self.log.exception("genai client initialization failed.")
        else:
            self.log.error("GEMINI_API_KEY is not set.")
        return client

    def _get_user_client(self, __user__: "UserData") -> genai.Client | None:
        # Register a user specific client if they have added their own API key.
        user_valves: Pipe.UserValves | None = __user__.get("valves")
        if (
            user_valves
            and user_valves.GEMINI_API_KEY
            and not self.clients.get(__user__["id"])
        ):
            self.log.info(
                f'Creating a new genai client for user {__user__.get("email")}'
            )
            self.clients[__user__.get("id")] = self._get_genai_client(
                api_key=user_valves.GEMINI_API_KEY,
                base_url=user_valves.GEMINI_API_BASE_URL,
            )
            self._log_with_data(
                "DEBUG", "Genai clients dict now looks like this:", self.clients
            )
        if user_client := self.clients.get(__user__.get("id")):
            return user_client
        else:
            return self.clients.get("default")

    async def _get_genai_models(self) -> list[types.Model]:
        """
        Gets valid Google models from the API.
        Returns a list of `genai.types.Model` objects.
        """
        google_models = None
        client = self.clients.get("default")
        if not client:
            self.log.warning("There is no usable genai client. Trying to create one.")
            # Try to create a client one more time.
            if client := self._get_genai_client():
                self.clients["default"] = client
            else:
                self.log.error("Can't initialize the client, returning no models.")
                return []
        # This executes if we have a working client.
        try:
            google_models = await client.aio.models.list(config={"query_base": True})
        except Exception:
            self.log.exception("Retriving models from Google API failed.")
            return []
        self.log.info(
            f"Retrieved {len(google_models)} models from Gemini Developer API."
        )
        # Filter Google models list down to generative models only.
        return [
            model
            for model in google_models
            if model.supported_actions and "generateContent" in model.supported_actions
        ]

    def _filter_models(self, google_models: list[types.Model]) -> list["ModelData"]:
        """
        Filters the genai model list down based on configured white- and blacklist.
        Returns a list[dict] that can be directly returned by the `pipes` method.
        """
        if not google_models:
            # TODO: Make error messages more specific.
            error_msg = "Error during getting the models from Google API, check logs."
            return [self._return_error_model(error_msg, exception=False)]

        self.last_whitelist = self.valves.MODEL_WHITELIST
        self.last_blacklist = self.valves.MODEL_BLACKLIST
        whitelist = (
            self.valves.MODEL_WHITELIST.replace(" ", "").split(",")
            if self.valves.MODEL_WHITELIST
            else []
        )
        blacklist = (
            self.valves.MODEL_BLACKLIST.replace(" ", "").split(",")
            if self.valves.MODEL_BLACKLIST
            else []
        )
        return [
            {
                "id": self._strip_prefix(model.name),
                "name": model.display_name,
                "description": model.description,
            }
            for model in google_models
            if model.name
            and model.display_name
            and any(fnmatch.fnmatch(model.name, f"models/{w}") for w in whitelist)
            and not any(fnmatch.fnmatch(model.name, f"models/{b}") for b in blacklist)
        ]

    def _get_safety_settings(self, model_name: str) -> list[types.SafetySetting]:
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

        self.log.debug(
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
        # TODO: [refac] pointless helper, remove.
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            return stripped
        except Exception:
            error_msg = "Error stripping prefix, using the original model name."
            self.log.exception(error_msg)
            return model_name

    # endregion

    # region Citations

    def _remove_citation_markers(self, text: str, sources: list["Source"]) -> str:
        original_text = text
        processed: set[str] = set()
        for source in sources:
            supports = [
                metadata["supports"]
                for metadata in source.get("metadata", [])
                if "supports" in metadata
            ]
            supports = [item for sublist in supports for item in sublist]
            for support in supports:
                support = types.GroundingSupport(**support)
                indices = support.grounding_chunk_indices
                segment = support.segment
                if not (indices and segment):
                    continue
                segment_text = segment.text
                if not segment_text:
                    continue
                # Using a shortened version because user could edit the assistant message in the front-end.
                # If citation segment get's edited, then the markers would not be removed. Shortening reduces the
                # chances of this happening.
                segment_end = segment_text[-32:]
                if segment_end in processed:
                    continue
                processed.add(segment_end)
                citation_markers = "".join(f"[{index + 1}]" for index in indices)
                # Find the position of the citation markers in the text
                pos = text.find(segment_text + citation_markers)
                if pos != -1:
                    # Remove the citation markers
                    text = (
                        text[: pos + len(segment_text)]
                        + text[pos + len(segment_text) + len(citation_markers) :]
                    )
        trim = len(original_text) - len(text)
        self.log.debug(
            f"Citation removal finished. Returning text str that is {trim} character shorter than the original input."
        )
        return text

    # endregion

    # region Usage data
    def _get_usage_data_event(
        self,
        response: types.GenerateContentResponse,
    ) -> "ChatCompletionEvent | None":
        """
        Extracts usage data from a GenerateContentResponse object.
        Returns None if any of the core metrics (prompt_tokens, completion_tokens, total_tokens)
        cannot be reliably determined.

        Args:
            response: The GenerateContentResponse object.

        Returns:
            A dictionary containing the usage data, formatted as a ResponseUsage type,
            or None if any core metrics are missing.
        """

        if not response.usage_metadata:
            self.log.warning(
                "Usage_metadata is missing from the response. Cannot reliably determine usage."
            )
            return None

        usage_data = response.usage_metadata.model_dump()
        usage_data["prompt_tokens"] = usage_data.pop("prompt_token_count")
        usage_data["completion_tokens"] = usage_data.pop("candidates_token_count")
        usage_data["total_tokens"] = usage_data.pop("total_token_count")
        # Remove null values and turn ModalityTokenCount into dict.
        for k, v in usage_data.copy().items():
            if k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                continue
            if not v:
                del usage_data[k]

        completion_event: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {"usage": usage_data},
        }
        return completion_event

    # endregion

    # region Other helpers
    def _log_with_data(
        self,
        log_level: str,
        msg: str,
        data: Any,
        truncate: bool = True,
        truncate_length: int = 512,
    ):
        """
        Logs a message and then lazily logs potentially complex data, formatted as JSON.

        The data logging uses the same log level, is raw (no Loguru formatting added),
        and only occurs if the log level is enabled. The data serialization can optionally
        truncate long strings/bytes.

        Both the initial message and the data log will show the caller of this
        `log_with_data` function in their trace information.

        Args:
            log_level: The logging level (e.g., "DEBUG", "INFO").
            msg: The primary log message.
            data: The data structure (dict, list, Pydantic model) to serialize and self.log.
            truncate: Whether to truncate long strings/bytes in the data (default True).
            truncate_length: Max length before truncation (default 64).
        """
        # Log the primary message normally, using depth=1 to point to the caller of log_with_data
        self.log.opt(depth=1).log(log_level, msg)

        # Lazy log the serialized data raw, using depth=1 to point to the caller of log_with_data
        self.log.opt(raw=True, lazy=True, depth=1).log(
            log_level,
            "{x}\n",  # Use a format string for the lambda argument, add newline for clarity
            x=lambda: self._serialize_data(
                data, truncate=truncate, max_length=truncate_length
            ),
        )

    def _serialize_data(self, data: Any, truncate: bool, max_length: int) -> str:
        """
        Recursively processes data (dicts, lists, Pydantic models) and returns
        a formatted JSON string. Optionally truncates long strings/bytes fields.
        The original input data remains unmodified.

        Args:
            data: A dictionary, list, or Pydantic BaseModel instance.
            truncate: If True, strings/bytes exceeding max_length will be truncated.
            max_length: The maximum length for strings/bytes before truncation (if truncate=True).

        Returns:
            A nicely formatted JSON string representation of the processed data.
        """

        def process_item(item: Any, truncate_flag: bool, length: int) -> Any:
            if isinstance(item, BaseModel):
                # Convert Pydantic model to dict and process recursively
                item_dict = item.model_dump()
                return process_item(item_dict, truncate_flag, length)
            elif isinstance(item, dict):
                # Process dictionary items recursively
                return {
                    key: process_item(value, truncate_flag, length)
                    for key, value in item.items()
                }
            elif isinstance(item, list):
                # Process list items recursively
                return [
                    process_item(element, truncate_flag, length) for element in item
                ]
            elif isinstance(item, str):
                # Truncate string if requested and necessary
                if truncate_flag and len(item) > length:
                    truncated_len = len(item) - length
                    return f"{item[:length]}[{truncated_len} chars truncated]"
                return (
                    item  # Return original string if not truncated or not long enough
                )
            elif isinstance(item, bytes):
                # Convert bytes to hex and truncate if requested and necessary
                hex_str = item.hex()
                if truncate_flag and len(hex_str) > length:
                    truncated_len = len(hex_str) - length
                    return f"{hex_str[:length]}[{truncated_len} chars truncated]"
                else:
                    return hex_str  # Return original hex string
            else:
                # Return other types as is
                return item

        # Deep copy to avoid modifying the original data structure
        copied_data = copy.deepcopy(data)
        processed = process_item(copied_data, truncate, max_length)
        # Serialize the processed data to a formatted JSON string
        return json.dumps(processed, indent=2, default=str)

    def _setup_plugin_logger(self):
        """
        Creates an independent logger instance for this plugin by temporarily
        removing and restoring global handlers during a deep copy.
        """
        try:
            # 1. Remove all handlers from the global logger.
            #    This returns it to a "clean" state for copying.
            global_logger.remove(None)

            # 2. Create a deep copy of the now handler-less global logger.
            self.log: "Logger" = copy.deepcopy(global_logger)
        finally:
            # 3. Restore the backend's global logger configuration *immediately*.
            #    Call the same function the backend uses for its initial setup.
            try:
                start_logger()  # Or whatever function sets up the backend logger
            except Exception as e:
                # Log an error if restoring global handlers fails, but proceed
                print(
                    f"CRITICAL ERROR: Failed to restore global logger: {e}",
                    file=sys.stderr,
                )
                # Try adding a basic handler back to global_logger.
                global_logger.add(sys.stderr, level="INFO")
        # 4. Configure the *copied* plugin logger instance. It starts empty, so no need to remove again.
        try:
            self.log.add(
                sys.stdout,
                level=self.log_level,
                format=stdout_format,  # Use the backend's format
            )
            self.log.debug(
                f"Initialized independent logger for {__name__} with level {self.log_level}."
            )
        except Exception as e:
            print(
                f"ERROR: Failed to add handler to plugin logger: {e}", file=sys.stderr
            )

    def _get_mime_type(self, file_uri: str) -> str:
        """
        Determines MIME type based on file extension using the mimetypes module.
        """
        mime_type, encoding = mimetypes.guess_type(file_uri)
        if mime_type is None:
            return "application/octet-stream"  # Default MIME type if unknown
        return mime_type

    def _extract_youtube_urls(self, text: str) -> list[str]:
        """
        Extracts YouTube URLs from a given text.
        Supports standard youtube.com/watch?v= URLs and shortened youtu.be URLs
        """
        youtube_urls = []
        # Match standard YouTube URLs
        for match in re.finditer(
            r"https?://(?:www\.)?youtube\.com/watch\?v=[^&\s]+", text
        ):
            youtube_urls.append(match.group(0))
        # Match shortened YouTube URLs
        for match in re.finditer(r"https?://(?:www\.)?youtu\.be/[^&\s]+", text):
            youtube_urls.append(match.group(0))

        if youtube_urls:
            # TODO: toast
            self.log.info(f"Extracted YouTube URLs: {youtube_urls}")

        return youtube_urls

    # endregion

    # endregion
