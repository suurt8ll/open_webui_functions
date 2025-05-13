"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Supports native image generation, grounding with Google Search and streaming. Uses google-genai.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.18.0
requirements: google-genai==1.13.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

# Supported features:
#   - Native image generation (image output), use "gemini-2.0-flash-exp-image-generation"
#   - Document understanding (PDF and plaintext files). (Gemini Manifold Companion >= 1.4.0 required)
#   - Display citations in the front-end.
#   - Image input
#   - YouTube video input (automatically detects youtube.com and youtu.be URLs in messages)
#   - Streaming
#   - Grounding with Google Search (this requires installing "Gemini Manifold Companion" >= 1.2.0 filter, see GitHub README)
#   - Permissive safety settings (Gemini Manifold Companion >= 1.3.0 required)
#   - Each user can decide to use their own API key.
#   - Token usage data
#   - Code execution tool. (Gemini Manifold Companion >= 1.1.0 required)

# Features that are supported by API but not yet implemented in the manifold:
#   TODO Audio input support.
#   TODO Video input support (other than YouTube URLs).

import inspect
from google import genai
from google.genai import types

import asyncio
import copy
import json
import time
from functools import cache
from fastapi.datastructures import State
import io
import mimetypes
import os
import uuid
import base64
import re
import fnmatch
import sys
from loguru import logger
from fastapi import Request
import pydantic_core
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

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


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
        THINKING_BUDGET: int = Field(
            ge=0,
            le=24576,
            default=8192,
            description="""Gemini 2.5 Flash only. Indicates the thinking budget in tokens.
            0 means no thinking. Default value is 8192.
            See <https://cloud.google.com/vertex-ai/generative-ai/docs/thinking> for more.""",
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
        THINKING_BUDGET: int = Field(
            ge=0,
            le=24576,
            default=8192,
            description="""Gemini 2.5 Flash only. Indicates the thinking budget in tokens.
            0 means no thinking. Default value is 8192.
            See <https://cloud.google.com/vertex-ai/generative-ai/docs/thinking> for more.""",
        )
        # TODO: Add more options that can be changed by the user.

    def __init__(self):
        # This hack makes the valves values available to the `__init__` method.
        # TODO: Get the id from the frontmatter instead of hardcoding it.
        valves = Functions.get_function_valves_by_id("gemini_manifold_google_genai")
        self.valves = self.Valves(**(valves if valves else {}))
        self._add_log_handler(self.valves.LOG_LEVEL)
        # Initialize the genai client with default API given in Valves.
        self.clients = {"default": self._get_genai_client()}

        log.success("Function has been initialized.")
        log.trace("Full self object:", payload=self.__dict__)

    async def pipes(self) -> list["ModelData"]:
        """Register all available Google models."""
        # Detect log level change inside self.valves
        self._add_log_handler(self.valves.LOG_LEVEL)

        # Clear cache if caching is disabled
        if not self.valves.CACHE_MODELS:
            log.debug("CACHE_MODELS is False, clearing model cache.")
            self._get_genai_models.cache_clear()

        log.info("Fetching and filtering models from Google API.")
        # Get and filter models (potentially cached based on API key, base URL, white- and blacklist)
        try:
            filtered_models = await self._get_genai_models(
                api_key=self.valves.GEMINI_API_KEY,
                base_url=self.valves.GEMINI_API_BASE_URL,
                whitelist_str=self.valves.MODEL_WHITELIST,
                blacklist_str=self.valves.MODEL_BLACKLIST,
            )
        except RuntimeError:
            error_msg = "Error getting the models from Google API, check the logs."
            return [self._return_error_model(error_msg, exception=False)]

        log.info(
            f"Finished processing models. Returning {len(filtered_models)} models to Open WebUI."
        )
        log.debug("Model list:", payload=filtered_models, _log_truncation_enabled=False)
        return filtered_models

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
                log.info("Using genai client with the default API key.")
        else:
            log.info(f"Using genai client with user {__user__.get('email')} API key.")

        log.trace("__metadata__:", payload=__metadata__)
        # Check if user is chatting with an error model for some reason.
        if "error" in __metadata__["model"]["id"]:
            error_msg = f"There has been an error during model retrival phase: {str(__metadata__['model'])}"
            raise ValueError(error_msg)

        # Get the message history directly from the backend.
        # This allows us to see data about sources and files data.
        chat_id = __metadata__.get("chat_id", "")
        if chat := Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=__user__["id"]):
            chat_content: "ChatObjectDataTD" = chat.chat  # type: ignore
            # Last message is the upcoming assistant response, at this point in the logic it's empty.
            messages_db = chat_content.get("messages")[:-1]
        else:
            warn_msg = f"Chat with ID - {chat_id} - not found. Can't filter out the citation marks."
            log.warning(warn_msg)
            messages_db = None

        system_prompt = self._pop_system_prompt(body.get("messages"))

        if messages_db and len(messages_db) != len(body.get("messages")):
            warn_msg = (
                f"Messages in the body ({len(body.get('messages'))}) and "
                f"messages in the database ({len(messages_db)}) do not match. "
                "This is likely due to a bug in Open WebUI. "
                "Cannot filter out citation marks or upload files."
            )
            log.warning(warn_msg)
            await self._emit_toast(warn_msg, __event_emitter__, "warning")
            messages_db = None

        features = __metadata__.get("features", {}) or {}
        log.info(
            "Converting Open WebUI's `body` dict into list of `Content` objects that `google-genai` understands."
        )
        contents = await self._genai_contents_from_messages(
            body.get("messages"),
            messages_db,
            features.get("upload_documents", False),
            __event_emitter__,
        )

        # Assemble GenerateContentConfig
        safety_settings: list[types.SafetySetting] | None = __metadata__.get(
            "safety_settings"
        )
        model_name = re.sub(r"^.*?[./]", "", body.get("model", ""))

        # Get UserValves if it exists.
        user_valves: Pipe.UserValves | None = __user__.get("valves")

        # API does not stream thoughts sadly. See https://github.com/googleapis/python-genai/issues/226#issuecomment-2631657100
        thinking_conf = None
        if model_name == "gemini-2.5-flash-preview-04-17":
            log.info(f"Model ID '{model_name}' allows adjusting the thinking settings.")
            thinking_conf = types.ThinkingConfig(
                thinking_budget=(
                    user_valves.THINKING_BUDGET
                    if user_valves
                    else self.valves.THINKING_BUDGET
                ),
                include_thoughts=None,
            )
        # TODO: Take defaults from the general front-end config.
        gen_content_conf = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            max_output_tokens=body.get("max_tokens"),
            stop_sequences=body.get("stop"),
            safety_settings=safety_settings,
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
                log.warning(
                    "Image Generation model does not support the system prompt message! Removing the system prompt."
                )
        gen_content_conf.tools = []
        if features.get("google_search_tool"):
            log.info("Using grounding with Google Search as a Tool.")
            gen_content_conf.tools.append(
                types.Tool(google_search=types.GoogleSearch())
            )
        elif features.get("google_search_retrieval"):
            log.info("Using grounding with Google Search Retrieval.")
            gs = types.GoogleSearchRetrieval(
                dynamic_retrieval_config=types.DynamicRetrievalConfig(
                    dynamic_threshold=features.get("google_search_retrieval_threshold")
                )
            )
            gen_content_conf.tools.append(types.Tool(google_search_retrieval=gs))
        # NB: It is not possible to use both Search and Code execution at the same time,
        # however, it can be changed later, so let's just handle it as a common error
        if features.get("google_code_execution"):
            log.info("Using code execution on Google side.")
            gen_content_conf.tools.append(
                types.Tool(code_execution=types.ToolCodeExecution())
            )
        gen_content_args = {
            "model": model_name,
            "contents": contents,
            "config": gen_content_conf,
        }
        log.debug("Passing these args to the Google API:", payload=gen_content_args)

        if body.get("stream", False):
            # Streaming response
            response_stream: AsyncIterator[types.GenerateContentResponse] = (
                await client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
            )
            log.info("Streaming enabled. Returning AsyncGenerator.")
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
                log.info("Non-streaming response finished successfully!")
                log.debug("Non-streaming response:", payload=res)
                await self._do_post_processing(
                    res, __event_emitter__, __metadata__, __request__
                )
                log.info(
                    "Streaming disabled. Returning full response as str. "
                    "With that Pipe.pipe method has finished it's run!"
                )
                return raw_text
            else:
                warn_msg = "Non-streaming response did not have any text inside it."
                raise ValueError(warn_msg)

    # region 1. Helper methods inside the Pipe class

    # region 1.1 Client initialization and model retrival from Google API
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
                log.success("Genai client successfully initialized!")
            except Exception:
                log.exception("genai client initialization failed.")
        else:
            log.error("GEMINI_API_KEY is not set.")
        return client

    def _get_user_client(self, __user__: "UserData") -> genai.Client | None:
        # Register a user specific client if they have added their own API key.
        user_valves: Pipe.UserValves | None = __user__.get("valves")
        if (
            user_valves
            and user_valves.GEMINI_API_KEY
            and not self.clients.get(__user__["id"])
        ):
            log.info(f"Creating a new genai client for user {__user__.get('email')}")
            self.clients[__user__.get("id")] = self._get_genai_client(
                api_key=user_valves.GEMINI_API_KEY,
                base_url=user_valves.GEMINI_API_BASE_URL,
            )
            log.debug("Genai clients dict now looks like this:", payload=self.clients)
        if user_client := self.clients.get(__user__.get("id")):
            return user_client
        else:
            return self.clients.get("default")

    def _return_error_model(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> "ModelData":
        """Returns a placeholder model for communicating error inside the pipes method to the front-end."""
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        return {
            "id": "error",
            "name": "[gemini_manifold] " + error_msg,
            "description": error_msg,
        }

    @cache
    async def _get_genai_models(
        self,
        api_key: str | None,
        base_url: str,
        whitelist_str: str,
        blacklist_str: str | None,
    ) -> list["ModelData"]:
        """
        Gets valid Google models from the API and filters them based on configured white- and blacklist.
        Returns a list[dict] that can be directly returned by the `pipes` method.
        The result is cached based on the provided api_key, base_url, whitelist_str, and blacklist_str.
        """
        # Get a client using the provided API key and base URL
        client = self._get_genai_client(api_key=api_key, base_url=base_url)

        if not client:
            log.error(
                "Can't initialize the client with provided API key and base URL, returning no models."
            )
            raise RuntimeError

        # This executes if we have a working client.
        google_models_pager = None
        try:
            # Get the AsyncPager object
            google_models_pager = await client.aio.models.list(
                config={"query_base": True}
            )
        except Exception:
            log.exception("Retrieving models from Google API failed.")
            raise RuntimeError

        # Iterate the pager to get the full list of models
        # This is where the actual API calls for subsequent pages happen if needed
        try:
            all_google_models = [model async for model in google_models_pager]
        except Exception:
            log.exception("Iterating Google models pager failed.")
            raise RuntimeError

        log.info(
            f"Retrieved {len(all_google_models)} models from Gemini Developer API."
        )
        log.trace("All models returned by Google:", payload=all_google_models)
        # Filter Google models list down to generative models only.
        generative_models = [
            model
            for model in all_google_models
            if model.supported_actions and "generateContent" in model.supported_actions
        ]

        # Raise if there are not generative models
        if not generative_models:
            raise RuntimeError

        # Filter based on whitelist and blacklist
        whitelist = whitelist_str.replace(" ", "").split(",") if whitelist_str else []
        blacklist = blacklist_str.replace(" ", "").split(",") if blacklist_str else []
        filtered_models: list["ModelData"] = [
            {
                "id": re.sub(r"^.*?[./]", "", model.name),
                "name": model.display_name,
                "description": model.description,
            }
            for model in generative_models
            if model.name
            and model.display_name
            and any(fnmatch.fnmatch(model.name, f"models/{w}") for w in whitelist)
            and not any(fnmatch.fnmatch(model.name, f"models/{b}") for b in blacklist)
        ]
        log.info(
            f"Filtered {len(generative_models)} raw models down to {len(filtered_models)} models based on white/blacklists."
        )
        return filtered_models

    # endregion 1.1 Client initialization and model retrival from Google API

    # region 1.2 Open WebUI's body.messages -> list[genai.types.Content] conversion

    def _pop_system_prompt(self, messages: list["Message"]) -> str | None:
        """
        Pops the system prompt from the messages list.
        System prompt is always the first message in the list.
        """
        if not messages:
            return None
        first_message = messages[0]
        if first_message.get("role") == "system":
            first_message = cast("SystemMessage", first_message)
            system_prompt = first_message.get("content")
            log.info("System prompt found in the messages list.")
            log.debug("System prompt:", payload=system_prompt)
            messages.pop(0)
            return system_prompt
        return None

    async def _genai_contents_from_messages(
        self,
        messages_body: list["Message"],
        messages_db: list["ChatMessageTD"] | None,
        upload_documents: bool,
        event_emitter: Callable[["Event"], Awaitable[None]],
    ) -> list[types.Content]:
        """Transforms `body.messages` list into list of `genai.types.Content` objects"""

        contents = []
        parts = []
        for i, message in enumerate(messages_body):
            role = message.get("role")
            if role == "user":
                message = cast("UserMessage", message)
                files = []
                if messages_db:
                    message_db = messages_db[i]
                    if upload_documents:
                        files = message_db.get("files", [])
                parts = await self._process_user_message(message, files, event_emitter)
            elif role == "assistant":
                message = cast("AssistantMessage", message)
                # Google API's assistant role is "model"
                role = "model"
                sources = None
                if messages_db:
                    message_db = messages_db[i]
                    sources = message_db.get("sources")
                parts = self._process_assistant_message(message, sources)
            else:
                warn_msg = f"Message {i} has an invalid role: {role}. Skipping to the next message."
                log.warning(warn_msg)
                await self._emit_toast(warn_msg, event_emitter, "warning")
                continue
            contents.append(types.Content(role=role, parts=parts))
        return contents

    async def _process_user_message(
        self,
        message: "UserMessage",
        files: list["FileAttachmentTD"],
        event_emitter: Callable[["Event"], Awaitable[None]],
    ) -> list[types.Part]:
        user_parts = []

        if files:
            log.info(f"Adding {len(files)} files to the user message.")
        for file in files:
            log.debug("Processing file:", payload=file)
            file_id = file.get("file", {}).get("id")
            document_bytes, mime_type = self._get_file_data(file_id)
            if not document_bytes or not mime_type:
                # Warnings are logged by the method above.
                continue

            if mime_type.startswith("text/") or mime_type == "application/pdf":
                log.debug(
                    f"{mime_type} is supported by Google API! Creating `types.Part` model for it."
                )
                user_parts.append(
                    types.Part.from_bytes(data=document_bytes, mime_type=mime_type)
                )
            else:
                warn_msg = f"{mime_type} is not supported by Google API! Skipping file {file_id}."
                log.warning(warn_msg)
                await self._emit_toast(warn_msg, event_emitter, "warning")

        user_content = message.get("content")
        if isinstance(user_content, str):
            user_content_list: list["Content"] = [
                {"type": "text", "text": user_content}
            ]
        elif isinstance(user_content, list):
            user_content_list = user_content
        else:
            warn_msg = f"User message content is not a string or list, skipping to the next message."
            log.warning(warn_msg)
            await self._emit_toast(warn_msg, event_emitter, "warning")
            return user_parts

        for c in user_content_list:
            c_type = c.get("type")
            if c_type == "text":
                c = cast("TextContent", c)
                # Don't process empty strings.
                if c_text := c.get("text"):
                    # Check for YouTube URLs in text content
                    # FIXME: Better ordering of Parts here.
                    youtube_urls = self._extract_youtube_urls(c_text)
                    if youtube_urls:
                        for url in youtube_urls:
                            user_parts.append(
                                types.Part(file_data=types.FileData(file_uri=url))
                            )

                    user_parts.extend(self._genai_parts_from_text(c_text))
            elif c_type == "image_url":
                c = cast("ImageContent", c)
                if img_part := self._genai_part_from_image_url(
                    c.get("image_url").get("url")
                ):
                    user_parts.append(img_part)
            else:
                warn_msg = f"User message content type {c_type} is not supported, skipping to the next message."
                log.warning(warn_msg)
                await self._emit_toast(warn_msg, event_emitter, "warning")
                continue

        return user_parts

    def _process_assistant_message(
        self, message: "AssistantMessage", sources: list["Source"] | None
    ) -> list[types.Part]:
        assistant_text = message.get("content")
        if sources:
            assistant_text = self._remove_citation_markers(assistant_text, sources)
        return self._genai_parts_from_text(assistant_text)

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
            log.exception(f"Error processing image URL: {image_url[:64]}[...]")
            return None

    def _genai_parts_from_text(self, text: str) -> list[types.Part]:
        """
        Turns raw text into list of genai.types.Parts objects.
        Extracts and converts markdown images to parts, preserving text order.
        """
        # TODO: Extract generated code and it's result into genai parts too.

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
                    log.debug(
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
                    log.exception("Error decoding base64 image:")
            elif match.group(4):  # File URL
                log.debug("Found API image link!", id=match.group(4))
                file_id = match.group(4)
                image_bytes, mime_type = self._get_file_data(file_id)
                if not image_bytes or not mime_type:
                    continue
                image_part = types.Part.from_bytes(
                    data=image_bytes, mime_type=mime_type
                )
                parts.append(image_part)

            last_pos = match.end()

        # Add remaining text
        remaining_text = text[last_pos:]
        if remaining_text.strip():
            parts.append(types.Part.from_text(text=remaining_text))

        return parts

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
            log.info(f"Extracted YouTube URLs: {youtube_urls}")

        return youtube_urls

    # endregion 1.2 Open WebUI's body.messages -> list[genai.types.Content] conversion

    # region 1.3 Model response streaming
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

        # Get thinking budget, UserValves takes priority.
        user_valves: Pipe.UserValves | None = __user__.get("valves")
        thinking_budget = (
            user_valves.THINKING_BUDGET if user_valves else self.valves.THINKING_BUDGET
        )

        # Start thinking timer (model name check is inside this method).
        model_name = gen_content_args.get("model", "")
        start_time, thinking_timer_task = await self._start_thinking_timer(
            model_name, event_emitter, thinking_budget
        )

        try:
            async for chunk in response_stream:
                final_response_chunk = chunk

                # Stop the timer when we receive the first chunk
                await self._cancel_thinking_timer(
                    thinking_timer_task,
                    start_time,
                    event_emitter,
                    model_name,
                    thinking_budget,
                )
                # Set timer task to None to avoid duplicate cancellation in finally block and subsequent stream chunks.
                thinking_timer_task = None

                if not (candidate := self._get_first_candidate(chunk.candidates)):
                    log.warning("Stream chunk has no candidates, skipping.")
                    continue
                if not (parts := candidate.content and candidate.content.parts):
                    log.warning(
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
            # Cancel the timer task if error occured before the stream could start.
            await self._cancel_thinking_timer(
                thinking_timer_task, None, event_emitter, model_name, thinking_budget
            )

            if not error_occurred:
                log.info(f"Stream finished successfully!")
                log.debug("Last chunk:", payload=final_response_chunk)
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
                log.exception(error_msg)
            log.debug("AsyncGenerator finished.")

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
        log.info("Uploading the model generated image to Open WebUI backend.")
        log.debug("Uploading to the configured storage provider.")
        try:
            # Dynamically check if 'tags' parameter exists
            sig = inspect.signature(Storage.upload_file)
            has_tags = "tags" in sig.parameters
        except Exception as e:
            log.error(f"Error checking Storage.upload_file signature: {e}")
            has_tags = False  # Default to old behavior

        try:
            if has_tags:
                # New version with tags support >=v0.6.6
                contents, image_path = Storage.upload_file(image, imagename, tags={})
            else:
                # Old version without tags <v0.6.5
                contents, image_path = Storage.upload_file(image, imagename)  # type: ignore
        except Exception:
            error_msg = "Error occurred during upload to the storage provider."
            log.exception(error_msg)
            return None
        # Add the image file to files database.
        log.debug("Adding the image file to Open WebUI files database.")
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
            log.warning(
                "Files.insert_new_file did not return anything. Image upload to Open WebUI database likely failed."
            )
            return None
        # Get the image url.
        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        log.success("Image upload finished!")
        return image_url

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
                log.warning(
                    f"Could not extract language name from {executable_code_part_lang_enum}. Default to python."
                )
        else:
            log.warning("Language Enum is None, defaulting to python.")

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

    # endregion 1.3 Model response streaming

    # region 1.4 Thinking status message
    def _get_budget_str(self, model_name: str, thinking_budget: int) -> str:
        return (
            f" • {thinking_budget} tokens budget"
            if model_name == "gemini-2.5-flash-preview-04-17" and thinking_budget > 0
            else ""
        )

    def is_thinking_model(self, model_id: str) -> bool:
        """Check if the model is a thinking model based on the valve pattern."""
        try:
            result = bool(
                re.search(self.valves.THINKING_MODEL_PATTERN, model_id, re.IGNORECASE)
            )
            return result
        except Exception:
            log.exception("Error checking if model is a thinking model")
            return False

    async def _start_thinking_timer(
        self,
        model_name: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        thinking_budget: int,
    ) -> tuple[float | None, asyncio.Task[None] | None]:
        # Check if this is a thinking model and exit early if not.
        # Exit also if thinking budget is explicitly set to 0 and Gemini 2.5 Flash is selected.
        if not self.is_thinking_model(model_name) or (
            thinking_budget == 0 and model_name == "gemini-2.5-flash-preview-04-17"
        ):
            return None, None
        # Indicates if emitted status messages should be visible in the front-end.
        hidden = not self.valves.EMIT_STATUS_UPDATES
        # Emit initial 'Thinking' status
        await self._emit_status(
            f"Thinking • 0s elapsed{self._get_budget_str(model_name, thinking_budget)}",
            event_emitter=event_emitter,
            done=False,
            hidden=hidden,
        )
        # Record the start time
        start_time = time.time()
        # Start the thinking timer
        # NOTE: It's important to note that the model could not be actually thinking
        # when the status message starts. API could be just slow or the chat data
        # payload could still be uploading.
        thinking_timer_task = asyncio.create_task(
            self._thinking_timer(
                event_emitter, model_name, thinking_budget, hidden=hidden
            )
        )
        return start_time, thinking_timer_task

    async def _thinking_timer(
        self,
        event_emitter: Callable[["Event"], Awaitable[None]],
        model_name: str,
        thinking_budget: int,
        hidden=False,
    ) -> None:
        """Asynchronous task to emit periodic status updates."""
        elapsed = 0
        try:
            log.info("Thinking timer started.")
            while True:
                await asyncio.sleep(self.valves.EMIT_INTERVAL)
                elapsed += self.valves.EMIT_INTERVAL
                # Format elapsed time
                if elapsed < 60:
                    time_str = f"{elapsed}s"
                else:
                    minutes, seconds = divmod(elapsed, 60)
                    time_str = f"{minutes}m {seconds}s"
                status_message = f"Thinking • {time_str} elapsed{self._get_budget_str(model_name, thinking_budget)}"
                await self._emit_status(
                    status_message,
                    event_emitter=event_emitter,
                    done=False,
                    hidden=hidden,
                )
        except asyncio.CancelledError:
            log.debug("Timer task cancelled.")
        except Exception:
            log.exception("Error in timer task")

    async def _cancel_thinking_timer(
        self,
        timer_task: asyncio.Task[None] | None,
        start_time: float | None,
        event_emitter: Callable[["Event"], Awaitable[None]],
        model_name: str,
        thinking_budget: int,
    ):
        # Check if task was already canceled.
        if not timer_task:
            return
        # Cancel the timer.
        timer_task.cancel()
        try:
            await timer_task
        except asyncio.CancelledError:
            log.info(f"Thinking timer task successfully cancelled.")
        except Exception:
            log.exception(f"Error cancelling thinking timer task.")
        # Indicates if emitted status messages should be visible in the front-end.
        hidden = not self.valves.EMIT_STATUS_UPDATES
        # Calculate elapsed time and emit final status message
        if start_time:
            total_elapsed = int(time.time() - start_time)
            if total_elapsed < 60:
                total_time_str = f"{total_elapsed}s"
            else:
                minutes, seconds = divmod(total_elapsed, 60)
                total_time_str = f"{minutes}m {seconds}s"

            final_status = f"Thinking completed • took {total_time_str}{self._get_budget_str(model_name, thinking_budget)}"
            await self._emit_status(
                final_status, event_emitter=event_emitter, done=True, hidden=hidden
            )
        else:
            # Hide the status message if stream failed.
            final_status = f"An error occured during the thinking phase."
            await self._emit_status(
                final_status, event_emitter=event_emitter, done=True, hidden=True
            )

    # endregion 1.4 Thinking status message

    # region 1.5 Post-processing
    async def _do_post_processing(
        self,
        model_response: types.GenerateContentResponse | None,
        event_emitter: Callable[["Event"], Awaitable[None]],
        metadata: dict[str, Any],
        request: Request,
        stream_error_happened: bool = False,
    ):
        """Handles emitting usage, grounding, and sources after the main response/stream is done."""
        log.info("Post-processing the model response.")
        if stream_error_happened:
            log.warning(
                "An error occured during the stream, cannot do post-processing."
            )
            # All the needed metadata is always in the last chunk, so if error happened then we cannot do anything.
            return
        if not model_response:
            log.warning("model_response is empty, cannot do post-processing.")
            return
        if not (candidate := self._get_first_candidate(model_response.candidates)):
            log.warning(
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
            log.error(error_msg)
            return
        else:
            log.debug(f"Response has correct finish reason: {finish_reason}.")

        # Emit token usage data.
        if usage_event := self._get_usage_data_event(model_response):
            log.debug("Emitting usage data:", payload=usage_event)
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
            log.debug(
                f"Found grounding metadata. Storing in in request's app state using key {storage_key}."
            )
            # Using shared `request.app.state` to pass grounding metadata to Filter.outlet.
            # This is necessary because the Pipe finishes during the initial `/api/completion` request,
            # while Filter.outlet is invoked by a separate, later `/api/chat/completed` request.
            # `request.state` does not persist across these distinct request lifecycles.
            app_state: State = request.app.state
            app_state._state[storage_key] = grounding_metadata_obj
        else:
            log.debug(f"Response {message_id} does not have grounding metadata.")

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
        log.debug(
            f"Citation removal finished. Returning text str that is {trim} character shorter than the original input."
        )
        return text

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
            log.warning(
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

    # endregion 1.5 Post-processing

    # region 1.6 Event emissions
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

    async def _emit_status(
        self,
        message: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        done: bool = False,
        hidden: bool = False,
    ) -> None:
        """Emit status updates asynchronously."""
        try:
            if not self.valves.EMIT_STATUS_UPDATES:
                return
            status_event: "StatusEvent" = {
                "type": "status",
                "data": {"description": message, "done": done, "hidden": hidden},
            }
            await event_emitter(status_event)
            log.debug(f"Emitted status:", payload=status_event)
        except Exception:
            log.exception("Error emitting status")

    async def _emit_error(
        self,
        error_msg: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
        warning: bool = False,
        exception: bool = True,
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""

        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
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

    # endregion 1.6 Event emissions

    # region 1.7 Utility helpers
    def _is_flat_dict(self, data: Any) -> bool:
        """
        Checks if a dictionary contains only non-dict/non-list values (is one level deep).
        """
        if not isinstance(data, dict):
            return False
        return not any(isinstance(value, (dict, list)) for value in data.values())

    def _truncate_long_strings(
        self, data: Any, max_len: int, truncation_marker: str, truncation_enabled: bool
    ) -> Any:
        """
        Recursively traverses a data structure (dicts, lists) and truncates
        long string values. Creates copies to avoid modifying original data.

        Args:
            data: The data structure (dict, list, str, int, float, bool, None) to process.
            max_len: The maximum allowed length for string values.
            truncation_marker: The string to append to truncated values.
            truncation_enabled: Whether truncation is enabled.

        Returns:
            A potentially new data structure with long strings truncated.
        """
        if not truncation_enabled or max_len <= len(truncation_marker):
            # If truncation is disabled or max_len is too small, return original
            # Make a copy only if it's a mutable type we might otherwise modify
            if isinstance(data, (dict, list)):
                return copy.deepcopy(data)  # Ensure deep copy for nested structures
            return data  # Primitives are immutable

        if isinstance(data, str):
            if len(data) > max_len:
                return data[: max_len - len(truncation_marker)] + truncation_marker
            return data  # Return original string if not truncated
        elif isinstance(data, dict):
            # Process dictionary items, creating a new dict
            return {
                k: self._truncate_long_strings(
                    v, max_len, truncation_marker, truncation_enabled
                )
                for k, v in data.items()
            }
        elif isinstance(data, list):
            # Process list items, creating a new list
            return [
                self._truncate_long_strings(
                    item, max_len, truncation_marker, truncation_enabled
                )
                for item in data
            ]
        else:
            # Return non-string, non-container types as is (they are immutable)
            return data

    def plugin_stdout_format(self, record: "Record") -> str:
        """
        Custom format function for the plugin's logs.
        Serializes and truncates data passed under the 'payload' key in extra.
        """

        # Configuration Keys
        LOG_OPTIONS_PREFIX = "_log_"
        TRUNCATION_ENABLED_KEY = f"{LOG_OPTIONS_PREFIX}truncation_enabled"
        MAX_LENGTH_KEY = f"{LOG_OPTIONS_PREFIX}max_length"
        TRUNCATION_MARKER_KEY = f"{LOG_OPTIONS_PREFIX}truncation_marker"
        DATA_KEY = "payload"

        original_extra = record["extra"]
        # Extract the data intended for serialization using the chosen key
        data_to_process = original_extra.get(DATA_KEY)

        serialized_data_json = ""
        if data_to_process is not None:
            try:
                serializable_data = pydantic_core.to_jsonable_python(
                    data_to_process, serialize_unknown=True
                )

                # Determine truncation settings
                truncation_enabled = original_extra.get(TRUNCATION_ENABLED_KEY, True)
                max_length = original_extra.get(MAX_LENGTH_KEY, 256)
                truncation_marker = original_extra.get(TRUNCATION_MARKER_KEY, "[...]")

                # If max_length was explicitly provided, force truncation enabled
                if MAX_LENGTH_KEY in original_extra:
                    truncation_enabled = True

                # Truncate long strings
                truncated_data = self._truncate_long_strings(
                    serializable_data,
                    max_length,
                    truncation_marker,
                    truncation_enabled,
                )

                # Serialize the (potentially truncated) data
                if self._is_flat_dict(truncated_data) and not isinstance(
                    truncated_data, list
                ):
                    json_string = json.dumps(
                        truncated_data, separators=(",", ":"), default=str
                    )
                    # Add a simple prefix if it's compact
                    serialized_data_json = " - " + json_string
                else:
                    json_string = json.dumps(truncated_data, indent=2, default=str)
                    # Prepend with newline for readability
                    serialized_data_json = "\n" + json_string

            except (TypeError, ValueError) as e:  # Catch specific serialization errors
                serialized_data_json = f" - {{Serialization Error: {e}}}"
            except (
                Exception
            ) as e:  # Catch any other unexpected errors during processing
                serialized_data_json = f" - {{Processing Error: {e}}}"

        # Add the final JSON string (or error message) back into the record
        record["extra"]["_plugin_serialized_data"] = serialized_data_json

        # Base template
        base_template = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

        # Append the serialized data
        base_template += "{extra[_plugin_serialized_data]}"
        # Append the exception part
        base_template += "\n{exception}"
        # Return the format string template
        return base_template.rstrip()

    @cache
    def _add_log_handler(self, log_level: str):
        """
        Adds or updates the loguru handler specifically for this plugin.
        Includes logic for serializing and truncating extra data.
        The handler is added only if the log_level has changed since the last call.
        """

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__

        # Get the desired level name and number
        desired_level_name = log_level
        try:
            # Use the public API to get level details
            desired_level_info = log.level(desired_level_name)
            desired_level_no = desired_level_info.no
        except ValueError:
            log.error(
                f"Invalid LOG_LEVEL '{desired_level_name}' configured for plugin {__name__}. Cannot add/update handler."
            )
            return  # Stop processing if the level is invalid

        # Access the internal state of the log
        handlers: dict[int, "Handler"] = log._core.handlers  # type: ignore
        handler_id_to_remove = None
        found_correct_handler = False

        for handler_id, handler in handlers.items():
            existing_filter = handler._filter  # Access internal attribute

            # Check if the filter matches our plugin_filter
            # Comparing function objects directly can be fragile if they are recreated.
            # Comparing by name and module is more robust for functions defined at module level.
            is_our_filter = (
                existing_filter is not None  # Make sure a filter is set
                and hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            )

            if is_our_filter:
                existing_level_no = handler.levelno
                log.trace(
                    f"Found existing handler {handler_id} for {__name__} with level number {existing_level_no}."
                )

                # Check if the level matches the desired level
                if existing_level_no == desired_level_no:
                    log.debug(
                        f"Handler {handler_id} for {__name__} already exists with the correct level '{desired_level_name}'."
                    )
                    found_correct_handler = True
                    break  # Found the correct handler, no action needed
                else:
                    # Found our handler, but the level is wrong. Mark for removal.
                    log.info(
                        f"Handler {handler_id} for {__name__} found, but log level differs "
                        f"(existing: {existing_level_no}, desired: {desired_level_no}). "
                        f"Removing it to update."
                    )
                    handler_id_to_remove = handler_id
                    break  # Found the handler to replace, stop searching

        # Remove the old handler if marked for removal
        if handler_id_to_remove is not None:
            try:
                log.remove(handler_id_to_remove)
                log.debug(f"Removed handler {handler_id_to_remove} for {__name__}.")
            except ValueError:
                # This might happen if the handler was somehow removed between the check and now
                log.warning(
                    f"Could not remove handler {handler_id_to_remove} for {__name__}. It might have already been removed."
                )
                # If removal failed but we intended to remove, we should still proceed to add
                # unless found_correct_handler is somehow True (which it shouldn't be if handler_id_to_remove was set).

        # Add a new handler if no correct one was found OR if we just removed an incorrect one
        if not found_correct_handler:
            log.add(
                sys.stdout,
                level=desired_level_name,
                format=self.plugin_stdout_format,
                filter=plugin_filter,
            )
            log.debug(
                f"Added new handler to loguru for {__name__} with level {desired_level_name}."
            )

    def _get_mime_type(self, file_uri: str) -> str:
        """
        Determines MIME type based on file extension using the mimetypes module.
        """
        mime_type, encoding = mimetypes.guess_type(file_uri)
        if mime_type is None:
            return "application/octet-stream"  # Default MIME type if unknown
        return mime_type

    def _get_first_candidate(
        self, candidates: list[types.Candidate] | None
    ) -> types.Candidate | None:
        """Selects the first candidate, logging a warning if multiple exist."""
        if not candidates:
            # Logging warnings is handled downstream.
            return None
        if len(candidates) > 1:
            log.warning("Multiple candidates found, defaulting to first candidate.")
        return candidates[0]

    def _get_file_data(self, file_id: str) -> tuple[bytes | None, str | None]:
        if not file_id:
            # TODO: Emit toast
            log.warning(f"file_id is empty. Cannot continue.")
            return None, None
        file_model = Files.get_file_by_id(file_id)
        if file_model is None:
            # TODO: Emit toast
            log.warning(f"File {file_id} not found in the backend's database.")
            return None, None
        if not (file_path := file_model.path):
            # TODO: Emit toast
            log.warning(
                f"File {file_id} was found in the database but it lacks `path` field. Cannot Continue."
            )
            return None, None
        if file_model.meta is None:
            # TODO: Emit toast
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta` field. Cannot continue."
            )
            return None, None
        if not (content_type := file_model.meta.get("content_type")):
            # TODO: Emit toast
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta.content_type` field. Cannot continue."
            )
            return None, None
        try:
            with open(file_path, "rb") as file:
                image_data = file.read()
            return image_data, content_type
        except FileNotFoundError:
            # TODO: Emit toast
            log.exception(f"File {file_path} not found on disk.")
            return None, content_type
        except Exception:
            # TODO: Emit toast
            log.exception(f"Error processing file {file_path}")
            return None, content_type

    # endregion 1.7 Utility helpers

    # endregion 1. Helper methods inside the Pipe class
