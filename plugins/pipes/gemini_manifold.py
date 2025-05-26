"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API. Supports native image generation, grounding with Google Search and streaming. Uses google-genai.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.18.2
requirements: google-genai==1.16.1
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
from aiocache import cached
from aiocache.base import BaseCache
from fastapi.datastructures import State
import io
import os
import uuid
import base64
import re
import fnmatch
import sys
from loguru import logger
from fastapi import Request
import pydantic_core
from pydantic import BaseModel, Field, field_validator
from collections.abc import AsyncIterator, Awaitable, Callable
from itertools import chain
from typing import (
    Any,
    AsyncGenerator,
    Literal,
    TYPE_CHECKING,
    cast,
)
from open_webui.models.chats import Chats
from open_webui.models.files import FileForm, Files
from open_webui.storage.provider import Storage
from open_webui.utils.misc import pop_system_message

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler  # type: ignore
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


class GenaiApiError(Exception):
    """Custom exception for errors during Genai API interactions."""

    def __init__(self, message):
        super().__init__(message)


class ContentBuilder:
    def __init__(
        self,
        event_emitter: Callable[["Event"], Awaitable[None]],
        metadata: dict,
        user_data: dict,
        body_messages: list["Message"],
    ):
        self.event_emitter = event_emitter
        self.system_prompt: str | None = None
        self.messages: list["Message"] = []
        self.messages_db: list["ChatMessageTD"] | None = None
        self.upload_documents: bool = False

        # Check if the user has the upload_documents feature enabled
        features = metadata.get("features", {}) or {}
        self.upload_documents = features.get("upload_documents", False)

        # Extract the system prompt from the body messages (if any)
        # and store the message list without the system prompt.
        system_message, self.messages = pop_system_message(body_messages)
        if system_message is not None:
            self.system_prompt = system_message["content"]

        # Ensure consistency between the messages list and the database.
        chat_id, user_id = metadata.get("chat_id", ""), user_data["id"]
        chat = Chats.get_chat_by_id_and_user_id(id=chat_id, user_id=user_id)

        if not chat:
            # Case 1: Chat not found, self.messages_db remains None
            warn_msg = f"Chat with ID - {chat_id} - not found. Can't filter out the citation marks."
            log.warning(warn_msg)
        elif len(messages_db := chat.chat.get("messages", [])[:-1]) != len(
            self.messages
        ):
            # Case 2: Chat found, but message lengths mismatch, self.messages_db remains None
            # (Last message is the upcoming assistant response, at this point in the logic it's empty, it's popped out)
            # This block is reached only if 'chat' is not None.
            warn_msg = (
                f"Messages in the body ({len(self.messages)}) and "
                f"messages in the database ({len(messages_db)}) do not match. "
                "This is likely due to a bug in Open WebUI. "
                "Cannot filter out citation marks or upload files."
            )
            log.warning(warn_msg)
        else:
            # Case 3: Chat found and message lengths match (Success!),
            # self.messages_db is set to the messages from the database.
            # This block is reached only if 'chat' is not None AND lengths match.
            self.messages_db = messages_db

    async def _emit_toast(
        self,
        msg: str,
        toastType: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        """Emits a toast message to the front-end."""
        event: "NotificationEvent" = {
            "type": "notification",
            "data": {"type": toastType, "content": msg},
        }
        await self.event_emitter(event)

    async def _emit_warning(self, msg: str) -> None:
        """Emits a warning message to the front-end and log it in the backend."""
        log.warning(msg)
        await self._emit_toast(msg, "warning")

    def _get_file_data(self, file_id: str) -> tuple[bytes | None, str | None]:
        if not file_id:
            log.warning("file_id is empty. Cannot continue.")
            return None, None
        file_model = Files.get_file_by_id(file_id)
        if file_model is None:
            log.warning(f"File {file_id} not found in the backend's database.")
            return None, None
        if not (file_path := file_model.path):
            log.warning(
                f"File {file_id} was found in the database but it lacks `path` field. Cannot Continue."
            )
            return None, None
        if file_model.meta is None:
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta` field. Cannot continue."
            )
            return None, None
        if not (content_type := file_model.meta.get("content_type")):
            log.warning(
                f"File {file_path} was found in the database but it lacks `meta.content_type` field. Cannot continue."
            )
            return None, None
        try:
            with open(file_path, "rb") as file:
                image_data = file.read()
            return image_data, content_type
        except FileNotFoundError:
            log.exception(f"File {file_path} not found on disk.")
            return None, content_type
        except Exception:
            log.exception(f"Error processing file {file_path}")
            return None, content_type

    def _genai_part_from_image_url(self, image_url: str) -> types.Part | None:
        """
        Processes an image URL and returns a genai.types.Part object from it.
        Handles GCS, data URIs, standard URLs, and Open WebUI internal file URLs.
        """
        try:
            file_url_match = re.match(r"/api/v1/files/([a-f0-9\-]+)/content", image_url)
            if file_url_match:  # It is an Open WebUI internal file URL
                file_id = file_url_match.group(1)
                image_bytes, mime_type = self._get_file_data(file_id)
                if image_bytes and mime_type:
                    return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                else:
                    log.warning(f"Could not retrieve data for file ID: {file_id}")
                    return None

            base64_image_match = re.match(r"data:(image/\w+);base64,(.+)", image_url)
            if base64_image_match:  # It is a base64-encoded image data URI
                return types.Part.from_bytes(
                    data=base64.b64decode(base64_image_match.group(2)),
                    mime_type=base64_image_match.group(1),
                )

            # For other URLs (http/https, gs://), let genai handle it
            return types.Part.from_uri(file_uri=image_url)
        except Exception:
            log.exception(f"Error processing image URL: {image_url[:64]}[...]")
            return None

    def _genai_parts_from_text(self, text: str) -> list[types.Part]:
        """
        Parses the input text to extract and convert various content types into a list of genai.types.Part objects.

        This function identifies markdown-formatted image URLs (including data URIs and Open WebUI internal file URLs)
        and YouTube video URLs within the provided text. It then segments the text and converts each segment
        (plain text, images, or YouTube URLs) into the appropriate genai.types.Part representation.

        Args:
            text: The input string containing text, potentially with markdown images or YouTube URLs.

        Returns:
            A list of genai.types.Part objects representing the parsed content.
            Returns an empty list if the input text is empty or contains only whitespace.
        """
        # Regex to find markdown images or YouTube URLs
        # Using named groups for clarity and easier extraction
        pattern = re.compile(
            r"!\[.*?\]\((?P<image_url>(?:data:(?:image/\w+);base64,[^)]+)|(?:/api/v1/files/[a-f0-9\-]+/content)))\)|"
            r"(?P<youtube_url>https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^&\s]+)"
        )

        parts: list[types.Part] = []
        last_pos = 0
        for match in pattern.finditer(text):
            # Process text before the first match / between matches
            if text := text[last_pos : match.start()].strip():
                parts.append(types.Part.from_text(text=text))

            if image_url := match.group("image_url"):  # It's a markdown image
                if image_part := self._genai_part_from_image_url(image_url):
                    parts.append(image_part)
            elif youtube_url := match.group("youtube_url"):  # It's a YouTube URL
                log.info(f"Found YouTube URL: {youtube_url}")
                parts.append(types.Part(file_data=types.FileData(file_uri=youtube_url)))

            last_pos = match.end()

        # Add remaining text after the last match
        # This also handles the case where no matches were found
        if remaining_text := text[last_pos:].strip():
            parts.append(types.Part.from_text(text=remaining_text))

        # If text was only whitespace, [] will be returned
        return parts

    def _remove_citation_markers(self, text: str, sources: list["Source"]) -> str:
        original_text = text
        processed: set[str] = set()
        for source in sources:
            supports = chain.from_iterable(
                metadata["supports"]
                for metadata in source.get("metadata", [])
                if "supports" in metadata
            )

            for support in supports:
                support = types.GroundingSupport(**support)
                indices = support.grounding_chunk_indices
                segment = support.segment
                if not (indices and segment and segment.text):
                    continue
                segment_text = segment.text

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

    async def _create_part_from_file_attachment(
        self, file_item: "FileAttachmentTD"
    ) -> types.Part | None:
        """
        Processes a single file attachment and returns a types.Part if supported,
        otherwise logs a warning and emits a toast.
        """
        file_id = file_item.get("file", {}).get("id", "unknown_id")
        log.debug(f"Processing file attachment: {file_id}", payload=file_item)

        document_bytes, mime_type = self._get_file_data(file_id)

        if not document_bytes or not mime_type:
            log.debug(f"No data or mime_type for file_id: {file_id}, skipping.")
            return None

        if mime_type.startswith("text/") or mime_type == "application/pdf":
            log.debug(
                f"MIME type {mime_type} for file {file_id} is supported. Creating types.Part."
            )
            return types.Part.from_bytes(data=document_bytes, mime_type=mime_type)

        warn_msg = f"MIME type {mime_type} for file {file_id} is not supported by Google API. Skipping file."
        await self._emit_warning(warn_msg)
        return None

    async def _process_user_message(
        self, message: "UserMessage", files: list["FileAttachmentTD"]
    ) -> list[types.Part]:
        user_parts: list[types.Part] = []

        # 1. Process file attachments
        if files:
            log.info(f"Processing {len(files)} file attachments for the user message.")
            for file_item in files:
                part = await self._create_part_from_file_attachment(file_item)
                if part:
                    user_parts.append(part)
        else:
            log.debug("No file attachments to process for the user message.")

        # 2. Determine user_content_list from message content
        raw_user_content = message.get("content")
        user_content_list: list["Content"] | None = None

        if isinstance(raw_user_content, str):
            log.debug("User message content is a string. Wrapping in a list structure.")
            user_content_list = [{"type": "text", "text": raw_user_content}]  # type: ignore
        elif isinstance(raw_user_content, list):
            log.debug("User message content is a list.")
            user_content_list = raw_user_content  # type: ignore
            # Assuming raw_user_content is list["Content"] if it's a list.
        else:
            type_name = type(raw_user_content).__name__
            warn_msg = (
                f"User message content is of type {type_name}, "
                "not a string or list. Skipping content processing."
            )
            await self._emit_warning(warn_msg)
            return user_parts

        # 3. Process content items from user_content_list
        log.info(
            f"Processing {len(user_content_list)} content items from the user message."
        )
        for item in user_content_list:
            kind = item.get("type")
            log.debug(f"Processing content item of type: {kind}")

            if kind == "text" and (text := item.get("text")):
                text_parts = self._genai_parts_from_text(text)
                user_parts.extend(text_parts)
            elif kind == "image_url" and (url := item.get("image_url", {}).get("url")):
                img_part = self._genai_part_from_image_url(url)
                if img_part:
                    user_parts.append(img_part)
            else:
                warn_msg = (
                    f"User message content item type '{kind}' is not supported. "
                    "Skipping this content item."
                )
                await self._emit_warning(warn_msg)

        log.info(
            f"Finished processing user message. Generated {len(user_parts)} parts in total."
        )
        return user_parts

    def _process_assistant_message(
        self, message: "AssistantMessage", sources: list["Source"] | None
    ) -> list[types.Part]:
        assistant_text = message.get("content")
        if sources:
            assistant_text = self._remove_citation_markers(assistant_text, sources)
        return self._genai_parts_from_text(assistant_text)

    async def build_contents(self) -> list[types.Content]:
        contents = []
        for i, message_data in enumerate(self.messages):
            current_role = message_data.get("role")
            parts = []

            # Leverage the guarantee: if self.messages_db exists, self.messages_db[i] is safe to access.
            message_db_item = self.messages_db[i] if self.messages_db else None

            if current_role == "user":
                user_message = cast("UserMessage", message_data)
                files = []
                if message_db_item and self.upload_documents:
                    files = message_db_item.get("files", [])
                parts = await self._process_user_message(user_message, files)
                contents.append(types.Content(role="user", parts=parts))

            elif current_role == "assistant":
                assistant_message = cast("AssistantMessage", message_data)
                sources = None
                if message_db_item:
                    sources = message_db_item.get("sources")
                parts = self._process_assistant_message(assistant_message, sources)
                # Google API's assistant role is "model"
                contents.append(types.Content(role="model", parts=parts))

            else:
                warn_msg = f"Message {i} has an invalid role: {current_role}. Skipping to the next message."
                await self._emit_warning(warn_msg)

        return contents


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
        USE_VERTEX_AI: bool = Field(
            default=False,
            description="""Whether to allow using Google Cloud Vertex AI.
            Default value is False.""",
        )
        VERTEX_PROJECT: str | None = Field(
            default=None,
            description="""The Google Cloud project ID to use with Vertex AI.
            Default value is None.
            If you set this value then the plugin will start using Vertex AI by default.
            Users can override this inside UserValves.""",
        )
        VERTEX_LOCATION: str = Field(
            default="global",
            description="""The Google Cloud region to use with Vertex AI.
            Default value is 'global'.""",
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
        ENABLE_URL_CONTEXT_TOOL: bool = Field(
            default=False,
            description="Enable the URL context tool to allow the model to fetch and use content from provided URLs. This tool is only compatible with specific models.",
        )

    class UserValves(BaseModel):
        # TODO: Add more options that can be changed by the user.
        GEMINI_API_KEY: str | None = Field(
            default=None,
            description="""Gemini Developer API key.
            Default value is None (uses the default from Valves, same goes for other options below).""",
        )
        GEMINI_API_BASE_URL: str | None = Field(
            default=None,
            description="""The base URL for calling the Gemini API
            Default value is None.""",
        )
        USE_VERTEX_AI: bool | None | Literal[""] = Field(
            default=None,
            description="""Whether to use Google Cloud Vertex AI instead of the standard Gemini API.
            Default value is None.""",
        )
        VERTEX_PROJECT: str | None = Field(
            default=None,
            description="""The Google Cloud project ID to use with Vertex AI.
            Default value is None.""",
        )
        VERTEX_LOCATION: str | None = Field(
            default=None,
            description="""The Google Cloud region to use with Vertex AI.
            Default value is None.""",
        )
        THINKING_BUDGET: int | None | Literal[""] = Field(
            default=None,
            description="""Gemini 2.5 Flash only. Indicates the thinking budget in tokens.
            0 means no thinking. Default value is None (uses the default from Valves).
            See <https://cloud.google.com/vertex-ai/generative-ai/docs/thinking> for more.""",
        )
        ENABLE_URL_CONTEXT_TOOL: bool = Field(
            default=False,
            description="Enable the URL context tool to allow the model to fetch and use content from provided URLs. This tool is only compatible with specific models.",
        )

        @field_validator("THINKING_BUDGET", mode="after")
        @classmethod
        def validate_thinking_budget_range(cls, v):
            if v is not None and v != "":
                if not (0 <= v <= 24576):
                    raise ValueError(
                        "THINKING_BUDGET must be between 0 and 24576, inclusive."
                    )
            return v

    def __init__(self):
        self.valves = self.Valves()

    async def pipes(self) -> list["ModelData"]:
        """Register all available Google models."""
        self._add_log_handler(self.valves.LOG_LEVEL)

        # Clear cache if caching is disabled
        if not self.valves.CACHE_MODELS:
            log.debug("CACHE_MODELS is False, clearing model cache.")
            cache_instance = getattr(self._get_genai_models, "cache")
            await cast(BaseCache, cache_instance).clear()

        log.info("Fetching and filtering models from Google API.")
        # Get and filter models (potentially cached based on API key, base URL, white- and blacklist)
        try:
            client_args = self._prepare_client_args(self.valves)
            client_args += [self.valves.MODEL_WHITELIST, self.valves.MODEL_BLACKLIST]
            filtered_models = await self._get_genai_models(*client_args)
        except GenaiApiError:
            error_msg = "Error getting the models from Google API, check the logs."
            return [self._return_error_model(error_msg, exception=True)]

        log.info(f"Returning {len(filtered_models)} models to Open WebUI.")
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
        self._add_log_handler(self.valves.LOG_LEVEL)

        # Apply settings from the user
        valves: Pipe.Valves = self._get_merged_valves(
            self.valves, __user__.get("valves")
        )
        log.debug(
            f"USE_VERTEX_AI: {valves.USE_VERTEX_AI}, VERTEX_PROJECT set: {bool(valves.VERTEX_PROJECT)}, API_KEY set: {bool(valves.GEMINI_API_KEY)}"
        )

        log.debug(
            f"Getting genai client (potentially cached) for user {__user__['email']}."
        )
        client = self._get_user_client(valves, __user__["email"])

        log.trace("__metadata__:", payload=__metadata__)
        # Check if user is chatting with an error model for some reason.
        if "error" in __metadata__["model"]["id"]:
            error_msg = f"There has been an error during model retrieval phase: {str(__metadata__['model'])}"
            raise ValueError(error_msg)

        features = __metadata__.get("features", {}) or {}
        content_builder = ContentBuilder(
            event_emitter=__event_emitter__,
            metadata=__metadata__,
            user_data=__user__,
            body_messages=body.get("messages"),
        )

        log.info(
            "Converting Open WebUI's `body` dict into list of `Content` objects that `google-genai` understands."
        )
        contents = await content_builder.build_contents()

        # Assemble GenerateContentConfig
        safety_settings: list[types.SafetySetting] | None = __metadata__.get(
            "safety_settings"
        )
        model_name = re.sub(r"^.*?[./]", "", body.get("model", ""))

        # API does not stream thoughts sadly. See https://github.com/googleapis/python-genai/issues/226#issuecomment-2631657100
        thinking_conf = None
        if model_name == "gemini-2.5-flash-preview-04-17":
            log.info(f"Model ID '{model_name}' allows adjusting the thinking settings.")
            thinking_conf = types.ThinkingConfig(
                thinking_budget=valves.THINKING_BUDGET,
                include_thoughts=None,
            )
        # TODO: Take defaults from the general front-end config.
        gen_content_conf = types.GenerateContentConfig(
            system_instruction=content_builder.system_prompt,
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

        # Add URL context tool if enabled and model is compatible
        if valves.ENABLE_URL_CONTEXT_TOOL:
            compatible_models_for_url_context = [
                "gemini-2.5-pro-preview-05-06",
                "gemini-2.5-flash-preview-05-20",
                "gemini-2.0-flash",
                "gemini-2.0-flash-live-001",
            ]
            if model_name in compatible_models_for_url_context:
                log.info(
                    f"Model {model_name} is compatible with URL context tool. Enabling."
                )
                gen_content_conf.tools.append(
                    types.Tool(url_context=types.UrlContext())
                )
            else:
                log.warning(
                    f"URL context tool is enabled, but model {model_name} is not in the compatible list. Skipping."
                )

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
                valves,
                gen_content_args,
                __event_emitter__,
                __metadata__,
                __user__["id"],
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
    @staticmethod
    @cache
    def _get_or_create_genai_client(
        api_key: str | None = None,
        base_url: str | None = None,
        use_vertex_ai: bool | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> genai.Client:
        """
        Creates a genai.Client instance or retrieves it from cache.
        Raises GenaiApiError on failure.
        """

        if not vertex_project and not api_key:
            msg = "Neither VERTEX_PROJECT nor GEMINI_API_KEY is set."
            raise GenaiApiError(msg)

        if use_vertex_ai and vertex_project:
            kwargs = {
                "vertexai": True,
                "project": vertex_project,
                "location": vertex_location,
            }
            api = "Vertex AI"
        else:  # Covers (use_vertex_ai and not vertex_project) OR (not use_vertex_ai)
            if use_vertex_ai and not vertex_project:
                log.warning(
                    "Vertex AI is enabled but no project is set. "
                    "Using Gemini Developer API."
                )
            # This also implicitly covers the case where api_key might be None,
            # which is handled by the initial check or the SDK.
            kwargs = {
                "api_key": api_key,
                "http_options": types.HttpOptions(base_url=base_url),
            }
            api = "Gemini Developer API"

        try:
            client = genai.Client(**kwargs)
            log.success(f"{api} Genai client successfully initialized.")
            return client
        except Exception as e:
            raise GenaiApiError(f"{api} Genai client initialization failed: {e}") from e

    def _get_user_client(self, valves: "Pipe.Valves", user_email: str) -> genai.Client:
        try:
            client_args = self._prepare_client_args(valves)
            client = self._get_or_create_genai_client(*client_args)
        except GenaiApiError as e:
            error_msg = f"Failed to initialize genai client for user {user_email}: {e}"
            # FIXME: include correct traceback.
            raise ValueError(error_msg) from e
        return client

    @staticmethod
    def _return_error_model(
        error_msg: str, warning: bool = False, exception: bool = True
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

    @staticmethod
    def strip_prefix(model_name: str) -> str:
        """
        Extract the model identifier using regex, handling various naming conventions.
        e.g., "gemini_manifold_google_genai.gemini-2.5-flash-preview-04-17" -> "gemini-2.5-flash-preview-04-17"
        e.g., "models/gemini-1.5-flash-001" -> "gemini-1.5-flash-001"
        e.g., "publishers/google/models/gemini-1.5-pro" -> "gemini-1.5-pro"
        """
        # Use regex to remove everything up to and including the last '/' or the first '.'
        stripped = re.sub(r"^(?:.*/|[^.]*\.)", "", model_name)
        return stripped

    @cached()
    async def _get_genai_models(
        self,
        api_key: str | None,
        base_url: str | None,
        use_vertex_ai: bool | None,
        vertex_project: str | None,
        vertex_location: str | None,
        whitelist_str: str,
        blacklist_str: str | None,
    ) -> list["ModelData"]:
        """
        Gets valid Google models from the API and filters them based on configured white- and blacklist.
        Returns a list[dict] that can be directly returned by the `pipes` method.
        The result is cached based on the provided api_key, base_url, whitelist_str, and blacklist_str.
        """
        # Get a client using the provided API key and base URL
        client = self._get_or_create_genai_client(
            api_key, base_url, use_vertex_ai, vertex_project, vertex_location
        )

        try:
            # Get the AsyncPager object
            google_models_pager = await client.aio.models.list(
                config={"query_base": True}
            )
            # Iterate the pager to get the full list of models
            # This is where the actual API calls for subsequent pages happen if needed
            all_google_models = [model async for model in google_models_pager]
        except Exception as e:
            raise GenaiApiError(f"Retrieving models from Google API failed: {e}") from e
        else:
            log.info(f"Retrieved {len(all_google_models)} models from Google API.")
            log.trace("All models returned by Google:", payload=all_google_models)

        # Filter Google models list down to generative models only.
        # FIXME: model.supported_actions is None with vertex models
        # see https://github.com/googleapis/python-genai/issues/679
        generative_models = []
        for model in all_google_models:
            actions = model.supported_actions
            if actions is None or "generateContent" in actions:
                generative_models.append(model)

        # Raise if there are not generative models
        if not generative_models:
            log.warning("No generative models found.")

        # Filter based on whitelist and blacklist
        def match(name, list_str):
            patterns = list_str.replace(" ", "").split(",") if list_str else []
            return any(fnmatch.fnmatch(name, pat) for pat in patterns)

        filtered_models: list["ModelData"] = []
        for model in generative_models:
            name = Pipe.strip_prefix(model.name)
            if name and match(name, whitelist_str) and not match(name, blacklist_str):
                filtered_models.append(
                    {
                        "id": name,
                        "name": model.display_name or name,
                        "description": model.description,
                    }
                )
        log.info(
            f"Filtered {len(generative_models)} models down to {len(filtered_models)} models based on white/blacklists."
        )
        return filtered_models

    @staticmethod
    def _prepare_client_args(
        source_valves: "Pipe.Valves | Pipe.UserValves",
    ) -> list[str | bool | None]:
        """Prepares arguments for _get_or_create_genai_client from source_valves."""
        ATTRS = [
            "GEMINI_API_KEY",
            "GEMINI_API_BASE_URL",
            "USE_VERTEX_AI",
            "VERTEX_PROJECT",
            "VERTEX_LOCATION",
        ]
        return [getattr(source_valves, attr) for attr in ATTRS]

    # endregion 1.1 Client initialization and model retrival from Google API

    # region 1.2 Open WebUI's body.messages -> list[genai.types.Content] conversion

    # endregion 1.2 Open WebUI's body.messages -> list[genai.types.Content] conversion

    # region 1.3 Model response streaming
    async def _stream_response_generator(
        self,
        response_stream: AsyncIterator[types.GenerateContentResponse],
        __request__: Request,
        valves: "Pipe.Valves",
        gen_content_args: dict,
        event_emitter: Callable[["Event"], Awaitable[None]],
        metadata: dict[str, Any],
        user_id: str,
    ) -> AsyncGenerator[str, None]:
        """
        Yields text chunks from the stream and spawns metadata processing task on completion.
        """
        final_response_chunk: types.GenerateContentResponse | None = None
        error_occurred = False

        # Start thinking timer (model name check is inside this method).
        model_name = gen_content_args.get("model", "")
        thinking_budget = valves.THINKING_BUDGET
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
                                user_id,
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
        self, inline_data, gen_content_args: dict, user_id: str, request: Request
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
                user_id,
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
            f"Thinking • 0s{self._get_budget_str(model_name, thinking_budget)}",
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
                status_message = f"Thinking • {time_str}{self._get_budget_str(model_name, thinking_budget)}"
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

        # Emit URL context metadata if available
        url_context_meta = candidate.url_context_metadata if candidate else None
        if (
            url_context_meta
            and hasattr(url_context_meta, "retrieved_urls")
            and url_context_meta.retrieved_urls
        ):
            # Convert UrlContextRetrievedUrl objects to a list of dicts
            # as the objects themselves might not be directly JSON serializable in the event.
            retrieved_urls_data = []
            for retrieved_url_obj in url_context_meta.retrieved_urls:
                url_data = {
                    "url": retrieved_url_obj.url,
                    "title": retrieved_url_obj.title,
                }
                # Optionally, include favicon if it exists and is needed by the frontend
                # if hasattr(retrieved_url_obj, 'favicon') and retrieved_url_obj.favicon:
                #     url_data["favicon"] = retrieved_url_obj.favicon
                retrieved_urls_data.append(url_data)

            if retrieved_urls_data:  # Ensure we have something to send
                url_event = {
                    "type": "chat:url_context",
                    "data": {"retrieved_urls": retrieved_urls_data},
                }
                await event_emitter(url_event)
                log.debug("Emitted URL context metadata:", payload=url_event)

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

    @staticmethod
    def _get_merged_valves(
        default_valves: "Pipe.Valves",
        user_valves: "Pipe.UserValves | None",
    ) -> "Pipe.Valves":
        """
        Merges UserValves into a base Valves configuration.

        The general rule is that if a field in UserValves is not None, it overrides
        the corresponding field in the default_valves. Otherwise, the default_valves
        field value is used.

        Exceptions:
        - If default_valves.REQUIRE_USER_API_KEY is True, then GEMINI_API_KEY and
          VERTEX_PROJECT in the merged result will be taken directly from
          user_valves (even if they are None), ignoring the values in default_valves.

        Args:
            default_valves: The base Valves object with default configurations.
            user_valves: An optional UserValves object with user-specific overrides.
                         If None, a copy of default_valves is returned.

        Returns:
            A new Valves object representing the merged configuration.
        """
        if user_valves is None:
            # If no user-specific valves are provided, return a copy of the default valves.
            return default_valves.model_copy(deep=True)

        # Start with the values from the base `Valves`
        merged_data = default_valves.model_dump()

        # Override with non-None values from `UserValves`
        # Iterate over fields defined in the UserValves model
        for field_name in Pipe.UserValves.model_fields:
            # getattr is safe as field_name comes from model_fields of user_valves' type
            user_value = getattr(user_valves, field_name)
            if user_value is not None and user_value != "":
                # Only update if the field is also part of the main Valves model
                # (keys of merged_data are fields of default_valves)
                if field_name in merged_data:
                    merged_data[field_name] = user_value

        # Apply special logic based on default_valves.REQUIRE_USER_API_KEY
        if default_valves.REQUIRE_USER_API_KEY:
            # If REQUIRE_USER_API_KEY is True, GEMINI_API_KEY and VERTEX_PROJECT
            # must be taken from UserValves, or be None if UserValves has them as None.
            # The base values from default_valves for these fields are effectively ignored in this case.
            # This assignment happens after the general loop, ensuring it takes final precedence.
            merged_data["GEMINI_API_KEY"] = user_valves.GEMINI_API_KEY
            merged_data["VERTEX_PROJECT"] = user_valves.VERTEX_PROJECT

        # Create a new Valves instance with the merged data.
        # Pydantic will validate the data against the Valves model definition during instantiation.
        return Pipe.Valves(**merged_data)

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

    # endregion 1.7 Utility helpers

    # endregion 1. Helper methods inside the Pipe class
