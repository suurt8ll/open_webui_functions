"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API and Vertex AI. Uses the newer google-genai SDK. Aims to support as many features from it as possible.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.21.0
requirements: google-genai==1.20.0
"""

# This is a helper function that provides a manifold for Google's Gemini Studio API and Vertex AI.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

# Supported features:
#   - Display thinking summary (Open WebUI >= 0.6.14 required)
#   - Thinking budget
#   - Gemini Reasoning Toggle (Reason filter required, see GitHub README)
#   - Native image generation (image output), use "gemini-2.0-flash-preview-image-generation"
#   - Document understanding (PDF and plaintext files). (Gemini Manifold Companion >= 1.4.0 filter required, see GitHub README)
#   - Image input
#   - YouTube video input (automatically detects youtube.com and youtu.be URLs in messages)
#   - Grounding with Google Search (Gemini Manifold Companion >= 1.2.0 required)
#   - Display citations in the front-end. (Gemini Manifold Companion >= 1.5.0 required)
#   - Permissive safety settings (Gemini Manifold Companion >= 1.3.0 required)
#   - Each user can decide to use their own API key.
#   - Token usage data
#   - Code execution tool. (Gemini Manifold Companion >= 1.1.0 required)
#   - URL context tool (Gemini Manifold Companion >= 1.5.0 required if you want to see citations in the front-end).
#   - Streaming and non-streaming responses.

# Features that are supported by API but not yet implemented in the manifold:
#   TODO Audio input support
#   TODO Video input support (other than YouTube URLs)
#   TODO Google Files API
#   TODO Native tool calling

from google import genai
from google.genai import types

import inspect
import copy
import json
from functools import cache
from aiocache import cached
from aiocache.base import BaseCache
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
from pydantic import BaseModel, Field, field_validator
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
from open_webui.storage.provider import Storage
from open_webui.models.functions import Functions

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler  # type: ignore
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)

# These tags will be "disabled" in the response, meaning that they will not be parsed by the backend.
SPECIAL_TAGS_TO_DISABLE = [
    "details",
    "think",
    "thinking",
    "reason",
    "reasoning",
    "thought",
    "Thought",
    "|begin_of_thought|",
    "code_interpreter",
    "|begin_of_solution|",
]
ZWS = "\u200b"


class GenaiApiError(Exception):
    """Custom exception for errors during Genai API interactions."""

    def __init__(self, message):
        super().__init__(message)


class Pipe:
    class Valves(BaseModel):
        GEMINI_API_KEY: str | None = Field(default=None)
        USER_MUST_PROVIDE_AUTH_CONFIG: bool = Field(
            default=False,
            description="""Whether to require users (including admins) to provide their own authentication configuration.
            User can provide these through UserValves. Setting this to True will disallow users from using Vertex AI.
            Default value is False.""",
        )
        AUTH_WHITELIST: str | None = Field(
            default=None,
            description="""Comma separated list of user emails that are allowed to bypassUSER_MUST_PROVIDE_AUTH_CONFIG and use the default authentication configuration.
            Default value is None (no users are whitelisted).""",
        )
        GEMINI_API_BASE_URL: str | None = Field(
            default=None,
            description="""The base URL for calling the Gemini API. 
            Default value is None.""",
        )
        USE_VERTEX_AI: bool = Field(
            default=False,
            description="""Whether to use Google Cloud Vertex AI instead of the standard Gemini API.
            If VERTEX_PROJECT is not set then the plugin will use the Gemini Developer API.
            Default value is False.
            Users can opt out of this by setting USE_VERTEX_AI to False in their UserValves.""",
        )
        VERTEX_PROJECT: str | None = Field(
            default=None,
            description="""The Google Cloud project ID to use with Vertex AI.
            Default value is None.""",
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
            description="""Whether to request models only on first load and when white- or blacklist changes. 
            Default value is True.""",
        )
        THINKING_BUDGET: int = Field(
            default=8192,
            ge=0,
            # The widest possible range is 0 (for Lite/Flash) to 32768 (for Pro).
            # Model-specific constraints are detailed in the description.
            le=32768,
            description="""Specifies the token budget for the model's internal thinking process,
            used for complex tasks like tool use. Applicable to Gemini 2.5 models.
            Default value is 8192.

            The valid token range depends on the specific model tier:
            - **Pro models**: Must be a value between 128 and 32,768.
            - **Flash and Lite models**: A value between 0 and 24,576. For these
              models, a value of 0 disables the thinking feature.

            See <https://cloud.google.com/vertex-ai/generative-ai/docs/thinking> for more details.""",
        )
        SHOW_THINKING_SUMMARY: bool = Field(
            default=True,
            description="""Whether to show the thinking summary in the response.
            This is only applicable for Gemini 2.5 models.
            Default value is True.""",
        )
        USE_FILES_API: bool = Field(
            default=True,
            description="""Save the image files using Open WebUI's API for files. 
            Default value is True.""",
        )
        THINKING_MODEL_PATTERN: str = Field(
            default=r"gemini-2.5",
            description="""Regex pattern to identify thinking models. 
            Default value is r"gemini-2.5".""",
        )
        ENABLE_URL_CONTEXT_TOOL: bool = Field(
            default=False,
            description="""Enable the URL context tool to allow the model to fetch and use content from provided URLs. 
            This tool is only compatible with specific models. Default value is False.""",
        )
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="""Select logging level. Use `docker logs -f open-webui` to view logs.
            Default value is INFO.""",
        )

    class UserValves(BaseModel):
        """Defines user-specific settings that can override the default `Valves`.

        The `UserValves` class provides a mechanism for individual users to customize
        their Gemini API settings for each request. This system is designed as a
        practical workaround for backend/frontend limitations, enabling per-user
        configurations.

        Think of the main `Valves` as the global, admin-configured template for the
        plugin. `UserValves` acts as a user-provided "overlay" or "patch" that
        is applied on top of that template at runtime.

        How it works:
        1.  **Default Behavior:** At the start of a request, the system merges the
            user's `UserValves` with the admin's `Valves`. If a field in
            `UserValves` has a value (i.e., is not `None` or an empty string `""`),
            it overrides the corresponding value from the main `Valves`. If a
            field is `None` or `""`, the admin's default is used.

        2.  **Special Authentication Logic:** A critical exception exists to enforce
            security and usage policies. If the admin sets `USER_MUST_PROVIDE_AUTH_CONFIG`
            to `True` in the main `Valves`, the merging logic changes for any user
            not on the `AUTH_WHITELIST`:
            - The user's `GEMINI_API_KEY` is taken directly from their `UserValves`,
              bypassing the admin's key entirely.
            - The ability to use the admin-configured Vertex AI is disabled
              (`USE_VERTEX_AI` is forced to `False`).
            This ensures that when required, users must use their own credentials
            and cannot fall back on the shared, system-level authentication.

        This two-tiered configuration allows administrators to set sensible defaults
        and enforce policies, while still giving users the flexibility to tailor
        certain parameters, like their API key or model settings, for their own use.
        """

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
            description="""Specifies the token budget for the model's internal thinking process,
            used for complex tasks like tool use. Applicable to Gemini 2.5 models.
            Default value is None.

            The valid token range depends on the specific model tier:
            - **Pro models**: Must be a value between 128 and 32,768.
            - **Flash and Lite models**: A value between 0 and 24,576. For these
              models, a value of 0 disables the thinking feature.

            See <https://cloud.google.com/vertex-ai/generative-ai/docs/thinking> for more details.""",
        )
        SHOW_THINKING_SUMMARY: bool | None | Literal[""] = Field(
            default=None,
            description="""Whether to show the thinking summary in the response.
            This is only applicable for Gemini 2.5 models.
            Default value is None.""",
        )
        USE_FILES_API: bool | None | Literal[""] = Field(
            default=None,
            description="""Save the image files using Open WebUI's API for files.
            Default value is None.""",
        )
        THINKING_MODEL_PATTERN: str | None = Field(
            default=None,
            description="""Regex pattern to identify thinking models.
            Default value is None.""",
        )
        ENABLE_URL_CONTEXT_TOOL: bool | None | Literal[""] = Field(
            default=None,
            description="""Enable the URL context tool to allow the model to fetch and use content from provided URLs. 
            This tool is only compatible with specific models. Default value is None.""",
        )

        @field_validator("THINKING_BUDGET", mode="after")
        @classmethod
        def validate_thinking_budget_range(cls, v):
            if v is not None and v != "":
                if not (0 <= v <= 32768):
                    raise ValueError(
                        "THINKING_BUDGET must be between 0 and 32768, inclusive."
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
    ) -> AsyncGenerator[dict, None] | str:
        self._add_log_handler(self.valves.LOG_LEVEL)

        # Apply settings from the user
        valves: Pipe.Valves = self._get_merged_valves(
            self.valves, __user__.get("valves"), __user__.get("email")
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

        thinking_conf = None
        if re.search(self.valves.THINKING_MODEL_PATTERN, model_name, re.IGNORECASE):
            log.info(f"Model ID '{model_name}' allows adjusting the thinking settings.")
            thinking_conf = types.ThinkingConfig(
                thinking_budget=valves.THINKING_BUDGET,
                include_thoughts=valves.SHOW_THINKING_SUMMARY,
            )

        # TODO: Check availability of companion filter too with this method.
        if self._is_function_active("gemini_reasoning_toggle"):
            # NOTE: Gemini 2.5 Pro supports reasoning budget but not toggling reasoning on/off.
            if re.search(
                r"gemini-2.5-(flash|lite)", model_name, re.IGNORECASE
            ) and not body.get("reason", False):
                log.info(
                    f"Model ID '{model_name}' allows turning off the reasoning feature. "
                    "Reasoning is currently toggled off in the UI. Setting thinking budget to 0."
                )
                thinking_conf = types.ThinkingConfig(
                    thinking_budget=0,
                    include_thoughts=valves.SHOW_THINKING_SUMMARY,
                )
        else:
            log.warning(
                "Gemini Reasoning Toggle filter is not active. "
                "Install or enable it if you want to toggle Gemini 2.5 Flash or Lite reasoning on/off."
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
            "gemini-2.0-flash-preview-image-generation" in model_name
            or "gemma" in model_name
        ):
            if "gemini-2.0-flash-preview-image-generation" in model_name:
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
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite-preview-06-17",
                "gemini-2.5-pro-preview-06-05",
                "gemini-2.5-pro-preview-05-06",
                "gemini-2.5-flash-preview-05-20",
                "gemini-2.0-flash",
                "gemini-2.0-flash-001",
                "gemini-2.0-flash-live-001",
            ]
            if model_name in compatible_models_for_url_context:
                if client.vertexai:
                    log.warning(
                        "URL context tool is enabled, but Vertex AI does not support it. Skipping."
                    )
                else:
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
                gen_content_args,
                __event_emitter__,
                __metadata__,
                __user__["id"],
            )
        else:
            # Non-streaming response.
            if "gemini-2.0-flash-preview-image-generation" in model_name:
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

    # region 1.1 Client initialization
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
            # FIXME: More detailed reason in the exception (tell user to set the API key).
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
        user_whitelist = (
            valves.AUTH_WHITELIST.split(",") if valves.AUTH_WHITELIST else []
        )
        log.debug(
            f"User whitelist: {user_whitelist}, user email: {user_email}, "
            f"USER_MUST_PROVIDE_AUTH_CONFIG: {valves.USER_MUST_PROVIDE_AUTH_CONFIG}"
        )
        if valves.USER_MUST_PROVIDE_AUTH_CONFIG and user_email not in user_whitelist:
            if not valves.GEMINI_API_KEY:
                error_msg = (
                    "User must provide their own authentication configuration. "
                    "Please set GEMINI_API_KEY in your UserValves."
                )
                raise ValueError(error_msg)
        try:
            client_args = self._prepare_client_args(valves)
            client = self._get_or_create_genai_client(*client_args)
        except GenaiApiError as e:
            error_msg = f"Failed to initialize genai client for user {user_email}: {e}"
            # FIXME: include correct traceback.
            raise ValueError(error_msg) from e
        return client

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

    # endregion 1.1 Client initialization

    # region 1.2 Model retrival from Google API
    @cached()  # aiocache.cached for async method
    async def _get_genai_models(
        self,
        api_key: str | None,
        base_url: str | None,
        use_vertex_ai: bool | None,  # User's preference from config
        vertex_project: str | None,
        vertex_location: str | None,
        whitelist_str: str,
        blacklist_str: str | None,
    ) -> list["ModelData"]:
        """
        Gets valid Google models from API(s) and filters them.
        If use_vertex_ai, vertex_project, and api_key are all provided,
        models are fetched from both Vertex AI and Gemini Developer API and merged.
        """
        all_raw_models: list[types.Model] = []

        # Condition for fetching from both sources
        fetch_both = bool(use_vertex_ai and vertex_project and api_key)

        if fetch_both:
            log.info(
                "Attempting to fetch models from both Gemini Developer API and Vertex AI."
            )
            gemini_models_list: list[types.Model] = []
            vertex_models_list: list[types.Model] = []

            # TODO: perf, consider parallelizing these two fetches
            # 1. Fetch from Gemini Developer API
            try:
                gemini_client = self._get_or_create_genai_client(
                    api_key=api_key,
                    base_url=base_url,
                    use_vertex_ai=False,  # Explicitly target Gemini API
                    vertex_project=None,
                    vertex_location=None,
                )
                gemini_models_list = await self._fetch_models_from_client_internal(
                    gemini_client, "Gemini Developer API"
                )
            except GenaiApiError as e:
                log.warning(
                    f"Failed to initialize or retrieve models from Gemini Developer API: {e}"
                )
            except Exception as e:
                log.warning(
                    f"An unexpected error occurred with Gemini Developer API models: {e}",
                    exc_info=True,
                )

            # 2. Fetch from Vertex AI
            try:
                vertex_client = self._get_or_create_genai_client(
                    use_vertex_ai=True,  # Explicitly target Vertex AI
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    api_key=None,  # API key is not used for Vertex AI with project auth
                    base_url=base_url,  # Pass base_url for potential Vertex custom endpoints
                )
                vertex_models_list = await self._fetch_models_from_client_internal(
                    vertex_client, "Vertex AI"
                )
            except GenaiApiError as e:
                log.warning(
                    f"Failed to initialize or retrieve models from Vertex AI: {e}"
                )
            except Exception as e:
                log.warning(
                    f"An unexpected error occurred with Vertex AI models: {e}",
                    exc_info=True,
                )

            # 3. Combine and de-duplicate
            # Prioritize models from Gemini Developer API in case of ID collision
            combined_models_dict: dict[str, types.Model] = {}

            for model in gemini_models_list:
                if model.name:
                    model_id = Pipe.strip_prefix(model.name)
                    if model_id and model_id not in combined_models_dict:
                        combined_models_dict[model_id] = model
                else:
                    log.trace(
                        f"Gemini model without a name encountered: {model.display_name or 'N/A'}"
                    )

            for model in vertex_models_list:
                if model.name:
                    model_id = Pipe.strip_prefix(model.name)
                    if model_id:
                        if model_id not in combined_models_dict:
                            combined_models_dict[model_id] = model
                        else:
                            log.info(
                                f"Duplicate model ID '{model_id}' from Vertex AI already sourced from Gemini API. Keeping Gemini API version."
                            )
                else:
                    log.trace(
                        f"Vertex AI model without a name encountered: {model.display_name or 'N/A'}"
                    )

            all_raw_models = list(combined_models_dict.values())

            log.info(
                f"Fetched {len(gemini_models_list)} models from Gemini API, "
                f"{len(vertex_models_list)} from Vertex AI. "
                f"Combined to {len(all_raw_models)} unique models."
            )

            if not all_raw_models and (gemini_models_list or vertex_models_list):
                log.warning(
                    "Models were fetched but resulted in an empty list after de-duplication, possibly due to missing names or empty/duplicate IDs."
                )

            if not all_raw_models and not gemini_models_list and not vertex_models_list:
                raise GenaiApiError(
                    "Failed to retrieve models: Both Gemini Developer API and Vertex AI attempts yielded no models."
                )

        else:  # Single source logic
            # Determine if we are effectively using Vertex AI or Gemini API
            # This depends on user's config (use_vertex_ai) and availability of project/key
            client_target_is_vertex = bool(use_vertex_ai and vertex_project)
            client_source_name = (
                "Vertex AI" if client_target_is_vertex else "Gemini Developer API"
            )
            log.info(
                f"Attempting to fetch models from a single source: {client_source_name}."
            )

            try:
                client = self._get_or_create_genai_client(
                    api_key=api_key,
                    base_url=base_url,
                    use_vertex_ai=client_target_is_vertex,  # Pass the determined target
                    vertex_project=vertex_project if client_target_is_vertex else None,
                    vertex_location=(
                        vertex_location if client_target_is_vertex else None
                    ),
                )
                all_raw_models = await self._fetch_models_from_client_internal(
                    client, client_source_name
                )

                if not all_raw_models:
                    raise GenaiApiError(
                        f"No models retrieved from {client_source_name}. This could be due to an API error, network issue, or no models being available."
                    )

            except GenaiApiError as e:
                raise GenaiApiError(
                    f"Failed to get models from {client_source_name}: {e}"
                ) from e
            except Exception as e:
                log.error(
                    f"An unexpected error occurred while configuring client or fetching models from {client_source_name}: {e}",
                    exc_info=True,
                )
                raise GenaiApiError(
                    f"An unexpected error occurred while retrieving models from {client_source_name}: {e}"
                ) from e

        # --- Common processing for all_raw_models ---

        if not all_raw_models:
            log.warning("No models available after attempting all configured sources.")
            return []

        log.info(f"Processing {len(all_raw_models)} unique raw models.")

        generative_models: list[types.Model] = []
        for model in all_raw_models:
            if model.name is None:
                log.trace(
                    f"Skipping model with no name during generative filter: {model.display_name or 'N/A'}"
                )
                continue
            actions = model.supported_actions
            if (
                actions is None or "generateContent" in actions
            ):  # Includes models if actions is None (e.g., Vertex)
                generative_models.append(model)
            else:
                log.trace(
                    f"Model '{model.name}' (ID: {Pipe.strip_prefix(model.name)}) skipped, not generative (actions: {actions})."
                )

        if not generative_models:
            log.warning(
                "No generative models found after filtering all retrieved models."
            )
            return []

        def match_patterns(
            name_to_check: str, list_of_patterns_str: str | None
        ) -> bool:
            if not list_of_patterns_str:
                return False
            patterns = [
                pat for pat in list_of_patterns_str.replace(" ", "").split(",") if pat
            ]  # Ensure pat is not empty
            return any(fnmatch.fnmatch(name_to_check, pat) for pat in patterns)

        filtered_models_data: list["ModelData"] = []
        for model in generative_models:
            # model.name is guaranteed non-None by generative_models filter logic
            stripped_name = Pipe.strip_prefix(model.name)  # type: ignore

            if not stripped_name:
                log.warning(
                    f"Model '{model.name}' (display: {model.display_name}) resulted in an empty ID after stripping. Skipping."
                )
                continue

            passes_whitelist = not whitelist_str or match_patterns(
                stripped_name, whitelist_str
            )
            passes_blacklist = not blacklist_str or not match_patterns(
                stripped_name, blacklist_str
            )

            if passes_whitelist and passes_blacklist:
                filtered_models_data.append(
                    {
                        "id": stripped_name,
                        "name": model.display_name or stripped_name,
                        "description": model.description,
                    }
                )
            else:
                log.trace(
                    f"Model ID '{stripped_name}' filtered out by whitelist/blacklist. Whitelist match: {passes_whitelist}, Blacklist pass: {passes_blacklist}"
                )

        log.info(
            f"Filtered {len(generative_models)} generative models down to {len(filtered_models_data)} models based on white/blacklists."
        )
        return filtered_models_data

    # TODO: Use cache for this method too?
    async def _fetch_models_from_client_internal(
        self, client: genai.Client, source_name: str
    ) -> list[types.Model]:
        """Helper to fetch models from a given client and handle common exceptions."""
        try:
            google_models_pager = await client.aio.models.list(
                config={"query_base": True}  # Fetch base models by default
            )
            models = [model async for model in google_models_pager]
            log.info(f"Retrieved {len(models)} models from {source_name}.")
            log.trace(
                f"All models returned by {source_name}:", payload=models
            )  # Can be verbose
            return models
        except Exception as e:
            log.error(f"Retrieving models from {source_name} failed: {e}")
            # Return empty list; caller decides if this is fatal for the whole operation.
            return []

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

    # endregion 1.2 Model retrival from Google API

    # region 1.3 Open WebUI's body.messages -> list[genai.types.Content] conversion

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
                    # YouTube URL extraction is now handled by _genai_parts_from_text
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

    @staticmethod
    def _enable_special_tags(text: str) -> str:
        """
        Reverses the action of _disable_special_tags by removing the ZWS
        from special tags. This is used to clean up history messages before
        sending them to the model, so it can understand the context correctly.
        """
        if not text:
            return ""

        # The regex finds '<ZWS' followed by an optional '/' and then one of the special tags.
        # The inner parentheses group the tags, so the optional '/' applies to all of them.
        REVERSE_TAG_REGEX = re.compile(
            r"<"
            + ZWS
            + r"(/?"
            + "("
            + "|".join(re.escape(tag) for tag in SPECIAL_TAGS_TO_DISABLE)
            + ")"
            + r")"
        )
        # The substitution restores the original tag, e.g., '<ZWS/think' becomes '</think'.
        restored_text, count = REVERSE_TAG_REGEX.subn(r"<\1", text)
        if count > 0:
            log.debug(f"Re-enabled {count} special tag(s) for model context.")

        return restored_text

    def _genai_parts_from_text(self, text: str) -> list[types.Part]:
        if not text:
            return []

        # Restore special tags that were disabled for front-end safety.
        # This ensures the model receives the original, intended text.
        text = self._enable_special_tags(text)

        parts: list[types.Part] = []
        last_pos = 0

        # Regex to find markdown images or YouTube URLs
        # Markdown image: !\[.*?\]\((data:(image/[^;]+);base64,([^)]+)|/api/v1/files/([a-f0-9\-]+)/content)\)
        # YouTube URL: (https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^&\s]+)
        pattern = re.compile(
            r"!\[.*?\]\((data:(image/[^;]+);base64,([^)]+)|/api/v1/files/([a-f0-9\-]+)/content)\)|"
            r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[^&\s]+)"
        )

        for match in pattern.finditer(text):
            # Add text before the current match
            text_segment = text[last_pos : match.start()]
            if text_segment.strip():  # Ensure non-empty, stripped text
                parts.append(types.Part.from_text(text=text_segment.strip()))

            # Determine if it's an image or a YouTube URL
            if match.group(1):  # It's a markdown image
                if match.group(2):  # Base64 encoded image
                    try:
                        mime_type = match.group(2)
                        base64_data = match.group(3)
                        image_part = types.Part.from_bytes(
                            data=base64.b64decode(base64_data),
                            mime_type=mime_type,
                        )
                        parts.append(image_part)
                    except Exception:
                        log.exception("Error decoding base64 image:")
                elif match.group(4):  # File URL for image
                    file_id = match.group(4)
                    image_bytes, mime_type = self._get_file_data(file_id)
                    if image_bytes and mime_type:
                        image_part = types.Part.from_bytes(
                            data=image_bytes, mime_type=mime_type
                        )
                        parts.append(image_part)
            elif match.group(5):  # It's a YouTube URL
                youtube_url = match.group(5)
                log.info(f"Found YouTube URL: {youtube_url}")
                if (
                    self.valves.USE_VERTEX_AI
                ):  # Vertex AI expects from_uri part instead of files
                    parts.append(
                        types.Part.from_uri(file_uri=youtube_url, mime_type="video/mp4")
                    )
                else:
                    parts.append(
                        types.Part(file_data=types.FileData(file_uri=youtube_url))
                    )

            last_pos = match.end()

        # Add remaining text after the last match
        remaining_text = text[last_pos:]
        if remaining_text.strip():  # Ensure non-empty, stripped text
            parts.append(types.Part.from_text(text=remaining_text.strip()))

        # If no matches were found at all (e.g. plain text), the original text (stripped) is added as a single part.
        if not parts and text.strip():
            parts.append(types.Part.from_text(text=text.strip()))

        # If parts list is empty and original text was only whitespace, return empty list.
        # Otherwise, if parts were added, or if it was plain text, it's handled above.
        # This check ensures that if text was "   ", we don't add a Part for "   ".
        # The .strip() in the conditions above should handle this, but as a safeguard:
        if not parts and not text.strip():
            return []

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

    # endregion 1.3 Open WebUI's body.messages -> list[genai.types.Content] conversion

    # region 1.4 Model response streaming
    async def _process_parts_to_structured_stream(
        self,
        response_stream: AsyncIterator[types.GenerateContentResponse],
        __request__: Request,
        gen_content_args: dict,
        user_id: str,
    ) -> AsyncGenerator[tuple[dict, int, types.GenerateContentResponse | None], None]:
        """
        Processes a stream of Gemini responses, yielding structured dictionaries,
        a substitution count for the ZWS safeguard, and the raw chunk.
        """
        try:
            async for chunk in response_stream:
                if not (candidate := self._get_first_candidate(chunk.candidates)):
                    log.warning("Stream chunk has no candidates, skipping.")
                    continue
                if not (parts := candidate.content and candidate.content.parts):
                    log.warning("Candidate has no content parts, skipping.")
                    continue

                for part in parts:
                    # Initialize variables at the start of each loop to satisfy the linter
                    # and ensure they always have a defined state.
                    payload: dict | None = None
                    count: int = 0
                    key: str = "content"

                    match part:
                        case types.Part(text=str(text), thought=True):
                            # It's a thought, so we'll use the "reasoning" key.
                            key = "reasoning"
                            sanitized_text, count = self._disable_special_tags(text)
                            payload = {key: sanitized_text}
                        case types.Part(text=str(text)):
                            # It's regular content, using the default "content" key.
                            sanitized_text, count = self._disable_special_tags(text)
                            payload = {key: sanitized_text}
                        case types.Part(inline_data=data):
                            # Image parts don't need tag disabling.
                            processed_text = self._process_image_part(
                                data, gen_content_args, user_id, __request__
                            )
                            if processed_text:
                                payload = {"content": processed_text}
                        case types.Part(executable_code=code):
                            processed_text = self._process_executable_code_part(code)
                            # Code blocks are already formatted and safe.
                            if processed_text:
                                payload = {"content": processed_text}
                        case types.Part(code_execution_result=result):
                            processed_text = self._process_code_execution_result_part(
                                result
                            )
                            # Code results are also safe.
                            if processed_text:
                                payload = {"content": processed_text}

                    if payload:
                        structured_chunk = {"choices": [{"delta": payload}]}
                        yield structured_chunk, count, chunk
        except Exception:
            raise

    async def _stream_response_generator(
        self,
        response_stream: AsyncIterator[types.GenerateContentResponse],
        __request__: Request,
        gen_content_args: dict,
        event_emitter: Callable[["Event"], Awaitable[None]],
        metadata: dict[str, Any],
        user_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Yields structured dictionary chunks from the stream, counts tag substitutions
        for a final toast notification, and handles post-processing.
        """
        final_response_chunk: types.GenerateContentResponse | None = None
        error_occurred = False
        total_substitutions = 0

        try:
            part_processor = self._process_parts_to_structured_stream(
                response_stream, __request__, gen_content_args, user_id
            )
            async for structured_chunk, count, raw_chunk in part_processor:
                if count > 0:
                    total_substitutions += count
                    log.debug(f"Disabled {count} special tag(s) in a chunk.")

                if raw_chunk:
                    final_response_chunk = raw_chunk
                yield structured_chunk

        except Exception as e:
            error_occurred = True
            error_msg = f"Stream ended with error: {e}"
            # FIXME: raise the error instead?
            await self._emit_error(error_msg, event_emitter)
        finally:
            if total_substitutions > 0 and not error_occurred:
                plural_s = "s" if total_substitutions > 1 else ""
                toast_msg = (
                    f"For clarity, {total_substitutions} special tag{plural_s} "
                    "were disabled in the response by injecting a zero-width space (ZWS)."
                )
                await self._emit_toast(toast_msg, event_emitter, "info")

            if not error_occurred:
                log.info("Stream finished successfully!")
                log.debug("Last chunk:", payload=final_response_chunk)

            try:
                await self._do_post_processing(
                    final_response_chunk,
                    event_emitter,
                    metadata,
                    __request__,
                    error_occurred,
                )
            except Exception as e:
                error_msg = f"Post-processing failed with error:\n\n{e}"
                await self._emit_toast(error_msg, event_emitter, "error")
                log.exception(error_msg)

            log.debug("AsyncGenerator finished.")

    @staticmethod
    def _disable_special_tags(text: str) -> tuple[str, int]:
        """
        Finds special tags in a text chunk and inserts a Zero-Width Space (ZWS)
        to prevent them from being parsed by the Open WebUI backend's legacy system.
        This is a safeguard against accidental tag generation by the model.
        """
        if not text:
            return "", 0

        # The regex finds '<' followed by an optional '/' and then one of the special tags.
        # The inner parentheses group the tags, so the optional '/' applies to all of them.
        TAG_REGEX = re.compile(
            r"<(/?"
            + "("
            + "|".join(re.escape(tag) for tag in SPECIAL_TAGS_TO_DISABLE)
            + ")"
            + r")"
        )
        # The substitution injects a ZWS, e.g., '</think>' becomes '<ZWS/think'.
        modified_text, num_substitutions = TAG_REGEX.subn(rf"<{ZWS}\1", text)
        return modified_text, num_substitutions

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

    # endregion 1.4 Model response streaming

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

        # TODO: Emit a toast message if url context retrieval was not successful.

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
        """Emits a toast notification to the front-end."""
        # TODO: Use this method in more places, even for info toasts.
        event: NotificationEvent = {
            "type": "notification",
            "data": {"type": toastType, "content": msg},
        }
        await event_emitter(event)

    # endregion 1.6 Event emissions

    # region 1.7 Logging

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

    # endregion 1.7 Logging

    # region 1.8 Utility helpers

    @staticmethod
    def _is_function_active(id: str) -> bool:
        # Get the filter's data from the database.
        companion_filter = Functions.get_function_by_id(id)
        # Return if the filter is installed and active.
        return bool(companion_filter and companion_filter.is_active)

    @staticmethod
    def _get_merged_valves(
        default_valves: "Pipe.Valves",
        user_valves: "Pipe.UserValves | None",
        user_email: str,
    ) -> "Pipe.Valves":
        """
        Merges UserValves into a base Valves configuration.

        The general rule is that if a field in UserValves is not None, it overrides
        the corresponding field in the default_valves. Otherwise, the default_valves
        field value is used.

        Exceptions:
        - If default_valves.USER_MUST_PROVIDE_AUTH_CONFIG is True, then GEMINI_API_KEY and
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

        user_whitelist = (
            default_valves.AUTH_WHITELIST.split(",")
            if default_valves.AUTH_WHITELIST
            else []
        )

        # Apply special logic based on default_valves.USER_MUST_PROVIDE_AUTH_CONFIG
        if (
            default_valves.USER_MUST_PROVIDE_AUTH_CONFIG
            and user_email not in user_whitelist
        ):
            # If USER_MUST_PROVIDE_AUTH_CONFIG is True and user is not in the whitelist,
            # then user must provide their own GEMINI_API_KEY
            # User is disallowed from using Vertex AI in this case.
            merged_data["GEMINI_API_KEY"] = user_valves.GEMINI_API_KEY
            merged_data["VERTEX_PROJECT"] = None
            merged_data["USE_VERTEX_AI"] = False

        # Create a new Valves instance with the merged data.
        # Pydantic will validate the data against the Valves model definition during instantiation.
        return Pipe.Valves(**merged_data)

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

    # endregion 1.8 Utility helpers

    # endregion 1. Helper methods inside the Pipe class
