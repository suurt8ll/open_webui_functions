"""
title: Gemini Manifold Companion
id: gemini_manifold_companion
description: A companion filter for "Gemini Manifold google_genai" pipe providing enhanced functionality.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.2.0
"""

# This filter can detect that a feature like web search or code execution is enabled in the front-end,
# set the feature back to False so Open WebUI does not run it's own logic and then
# pass custom values to "Gemini Manifold google_genai" that signal which feature was enabled and intercepted.

from google.genai import types

import sys
import asyncio
import aiohttp
from fastapi import Request
from fastapi.datastructures import State
from loguru import logger
from pydantic import BaseModel, Field
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TYPE_CHECKING, cast

from open_webui.utils.logger import stdout_format
from open_webui.models.functions import Functions

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.

# according to https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini
ALLOWED_GROUNDING_MODELS = [
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-001",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
]

# according to https://ai.google.dev/gemini-api/docs/code-execution
ALLOWED_CODE_EXECUTION_MODELS = [
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-001",
]


# Default timeout for URL resolution
# TODO: Move to Pipe.Valves.
DEFAULT_URL_TIMEOUT = aiohttp.ClientTimeout(total=10)  # 10 seconds total timeout

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


class Filter:

    class Valves(BaseModel):
        SET_TEMP_TO_ZERO: bool = Field(
            default=False,
            description="""Decide if you want to set the temperature to 0 for grounded answers, 
            Google reccomends it in their docs.""",
        )
        GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD: float | None = Field(
            default=None,
            description="""See https://ai.google.dev/gemini-api/docs/grounding?lang=python#dynamic-threshold for more information.
            Only supported for 1.0 and 1.5 models""",
        )
        LOG_LEVEL: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
            Field(
                default="INFO",
                description="Select logging level. Use `docker logs -f open-webui` to view logs.",
            )
        )

    # TODO: Support user settting through UserValves.

    def __init__(self):
        # This hack makes the valves values available to the `__init__` method.
        # TODO: Get the id from the frontmatter instead of hardcoding it.
        valves = Functions.get_function_valves_by_id("gemini_manifold_companion")
        self.valves = self.Valves(**(valves if valves else {}))
        # FIXME: There is no trigger for changing the log level if it changes inside Pipe.Valves
        self._add_log_handler()
        log.info("Filter function has been initialized!")

    def inlet(self, body: dict) -> dict:
        """Modifies the incoming request payload before it's sent to the LLM. Operates on the `form_data` dictionary."""

        # Exit early if we are filtering an unsupported model.
        model_name: str = body.get("model", "")

        # Extract and use base model name in case of custom Workspace models
        metadata = body.get("metadata", {})
        base_model_name = (
            metadata.get("model", {}).get("info", {}).get("base_model_id", None)
        )
        if base_model_name:
            model_name = base_model_name

        canonical_model_name = model_name.replace("gemini_manifold_google_genai.", "")

        if (
            "gemini_manifold_google_genai." not in model_name
            or canonical_model_name not in ALLOWED_GROUNDING_MODELS
            or canonical_model_name not in ALLOWED_CODE_EXECUTION_MODELS
        ):
            return body

        features = body.get("features", {})
        log.debug(f"Features: {features}")

        # Ensure metadata structure exists and add new feature
        metadata = body.setdefault("metadata", {})
        metadata_features = metadata.setdefault("features", {})

        if canonical_model_name in ALLOWED_GROUNDING_MODELS:
            web_search_enabled = (
                features.get("web_search", False)
                if isinstance(features, dict)
                else False
            )
            if web_search_enabled:
                log.info(
                    "Search feature is enabled, disabling it and adding custom feature called grounding_w_google_search."
                )
                # Disable web_search
                features["web_search"] = False
                # Use "Google Search Retrieval" for 1.0 and 1.5 models and "Google Search as a Tool for >=2.0 models".
                if "1.0" in model_name or "1.5" in model_name:
                    metadata_features["google_search_retrieval"] = True
                    metadata_features["google_search_retrieval_threshold"] = (
                        self.valves.GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD
                    )
                else:
                    metadata_features["google_search_tool"] = True
                # Google suggest setting temperature to 0 if using grounding:
                # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-with-google-search#:~:text=For%20ideal%20results%2C%20use%20a%20temperature%20of%200.0.
                if self.valves.SET_TEMP_TO_ZERO:
                    log.info("Setting temperature to 0.")
                    body["temperature"] = 0

        if canonical_model_name in ALLOWED_CODE_EXECUTION_MODELS:
            code_execution_enabled = (
                features.get("code_interpreter", False)
                if isinstance(features, dict)
                else False
            )
            if code_execution_enabled:
                log.info(
                    "Code interpreter feature is enabled, disabling it and adding custom feature called google_code_execution."
                )
                # Disable code_interpreter
                features["code_interpreter"] = False
                metadata_features["google_code_execution"] = True

        # TODO: Filter out the citation markers here.

        return body

    def stream(self, event: dict) -> dict:
        """Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data."""
        return event

    async def outlet(
        self,
        body: "Body",
        __request__: Request,
        __metadata__: dict[str, Any],
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        **kwargs,
    ) -> "Body":
        """Modifies the complete response payload after it's received from the LLM. Operates on the final `body` dictionary."""

        chat_id: str = __metadata__.get("chat_id", "")
        message_id: str = __metadata__.get("message_id", "")
        storage_key = f"grounding_{chat_id}_{message_id}"

        app_state: State = __request__.app.state
        log.info(f"Seeing if there is attribute {storage_key} in request state.")
        stored_metadata: types.GroundingMetadata | None = getattr(
            app_state, storage_key, None
        )
        if stored_metadata:
            log.info("Found grounding metadata, processing citations.")

            current_content = body["messages"][-1]["content"]
            if isinstance(current_content, list):
                # Extract text from list (could be empty if only images)
                text_to_use = ""
                for item in current_content:
                    if item.get("type") == "text":
                        item = cast("TextContent", item)
                        text_to_use = item["text"]
                        break
            else:
                text_to_use = current_content

            cited_text = await self._get_text_w_citation_markers(
                stored_metadata,
                text_to_use,
                __event_emitter__,
            )
            log.debug(f"Text with citations:\n{cited_text}")

            if cited_text:
                content = body["messages"][-1]["content"]
                if isinstance(content, list):
                    # Update existing text element if present
                    for item in content:
                        if item.get("type") == "text":
                            item = cast("TextContent", item)
                            item["text"] = cited_text
                            break  # Stop after first text element
                else:
                    body["messages"][-1]["content"] = cited_text

            await self._emit_status_event_w_queries(stored_metadata, __event_emitter__)
            delattr(app_state, storage_key)
        else:
            log.info("No grounding metadata found in request state.")

        return body

    # region Helper methods inside the Pipe class

    async def _get_text_w_citation_markers(
        self,
        grounding_metadata: types.GroundingMetadata,
        raw_str: str,
        event_emitter: Callable[["Event"], Awaitable[None]],
    ) -> str | None:
        """
        Returns the model response with citation markers and launches background URL resolution.
        """

        supports = grounding_metadata.grounding_supports
        grounding_chunks = grounding_metadata.grounding_chunks
        if not supports or not grounding_chunks:
            log.info(
                "Grounding metadata missing supports or chunks, can't insert citation markers."
            )
            return

        initial_metadatas: list[tuple[int, str]] = []
        for i, g_c in enumerate(grounding_chunks):
            web_info = g_c.web
            if web_info and web_info.uri:
                initial_metadatas.append((i, web_info.uri))

        source_metadatas_template: list["SourceMetadata"] = [
            {
                "source": None,
                "original_url": None,
                "supports": [],
            }
            for _ in grounding_chunks
        ]

        # Ensure raw_str is not empty before processing
        final_content_with_markers = raw_str
        if raw_str:
            try:
                modified_content_bytes = bytearray(raw_str.encode("utf-8"))
                for support in reversed(supports):
                    segment = support.segment
                    indices = support.grounding_chunk_indices

                    if not (
                        indices is not None
                        and segment
                        and segment.end_index is not None
                    ):
                        log.debug(f"Skipping support due to missing data: {support}")
                        continue

                    end_pos = segment.end_index
                    citation_markers = "".join(f"[{index + 1}]" for index in indices)
                    encoded_citation_markers = citation_markers.encode("utf-8")

                    modified_content_bytes[end_pos:end_pos] = encoded_citation_markers

                final_content_with_markers = modified_content_bytes.decode("utf-8")
            except Exception as e:
                log.error(
                    f"Error injecting citation markers: {e}. Emitting original text."
                )
                final_content_with_markers = raw_str  # Fallback to original text

        else:
            log.warning("Raw string is empty, cannot inject citation markers.")

        if initial_metadatas:
            await self._resolve_and_emit_sources(
                initial_metadatas=initial_metadatas,
                source_metadatas_template=source_metadatas_template,
                supports=supports,
                event_emitter=event_emitter,
            )
        else:
            log.info("No source URIs found, skipping background URL resolution task.")

        return final_content_with_markers

    async def _resolve_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
        timeout: aiohttp.ClientTimeout = DEFAULT_URL_TIMEOUT,
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> str:
        """Resolves a given URL using the provided aiohttp session, with multiple retries on failure."""

        if not url:
            return ""

        for attempt in range(max_retries + 1):
            try:
                async with session.get(
                    url,
                    allow_redirects=True,
                    timeout=timeout,
                ) as response:
                    final_url = str(response.url)
                    log.debug(
                        f"Resolved URL '{url}' to '{final_url}' after {attempt} retries"
                    )
                    return final_url
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt == max_retries:
                    log.error(
                        f"Failed to resolve URL '{url}' after {max_retries + 1} attempts: {e}"
                    )
                    return url
                else:
                    delay = min(base_delay * (2**attempt), 10.0)
                    log.warning(
                        f"Retry {attempt + 1}/{max_retries + 1} for URL '{url}': {e}. Waiting {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            except Exception as e:
                log.error(f"Unexpected error resolving URL '{url}': {e}")
                return url
        return url  # Shouldn't reach here if loop is correct

    async def _resolve_and_emit_sources(
        self,
        initial_metadatas: list[tuple[int, str]],
        source_metadatas_template: list["SourceMetadata"],
        supports: list[types.GroundingSupport],
        event_emitter: Callable[["Event"], Awaitable[None]],
    ):
        """
        Resolves URLs in the background and emits a chat completion event
        containing only the source information.
        """
        resolved_uris_map = {}
        try:
            urls_to_resolve = [url for _, url in initial_metadatas]
            resolved_uris: list[str] = []

            log.info(f"Resolving {len(urls_to_resolve)} source URLs...")
            async with aiohttp.ClientSession() as session:
                tasks = [self._resolve_url(session, url) for url in urls_to_resolve]
                resolved_uris = await asyncio.gather(*tasks)
            log.info("URL resolution completed.")

            resolved_uris_map = dict(zip(urls_to_resolve, resolved_uris))

        except Exception as e:
            log.error(f"Error during URL resolution: {e}")
            # Populate map with original URLs as fallback if resolution fails
            resolved_uris_map = {url: url for _, url in initial_metadatas}

        # Populate a *copy* of the template
        populated_metadatas = [m.copy() for m in source_metadatas_template]

        for chunk_index, original_uri in initial_metadatas:
            # Use resolved URI if available, otherwise fallback to original
            resolved_uri = resolved_uris_map.get(original_uri, original_uri)
            if 0 <= chunk_index < len(populated_metadatas):
                populated_metadatas[chunk_index]["original_url"] = original_uri
                populated_metadatas[chunk_index]["source"] = resolved_uri
            else:
                log.warning(
                    f"Chunk index {chunk_index} out of bounds when populating resolved URLs."
                )

        # Add support metadata back
        for support in supports:
            segment = support.segment
            indices = support.grounding_chunk_indices
            if not (indices is not None and segment and segment.end_index is not None):
                continue

            for index in indices:
                if 0 <= index < len(populated_metadatas):
                    populated_metadatas[index]["supports"].append(support.model_dump())  # type: ignore
                else:
                    log.warning(
                        f"Invalid grounding chunk index {index} found in support during background processing."
                    )

        # Filter out metadata entries that didn't have an original URL
        # (Shouldn't happen with the fallback logic, but keep as safeguard)
        valid_source_metadatas = [
            m for m in populated_metadatas if m.get("original_url") is not None
        ]

        sources_list: list["Source"] = []
        if valid_source_metadatas:
            # Ensure document list matches metadata length
            doc_list = [""] * len(valid_source_metadatas)
            sources_list.append(
                {
                    "source": {"name": "web_search"},
                    "document": doc_list,
                    "metadata": valid_source_metadatas,
                }
            )

        # Prepare and emit the sources-only event
        event: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {"sources": sources_list},
        }
        await event_emitter(event)
        log.info("Emitted sources event.")

    async def _emit_status_event_w_queries(
        self,
        grounding_metadata: types.GroundingMetadata,
        event_emitter: Callable[["Event"], Awaitable[None]],
    ) -> None:
        """
        Creates a StatusEvent with Google search URLs based on the web_search_queries
        in the GenerateContentResponse.
        """
        if not grounding_metadata.web_search_queries:
            log.warning("Grounding metadata does not contain any search queries.")
            return None

        search_queries = grounding_metadata.web_search_queries
        if not search_queries:
            log.debug("web_search_queries list is empty.")
            return None
        google_search_urls = [
            f"https://www.google.com/search?q={query}" for query in search_queries
        ]

        status_event_data: StatusEventData = {
            "action": "web_search",
            "description": "This response was grounded with Google Search",
            "urls": google_search_urls,
        }
        status_event: StatusEvent = {
            "type": "status",
            "data": status_event_data,
        }
        log.debug(f"Emitting StatusEvent: {status_event}")
        await event_emitter(status_event)

    def _get_first_candidate(
        self, candidates: list[types.Candidate] | None
    ) -> types.Candidate | None:
        """Selects the first candidate, logging a warning if multiple exist."""
        if not candidates:
            log.warning("Received chunk with no candidates, skipping processing.")
            return None
        if len(candidates) > 1:
            log.warning("Multiple candidates found, defaulting to first candidate.")
        return candidates[0]

    def _add_log_handler(self):
        """Adds handler to the root loguru instance for this plugin if one does not exist already."""

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__  # Filter by module name

        # Access the internal state of the log
        handlers: dict[int, "Handler"] = log._core.handlers  # type: ignore
        for key, handler in handlers.items():
            existing_filter = handler._filter
            # FIXME: Check log level too.
            if (
                hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            ):
                log.debug("Handler for this plugin is already present!")
                return

        log.add(
            sys.stdout,
            level=self.valves.LOG_LEVEL,
            format=stdout_format,
            filter=plugin_filter,
        )
        log.info(
            f"Added new handler to loguru with level {self.valves.LOG_LEVEL} and filter {__name__}."
        )

    # endregion
