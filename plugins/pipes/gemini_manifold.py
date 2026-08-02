"""
title: Gemini Manifold google_genai
id: gemini_manifold_google_genai
description: Manifold function for Gemini Developer API and Vertex AI. Uses the newer google-genai SDK. Aims to support as many features from it as possible.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 2.1.0
requirements: google-genai==2.14.0
"""

# I change these only when I make a release to avoid PR merge conflicts.
# If you are making a PR then please do not change these values.
VERSION = "2.1.0"
# This is the recommended version for the companion filter.
# Older versions might still work, but backward compatibility is not guaranteed
# during the development of this personal use plugin.
RECOMMENDED_COMPANION_VERSION = "2.1.0"


# Keys `title`, `id` and `description` in the frontmatter above are used for my own development purposes.
# They don't have any effect on the plugin's functionality.


# This is a helper function that provides a manifold for Google's Gemini Studio API and Vertex AI.
# Be sure to check out my GitHub repository for more information! Contributions, questions and suggestions are very welcome.

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from google.cloud import storage
from google.api_core import exceptions

import json
import time
import copy
from urllib.parse import urlparse, parse_qs
import xxhash
import asyncio
import aiofiles
from aiocache import cached
from aiocache.base import BaseCache
from aiocache.serializers import NullSerializer
from aiocache.backends.memory import SimpleMemoryCache
from functools import cache
from datetime import datetime, timezone
from fastapi.datastructures import State
import io
import mimetypes
import uuid
import base64
import re
import fnmatch
import difflib
from loguru import logger
from fastapi import Request, FastAPI
import pydantic_core
from pydantic import BaseModel, Field, field_validator
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import (
    Any,
    Final,
    AsyncGenerator,
    Literal,
    TYPE_CHECKING,
    cast,
)

from open_webui.models.chats import Chats
from open_webui.models.files import FileForm, Files
from open_webui.storage.provider import Storage
from open_webui.models.functions import Functions
from open_webui.utils.misc import pop_system_message

# This block is skipped at runtime.
if TYPE_CHECKING:
    from plugins.filters.gemini_manifold_companion import EventEmitter
    # Imports custom type definitions (TypedDicts) for static analysis purposes (mypy/pylance).
    from utils.manifold_types import *

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)

def _log_and_toast(
    event_emitter: "EventEmitter",
    msg: str,
    level: Literal["info", "warning", "error"] = "warning",
) -> None:
    """Logs `msg` at the caller's stack location and mirrors it to the front-end as a toast."""
    log.opt(depth=1).log(level, msg)
    event_emitter.emit_toast(msg, level)

# A mapping of finish reason names (str) to human-readable descriptions.
# This allows handling of reasons that may not be defined in the current SDK version.
FINISH_REASON_DESCRIPTIONS: Final = {
    "FINISH_REASON_UNSPECIFIED": "The reason for finishing is not specified.",
    "STOP": "Natural stopping point or stop sequence reached.",
    "MAX_TOKENS": "The maximum number of tokens was reached.",
    "SAFETY": "The response was blocked due to safety concerns.",
    "RECITATION": "The response was blocked due to potential recitation of copyrighted material.",
    "LANGUAGE": "The response was stopped because of an unsupported language.",
    "OTHER": "The response was stopped for an unspecified reason.",
    "BLOCKLIST": "The response was blocked due to a word on a blocklist.",
    "PROHIBITED_CONTENT": "The response was blocked for containing prohibited content.",
    "SPII": "The response was blocked for containing sensitive personally identifiable information.",
    "MALFORMED_FUNCTION_CALL": "The model generated an invalid function call.",
    "IMAGE_SAFETY": "Generated image was blocked due to safety concerns.",
    "UNEXPECTED_TOOL_CALL": "The model generated an invalid tool call.",
    "IMAGE_PROHIBITED_CONTENT": "Generated image was blocked for containing prohibited content.",
    "NO_IMAGE": "The model was expected to generate an image, but it did not.",
    "IMAGE_OTHER": (
        "Image generation stopped for other reasons, possibly related to safety or quality. "
        "Try a different image or prompt."
    ),
}

# Finish reasons that are considered normal and do not require user notification.
NORMAL_REASONS: Final = {types.FinishReason.STOP, types.FinishReason.MAX_TOKENS}

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

    pass


class FilesAPIError(Exception):
    """Custom exception for errors during Files API operations."""

    pass


class UploadStatusManager:
    """
    Manages and centralizes status updates for concurrent file uploads.

    This manager is self-configuring. It discovers the number of files that
    require an actual upload at runtime, only showing a status message to the
    user when network activity is necessary.

    The communication protocol uses tuples sent via an asyncio.Queue:
    - ('REGISTER_UPLOAD',): Sent by a worker when it determines an upload is needed.
    - ('COMPLETE_UPLOAD',): Sent by a worker when its upload is finished.
    - ('FINALIZE',): Sent by the orchestrator when all workers are done.
    """

    def __init__(
        self,
        event_emitter: "EventEmitter",
    ):
        self.event_emitter = event_emitter
        self.queue = asyncio.Queue()
        self.total_uploads_expected = 0
        self.uploads_completed = 0
        self.finalize_received = False
        self.is_active = False

    async def run(self) -> None:
        """
        Runs the manager loop, listening for updates and emitting status to the UI.
        This should be started as a background task using asyncio.create_task().
        """
        while not (
            self.finalize_received
            and self.total_uploads_expected == self.uploads_completed
        ):
            msg = await self.queue.get()
            msg_type = msg[0]

            if msg_type == "REGISTER_UPLOAD":
                self.is_active = True
                self.total_uploads_expected += 1
                await self._emit_progress_update()
            elif msg_type == "COMPLETE_UPLOAD":
                self.uploads_completed += 1
                await self._emit_progress_update()
            elif msg_type == "FINALIZE":
                self.finalize_received = True

            self.queue.task_done()

        log.debug("UploadStatusManager finished its run.")

    async def _emit_progress_update(self) -> None:
        """Emits the current progress to the front-end if uploads are active."""
        if not self.is_active:
            return

        is_done = (
            self.total_uploads_expected > 0
            and self.uploads_completed == self.total_uploads_expected
        )

        if is_done:
            message = f"Upload complete. {self.uploads_completed} file(s) processed."
        else:
            # Show "Uploading 1 of N..."
            message = f"Uploading file {self.uploads_completed + 1} of {self.total_uploads_expected}..."

        self.event_emitter.emit_status(message, done=is_done, indent_level=1)


class FilesAPIManager:
    """
    Manages uploading, caching, and retrieving files using the Google Gemini Files API.

    This class provides a stateless and efficient way to handle files by using a fast,
    non-cryptographic hash (xxHash) of the file's content as the primary identifier.
    This enables content-addressable storage, preventing duplicate uploads of the
    same file. It uses a multi-tiered approach:

    1. Hot Path (In-Memory Caches): For instantly retrieving file objects and hashes
       for recently used files.
    2. Warm Path (Stateless GET): For quickly recovering file state after a server
       restart by using a deterministic name (derived from the content hash) and a
       single `get` API call.
    3. Cold Path (Upload): As a last resort, for uploading new files or re-uploading
       expired ones.
    """

    def __init__(
        self,
        client: genai.Client,
        file_cache: SimpleMemoryCache,
        id_hash_cache: SimpleMemoryCache,
        event_emitter: "EventEmitter",
    ):
        """
        Initializes the FilesAPIManager.

        Args:
            client: An initialized `google.genai.Client` instance.
            file_cache: An aiocache instance for mapping `content_hash -> types.File`.
                        Must be configured with `aiocache.serializers.NullSerializer`.
            id_hash_cache: An aiocache instance for mapping `owui_file_id -> content_hash`.
                           This is an optimization to avoid re-hashing known files.
            event_emitter: An abstract class for emitting events to the front-end.
        """
        self.client = client
        self.file_cache = file_cache
        self.id_hash_cache = id_hash_cache
        self.event_emitter = event_emitter
        # A dictionary to manage locks for concurrent uploads.
        # The key is a composite of api_key_hash and content_hash.
        self.upload_locks: dict[str, asyncio.Lock] = {}
        self.api_key_hash = self._get_api_key_hash()

    def _get_api_key_hash(self) -> str:
        """
        Returns a hash of the API key for use in cache keys.

        Returns 'no_key' if the client is not using an API key (e.g., Vertex AI with ADC).
        """
        # The genai.Client object doesn't expose the API key directly.
        # It's stored in the internal _api_client.
        api_key = getattr(self.client._api_client, "api_key", None)
        if not api_key:
            # This could happen if using Vertex AI with Application Default Credentials
            return "no_key"
        return xxhash.xxh64(api_key.encode("utf-8")).hexdigest()

    def _get_file_cache_key(self, content_hash: str) -> str:
        """Gets the namespaced key for the file cache."""
        return f"{self.api_key_hash}:{content_hash}"

    def _get_lock_key(self, content_hash: str) -> str:
        """Gets the namespaced key for upload locks."""
        # Although the deterministic_name is content-based, the file's ownership
        # is tied to the API key (project). Locking per API key + content hash
        # allows concurrent uploads of the same file for different users.
        return f"{self.api_key_hash}:{content_hash}"

    async def get_or_upload_file(
        self,
        file_bytes: bytes,
        mime_type: str,
        *,
        owui_file_id: str | None = None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """
        The main public method to get a file, using caching, recovery, or uploading.

        This method uses a fast content hash (xxHash) as the primary key for all
        caching and remote API interactions to ensure deduplication and performance.
        It is safe from race conditions during concurrent uploads.

        Args:
            file_bytes: The raw byte content of the file. Required.
            mime_type: The MIME type of the file (e.g., 'image/png'). Required.
            owui_file_id: The unique ID of the file from Open WebUI, if available.
                      RECOMMENDED_COMPANION_VERSION    Used for logging and as a key for the hash cache optimization.
            status_queue: An optional asyncio.Queue to report upload lifecycle events.

        Returns:
            An `ACTIVE` `google.genai.types.File` object.

        Raises:
            FilesAPIError: If the file fails to upload or process.
        """
        # Step 1: Get the fast content hash, using the ID cache as an optimization if possible.
        content_hash = await self._get_content_hash(file_bytes, owui_file_id)

        # Step 2: The Hot Path (Check Local File Cache)
        # A cache hit means the file is valid and we can return immediately.
        file_cache_key = self._get_file_cache_key(content_hash)
        cached_file: types.File | None = await self.file_cache.get(file_cache_key)
        if cached_file:
            log_id = f"OWUI ID: {owui_file_id}" if owui_file_id else "anonymous file"
            log.debug(
                f"Cache HIT for file hash {content_hash} ({log_id}). Returning immediately."
            )
            return cached_file

        # On cache miss, acquire a lock specific to this file's content to prevent race conditions.
        # dict.setdefault is atomic, ensuring only one lock is created per hash.
        lock_key = self._get_lock_key(content_hash)
        lock = self.upload_locks.setdefault(lock_key, asyncio.Lock())
        if lock.locked():
            log.debug(
                f"Lock for key {lock_key} is held by another task. "
                f"This call will now wait for the lock to be released."
            )

        async with lock:
            # Step 2.5: Double-Checked Locking
            # After acquiring the lock, check the cache again. Another task might have
            # completed the upload while we were waiting for the lock.
            cached_file = await self.file_cache.get(file_cache_key)
            if cached_file:
                log.debug(
                    f"Cache HIT for file hash {content_hash} after acquiring lock. Returning."
                )
                return cached_file

            # Step 3: The Warm/Cold Path (On Cache Miss)
            # The file ID (name after "files/") must be <= 40 chars.
            # "owui-" (5) + hash (16) + "-" (1) + hash (16) = 38 chars.
            deterministic_name = f"files/owui-{self.api_key_hash}-{content_hash}"
            log.debug(
                f"Cache MISS for hash {content_hash}. Attempting stateless recovery with GET: {deterministic_name}"
            )

            try:
                # Attempt to get the file (Warm Path)
                file = await self.client.aio.files.get(name=deterministic_name)
                if not file.name:
                    raise FilesAPIError(
                        f"Stateless recovery for {deterministic_name} returned a file without a name."
                    )

                log.debug(
                    f"Stateless recovery successful for {deterministic_name}. File exists on server."
                )
                active_file = await self._poll_for_active_state(file.name, owui_file_id)

                ttl_seconds = self._calculate_ttl(active_file.expiration_time)
                await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)

                return active_file
            except genai_errors.ClientError as e:
                # NOTE: The Gemini Files API returns 403 Forbidden when trying to GET
                # a file that either does not exist or belongs to another project.
                # We treat 403 as the "not found" signal for our warm path and
                # include 404 for forward compatibility.
                if e.code == 403 or e.code == 404:
                    log.info(
                        f"File {deterministic_name} not found on server (received {e.code}). Proceeding to upload."
                    )
                    # Proceed to upload (Cold Path)
                    return await self._upload_and_process_file(
                        content_hash,
                        file_bytes,
                        mime_type,
                        deterministic_name,
                        owui_file_id,
                        status_queue,
                    )
                else:
                    log.exception(
                        f"An unhandled client error (code: {e.code}) occurred during stateless recovery for {deterministic_name}."
                    )
                    self.event_emitter.emit_toast(
                        f"API error for file: {e.code}. Please check permissions.",
                        "error",
                    )
                    raise FilesAPIError(
                        f"Failed to check file status for {deterministic_name}: {e}"
                    ) from e
            except Exception as e:
                log.exception(
                    f"An unexpected error occurred during stateless recovery for {deterministic_name}."
                )
                self.event_emitter.emit_toast(
                    "Unexpected error retrieving a file. Please try again.",
                    "error",
                )
                raise FilesAPIError(
                    f"Failed to check file status for {deterministic_name}: {e}"
                ) from e
            finally:
                # Clean up the lock from the dictionary once processing is complete
                # for this hash, preventing memory growth over time.
                # This is safe because any future request for this hash will hit the cache.
                if lock_key in self.upload_locks:
                    del self.upload_locks[lock_key]

    async def _get_content_hash(
        self, file_bytes: bytes, owui_file_id: str | None
    ) -> str:
        """
        Retrieves the file's content hash, using a cache for known IDs or computing it.

        This acts as a memoization layer for the hashing process, avoiding
        re-computation for files with a known Open WebUI ID. For anonymous files
        (owui_file_id=None), it will always compute the hash.
        """
        if owui_file_id:
            # First, check the ID-to-Hash cache for known files.
            # This cache is NOT namespaced by API key, as the mapping from
            # an OWUI file ID to its content hash is constant.
            cached_hash: str | None = await self.id_hash_cache.get(owui_file_id)
            if cached_hash:
                log.trace(f"Hash cache HIT for OWUI ID {owui_file_id}.")
                return cached_hash

        # If not in cache or if file is anonymous, compute the fast hash.
        log.trace(
            f"Hash cache MISS for OWUI ID {owui_file_id if owui_file_id else 'N/A'}. Computing hash."
        )
        content_hash = xxhash.xxh64(file_bytes).hexdigest()

        # If there was an ID, store the newly computed hash for next time.
        if owui_file_id:
            await self.id_hash_cache.set(owui_file_id, content_hash)

        return content_hash

    def _calculate_ttl(self, expiration_time: datetime | None) -> float | None:
        """Calculates the TTL in seconds from an expiration datetime."""
        if not expiration_time:
            return None

        now_utc = datetime.now(timezone.utc)
        if expiration_time <= now_utc:
            return 0

        return (expiration_time - now_utc).total_seconds()

    async def _upload_and_process_file(
        self,
        content_hash: str,
        file_bytes: bytes,
        mime_type: str,
        deterministic_name: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue | None = None,
    ) -> types.File:
        """Handles the full upload and post-upload processing workflow."""

        # Register with the manager that an actual upload is starting.
        if status_queue:
            await status_queue.put(("REGISTER_UPLOAD",))

        log.info(f"Starting upload for {deterministic_name}...")

        try:
            file_io = io.BytesIO(file_bytes)
            upload_config = types.UploadFileConfig(
                name=deterministic_name, mime_type=mime_type
            )
            uploaded_file = await self.client.aio.files.upload(
                file=file_io, config=upload_config
            )
            if not uploaded_file.name:
                raise FilesAPIError(
                    f"File upload for {deterministic_name} did not return a file name."
                )

            log.debug(f"{uploaded_file.name} uploaded.")
            log.trace("Uploaded file details:", payload=uploaded_file)

            # Check if the file is already active. If so, we can skip polling.
            if uploaded_file.state == types.FileState.ACTIVE:
                log.debug(
                    f"File {uploaded_file.name} is already ACTIVE. Skipping poll."
                )
                active_file = uploaded_file
            else:
                # If not active, proceed with the original polling logic.
                log.debug(
                    f"{uploaded_file.name} uploaded with state {uploaded_file.state}. Polling for ACTIVE state."
                )
                active_file = await self._poll_for_active_state(
                    uploaded_file.name, owui_file_id
                )
                log.debug(f"File {active_file.name} is now ACTIVE.")

            # Calculate TTL and set in the main file cache using the content hash as the key.
            ttl_seconds = self._calculate_ttl(active_file.expiration_time)
            file_cache_key = self._get_file_cache_key(content_hash)
            await self.file_cache.set(file_cache_key, active_file, ttl=ttl_seconds)
            log.debug(
                f"Cached new file object for hash {content_hash} with TTL: {ttl_seconds}s."
            )

            return active_file
        except Exception as e:
            log.exception(f"File upload or processing failed for {deterministic_name}.")
            self.event_emitter.emit_toast(
                "Upload failed for a file. Please check connection and try again.",
                "error",
            )
            raise FilesAPIError(f"Upload failed for {deterministic_name}: {e}") from e
        finally:
            # Report completion (success or failure) to the status manager.
            # This ensures the progress counter always advances.
            if status_queue:
                await status_queue.put(("COMPLETE_UPLOAD",))

    async def _poll_for_active_state(
        self,
        file_name: str,
        owui_file_id: str | None,
        timeout: int = 60,
        poll_interval: int = 1,
    ) -> types.File:
        """Polls the file's status until it is ACTIVE or fails."""
        end_time = time.monotonic() + timeout
        while time.monotonic() < end_time:
            try:
                file = await self.client.aio.files.get(name=file_name)
            except Exception as e:
                raise FilesAPIError(
                    f"Polling failed: Could not get status for {file_name}. Reason: {e}"
                ) from e

            if file.state == types.FileState.ACTIVE:
                return file
            if file.state == types.FileState.FAILED:
                log_id = f"'{owui_file_id}'" if owui_file_id else "an uploaded file"
                error_message = f"File processing failed on server for {file_name}."
                toast_message = f"Google could not process {log_id}."
                if file.error:
                    reason = f"Reason: {file.error.message} (Code: {file.error.code})"
                    error_message += f" {reason}"
                    toast_message += f" Reason: {file.error.message}"

                self.event_emitter.emit_toast(toast_message, "error")
                raise FilesAPIError(error_message)

            state_name = file.state.name if file.state else "UNKNOWN"
            log.trace(
                f"File {file_name} is still {state_name}. Waiting {poll_interval}s..."
            )
            await asyncio.sleep(poll_interval)

        raise FilesAPIError(
            f"File {file_name} did not become ACTIVE within {timeout} seconds."
        )


class GeminiContentBuilder:
    """Builds a list of `google.genai.types.Content` objects from the OWUI's body payload."""

    def __init__(
        self,
        messages_body: list["Message"],
        metadata_body: "Metadata",
        user_data: "UserData",
        event_emitter: "EventEmitter",
        valves: "Pipe.Valves",
        files_api_manager: "FilesAPIManager",
    ):
        self.messages_body = messages_body
        self.upload_documents = (metadata_body.get("features", {}) or {}).get(
            "upload_documents", False
        )
        # Identify if this is a background task (title/tags/etc) to optimize context.
        self.is_task = bool(metadata_body.get("task"))
        self.event_emitter = event_emitter
        self.valves = valves
        self.files_api_manager = files_api_manager

        chat_id = metadata_body.get("chat_id")
        chat_id_str = chat_id if isinstance(chat_id, str) else ""
        self.is_temp_chat = (
            not chat_id_str or "local" in chat_id_str or "temporary" in chat_id_str
        )
        self.vertexai = self.files_api_manager.client.vertexai

        self.system_prompt, self.messages_body = self._extract_system_prompt(
            self.messages_body
        )
        self.metadata_body = metadata_body
        self.user_data = user_data
        self.messages_db = None

    async def build_contents(self) -> list[types.Content]:
        """
        The main public method to generate the contents list by processing all
        message turns concurrently and using a self-configuring status manager.
        """
        # Fetch chat history and cumulative usage from the DB (async APIs).
        self.messages_db = await self._fetch_and_validate_chat_history(
            self.metadata_body, self.user_data
        )
        log.trace("Database messages: ", payload=self.messages_db)
        # Retrieve cumulative usage from the DB history and inject it into metadata.
        # This will be picked up later when constructing the final usage payload.
        c_tokens, c_cost = self._retrieve_previous_usage_data()
        self.metadata_body["cumulative_tokens"] = c_tokens
        self.metadata_body["cumulative_cost"] = c_cost

        # 1. Set up and launch the status manager. It will activate itself if needed.
        status_manager = UploadStatusManager(self.event_emitter)
        manager_task = asyncio.create_task(status_manager.run())

        # 2. Create and run concurrent processing tasks for each message turn.
        tasks = [
            self._process_message_turn(i, message, status_manager.queue)
            for i, message in enumerate(self.messages_body)
        ]
        log.debug(f"Starting concurrent processing of {len(tasks)} message turns.")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Signal to the manager that no more uploads will be registered.
        await status_manager.queue.put(("FINALIZE",))

        # 4. Wait for the manager to finish processing all reported uploads.
        await manager_task

        # 5. Filter and assemble the final contents list.
        contents: list[types.Content] = []
        for i, res in enumerate(results):
            if isinstance(res, types.Content):
                contents.append(res)
            elif isinstance(res, Exception):
                log.error(
                    f"An error occurred while processing message {i} concurrently.",
                    payload=res,
                )
        return contents

    def _retrieve_previous_usage_data(self) -> tuple[int | None, float | None]:
        """
        Retrieves the cumulative token count and cost from the last assistant message in the database.

        Returns:
            - (0, 0.0) if it's the start of a conversation (no previous assistant message).
            - (tokens, cost) if the previous assistant message has valid cumulative data.
            - (None, None) if the chain is broken (previous message exists but lacks data)
              or if DB history is unavailable (e.g., temp chat).
        """
        if not self.messages_db:
            return None, None

        for msg in reversed(self.messages_db):
            if msg.get("role") == "assistant":
                usage = msg.get("usage", {})
                # These keys must be populated by the plugin in previous turns
                c_tokens = usage.get("cumulative_token_count")
                c_cost = usage.get("cumulative_total_cost")

                if c_tokens is not None and c_cost is not None:
                    return c_tokens, c_cost
                else:
                    # Previous assistant message exists but lacks cumulative data.
                    # This indicates a broken chain (old message or different plugin).
                    return None, None

        # No assistant message found in history, implying this is the first turn.
        return 0, 0.0

    @staticmethod
    def _extract_system_prompt(
        messages: list["Message"],
    ) -> tuple[str | None, list["Message"]]:
        """Extracts the system prompt and returns it along with the modified message list."""
        system_message, remaining_messages = pop_system_message(messages)  # type: ignore
        system_prompt: str | None = (system_message or {}).get("content")
        return system_prompt, remaining_messages  # type: ignore

    async def _fetch_and_validate_chat_history(
        self, metadata_body: "Metadata", user_data: "UserData"
    ) -> list["ChatMessageTD"] | None:
        """
        Reconstructs the active chat branch from history. Removes the trailing
        assistant placeholder and strictly validates that the DB history length
        matches the request body.

        Every fallback to the active memory payload is logged and surfaced to the
        user via a toast with a reason-specific message.
        """
        if self.is_temp_chat:
            # Expected path, not a failure: temp chats have no DB history by design.
            _log_and_toast(
                self.event_emitter,
                "Temporary chat detected; skipping database history fetch and using the active memory payload.",
                "info",
            )
            return None

        chat_id = metadata_body.get("chat_id")
        if not chat_id:
            _log_and_toast(
                self.event_emitter,
                "No chat_id in metadata; cannot fetch database history. Using the active memory payload.",
                "warning",
            )
            return None

        chat = await Chats.get_chat_by_id_and_user_id(
            id=chat_id, user_id=user_data["id"]
        )
        if not chat:
            _log_and_toast(
                self.event_emitter,
                f"Chat {chat_id} was not found in the database; using the active memory payload.",
            )
            return None

        chat_content: "ChatObjectDataTD" = chat.chat  # type: ignore
        history_data = chat_content.get("history", {})
        messages_dict = history_data.get("messages", {})
        current_id = history_data.get("currentId")

        if not messages_dict or not current_id:
            _log_and_toast(
                self.event_emitter,
                "Chat history is empty or lacks a currentId; using the active memory payload.",
                "warning",
            )
            return None

        # 1. Walk up the parentId chain to reconstruct the linear conversation branch.
        messages_db: list["ChatMessageTD"] = []
        curr_id = current_id

        while curr_id and curr_id in messages_dict:
            msg = messages_dict[curr_id]
            messages_db.insert(0, msg)
            curr_id = msg.get("parentId")

        # 2. Handle the trailing assistant placeholder.
        # OWUI often inserts an empty assistant message entry for the turn currently
        # being processed. We remove it to align with the 'messages_body' which
        # only contains previous turns plus the current user message.
        if messages_db and messages_db[-1].get("role") == "assistant":
            messages_db.pop()

        # 3. Strict validation.
        # If the reconstructed history (minus placeholder) doesn't exactly match the
        # length of the request body (minus system prompt), we bail out.
        # This prevents misaligned metadata mapping.
        if len(messages_db) != len(self.messages_body):
            _log_and_toast(
                self.event_emitter,
                f"Strict length mismatch: DB={len(messages_db)}, Body={len(self.messages_body)}. "
                "Using the active memory payload.",
                "warning",
            )
            return None

        return messages_db

    async def _process_message_turn(
        self, i: int, message: "Message", status_queue: asyncio.Queue
    ) -> types.Content | None:
        """
        Processes a single message turn, handling user and assistant roles,
        and returns a complete `types.Content` object. Designed to be run concurrently.
        """
        role = message.get("role")
        parts: list[types.Part] = []

        if role == "user":
            message = cast("UserMessage", message)
            # Logic for retrieving files is now handled inside _process_user_message
            # to allow for finer control over which file types are included based on task mode.
            parts = await self._process_user_message(i, message, status_queue)
            # Case 1: User content is completely empty (no text, no files).
            if not parts:
                _log_and_toast(
                    self.event_emitter,
                    f"Your message #{i + 1} was completely empty. The assistant will ask for clarification.",
                    "warning",
                )
                clarification_prompt = (
                    "The user sent an empty message. Please ask the user for "
                    "clarification on what they would like to ask or discuss."
                )
                # This will become the only part for this user message.
                parts = await self._genai_parts_from_text(
                    clarification_prompt, status_queue
                )
            else:
                # Case 2: User has sent content, check if it includes text.
                has_text_component = any(p.text for p in parts if p.text)
                if not has_text_component:
                    # The user sent content (e.g., files) but no accompanying text.
                    if self.vertexai:
                        # Vertex AI requires a text part in multi-modal messages.
                        _log_and_toast(
                            self.event_emitter,
                            f"For your message #{i + 1}, a default prompt was added as text is required "
                            "for requests with attachments when using Vertex AI.",
                            "warning",
                        )
                        default_prompt_text = (
                            "The user did not send any text message with the additional context. "
                            "Answer by summarizing the newly added context."
                        )
                        default_text_parts = await self._genai_parts_from_text(
                            default_prompt_text, status_queue
                        )
                        parts.extend(default_text_parts)
                    else:
                        # Google Developer API allows no-text user content.
                        log.debug(
                            f"User message at index {i} lacks a text component for Google Developer API. "
                            "Proceeding with non-text parts only."
                        )
        elif role == "assistant":
            message = cast("AssistantMessage", message)
            # Google API's assistant role is "model"
            role = "model"
            message_db = self.messages_db[i] if self.messages_db else None
            sources = message_db.get("sources") if message_db else None
            parts = await self._process_assistant_message(
                i, message, message_db, sources, status_queue
            )
        else:
            _log_and_toast(
                self.event_emitter,
                f"Message {i} has an invalid role: {role}. Skipping to the next message.",
                "warning",
            )
            return None

        # Only create a Content object if there are parts to include.
        if parts:
            return types.Content(parts=parts, role=role)
        return None

    async def _process_user_message(
        self,
        i: int,
        message: "UserMessage",
        status_queue: asyncio.Queue,
    ) -> list[types.Part]:
        user_parts: list[types.Part] = []
        db_files_processed = False

        # PATH 1: Database is available (Normal Chat).
        if self.messages_db:
            message_db = self.messages_db[i]
            files: list["FileAttachmentTD"] = message_db.get("files", [])

            if files:
                db_files_processed = True
                upload_tasks = []

                for file in files:
                    content_type = file.get("content_type", "")
                    # MIME types for images always start with 'image/' (e.g., image/png, image/jpeg)
                    is_image = content_type.startswith("image/")

                    # Optimization: Task models (titles, tags, etc.) skip heavy documents
                    # but keep images as they provide high context value for low token cost.
                    should_include = is_image or (
                        self.upload_documents and not self.is_task
                    )

                    if not should_include:
                        log.debug(
                            f"Skipping {content_type} '{file.get('id')}' "
                            f"(is_task={self.is_task}, upload_documents={self.upload_documents})"
                        )
                        continue

                    # We always use the internal API endpoint to fetch file content from the DB.
                    # Even for images, the 'url' field in the attachment object is often
                    # just the UUID, which isn't fetchable on its own.
                    if file_id := file.get("id"):
                        uri = f"/api/v1/files/{file_id}/content"
                        upload_tasks.append(
                            self._genai_part_from_uri(uri, status_queue)
                        )
                    else:
                        _log_and_toast(
                            self.event_emitter,
                            f"Encountered a malformed file object in message #{i + 1} "
                            "without an ID; it will not be injected into the model's context.",
                            "warning",
                        )

                if upload_tasks:
                    log.info(f"Processing {len(upload_tasks)} file(s) from database.")
                    results = await asyncio.gather(*upload_tasks)
                    user_parts.extend(part for part in results if part)

        # Now, process the content from the message payload.
        user_content = message.get("content")
        if isinstance(user_content, str):
            user_content_list: list["Content"] = [
                {"type": "text", "text": user_content}
            ]
        elif isinstance(user_content, list):
            user_content_list = user_content
        else:
            _log_and_toast(
                self.event_emitter,
                f"Message #{i + 1} has invalid content (not a string or list); "
                "skipping the malformed content.",
                "warning",
            )
            return user_parts

        for c in user_content_list:
            c_type = c.get("type")
            if c_type == "text":
                c = cast("TextContent", c)
                if c_text := c.get("text"):
                    user_parts.extend(
                        await self._genai_parts_from_text(c_text, status_queue)
                    )

            # PATH 2: Temporary Chat Image Handling.
            # FIXME: this puts images to the end of the message, see if it matters where they are.
            elif c_type == "image_url" and not db_files_processed:
                log.info("Processing image from payload (temporary chat mode).")
                c = cast("ImageContent", c)
                if uri := c.get("image_url", {}).get("url"):
                    if part := await self._genai_part_from_uri(uri, status_queue):
                        user_parts.append(part)

        return user_parts

    async def _rehydrate_assistant_parts(
        self,
        gemini_parts: list[dict[str, Any]],
        status_queue: asyncio.Queue,
    ) -> list[types.Part]:
        """
        Reconstructs `types.Part` objects from dictionaries, rehydrating file-based parts
        by fetching their content from the OWUI backend.
        """
        rehydrated_parts: list[types.Part] = []
        for part_dict in gemini_parts:
            part = types.Part.model_validate(part_dict)

            if part.file_data and (file_uri := part.file_data.file_uri):
                if not file_uri.startswith("/api/v1/files/"):
                    raise ValueError(
                        f"Unsupported file_uri in assistant history: {file_uri}. "
                        "Only local Open WebUI files are supported for reconstruction."
                    )

                file_id = file_uri.split("/")[4]
                file_bytes, mime_type = await self._get_file_data(file_id)

                if not (file_bytes and mime_type):
                    raise ValueError(
                        f"Could not retrieve content for file_id '{file_id}' from assistant history."
                    )

                # Force raw bytes (inline_data) to preserve exact history context for the model.
                # This ensures we don't convert original inline images into Files API references.
                rehydrated_part = await self._create_genai_part_from_file_data(
                    file_bytes, mime_type, file_id, status_queue, force_raw=True
                )
                part.inline_data = rehydrated_part.inline_data
                part.file_data = rehydrated_part.file_data
                rehydrated_parts.append(part)
            else:
                rehydrated_parts.append(part)

        return rehydrated_parts

    def _pop_thoughts(self, content: str) -> tuple[str, list[str]]:
        """
        Identifies and removes thought blocks from the content.

        A thought is defined as text between <think> and </think>\n.
        This method handles multiple thought blocks if they are peppered
        throughout the message.

        :param content: The raw message content from the assistant.
        :return: A tuple containing (cleaned_content, list_of_extracted_thoughts).
        """
        # The pattern looks for the <​think> tag, captures everything inside (non-greedy),
        # and matches the <​/think> tag plus a potential trailing newline.
        # re.DOTALL allows the '.' character to match newlines within the capture group.
        thought_pattern = re.compile(r"<think>(.*?)</think>\n?", re.DOTALL)

        thoughts = thought_pattern.findall(content)
        # Replace all occurrences with an empty string to get the "clean" content.
        cleaned_content = thought_pattern.sub("", content)

        return cleaned_content, thoughts

    async def _process_assistant_message(
        self,
        i: int,
        message_body: "AssistantMessage",
        message_db: "ChatMessageTD | None",
        sources: list["Source"] | None,
        status_queue: asyncio.Queue,
    ) -> list[types.Part]:
        """
        Processes an assistant message, prioritizing reconstruction from stored 'gemini_parts'
        if available and unmodified. Falls back to processing the text content if parts
        are missing or if the user has edited the message.
        """
        gemini_parts = message_db.get("gemini_parts") if message_db else None
        original_content = message_db.get("original_content") if message_db else None
        current_content = message_body.get("content", "")

        # 1. Pop thoughts out before any comparison or citation stripping.
        # We store 'thoughts' for future use (e.g., adding to a metadata field).
        current_content, thoughts = self._pop_thoughts(current_content)

        # 2. Strip citations as before.
        if sources:
            current_content = self._remove_citation_markers(current_content, sources)

        # --- PATH 1: Restore from stored parts (ideal case) ---
        if gemini_parts and original_content is not None:
            # Now current_content has no thoughts and no citations,
            # making it directly comparable to original_content.
            if current_content.strip() == original_content.strip():
                log.debug(
                    f"Reconstructing assistant message at index {i} from stored parts."
                )
                try:
                    return await self._rehydrate_assistant_parts(
                        gemini_parts, status_queue
                    )
                except (pydantic_core.ValidationError, TypeError, ValueError) as exc:
                    _log_and_toast(
                        self.event_emitter,
                        f"Failed to reconstruct assistant message #{i + 1} from stored data: {exc}. "
                        "Falling back to plain text processing.",
                        "warning",
                    )
            else:
                # A meaningful edit was detected after accounting for whitespace.
                _log_and_toast(
                    self.event_emitter,
                    f"An edit was detected in assistant message #{i + 1}. "
                    "Using the edited text, which may affect model context for this turn.",
                    "warning",
                )

                diff = difflib.unified_diff(
                    original_content.strip().splitlines(keepends=True),
                    current_content.strip().splitlines(keepends=True),
                    fromfile="original_content_stripped",
                    tofile="current_content_stripped",
                )
                diff_str = "".join(diff)
                log.warning(
                    f"Edited content diff for assistant message {i}:\n{diff_str}"
                )
        elif message_db:
            _log_and_toast(
                self.event_emitter,
                f"Assistant message #{i + 1} is missing stored high-fidelity data from the database. "
                "Falling back to plain text reconstruction, which may affect model context for this turn.",
                "warning",
            )

        # --- PATH 2: Fallback to processing text content ---
        # This path is used for non-Gemini messages, edited messages, or on reconstruction failure.
        log.debug(f"Processing assistant message {i} content as plain text.")
        return await self._genai_parts_from_text(current_content, status_queue)

    async def _genai_parts_from_text(
        self, text: str, status_queue: asyncio.Queue
    ) -> list[types.Part]:
        if not text:
            return []

        text = self._enable_special_tags(text)
        parts: list[types.Part] = []
        last_pos = 0

        # Conditionally build a regex to find media links.
        # If YouTube parsing is disabled, the regex will only find markdown image links,
        # leaving YouTube URLs to be treated as plain text.
        markdown_part = r"!\[.*?\]\(([^)]+)\)"  # Group 1: Markdown URI
        youtube_part = r"(https?://(?:(?:www|music)\.)?youtube\.com/(?:watch\?v=|shorts/|live/)[^\s)]+|https?://youtu\.be/[^\s)]+)"  # Group 2: YouTube URL
        if self.valves.PARSE_YOUTUBE_URLS:
            pattern = re.compile(f"{markdown_part}|{youtube_part}")
            process_youtube = True
        else:
            pattern = re.compile(markdown_part)
            process_youtube = False
            log.info(
                "YouTube URL parsing is disabled. URLs will be treated as plain text."
            )

        for match in pattern.finditer(text):
            # Add the text segment that precedes the media link
            if text_segment := text[last_pos : match.start()].strip():
                parts.append(types.Part.from_text(text=text_segment))

            # The URI is in group 1 for markdown, or group 2 for YouTube.
            if process_youtube:
                uri = match.group(1) or match.group(2)
            else:
                uri = match.group(1)

            if not uri:
                _log_and_toast(
                    self.event_emitter,
                    f"Failed to extract URI from match: {match.group(0)}. Skipping.",
                    "warning",
                )
                continue

            # Delegate all URI processing to the unified helper
            if media_part := await self._genai_part_from_uri(uri, status_queue):
                parts.append(media_part)

            last_pos = match.end()

        # Add any remaining text after the last media link
        if remaining_text := text[last_pos:].strip():
            parts.append(types.Part.from_text(text=remaining_text))

        # If no media links were found, the whole text is a single part
        if not parts and text.strip():
            parts.append(types.Part.from_text(text=text.strip()))

        return parts

    async def _genai_part_from_uri(
        self, uri: str, status_queue: asyncio.Queue
    ) -> types.Part | None:
        """
        Processes any resource URI and returns a genai.types.Part.
        This is the central dispatcher for all media processing, handling data URIs,
        local API file paths, and YouTube URLs.
        """
        if not uri:
            log.warning("Received an empty URI, skipping.")
            return None

        try:
            file_bytes: bytes | None = None
            mime_type: str | None = None
            owui_file_id: str | None = None

            # Step 1: Extract bytes and mime_type from the URI if applicable
            if uri.startswith("data:image"):
                match = re.match(r"data:(image/\w+);base64,(.+)", uri)
                if not match:
                    raise ValueError("Invalid data URI for image.")
                mime_type, base64_data = match.group(1), match.group(2)
                file_bytes = base64.b64decode(base64_data)
            elif uri.startswith("/api/v1/files/"):
                log.info(f"Processing local API file URI: {uri}")
                file_id = uri.split("/")[4]
                owui_file_id = file_id
                file_bytes, mime_type = await self._get_file_data(file_id)
            elif "youtube.com/" in uri or "youtu.be/" in uri:
                log.info(f"Found YouTube URL: {uri}")
                return self._genai_part_from_youtube_uri(uri)
            # TODO: Google Cloud Storage bucket support.
            # elif uri.startswith("gs://"): ...
            else:
                _log_and_toast(
                    self.event_emitter,
                    f"Unsupported URI: '{uri[:64]}...' Links must be to YouTube or a supported file type.",
                    "warning",
                )
                return None

            # Step 2: If we have bytes, create the Part using the modularized helper
            if file_bytes and mime_type:
                return await self._create_genai_part_from_file_data(
                    file_bytes=file_bytes,
                    mime_type=mime_type,
                    owui_file_id=owui_file_id,
                    status_queue=status_queue,
                )

            return None  # Return None if bytes/mime_type could not be determined

        except FilesAPIError as e:
            _log_and_toast(
                self.event_emitter,
                f"Files API failed for URI '{uri[:64]}[...]': {e}",
                "error",
            )
            return None
        except Exception as e:
            _log_and_toast(
                self.event_emitter,
                f"Error processing URI: {uri[:64]}[...]: {e}",
                "error",
            )
            return None

    async def _create_genai_part_from_file_data(
        self,
        file_bytes: bytes,
        mime_type: str,
        owui_file_id: str | None,
        status_queue: asyncio.Queue,
        force_raw: bool = False,
    ) -> types.Part:
        """
        Creates a `types.Part` from file bytes, deciding whether to use the
        Google Files API or send raw bytes based on configuration and context.
        """
        # TODO: The Files API is strict about MIME types (e.g., text/plain,
        # application/pdf). In the future, inspect the content of files
        # with unsupported text-like MIME types (e.g., 'application/json',
        # 'text/markdown'). If the content is detected as plaintext,
        # override the `mime_type` variable to 'text/plain' to allow the upload.

        # Determine whether to use the Files API based on the specified conditions.
        use_files_api = True
        reason = ""

        if force_raw:
            reason = "raw bytes are forced (e.g. for assistant history reconstruction)"
            use_files_api = False
        elif not self.valves.USE_FILES_API:
            reason = "disabled by user setting (USE_FILES_API=False)"
            use_files_api = False
        elif self.vertexai:
            reason = "the active client is configured for Vertex AI, which does not support the Files API"
            use_files_api = False
        elif self.is_temp_chat:
            reason = "temporary chat mode is active"
            use_files_api = False

        if use_files_api:
            log.info("Using Google Files API for resource.")
            gemini_file = await self.files_api_manager.get_or_upload_file(
                file_bytes=file_bytes,
                mime_type=mime_type,
                owui_file_id=owui_file_id,
                status_queue=status_queue,
            )
            return types.Part(
                file_data=types.FileData(
                    file_uri=gemini_file.uri,
                    mime_type=gemini_file.mime_type,
                )
            )
        else:
            log.info(f"Sending raw bytes because {reason}.")
            return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    def _genai_part_from_youtube_uri(self, uri: str) -> types.Part | None:
        """Creates a Gemini Part from a YouTube URL, with optional video metadata.

        Handles standard (`watch?v=`), short (`youtu.be/`), mobile (`shorts/`),
        and live (`live/`) URLs. Metadata is parsed for the Gemini Developer API
        but ignored for Vertex AI, which receives a simple URI Part.

        - **Start/End Time**: `?t=<value>` and `#end=<value>`. The value can be a
          flexible duration (e.g., "1m30s", "90") and will be converted to seconds.
        - **Frame Rate**: Can be specified in two ways (if both are present,
          `interval` takes precedence):
          - **Interval**: `#interval=<value>` (e.g., `#interval=10s`, `#interval=0.5s`).
            The value is a flexible duration converted to seconds, then to FPS (1/interval).
          - **FPS**: `#fps=<value>` (e.g., `#fps=2.5`).
          The final FPS value must be in the range (0, 24].

        Args:
            uri: The raw YouTube URL from the user.
            is_vertex_client: If True, creates a simple Part for Vertex AI.

        Returns:
            A `types.Part` object, or `None` if the URI is not a valid YouTube link.
        """
        # Convert YouTube Music URLs to standard YouTube URLs for consistent parsing.
        if "music.youtube.com" in uri:
            uri = uri.replace("music.youtube.com", "www.youtube.com")
            log.info(f"Converted YouTube Music URL to standard URL: {uri}")

        # Regex to capture the 11-character video ID from various YouTube URL formats.
        video_id_pattern = re.compile(
            r"(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/|live/)|youtu.be/)([a-zA-Z0-9_-]{11})"
        )

        match = video_id_pattern.search(uri)
        if not match:
            log.warning(f"Could not extract a valid YouTube video ID from URI: {uri}")
            return None

        video_id = match.group(1)
        canonical_uri = f"https://www.youtube.com/watch?v={video_id}"

        # --- Branching logic for Vertex AI vs. Gemini Developer API ---
        if self.vertexai:
            return types.Part.from_uri(file_uri=canonical_uri, mime_type="video/mp4")
        else:
            parsed_uri = urlparse(uri)
            query_params = parse_qs(parsed_uri.query)
            fragment_params = parse_qs(parsed_uri.fragment)

            start_offset: str | None = None
            end_offset: str | None = None
            fps: float | None = None

            # Start time from query `t`. Convert flexible format to "Ns".
            if "t" in query_params:
                raw_start = query_params["t"][0]
                if (
                    total_seconds := self._parse_duration_to_seconds(raw_start)
                ) is not None:
                    start_offset = f"{total_seconds}s"

            # End time from fragment `end`. Convert flexible format to "Ns".
            if "end" in fragment_params:
                raw_end = fragment_params["end"][0]
                if (
                    total_seconds := self._parse_duration_to_seconds(raw_end)
                ) is not None:
                    end_offset = f"{total_seconds}s"

            # Frame rate from fragment `interval` or `fps`. `interval` takes precedence.
            if "interval" in fragment_params:
                raw_interval = fragment_params["interval"][0]
                if (
                    interval_seconds := self._parse_duration_to_seconds(raw_interval)
                ) is not None and interval_seconds > 0:
                    calculated_fps = 1.0 / interval_seconds
                    if 0.0 < calculated_fps <= 24.0:
                        fps = calculated_fps
                    else:
                        log.warning(
                            f"Interval '{raw_interval}' results in FPS '{calculated_fps}' which is outside the valid range (0.0, 24.0]. Ignoring."
                        )

            # Fall back to `fps` param if not set by `interval`.
            if fps is None and "fps" in fragment_params:
                try:
                    fps_val = float(fragment_params["fps"][0])
                    if 0.0 < fps_val <= 24.0:
                        fps = fps_val
                    else:
                        log.warning(
                            f"FPS value '{fps_val}' is outside the valid range (0.0, 24.0]. Ignoring."
                        )
                except (ValueError, IndexError):
                    log.warning(
                        f"Invalid FPS value in fragment: {fragment_params.get('fps')}. Ignoring."
                    )

            video_metadata: types.VideoMetadata | None = None
            if start_offset or end_offset or fps is not None:
                video_metadata = types.VideoMetadata(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    fps=fps,
                )

            return types.Part(
                file_data=types.FileData(file_uri=canonical_uri),
                video_metadata=video_metadata,
            )

    def _parse_duration_to_seconds(self, duration_str: str) -> float | None:
        """Converts a human-readable duration string to total seconds.

        Supports formats like "1h30m15s", "90m", "3600s", or just "90".
        Also supports float values like "0.5s" or "90.5".
        Returns total seconds as a float, or None if the string is invalid.
        """
        # First, try to convert the whole string as a plain number (e.g., "90", "90.5").
        try:
            return float(duration_str)
        except ValueError:
            # If it fails, it might be a composite duration like "1m30s", so we parse it below.
            pass

        total_seconds = 0.0
        # Regex to find number-unit pairs (e.g., 1h, 30.5m, 15s). Supports floats.
        parts = re.findall(r"(\d+(?:\.\d+)?)\s*(h|m|s)?", duration_str, re.IGNORECASE)

        if not parts:
            # log.warning(f"Could not parse duration string: {duration_str}")
            return None

        for value, unit in parts:
            val = float(value)
            unit = (unit or "s").lower()  # Default to seconds if no unit
            if unit == "h":
                total_seconds += val * 3600
            elif unit == "m":
                total_seconds += val * 60
            elif unit == "s":
                total_seconds += val

        return total_seconds

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

    @staticmethod
    async def _get_file_data(file_id: str) -> tuple[bytes | None, str | None]:
        """
        Asynchronously retrieves file metadata from the database and its content from disk.
        """
        # TODO: Emit toasts on unexpected conditions.
        if not file_id:
            log.warning("file_id is empty. Cannot continue.")
            return None, None

        # Await the async database call directly.
        try:
            file_model = await Files.get_file_by_id(file_id)
        except Exception as e:
            log.exception(
                f"An unexpected error occurred during database call for file_id {file_id}: {e}"
            )
            return None, None

        if file_model is None:
            # The get_file_by_id method already handles and logs the specific exception,
            # so we just need to handle the None return value.
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

        if file_path.startswith("gs://"):
            try:
                # Initialize the GCS client
                storage_client = storage.Client()

                # Parse the GCS path
                # The path should be in the format "gs://bucket-name/object-name"
                if len(file_path.split("/", 3)) < 4:
                    raise ValueError(
                        f"Invalid GCS path: '{file_path}'. "
                        "Path must be in the format 'gs://bucket-name/object-name'."
                    )

                bucket_name, blob_name = file_path.removeprefix("gs://").split("/", 1)

                # Get the bucket and blob (file object)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                # Download the file's content as bytes
                print(f"Reading from GCS: {file_path}")
                return blob.download_as_bytes(), content_type
            except exceptions.NotFound:
                print(f"Error: GCS object not found at {file_path}")
                raise
            except Exception as e:
                print(f"An error occurred while reading from GCS: {e}")
                raise
        try:
            async with aiofiles.open(file_path, "rb") as file:
                file_data = await file.read()
            return file_data, content_type
        except FileNotFoundError:
            log.exception(f"File {file_path} not found on disk.")
            return None, content_type
        except Exception:
            log.exception(f"Error processing file {file_path}")
            return None, content_type

    @staticmethod
    def _remove_citation_markers(text: str, sources: list["Source"]) -> str:
        # FIXME: this should be moved to `Filter.inlet`
        # FIXME: `text` still contains ZWS here, they need to be removed.
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


_SHARED_VALVE_DESCS = {
    "GEMINI_FREE_API_KEY": "Free Gemini Developer API key.",
    "GEMINI_PAID_API_KEY": "Paid Gemini Developer API key.",
    "GEMINI_API_BASE_URL": "Custom base URL for calling the Gemini API.",
    "USE_VERTEX_AI": (
        "Whether to use Google Cloud Vertex AI instead of the standard Gemini API.\n\n"
        "*Requires `VERTEX_PROJECT` to be set.*"
    ),
    "VERTEX_PROJECT": "Google Cloud project ID to use with Vertex AI.",
    "VERTEX_LOCATION": "Google Cloud region/location for Vertex AI (e.g., `global`, `us-central1`).",
    "ENABLE_FREE_TIER_FALLBACK": (
        "Automatically switch to the Paid API if a Free API request fails due to quota limits (`429`) or model overload (`503`).\n\n"
        "*Requires both Free and Paid API keys to be configured.*"
    ),
    "TASK_MODEL_ROUTING": (
        "Determines API routing strategy for task models (e.g. title generation):\n"
        "- `only_free`: Use only the Free API.\n"
        "- `free_fallback`: Try Free API first, fallback to Paid on failure.\n"
        "- `only_paid`: Bypass Free API and use Paid API directly (or Vertex AI if enabled).\n"
        "- `match_main`: Follow the same routing logic as the main chat generation."
    ),
    "THINKING_CONFIG_RULES": (
        "JSON string mapping model name regex patterns to default thinking budget (int) or thinking level (str: `MINIMAL`, `LOW`, `MEDIUM`, `HIGH`).\n\n"
        "Determines thinking parameters per model. **Note that key order matters:** the first matching regex pattern is used, so place more specific patterns before broader catch-all patterns.\n\n"
        "For details on backend defaults and model support, see [Google Cloud Thinking Docs](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking#budget)."
    ),
    "SHOW_THINKING_SUMMARY": "Whether to display the thinking process summary in responses (Gemini 2.5 and 3 models).",
    "USE_FILES_API": (
        "Whether to use the Google Files API for uploading files (enables caching and performance benefits).\n\n"
        "If disabled, raw file bytes are sent directly in request payloads."
    ),
    "PARSE_YOUTUBE_URLS": (
        "Whether to parse YouTube video URLs from user messages and provide content as context.\n\n"
        "If disabled, YouTube links are treated as plain text."
    ),
    "MAPS_GROUNDING_COORDINATES": (
        "Latitude and longitude coordinates for location-aware Google Maps grounding.\n\n"
        "Expected format: `latitude,longitude` (e.g., `40.7128,-74.0060`)."
    ),
    "IMAGE_RESOLUTION": "Output resolution for generated images (Gemini 3 Pro Image only).",
    "IMAGE_ASPECT_RATIO": "Aspect ratio for image generation (Gemini 3 Pro Image & 2.5 Flash Image).",
}

_ADMIN_VALVE_DESCS = {
    "USER_MUST_PROVIDE_AUTH_CONFIG": (
        "Require all users (including admins) to provide their own authentication credentials via `UserValves`.\n\n"
        "Setting this to `True` prevents non-whitelisted users from using Vertex AI."
    ),
    "AUTH_WHITELIST": (
        "Comma-separated list of user email addresses allowed to bypass `USER_MUST_PROVIDE_AUTH_CONFIG` and use default system credentials."
    ),
    "MODEL_WHITELIST": (
        "Comma-separated list of allowed model names.\n\n"
        "Supports `fnmatch` wildcard patterns: `*`, `?`, `[seq]`, `[!seq]`."
    ),
    "MODEL_BLACKLIST": (
        "Comma-separated list of blacklisted model names.\n\n"
        "Supports `fnmatch` wildcard patterns: `*`, `?`, `[seq]`, `[!seq]`."
    ),
    "CACHE_MODELS": "Whether to cache available models on startup and refresh only when whitelist or blacklist rules change.",
    "USE_ENTERPRISE_SEARCH": "Enable Enterprise Search tool allowing models to fetch and ground content from specified web URLs.",
}


def _format_valve_desc(text: str, default: Any = None, is_user: bool = False) -> str:
    """Formats Markdown descriptions for Valves and UserValves fields."""
    text = text.strip()
    sep = "\n\n---\n\n"
    if is_user:
        return f"{text}\n\n*If not set, the admin's setting is used.*{sep}"
    formatted_default = f"`{default}`" if default is not None else "`None`"
    return f"{text}\n\n**Default:** {formatted_default}{sep}"


class Pipe:

    @staticmethod
    def _validate_coordinates_format(v: str | None) -> str | None:
        """Reusable validator for 'latitude,longitude' format."""
        if v is not None and v != "":
            try:
                parts = v.split(",")
                if len(parts) != 2:
                    raise ValueError(
                        "Must contain exactly two parts separated by a comma."
                    )

                lat_str, lon_str = parts
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())

                if not (-90 <= lat <= 90):
                    raise ValueError("Latitude must be between -90 and 90.")
                if not (-180 <= lon <= 180):
                    raise ValueError("Longitude must be between -180 and 180.")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid format for MAPS_GROUNDING_COORDINATES: '{v}'. "
                    f"Expected 'latitude,longitude' (e.g., '40.7128,-74.0060'). Original error: {e}"
                )
        return v

    @staticmethod
    def _validate_thinking_config_rules(v: str | None) -> str | None:
        """Validates that THINKING_CONFIG_RULES is a valid JSON string mapping regex patterns to int or str."""
        if v is not None and v.strip() != "":
            try:
                data = json.loads(v)
                if not isinstance(data, dict):
                    raise ValueError("Must be a JSON object (dict).")
                for key, val in data.items():
                    re.compile(key)
                    if not isinstance(val, (int, str)):
                        raise ValueError(
                            f"Value for pattern '{key}' must be an int or str, got {type(val).__name__}."
                        )
            except Exception as e:
                raise ValueError(
                    f"Invalid JSON format or structure for THINKING_CONFIG_RULES: {e}"
                )
        return v

    class Valves(BaseModel):
        GEMINI_FREE_API_KEY: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["GEMINI_FREE_API_KEY"], default=None
            ),
        )
        GEMINI_PAID_API_KEY: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["GEMINI_PAID_API_KEY"], default=None
            ),
        )
        USER_MUST_PROVIDE_AUTH_CONFIG: bool = Field(
            default=False,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["USER_MUST_PROVIDE_AUTH_CONFIG"], default=False
            ),
        )
        AUTH_WHITELIST: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["AUTH_WHITELIST"], default=None
            ),
        )
        GEMINI_API_BASE_URL: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["GEMINI_API_BASE_URL"], default=None
            ),
        )
        USE_VERTEX_AI: bool = Field(
            default=False,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["USE_VERTEX_AI"], default=False
            ),
        )
        VERTEX_PROJECT: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["VERTEX_PROJECT"], default=None
            ),
        )
        VERTEX_LOCATION: str = Field(
            default="global",
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["VERTEX_LOCATION"], default="global"
            ),
        )
        ENABLE_FREE_TIER_FALLBACK: bool = Field(
            default=False,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["ENABLE_FREE_TIER_FALLBACK"], default=False
            ),
        )
        TASK_MODEL_ROUTING: Literal[
            "only_free",
            "free_fallback",
            "only_paid",
            "match_main",
        ] = Field(
            default="match_main",
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["TASK_MODEL_ROUTING"], default="match_main"
            ),
        )
        MODEL_WHITELIST: str = Field(
            default="*",
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["MODEL_WHITELIST"], default="*"
            ),
        )
        MODEL_BLACKLIST: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["MODEL_BLACKLIST"], default=None
            ),
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["CACHE_MODELS"], default=True
            ),
        )
        THINKING_CONFIG_RULES: str = Field(
            default='{"$a": 0}',
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["THINKING_CONFIG_RULES"], default='{"$a": 0}'
            ),
        )
        SHOW_THINKING_SUMMARY: bool = Field(
            default=True,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["SHOW_THINKING_SUMMARY"], default=True
            ),
        )
        USE_FILES_API: bool = Field(
            default=True,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["USE_FILES_API"], default=True
            ),
        )
        PARSE_YOUTUBE_URLS: bool = Field(
            default=True,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["PARSE_YOUTUBE_URLS"], default=True
            ),
        )
        USE_ENTERPRISE_SEARCH: bool = Field(
            default=False,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["USE_ENTERPRISE_SEARCH"], default=False
            ),
        )
        MAPS_GROUNDING_COORDINATES: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["MAPS_GROUNDING_COORDINATES"], default=None
            ),
        )
        IMAGE_RESOLUTION: Literal["1K", "2K", "4K"] = Field(
            default="1K",
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["IMAGE_RESOLUTION"], default="1K"
            ),
        )
        IMAGE_ASPECT_RATIO: Literal[
            "1:1",
            "2:3",
            "3:2",
            "3:4",
            "4:3",
            "4:5",
            "5:4",
            "9:16",
            "16:9",
            "21:9",
        ] = Field(
            default="16:9",
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["IMAGE_ASPECT_RATIO"], default="16:9"
            ),
        )

        @field_validator("THINKING_CONFIG_RULES", mode="after")
        @classmethod
        def validate_thinking_config_rules(cls, v: str | None):
            return Pipe._validate_thinking_config_rules(v)

        @field_validator("MAPS_GROUNDING_COORDINATES", mode="after")
        @classmethod
        def validate_coordinates_format(cls, v: str | None):
            return Pipe._validate_coordinates_format(v)

    class UserValves(BaseModel):
        GEMINI_FREE_API_KEY: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["GEMINI_FREE_API_KEY"], is_user=True
            ),
        )
        GEMINI_PAID_API_KEY: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["GEMINI_PAID_API_KEY"], is_user=True
            ),
        )
        GEMINI_API_BASE_URL: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["GEMINI_API_BASE_URL"], is_user=True
            ),
        )
        USE_VERTEX_AI: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["USE_VERTEX_AI"], is_user=True
            ),
        )
        VERTEX_PROJECT: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["VERTEX_PROJECT"], is_user=True
            ),
        )
        VERTEX_LOCATION: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["VERTEX_LOCATION"], is_user=True
            ),
        )
        ENABLE_FREE_TIER_FALLBACK: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["ENABLE_FREE_TIER_FALLBACK"], is_user=True
            ),
        )
        TASK_MODEL_ROUTING: (
            Literal["only_free", "free_fallback", "only_paid", "match_main", ""] | None
        ) = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["TASK_MODEL_ROUTING"], is_user=True
            ),
        )
        THINKING_CONFIG_RULES: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["THINKING_CONFIG_RULES"], is_user=True
            ),
        )
        SHOW_THINKING_SUMMARY: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["SHOW_THINKING_SUMMARY"], is_user=True
            ),
        )
        USE_FILES_API: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["USE_FILES_API"], is_user=True
            ),
        )
        PARSE_YOUTUBE_URLS: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["PARSE_YOUTUBE_URLS"], is_user=True
            ),
        )
        MAPS_GROUNDING_COORDINATES: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["MAPS_GROUNDING_COORDINATES"], is_user=True
            ),
        )
        IMAGE_RESOLUTION: Literal["1K", "2K", "4K"] | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["IMAGE_RESOLUTION"], is_user=True
            ),
        )
        IMAGE_ASPECT_RATIO: (
            Literal[
                "1:1",
                "2:3",
                "3:2",
                "3:4",
                "4:3",
                "4:5",
                "5:4",
                "9:16",
                "16:9",
                "21:9",
            ]
            | None
        ) = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["IMAGE_ASPECT_RATIO"], is_user=True
            ),
        )

        @field_validator("THINKING_CONFIG_RULES", mode="after")
        @classmethod
        def validate_thinking_config_rules(cls, v: str | None):
            return Pipe._validate_thinking_config_rules(v)

        @field_validator("MAPS_GROUNDING_COORDINATES", mode="after")
        @classmethod
        def validate_coordinates_format(cls, v: str | None):
            return Pipe._validate_coordinates_format(v)

    def __init__(self):
        self.valves = self.Valves()
        self.file_content_cache = SimpleMemoryCache(serializer=NullSerializer())
        self.file_id_to_hash_cache = SimpleMemoryCache(serializer=NullSerializer())
        log.success("Function has been initialized.")

    async def pipes(self) -> list["ModelData"]:
        """Register all available Google models."""
        log.debug("pipes method has been called.")

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
        log.debug(f"Model list:", payload=[model["id"] for model in filtered_models])
        log.trace("List of dicts that Pipe.pipes() will return:", payload=filtered_models, _log_truncation_enabled=False)
        log.debug("pipes method has finished.")

        return filtered_models

    async def pipe(
        self,
        body: "Body",
        __user__: "UserData",
        __request__: Request,
        __metadata__: "Metadata",
    ) -> AsyncGenerator[dict | str, None] | dict:

        log.debug(
            f"pipe method has been called. Gemini Manifold google_genai version is {VERSION}"
        )
        log.trace("__metadata__:", payload=__metadata__)
        features = __metadata__.get("features", {}) or cast("Features", {})

        # Check the version of the companion filter
        self._check_companion_filter_version(features)

        # Retrieve model configuration from app state
        app_state: State = __request__.app.state
        model_config: dict[str, Any] = await self._wait_for_state_value(
            app_state,
            key="gemini_model_config",
            description="Gemini model configuration",
        )

        merged_custom_params = self._resolve_custom_params(body, __metadata__)
        __metadata__["merged_custom_params"] = merged_custom_params

        model_id = self._get_model_name(body)
        __metadata__["canonical_model_id"] = model_id

        # 1. Capture the raw state of keys before any overrides
        valves: Pipe.Valves = self._get_merged_valves(
            self.valves, __user__.get("valves"), __user__.get("email")
        )

        if task_type := __metadata__.get("task"):
            log.info(f"{task_type=}, disabling event emissions, YouTube URL parsing and document processing.")
            # We disable YouTube parsing for task models to minimize latency and token costs,
            # as simple tasks like title or tag generation do not require video context.
            valves.PARSE_YOUTUBE_URLS = False
            # TODO: disable tools. for now I assume that model will know to not use them even if enabled, but if that's not the case then this needs to be addressed.
            # TODO: use the structured outputs feature to ensure a valid json at all times?

        # 3. Determine Execution Order
        execution_order = await self._determine_execution_order(
            valves=valves,
            __metadata__=__metadata__,
            model_config=model_config,
            features=features,
        )

        chat_id = __metadata__.get("chat_id")
        message_id = __metadata__.get("message_id")
        log.debug(f"Chat ID: {chat_id}, Message ID: {message_id}")

        event_emitter = await self._get_event_emitter(
            app_state=app_state,
            chat_id=chat_id,
            message_id=message_id,
            is_task=bool(__metadata__.get("task")),
        )

        # --- Execution Loop ---
        for attempt_idx, tier in enumerate(execution_order):
            is_last_attempt = attempt_idx == len(execution_order) - 1

            # Create a "Tier-Specific" valves object for this attempt.
            # We copy to ensure we don't pollute the original valves or metadata.
            current_valves = copy.copy(valves)

            if tier == "free":
                current_valves.GEMINI_PAID_API_KEY = None
                current_valves.USE_VERTEX_AI = False
                __metadata__["is_paid_api"] = False
            elif tier == "paid":
                current_valves.GEMINI_FREE_API_KEY = None
                current_valves.USE_VERTEX_AI = False
                __metadata__["is_paid_api"] = True
            elif tier == "vertex":
                current_valves.USE_VERTEX_AI = True
                __metadata__["is_paid_api"] = True

            try:
                log.info(f"Starting generation attempt on tier: {tier}")

                # Execute the attempt. This encapsulates client creation,
                # file uploads (scoped to key), and the API call.
                return await self._execute_generation_attempt(
                    tier=tier,
                    valves=current_valves,
                    body=body,
                    __user__=__user__,
                    __metadata__=__metadata__,
                    __request__=__request__,
                    event_emitter=event_emitter,
                    model_config=model_config,
                )

            except Exception as e:
                # Error Handling & Routing
                error_str = str(e).upper()

                # We catch Quota (429), Permission (403), and Overload/Unavailable (503) errors.
                # 503 is crucial for Free Tier which has strict load balancing.
                is_fallback_eligible = (
                    "429" in error_str
                    or "403" in error_str
                    or "503" in error_str
                    or "UNAVAILABLE" in error_str
                    or (isinstance(e, genai_errors.ClientError) and e.code in [429, 403])
                    or (isinstance(e, genai_errors.ServerError) and e.code in [503])
                )

                should_retry = not is_last_attempt and tier == "free" and is_fallback_eligible

                if should_retry:
                    reason = "quota exceeded" if "429" in error_str else "model overloaded"
                    log.warning(f"Free Tier {reason} (Error: {e}). Switching to Paid API...")
                    event_emitter.emit_status(
                        f"Free Tier {reason}, switching to Paid API...", done=False
                    )
                    continue 

                # If we can't retry, re-raise the error to stop execution
                log.error(f"Error during request execution (Tier: {tier}): {e}")
                raise e

        raise ValueError("Exhausted execution options without result.")

    # region 1. Helper methods inside the Pipe class

    # region 1.1 Client initialization
    @staticmethod
    @cache
    def _get_or_create_genai_client(
        free_api_key: str | None = None,
        paid_api_key: str | None = None,
        base_url: str | None = None,
        use_vertex_ai: bool | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
    ) -> genai.Client:
        """
        Creates a genai.Client instance or retrieves it from cache.
        Raises GenaiApiError on failure.
        """

        # Prioritize the free key, then fall back to the paid key.
        api_key = free_api_key or paid_api_key

        if not vertex_project and not api_key:
            # FIXME: More detailed reason in the exception (tell user to set the API key).
            msg = "Neither VERTEX_PROJECT nor a Gemini API key (free or paid) is set."
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
            if not valves.GEMINI_FREE_API_KEY and not valves.GEMINI_PAID_API_KEY:
                error_msg = (
                    "User must provide their own authentication configuration. "
                    "Please set GEMINI_FREE_API_KEY or GEMINI_PAID_API_KEY in your UserValves."
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
            "GEMINI_FREE_API_KEY",
            "GEMINI_PAID_API_KEY",
            "GEMINI_API_BASE_URL",
            "USE_VERTEX_AI",
            "VERTEX_PROJECT",
            "VERTEX_LOCATION",
        ]
        return [getattr(source_valves, attr, None) for attr in ATTRS]

    # endregion 1.1 Client initialization

    # region 1.2 Model retrival from Google API

    @cached()  # aiocache.cached for async method
    async def _get_genai_models(
        self,
        free_api_key: str | None = None,
        paid_api_key: str | None = None,
        base_url: str | None = None,
        use_vertex_ai: bool | None = None,
        vertex_project: str | None = None,
        vertex_location: str | None = None,
        whitelist_str: str = "*",
        blacklist_str: str | None = None,
    ) -> list["ModelData"]:
        """
        Gets valid Google models from API(s) and filters them.
        If use_vertex_ai, vertex_project, and api_key are all provided,
        models are fetched from both Vertex AI and Gemini Developer API and merged.
        """
        all_raw_models: list[types.Model] = []

        # Condition for fetching from both sources
        fetch_both = bool(use_vertex_ai and vertex_project and (free_api_key or paid_api_key))

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
                    free_api_key=free_api_key,
                    paid_api_key=paid_api_key,
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
                    base_url=base_url,  # Pass base_url for potential Vertex custom endpoints
                )
                vertex_models_list = await self._fetch_models_from_client_internal(
                    vertex_client, "Vertex AI"
                )
            except GenaiApiError as e:
                log.warning(
                    f"Failemodel_configd to initialize or retrieve models from Vertex AI: {e}"
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
                    model_id = self._strip_api_prefix(model.name)
                    if model_id and model_id not in combined_models_dict:
                        combined_models_dict[model_id] = model
                else:
                    log.trace(
                        f"Gemini model without a name encountered: {model.display_name or 'N/A'}"
                    )

            for model in vertex_models_list:
                if model.name:
                    model_id = self._strip_api_prefix(model.name)
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
                    free_api_key=free_api_key,
                    paid_api_key=paid_api_key,
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
                    f"Model '{model.name}' (ID: {self._strip_api_prefix(model.name)}) skipped, not generative (actions: {actions})."
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
            assert model.name is not None
            stripped_name = self._strip_api_prefix(model.name)

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
    def _strip_api_prefix(model_name: str) -> str:
        """
        Extract the model identifier by removing API resource prefixes.
        e.g., "models/gemini-1.5-flash-001" -> "gemini-1.5-flash-001"
        e.g., "publishers/google/models/gemini-1.5-pro" -> "gemini-1.5-pro"
        Does NOT handle the manifold pipe prefix (e.g. "gemini_manifold_google_genai.").
        """
        # Remove everything up to the last '/'
        return model_name.split("/")[-1]

    @staticmethod
    def _get_model_name(body: "Body") -> str:
        """
        Extracts the canonical model name from the request body.

        Handles standard model names and custom workspace models by prioritizing
        the base_model_id found in metadata.

        Args:
            body: The request body dictionary.

        Returns:
            The canonical model name (prefix removed).
        """
        # 1. Get the initially requested model name from the top level
        effective_model_name: str = body.get("model", "")
        initial_model_name = effective_model_name
        base_model_name = None

        # 2. Check for a base model ID in the metadata for custom models
        if metadata := body.get("metadata"):
            # Safely navigate the nested structure: metadata -> model -> info -> base_model_id
            base_model_name = (
                metadata.get("model", {}).get("info", {}).get("base_model_id", None)
            )
            # If a base model ID is found, it overrides the initially requested name
            if base_model_name:
                effective_model_name = base_model_name

        # 3. Create the canonical model name by removing the manifold prefix
        canonical_model_name = effective_model_name.replace(
            "gemini_manifold_google_genai.", ""
        )

        # 4. Log the relevant names for debugging purposes
        log.debug(
            f"Model Name Extraction: initial='{initial_model_name}', "
            f"base='{base_model_name}', effective='{effective_model_name}', "
            f"canonical='{canonical_model_name}'"
        )

        # 5. Return only the canonical name
        return canonical_model_name

    @staticmethod
    def _is_image_model(model_id: str, config: dict) -> bool:
        """Check if the model is an image generation model using provided config."""
        if model_id in config:
            return config[model_id].get("capabilities", {}).get("image_generation", False)

        return False

    @staticmethod
    def _parse_gemini_version(model_id: str) -> float:
        """Extracts the Gemini model version number from model ID.

        If not a Gemini model or version is missing, logs a warning and returns 3.0.
        """
        model_id_lower = model_id.lower()
        if "gemini" in model_id_lower:
            match = re.search(r"gemini-(\d+(?:\.\d+)?)", model_id_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        log.warning(
            f"Model '{model_id}' is either not a Gemini model or missing a standard version number. "
            "Treating as Gemini 3+ model."
        )
        return 3.0

    # endregion 1.2 Model retrival from Google API

    # region 1.3 GenerateContentConfig assembly

    @classmethod
    def _resolve_valve_thinking_config(
        cls, rules_json: str, model_id: str
    ) -> tuple[int | None, str | None]:
        """Evaluates THINKING_CONFIG_RULES regex patterns against model_id to determine default budget/level."""
        if not rules_json or not rules_json.strip():
            return None, None

        try:
            rules = json.loads(rules_json)
            if not isinstance(rules, dict):
                return None, None
        except Exception as e:
            log.warning(f"Failed to parse THINKING_CONFIG_RULES JSON: {e}")
            return None, None

        matching_hits: list[tuple[str, int | str]] = []
        for pattern, val in rules.items():
            try:
                if re.search(pattern, model_id):
                    matching_hits.append((pattern, val))
            except re.error as e:
                log.warning(
                    f"Invalid regex pattern '{pattern}' in THINKING_CONFIG_RULES: {e}"
                )

        if not matching_hits:
            return None, None

        model_version = cls._parse_gemini_version(model_id)
        is_under_v3 = model_version < 3.0

        level_hits = [val for _, val in matching_hits if isinstance(val, str)]
        budget_hits = [val for _, val in matching_hits if isinstance(val, int)]

        if is_under_v3:
            if budget_hits:
                return budget_hits[0], None
            elif level_hits:
                log.warning(
                    f"Model '{model_id}' (version {model_version}) is older than Gemini 3 and does not support thinking_level. "
                    f"Regex hit specified level '{level_hits[0]}', but no budget rule hit. Falling back to API default."
                )
                return None, None
        else:
            if level_hits:
                return None, level_hits[0]
            elif budget_hits:
                return budget_hits[0], None

        return None, None

    async def _build_gen_content_config(
        self,
        body: "Body",
        __metadata__: "Metadata",
        valves: "Valves",
        config: dict,
    ) -> types.GenerateContentConfig:
        """Assembles the GenerateContentConfig for a Gemini API request."""
        features = __metadata__.get("features", {}) or {}
        is_vertex_ai = __metadata__.get("is_vertex_ai", False)

        log.debug(
            "Features extracted from metadata (UI toggles and config):",
            payload=features,
        )

        safety_settings: list[types.SafetySetting] | None = __metadata__.get(
            "safety_settings"
        )

        thinking_conf = None
        # We are ensured to have a valid model ID at this point.
        model_id: str = __metadata__.get("canonical_model_id", "")
        is_thinking_model = False
        if model_id in config:
            is_thinking_model = (
                config[model_id].get("capabilities", {}).get("thinking", False)
            )

        log.debug(
            f"Model '{model_id}' is classified as a reasoning model: {bool(is_thinking_model)}. "
        )

        if is_thinking_model:
            # Precedence level 1 (lowest): Valves option
            chosen_budget, chosen_level = self._resolve_valve_thinking_config(
                valves.THINKING_CONFIG_RULES, model_id
            )
            log.info(
                f"Resolved valve thinking config for model '{model_id}': "
                f"budget={chosen_budget}, level={chosen_level}"
            )

            # Precedence level 2 (medium): Merged params override
            merged_params = __metadata__.get("merged_custom_params", {})
            if reasoning_effort := merged_params.get("reasoning_effort"):
                log.info(
                    f"Found `reasoning_effort` custom parameter: '{reasoning_effort}'. Overriding valve settings."
                )

                try:
                    budget = round(float(reasoning_effort))
                    log.info(
                        f"Interpreting `reasoning_effort` as a thinking budget: {budget}"
                    )
                    chosen_budget = budget
                    chosen_level = None
                except (ValueError, TypeError):
                    if isinstance(reasoning_effort, str):
                        effort_level_str = reasoning_effort.upper()
                        valid_levels = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
                        if effort_level_str in valid_levels:
                            log.info(
                                f"Interpreting `reasoning_effort` as a thinking level: {effort_level_str}"
                            )
                            chosen_level = effort_level_str
                            chosen_budget = None
                        else:
                            log.warning(
                                f"Invalid `reasoning_effort` string value: '{reasoning_effort}'. "
                                f"Valid values are {sorted(valid_levels)}. "
                                "Falling back to valve settings."
                            )
                    else:
                        log.warning(
                            f"Unsupported type for `reasoning_effort`: {type(reasoning_effort)}. "
                            "Expected a number or string. Falling back to valve settings."
                        )

            # Precedence level 3 (highest): Toggle filter override
            is_avail, is_on = await self._get_toggleable_feature_status(
                "gemini_reasoning_toggle", __metadata__
            )
            if is_avail and not is_on:
                is_25_flash_or_lite = bool(
                    re.search(
                        r"gemini-2\.5-(?:flash|flash-lite)", model_id, re.IGNORECASE
                    )
                )
                if is_25_flash_or_lite:
                    log.info(
                        f"Model '{model_id}' is Gemini 2.5 Flash/Flash-Lite and reasoning toggle is OFF in UI. "
                        "Overwriting `thinking_budget` to 0 to disable reasoning."
                    )
                    chosen_budget = 0
                    chosen_level = None
                else:
                    log.info(
                        f"Reasoning toggle is OFF in UI, but model '{model_id}' does not support disabling thinking. "
                        "Ignoring toggle setting."
                    )

            model_version = self._parse_gemini_version(model_id)
            if model_version < 3.0 and chosen_level is not None:
                log.warning(
                    f"Model '{model_id}' (version {model_version}) does not support `thinking_level`. "
                    f"Clearing requested thinking level '{chosen_level}'."
                )
                chosen_level = None

            thinking_kwargs: dict[str, Any] = {
                "include_thoughts": valves.SHOW_THINKING_SUMMARY
            }

            if chosen_level is not None:
                if hasattr(types, "ThinkingLevel") and hasattr(
                    types.ThinkingLevel, chosen_level
                ):
                    thinking_kwargs["thinking_level"] = types.ThinkingLevel[
                        chosen_level
                    ]
                else:
                    thinking_kwargs["thinking_level"] = chosen_level
            elif chosen_budget is not None:
                thinking_kwargs["thinking_budget"] = chosen_budget

            thinking_conf = types.ThinkingConfig(**thinking_kwargs)
            log.info("Final thinking config payload:", payload=thinking_conf)

        # TODO: Take defaults from the general front-end config.
        # system_instruction is intentionally left unset here. It will be set by the caller.
        gen_content_conf = types.GenerateContentConfig(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            top_k=body.get("top_k"),
            max_output_tokens=body.get("max_tokens"),
            stop_sequences=body.get("stop"),
            safety_settings=safety_settings,
            thinking_config=thinking_conf,
        )
        gen_content_conf.response_modalities = ["TEXT"]

        # Optimization: Task models (titles, tags, search queries) do not require tools.
        # Disabling them here prevents unnecessary tool-use overhead and reduces latency.
        if __metadata__.get("task"):
            log.debug("Task model detected. Skipping tool configuration.")
            return gen_content_conf

        if self._is_image_model(model_id, config):
            gen_content_conf.response_modalities.append("IMAGE")
            if "gemini-3-pro-image" in model_id and valves.IMAGE_RESOLUTION:
                log.debug(f"Setting image resolution to {valves.IMAGE_RESOLUTION}")
                if not gen_content_conf.image_config:
                    gen_content_conf.image_config = types.ImageConfig()
                gen_content_conf.image_config.image_size = valves.IMAGE_RESOLUTION

            if (
                "gemini-3-pro-image" in model_id or "gemini-2.5-flash-image" in model_id
            ) and valves.IMAGE_ASPECT_RATIO:
                log.debug(f"Setting image aspect ratio to {valves.IMAGE_ASPECT_RATIO}")
                if not gen_content_conf.image_config:
                    gen_content_conf.image_config = types.ImageConfig()
                gen_content_conf.image_config.aspect_ratio = valves.IMAGE_ASPECT_RATIO

        gen_content_conf.tools = []

        if features.get("google_search_tool"):
            if valves.USE_ENTERPRISE_SEARCH and is_vertex_ai:
                log.info("Using grounding with Enterprise Web Search as a Tool.")
                gen_content_conf.tools.append(
                    types.Tool(enterprise_web_search=types.EnterpriseWebSearch())
                )
            else:
                log.info("Using grounding with Google Search as a Tool.")
                gen_content_conf.tools.append(
                    types.Tool(google_search=types.GoogleSearch())
                )

        # NB: It is not possible to use both Search and Code execution at the same time,
        # however, it can be changed later, so let's just handle it as a common error
        if features.get("google_code_execution"):
            log.info("Using code execution on Google side.")
            gen_content_conf.tools.append(
                types.Tool(code_execution=types.ToolCodeExecution())
            )

        # Determine if URL context tool should be enabled.
        is_avail, is_on = await self._get_toggleable_feature_status(
            "gemini_url_context_toggle", __metadata__
        )
        enable_url_context = False
        if is_avail:
            # If the toggle filter is configured, it overrides the valve setting.
            enable_url_context = is_on

        if enable_url_context:
            # Check capability from config
            is_compatible = False
            if model_id in config:
                is_compatible = config[model_id].get("capabilities", {}).get("url_context", False)

            if is_compatible:
                if is_vertex_ai and (len(gen_content_conf.tools) > 0):
                    log.warning(
                        "URL context tool is enabled, but Vertex AI is used with other tools. Skipping."
                    )
                else:
                    log.info(
                        f"Model {model_id} is compatible with URL context tool. Enabling."
                    )
                    gen_content_conf.tools.append(
                        types.Tool(url_context=types.UrlContext())
                    )
            else:
                log.warning(
                    f"URL context tool is enabled, but model {model_id} does not support it (see capabilities.url_context in gemini_models.yaml). Skipping."
                )

        # Determine if Google Maps grounding should be enabled.
        is_avail, is_on = await self._get_toggleable_feature_status(
            "gemini_maps_grounding_toggle", __metadata__
        )
        if is_avail and is_on:
            log.info("Enabling Google Maps grounding tool.")
            gen_content_conf.tools.append(
                types.Tool(google_maps=types.GoogleMaps())
            )

            if valves.MAPS_GROUNDING_COORDINATES:
                try:
                    lat_str, lon_str = valves.MAPS_GROUNDING_COORDINATES.split(",")
                    latitude = float(lat_str.strip())
                    longitude = float(lon_str.strip())

                    log.info(
                        "Using coordinates for Maps grounding: "
                        f"lat={latitude}, lon={longitude}"
                    )

                    lat_lng = types.LatLng(latitude=latitude, longitude=longitude)

                    # Ensure tool_config and retrieval_config exist before assigning lat_lng.
                    if not gen_content_conf.tool_config:
                        gen_content_conf.tool_config = types.ToolConfig()
                    if not gen_content_conf.tool_config.retrieval_config:
                        gen_content_conf.tool_config.retrieval_config = (
                            types.RetrievalConfig()
                        )

                    gen_content_conf.tool_config.retrieval_config.lat_lng = lat_lng

                except (ValueError, TypeError) as e:
                    # This should not happen due to the Pydantic validator, but it's good practice to be safe.
                    log.error(
                        "Failed to parse MAPS_GROUNDING_COORDINATES: "
                        f"'{valves.MAPS_GROUNDING_COORDINATES}'. Error: {e}"
                    )

        return gen_content_conf

    # endregion 1.3 GenerateContentConfig assembly

    # region 1.4 Model response processing

    async def _aggregate_to_dict(
        self,
        generator: AsyncGenerator[dict | str, None],
    ) -> dict:
        """
        Consumes the unified response generator and aggregates the chunks into a
        single OpenAI Chat Completion dictionary. This keeps our processing pipeline
        unified while properly satisfying OWUI's non-streaming request expectations.
        """
        content = ""
        reasoning_content = ""
        usage = None

        async for chunk in generator:
            if isinstance(chunk, str):
                # Skip string yields (like "data: [DONE]") used for streaming protocol
                continue

            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    content += delta["content"]
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    reasoning_content += delta["reasoning_content"]

            if "usage" in chunk:
                usage = chunk["usage"]

        # Only add the reasoning key if there was actually reasoning content
        message: dict[str, str] = {
            "role": "assistant",
            "content": content,
        }
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        return {
            "choices": [
                {
                    "message": message,
                }
            ],
            "usage": usage or {},
        }

    async def _execute_generation_attempt(
        self,
        tier: str,
        valves: "Pipe.Valves",
        body: "Body",
        __user__: "UserData",
        __metadata__: "Metadata",
        __request__: Request,
        event_emitter: "EventEmitter",
        model_config: dict,
    ) -> AsyncGenerator[dict | str, None] | dict:
        """
        Executes a single generation attempt with a specific tier configuration.
        Constructs a fresh client and file manager to ensure assets are
        scoped to the correct API key/project.
        """

        # 1. Client Creation
        client = self._get_user_client(valves, __user__["email"])
        __metadata__["is_vertex_ai"] = client.vertexai
        api_name = "Vertex AI Gemini API" if client.vertexai else "Gemini Developer API"

        # 2. Files API Manager (Scoped to the current client)
        files_api_manager = FilesAPIManager(
            client=client,
            file_cache=self.file_content_cache,
            id_hash_cache=self.file_id_to_hash_cache,
            event_emitter=event_emitter,
        )

        # 3. Content Builder (Re-uploads files if client changed)
        builder = GeminiContentBuilder(
            messages_body=body.get("messages"),
            metadata_body=__metadata__,
            user_data=__user__,
            event_emitter=event_emitter,
            valves=valves,
            files_api_manager=files_api_manager,
        )

        event_emitter.emit_status("Preparing request...")
        contents = await builder.build_contents()

        # 4. Configuration Building
        gen_content_conf = await self._build_gen_content_config(
            body, __metadata__, valves, model_config
        )
        gen_content_conf.system_instruction = builder.system_prompt

        model_id = __metadata__.get("canonical_model_id", "")

        # Check for image/system prompt compatibility
        is_image_model = self._is_image_model(model_id, model_config)
        if (
            is_image_model or "gemma" in model_id
        ) and gen_content_conf.system_instruction:
            gen_content_conf.system_instruction = None
            log.warning(
                f"Model '{model_id}' does not support system prompts. Removing."
            )

        gen_content_args = {
            "model": model_id,
            "contents": contents,
            "config": gen_content_conf,
        }
        log.debug(
            f"Passing args to {api_name} (Tier: {tier}):", payload=gen_content_args
        )

        # 5. Stream Setup
        # 'is_streaming_request' tracks what Open WebUI expects to receive.
        # 'use_streaming_api' tracks how we actually call the Google SDK.
        is_streaming_request = body.get("stream", True)
        use_streaming_api = is_streaming_request

        # If a high-resolution image is requested with the gemini-3-pro-image model,
        # the Google GenAI SDK's streaming method often raises a "chunk too big" error
        # during the transfer of the generated image bytes. We avoid this by forcing
        # a non-streaming SDK call, while still yielding the result as a stream to OWUI.
        if (
            use_streaming_api
            and valves.IMAGE_RESOLUTION in ["2K", "4K"]
            # FIXME: Nano Banana 2 supports resolutions too now.
            and "gemini-3-pro-image" in model_id
        ):
            log.info(
                f"Forcing non-streaming SDK call due to {valves.IMAGE_RESOLUTION} resolution "
                "to avoid GenAI SDK 'chunk too big' errors."
            )
            use_streaming_api = False

        request_type_str = "streaming" if use_streaming_api else "non-streaming"
        request_type_msg = f"Sending {request_type_str} request to {api_name}..."
        log.info(request_type_msg)
        event_emitter.emit_status(request_type_msg)

        # 6. Execution & Peek Logic
        api_request_start_time = time.monotonic()
        if use_streaming_api:
            stream = await client.aio.models.generate_content_stream(**gen_content_args)  # type: ignore
        else:
            # When we use the non-streaming SDK call, we wrap the single response
            # in an iterator so that our unified processor can still treat it like a stream.
            response = await client.aio.models.generate_content(**gen_content_args)

            async def one_shot_iter():
                yield response

            stream = one_shot_iter()

        iterator = stream.__aiter__()

        try:
            first_chunk = await iterator.__anext__()
            first_token_time = time.monotonic()
        except StopAsyncIteration:
            raise ValueError("API returned an empty response stream.")

        # Success: Reconstruct the stream including the peeked chunk
        async def reconstructed_stream():
            yield first_chunk
            async for chunk in iterator:
                yield chunk

        log.info(f"Request successful ({tier}). Passing stream to unified processor.")

        processor = self._unified_response_processor(
            reconstructed_stream(),
            __request__.app,
            event_emitter,
            __metadata__,
            api_request_start_time=api_request_start_time,
            first_token_time=first_token_time,
        )

        # If OWUI requested a stream, we return the AsyncGenerator.
        # If OWUI requested a single object, we aggregate the chunks back into a dict.
        if is_streaming_request:
            return processor
        else:
            return await self._aggregate_to_dict(processor)

    def _check_free_tier_eligibility(
        self,
        model_id: str,
        model_config: dict,
        features: "Features",
    ) -> bool:
        """
        Determines if the request is eligible for the Free Tier based on model config
        and requested features (e.g., grounding).
        """
        # 1. Check if model is configured as having a free tier in YAML
        if model_id not in model_config:
            return False

        pricing = model_config[model_id].get("pricing", {})
        if not pricing.get("free_tier", False):
            return False

        # 2. Check for feature exclusions (e.g. Google Search is often Paid only)
        excluded_features = pricing.get("excluded_features", [])

        # Check Search
        is_search_requested = features.get("google_search_tool")
        if is_search_requested and "search_grounding" in excluded_features:
            log.info(
                f"Free Tier ineligible: Search requested but excluded for {model_id}."
            )
            return False

        # Check Maps
        # Note: We check the raw feature toggle presence here, assuming the toggle logic put it in features
        if (
            features.get("gemini_maps_grounding_toggle")
            and "grounding_google_maps" in excluded_features
        ):
            log.info(
                f"Free Tier ineligible: Maps requested but excluded for {model_id}."
            )
            return False

        return True

    async def _unified_response_processor(
        self,
        response_stream: AsyncIterator[types.GenerateContentResponse],
        app: FastAPI,
        event_emitter: "EventEmitter",
        __metadata__: "Metadata",
        api_request_start_time: float | None = None,
        first_token_time: float | None = None,
    ) -> AsyncGenerator[dict | str, None]:
        """
        Processes an async iterator of GenerateContentResponse objects, yielding
        structured dictionary chunks for the Open WebUI frontend.

        This single method handles both streaming and non-streaming (via an adapter)
        responses, eliminating code duplication. It processes all parts within each
        response chunk, counts tag substitutions for a final toast notification,
        and handles post-processing in a finally block.
        """
        final_response_chunk: types.GenerateContentResponse | None = None
        error_occurred = False
        total_substitutions = 0
        first_chunk_received = False
        chunk_counter = 0
        in_think = False
        last_title: str | None = None
        response_parts: list[types.Part] = []
        content_parts_text: list[str] = []

        try:
            async for chunk in response_stream:
                candidate = self._get_first_candidate(chunk.candidates)
                content = candidate.content if candidate else None
                log.trace(f"Processing response chunk #{chunk_counter}, first candidate's content:", payload=content)
                chunk_counter += 1
                final_response_chunk = chunk  # Keep the latest chunk for metadata

                if not first_chunk_received:
                    # This is the first (and possibly only) chunk.
                    event_emitter.emit_status("Response received", done=True)
                    first_chunk_received = True

                if not (parts := chunk.parts):
                    log.warning("Chunk has no parts, skipping.")
                    continue

                response_parts.extend(parts)

                # This inner loop makes the method robust. It handles a single chunk
                # with many parts (non-streaming) or many chunks with one part (streaming).
                for part in parts:
                    # Handle thought titles and transitions between reasoning and normal content.
                    if part.thought:
                        if not in_think:
                            # TODO: emit an status indicating that reasoning has started. include budget or level if set.
                            in_think = True

                        # Attempt to extract a title from any text within a thought part.
                        # TODO: refactor it to a helper method?
                        if isinstance(part.text, str):
                            try:
                                title: str | None = None
                                # Prefer markdown-style "### Heading" titles.
                                for m in re.finditer(
                                    r"(^|\n)###\s+(.+?)(?=\n|$)", part.text or ""
                                ):
                                    title = m.group(2).strip()
                                # Fall back to bold "**Title**" lines if no heading was found.
                                if not title:
                                    for m in re.finditer(
                                        r"(^|\n)\s*\*\*(.+?)\*\*\s*(?=\n|$)",
                                        part.text or "",
                                    ):
                                        title = (m.group(2) or "").strip()
                                if title:
                                    # Trim common surrounding quotes.
                                    title = title.strip('"“”‘’').strip()
                                if title and title != last_title:
                                    last_title = title
                                    event_emitter.emit_status(
                                        title,
                                        done=False,
                                        hidden=False,
                                        is_thought=True,
                                        indent_level=1,
                                    )
                            except Exception:
                                # Thought titles are a best-effort feature; failures should not break the stream.
                                pass
                    elif in_think:
                        # Terminate the 'in_think' state only when a non-thought part with actual content arrives.
                        # This prevents empty text parts from prematurely ending the thought block in the UI.
                        has_content = (
                            (isinstance(part.text, str) and part.text)
                            or part.inline_data
                            or part.executable_code
                            or part.code_execution_result
                        )
                        if has_content:
                            in_think = False
                            # Clear the last thought title when normal content begins.
                            event_emitter.emit_status(
                                "Thinking finished",
                                done=True,
                                is_thought=False,
                            )

                    payload, count = await self._process_part(
                        part,
                        app,
                        __metadata__,
                    )

                    if payload:
                        # Collect the original content text before it's sent to the frontend.
                        # We only care about the "content" key for the final message.
                        if "content" in payload and payload["content"]:
                            content_parts_text.append(payload["content"])

                        if count > 0:
                            total_substitutions += count
                            log.debug(f"Disabled {count} special tag(s) in a part.")

                        structured_chunk = {"choices": [{"delta": payload}]}
                        yield structured_chunk

        except Exception as e:
            error_occurred = True
            error_msg = f"Response processing ended with error: {e}"
            log.exception(error_msg)
            event_emitter.emit_error(error_msg)

        finally:
            # The async for loop has completed, meaning we have received all data
            # from the API. Now, we perform final internal processing.

            if total_substitutions > 0 and not error_occurred:
                plural_s = "s" if total_substitutions > 1 else ""
                toast_msg = (
                    f"For clarity, {total_substitutions} special tag{plural_s} "
                    "were disabled in the response by injecting a zero-width space (ZWS)."
                )
                event_emitter.emit_toast(toast_msg, "info")

            # Calculate usage data using the last received chunk.
            # If the chunk contains usage metadata, we yield it so the backend persists it to DB.
            if final_response_chunk and (
                usage_data := self._get_usage_data(
                    final_response_chunk,
                    app.state,
                    __metadata__,
                    event_emitter.start_time,
                    api_request_start_time,
                    first_token_time,
                )
            ):
                # Yielding this dictionary allows the OWUI proxy to catch and save usage.
                yield {"usage": usage_data}

            if not error_occurred:
                # 'data: [DONE]' should be the last thing yielded in a successful stream
                # to signify the protocol-level end of the OpenAI-compatible stream.
                yield "data: [DONE]"
                log.info("Response processing finished successfully!")

                # We manually upsert custom metadata to the database because Open WebUI
                # does not have a native way to persist non-standard message keys
                # through the standard pipe return/yield stream.
                chat_id = __metadata__.get("chat_id")
                message_id = __metadata__.get("message_id")
                is_task = __metadata__.get("task")

                if chat_id and message_id and not is_task:
                    db_payload = {}

                    if response_parts:
                        db_payload["gemini_parts"] = [
                            part.model_dump(mode="json", exclude_none=True)
                            for part in response_parts
                        ]

                    if content_parts_text:
                        db_payload["original_content"] = "".join(content_parts_text)

                    if db_payload:
                        try:
                            # We call the backend model directly. This updates the JSON blob
                            # in the 'chat' table, which the frontend uses as the source of truth.
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                id=chat_id,
                                message_id=message_id,
                                message=db_payload,
                            )
                            log.debug(
                                f"Successfully persisted Gemini metadata to message {message_id}"
                            )
                        except Exception as e:
                            log.error(f"Failed to persist Gemini metadata to DB: {e}")

            try:
                await self._do_post_processing(
                    final_response_chunk,
                    event_emitter,
                    app.state,
                    __metadata__,
                    stream_error_happened=error_occurred,
                )
            except Exception as e:
                error_msg = f"Post-processing failed with error:\n\n{e}"
                event_emitter.emit_toast(error_msg, "error")
                log.exception(error_msg)

            log.debug("Unified response processor has finished.")

    async def _process_part(
        self,
        part: types.Part,
        app: FastAPI,  # We need the app to generate URLs for model returned images.
        __metadata__: "Metadata",
    ) -> tuple[dict | None, int]:
        """
        Processes a single `types.Part` object and returns a payload dictionary
        for the Open WebUI stream, along with a count of tag substitutions.
        The payload key is 'reasoning_content' for thought parts and 'content' for others.
        """
        # Determine the payload key based on whether the part is a thought.
        key = "reasoning_content" if part.thought else "content"
        payload_content: str | None = None
        count: int = 0

        match part:
            case types.Part(text=str(text)):
                # It's regular content or a thought with text.
                sanitized_text, count = self._disable_special_tags(text)
                payload_content = sanitized_text
            case types.Part(inline_data=data) if data:
                # An image part, which can be part of a thought or regular content.
                # Image parts don't need tag disabling.
                processed_text, image_url = await self._process_image_part(
                    data, __metadata__, app
                )
                payload_content = processed_text

                # Transform inline_data into file_data to avoid storing raw bytes in the database.
                # This mutates the part object which is held by reference in `response_parts`.
                if image_url and data.mime_type:
                    part.inline_data = None
                    part.file_data = types.FileData(
                        file_uri=image_url, mime_type=data.mime_type
                    )
            case types.Part(executable_code=code) if code:
                # Code blocks are already formatted and safe.
                if processed_text := self._process_executable_code_part(code):
                    payload_content = processed_text
            case types.Part(code_execution_result=result) if result:
                # Code results are also safe.
                if processed_text := self._process_code_execution_result_part(result):
                    payload_content = processed_text

        if payload_content is not None:
            return {key: payload_content}, count

        return None, 0

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

    async def _process_image_part(
        self,
        inline_data: types.Blob,
        __metadata__: "Metadata",
        app: FastAPI,
    ) -> tuple[str, str | None]:
        """
        Handles image data by saving it to the Open WebUI backend and returning a markdown link
        and the URL.
        """
        mime_type = inline_data.mime_type
        image_data = inline_data.data
        image_url = None

        if mime_type and image_data:
            image_url = await self._upload_image(
                image_data,
                mime_type,
                __metadata__,
                app,
            )
        else:
            log.warning(
                "Image part has no mime_type or data, cannot upload image. "
                "Returning a placeholder message."
            )

        markdown_text = (
            f"![Generated Image]({image_url})"
            if image_url
            else "*An error occurred while trying to store this model generated image.*"
        )
        return markdown_text, image_url

    async def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        __metadata__: "Metadata",
        app: FastAPI,
    ) -> str | None:
        """
        Helper method that uploads a generated image to the configured Open WebUI storage provider.
        Returns the url to the uploaded image.
        """
        image_format = mimetypes.guess_extension(mime_type) or ".png"
        id = str(uuid.uuid4())
        name = f"generated-image{image_format}"

        # The final filename includes the unique ID to prevent collisions.
        imagename = f"{id}_{name}"
        image = io.BytesIO(image_data)

        # Create a clean, precise metadata object linking to the generation context.
        image_metadata = {
            "model": __metadata__.get("canonical_model_id"),
            "chat_id": __metadata__.get("chat_id"),
            "message_id": __metadata__.get("message_id"),
        }

        log.info("Uploading the model-generated image to the Open WebUI backend.")

        try:
            contents, image_path = await asyncio.to_thread(
                Storage.upload_file, image, imagename, tags={}
            )
        except Exception:
            log.exception("Error occurred during upload to the storage provider.")
            return None

        log.debug("Adding the image file to the Open WebUI files database.")
        file_item = await Files.insert_new_file(
            __metadata__.get("user_id"),
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
            log.warning("Image upload to Open WebUI database likely failed.")
            return None

        image_url: str = app.url_path_for(
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

    # endregion 1.4 Model response processing

    # region 1.5 Post-processing
    async def _do_post_processing(
        self,
        model_response: types.GenerateContentResponse | None,
        event_emitter: "EventEmitter",
        app_state: State,
        __metadata__: "Metadata",
        *,
        stream_error_happened: bool = False,
    ):
        """Handles grounding and sources after the main response/stream is done."""
        log.info("Post-processing the model response.")

        if stream_error_happened:
            log.warning("Response processing failed due to stream error.")
            event_emitter.emit_status("Response failed [Stream Error]", done=True)
            return

        if not model_response:
            log.warning("Response processing skipped: Model response was empty.")
            event_emitter.emit_status(
                "Response failed [Empty Response]", done=True
            )
            return

        if not (candidate := self._get_first_candidate(model_response.candidates)):
            log.warning("Response processing skipped: No candidates found.")
            event_emitter.emit_status(
                "Response failed [No Candidates]", done=True
            )
            return

        # --- Construct detailed finish reason message ---
        reason_name = getattr(candidate.finish_reason, "name", "UNSPECIFIED")
        reason_description = FINISH_REASON_DESCRIPTIONS.get(reason_name)
        finish_message = (
            candidate.finish_message.strip() if candidate.finish_message else None
        )

        details_parts = [part for part in (reason_description, finish_message) if part]
        details_str = f": {' '.join(details_parts)}" if details_parts else ""
        full_finish_details = f"[{reason_name}]{details_str}"

        # --- Determine final status and emit toast for errors ---
        is_normal_finish = candidate.finish_reason in NORMAL_REASONS

        if is_normal_finish:
            log.debug(f"Response finished normally. {full_finish_details}")
            status_prefix = "Response finished"
        else:
            log.error(f"Response finished with an error. {full_finish_details}")
            status_prefix = "Response failed"
            event_emitter.emit_toast(
                f"An error occurred. {full_finish_details}",
                "error",
            )

        # For the most common success case (STOP), we don't need to show the reason.
        final_reason_str = "" if reason_name == "STOP" else f" [{reason_name}]"
        event_emitter.emit_status(
            f"{status_prefix}{final_reason_str}",
            done=True,
            is_successful_finish=is_normal_finish,
        )

        # TODO: Emit a toast message if url context retrieval was not successful.

        storage_payload: dict[str, Any] = {}

        if candidate and (grounding_metadata_obj := candidate.grounding_metadata):
            storage_payload["grounding"] = grounding_metadata_obj

        self._store_data_in_state(app_state, __metadata__, storage_payload)

    @staticmethod
    def _calculate_cost(token_count: int, pricing_tiers: list[dict]) -> float:
        """
        Calculates cost based on tiered pricing structure (in USD)
        """
        if not pricing_tiers or token_count <= 0:
            return 0.0

        total_cost = 0.0
        remaining_tokens = token_count

        for tier in pricing_tiers:
            price_per_million = tier.get("price_per_million", 0.0)
            tier_limit = tier.get("up_to_tokens")  # None means unlimited

            if tier_limit is None:
                # Last tier with no limit - use all remaining tokens
                tokens_in_tier = remaining_tokens
            else:
                # Limited tier - use up to the tier limit
                tokens_in_tier = min(remaining_tokens, tier_limit)

            tier_cost = (tokens_in_tier / 1_000_000) * price_per_million
            total_cost += tier_cost
            remaining_tokens -= tokens_in_tier

            if remaining_tokens <= 0:
                break

        return total_cost

    def _get_usage_data(
        self,
        response: types.GenerateContentResponse,
        app_state: State,
        metadata: "Metadata",
        start_time: float,
        api_request_start_time: float | None = None,
        first_token_time: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Extracts usage data from a GenerateContentResponse object.
        Calculates and includes cost based on pricing from YAML configuration.
        Adds cumulative tokens and cost if previous history data is available.
        Returns None if usage metadata is not present.
        """
        if not response.usage_metadata:
            log.warning(
                "Usage metadata is missing from the response. Cannot determine usage."
            )
            return None

        # Dump the raw token usage details, excluding any fields that are None.
        token_details = response.usage_metadata.model_dump(exclude_none=True)

        is_paid_api = metadata.get("is_paid_api", True)
        model_id = metadata.get("canonical_model_id", "")

        cost_details: dict[str, float] = {
            "input_cost": 0.0,
            "cache_cost": 0.0,
            "output_cost": 0.0,
            "image_output_cost": 0.0,
            "total_cost": 0.0,
        }

        if not is_paid_api:
            log.debug(
                "Using free API, costs are not applicable and will be reported as 0."
            )
        else:
            # For paid APIs, attempt to calculate cost.
            try:
                model_config = app_state._state.get("gemini_model_config", {})
                if model_id in model_config:
                    pricing = model_config[model_id].get("pricing", {})

                    if pricing:
                        total_cost = input_cost = cache_cost = output_cost = (
                            image_output_cost
                        ) = 0.0

                        # Calculate input cost (non-cached tokens)
                        prompt_tokens = token_details.get("prompt_token_count", 0)
                        cached_tokens = token_details.get(
                            "cached_content_token_count", 0
                        )
                        non_cached_input_tokens = prompt_tokens - cached_tokens

                        if non_cached_input_tokens > 0 and "input" in pricing:
                            input_cost = self._calculate_cost(
                                non_cached_input_tokens, pricing["input"]
                            )
                            total_cost += input_cost

                        # Calculate cached input cost (if applicable)
                        if cached_tokens > 0 and "caching" in pricing:
                            cache_cost = self._calculate_cost(
                                cached_tokens, pricing["caching"]
                            )
                            total_cost += cache_cost

                        # Calculate output cost (image + text separately)
                        completion_tokens = token_details.get(
                            "candidates_token_count", 0
                        )
                        if completion_tokens > 0:
                            # If there is an image generated, it would be in candidates_tokens_details
                            candidates_details = token_details.get(
                                "candidates_tokens_details", []
                            )
                            image_tokens = 0
                            for detail in candidates_details or []:
                                if detail.get("modality") == "IMAGE":
                                    image_tokens += detail.get("token_count", 0)
                            text_tokens = completion_tokens - image_tokens

                            # Calculate text output cost
                            if text_tokens > 0 and "output" in pricing:
                                output_cost += self._calculate_cost(
                                    text_tokens, pricing["output"]
                                )

                            # Calculate image output cost
                            if image_tokens > 0 and "image_output" in pricing:
                                image_output_cost += self._calculate_cost(
                                    image_tokens, pricing["image_output"]
                                )
                            elif image_tokens > 0 and "output" in pricing:
                                image_output_cost += self._calculate_cost(
                                    image_tokens, pricing["output"]
                                )

                            total_cost += output_cost + image_output_cost

                        cost_details = {
                            "input_cost": round(input_cost, 6),
                            "cache_cost": round(cache_cost, 6),
                            "output_cost": round(output_cost, 6),
                            "image_output_cost": round(image_output_cost, 6),
                            "total_cost": round(total_cost, 6),
                        }
                        log.debug(
                            f"Calculated cost for model {model_id}:",
                            payload=cost_details,
                        )
                    else:
                        log.debug(
                            f"No pricing data found for model {model_id}. Cost details will be empty."
                        )
                else:
                    log.debug(
                        f"Model {model_id} not found in config. Cost details will be empty."
                    )
            except Exception as e:
                log.warning(
                    f"Failed to calculate cost: {e}. Cost details will be empty."
                )

        input_tokens = token_details.get("prompt_token_count", 0) + token_details.get(
            "tool_use_prompt_token_count", 0
        )
        output_tokens = token_details.get(
            "candidates_token_count", 0
        ) + token_details.get("thoughts_token_count", 0)

        usage_payload = {
            "token_details": token_details,
            "cost_details": cost_details,
        }

        # --- Calculate and append cumulative usage ---
        prev_tokens = metadata.get("cumulative_tokens")
        prev_cost = metadata.get("cumulative_cost")

        # Only add cumulative data if the chain is unbroken (previous data exists)
        if prev_tokens is not None and prev_cost is not None:
            current_tokens = token_details.get("total_token_count", 0)
            current_cost = cost_details.get("total_cost", 0.0)

            usage_payload["cumulative_token_count"] = prev_tokens + current_tokens
            usage_payload["cumulative_total_cost"] = round(prev_cost + current_cost, 6)

        now = time.monotonic()
        if api_request_start_time is not None and first_token_time is not None:
            usage_payload["time_to_first_token"] = round(
                first_token_time - api_request_start_time, 2
            )

        if api_request_start_time is not None:
            usage_payload["generation_time"] = round(now - api_request_start_time, 2)

        usage_payload["completion_time"] = round(now - start_time, 2)

        # Top-level token counts required by Open WebUI's admin dashboard
        usage_payload["input_tokens"] = input_tokens
        usage_payload["output_tokens"] = output_tokens

        return usage_payload

    # endregion 1.5 Post-processing

    # region 1.6 __request__.app.state

    @staticmethod
    def _store_data_in_state(
        app_state: State,
        __metadata__: "Metadata",
        data: dict[str, Any],
    ):
        """
        Stores multiple values in the app state, namespaced by chat and message ID.
        Exits early if this is a task model (e.g. title generation) to prevent
        state bloat and interference with the main chat's filter logic.
        """
        # TODO: code a separate dataclass that handles all stuff that I need to store in app state
        if __metadata__.get("task"):
            return

        chat_id = __metadata__.get("chat_id")
        message_id = __metadata__.get("message_id")

        if not chat_id or not message_id:
            log.warning(
                "Skipping state storage: chat_id or message_id missing from metadata."
            )
            return

        for key_suffix, value in data.items():
            key = f"{key_suffix}_{chat_id}_{message_id}"
            log.debug(f"Storing data in app state with key '{key}'.")
            # Using shared `request.app.state` to pass data to Filter.outlet.
            # This is necessary because Pipe.pipe and Filter.outlet operate on different requests.
            app_state._state[key] = value

    @staticmethod
    def _get_and_clear_data_from_state(
        app_state: State,
        chat_id: str,
        message_id: str,
        key_suffix: str,
        clear_after_read: bool,
    ) -> Any | None:
        """Retrieves data from the app state using a namespaced key.

        Deletes the value only when clear_after_read is True.
        """
        key = f"{key_suffix}_{chat_id}_{message_id}"
        value = getattr(app_state, key, None)
        if value is None:
            return None

        if clear_after_read:
            log.debug(f"Retrieved and cleared data from app state for key '{key}'.")
            try:
                delattr(app_state, key)
            except AttributeError:
                # This case is unlikely but handles a race condition where the attribute might already be gone.
                log.warning(
                    f"State key '{key}' was already gone before deletion attempt."
                )
        else:
            log.debug(
                f"Retrieved data from app state for key '{key}' without clearing it."
            )
        return value

    async def _wait_for_state_value(
        self,
        app_state: State,
        key: str,
        timeout: float = 10.0,
        poll_interval: float = 0.1,
        description: str = "state value",
    ) -> Any:
        """Polls app state until the requested key is populated or timeout expires."""
        if value := app_state._state.get(key):
            return value

        log.warning(
            f"'{key}' not found in state ({description}). Waiting up to {timeout}s for companion filter inlet..."
        )
        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            await asyncio.sleep(poll_interval)
            if value := app_state._state.get(key):
                log.debug(
                    f"'{key}' became available after {time.monotonic() - start_time:.2f}s."
                )
                return value

        raise ValueError(
            f"FATAL: '{key}' ({description}) not found in app state after {timeout}s timeout. "
            "Please ensure the Gemini Manifold Companion filter is installed and enabled."
        )

    async def _get_event_emitter(
        self,
        app_state: State,
        chat_id: str | None,
        message_id: str | None,
        is_task: bool,
    ) -> "EventEmitter":
        """Resolves the appropriate event emitter from app state for tasks or regular generation."""
        if is_task:
            return await self._wait_for_state_value(
                app_state,
                key="gemini_dummy_event_emitter",
                description="dummy event emitter for task model",
            )

        event_emitter = (
            self._get_and_clear_data_from_state(
                app_state,
                chat_id,
                message_id,
                key_suffix="gemini_event_emitter",
                clear_after_read=False,
            )
            if chat_id and message_id
            else None
        )
        if event_emitter is not None:
            return event_emitter

        log.warning(
            "No event emitter found in state for this request. Companion filter's inlet did not run? "
            "Falling back to dummy event emitter."
        )
        if dummy_emitter := app_state._state.get("gemini_dummy_event_emitter"):
            return dummy_emitter

        raise ValueError(
            "Neither request event emitter nor dummy event emitter is available in app state."
        )

    # endregion 1.6 __request__.app.state

    # region 1.7 Utility helpers

    async def _determine_execution_order(
        self,
        valves: "Pipe.Valves",
        __metadata__: "Metadata",
        model_config: dict[str, Any],
        features: "Features",
    ) -> list[str]:
        """
        Calculates the sequence of execution tiers (free, paid, vertex).
        Returns an empty list if no valid routing configuration is found.
        """
        model_id = __metadata__.get("canonical_model_id", "")
        has_free_key = bool(valves.GEMINI_FREE_API_KEY)
        has_paid_key = bool(valves.GEMINI_PAID_API_KEY)

        # Retrieve Toggle Statuses
        vertex_available, vertex_toggled_on = await self._get_toggleable_feature_status(
            "gemini_vertex_ai_toggle", __metadata__
        )
        # Vertex is only viable if toggled on AND we have a project ID
        can_use_vertex = (
            vertex_available and vertex_toggled_on and bool(valves.VERTEX_PROJECT)
        )

        paid_toggle_available, paid_toggled_on = await self._get_toggleable_feature_status(
            "gemini_paid_api", __metadata__
        )

        task_type = __metadata__.get("task")
        routing_strategy = valves.TASK_MODEL_ROUTING if task_type else "match_main"

        is_free_eligible = self._check_free_tier_eligibility(
            model_id, model_config, features
        )

        execution_order: list[str] = []

        match routing_strategy:
            case "only_free":
                log.debug("Task model routing override: only_free")
                if has_free_key:
                    if not is_free_eligible:
                        log.warning(
                            f"Task model '{model_id}' forced to 'only_free' but is ineligible. "
                            "Expect an upstream API error."
                        )
                    execution_order = ["free"]
                else:
                    log.error(
                        "Routing strategy 'only_free' requested, but no Free API Key is configured."
                    )

            case "free_fallback":
                log.debug("Task model routing override: free_fallback")
                # 1. Attempt Free if keys exist and model is eligible
                if has_free_key and is_free_eligible:
                    execution_order.append("free")

                # 2. Add Paid fallbacks
                if can_use_vertex:
                    execution_order.append("vertex")
                elif has_paid_key:
                    execution_order.append("paid")

                if not execution_order:
                    log.warning(
                        "Strategy 'free_fallback' could not find any viable tiers (check keys/eligibility)."
                    )

            case "only_paid":
                log.debug("Task model routing override: only_paid")
                if can_use_vertex:
                    execution_order = ["vertex"]
                elif has_paid_key:
                    execution_order = ["paid"]
                else:
                    log.error(
                        "Routing strategy 'only_paid' requested, but neither Vertex AI nor Paid API is configured."
                    )

            case _:
                # Default Routing Logic ("match_main")
                if can_use_vertex:
                    execution_order = ["vertex"]
                elif paid_toggle_available and paid_toggled_on:
                    if has_paid_key:
                        execution_order = ["paid"]
                    else:
                        log.error(
                            "Paid API toggle is ON, but GEMINI_PAID_API_KEY is missing."
                        )
                else:
                    # Logic for standard/un-toggled flow
                    if has_free_key:
                        if is_free_eligible:
                            execution_order = ["free"]
                            if valves.ENABLE_FREE_TIER_FALLBACK:
                                if can_use_vertex:
                                    execution_order.append("vertex")
                                elif has_paid_key:
                                    execution_order.append("paid")
                        else:
                            # If model isn't free-eligible, jump straight to paid tiers.
                            # If no paid tier exists, we try free anyway to let the API return the specific error.
                            if can_use_vertex:
                                execution_order = ["vertex"]
                            elif has_paid_key:
                                execution_order = ["paid"]
                            else:
                                log.warning(
                                    f"Model '{model_id}' is ineligible for free tier and no paid keys found."
                                )
                    elif can_use_vertex:
                        execution_order = ["vertex"]
                    elif has_paid_key:
                        execution_order = ["paid"]

        log.debug(
            f"Routing strategy for {model_id} ({routing_strategy}): {execution_order}"
        )
        return execution_order

    def _resolve_custom_params(
        self, body: "Body", __metadata__: "Metadata"
    ) -> dict[str, Any]:
        """
        Resolves custom parameters from the model page and chat controls.
        Chat control settings usually take precedence, but we ignore them
        if this is a task model (e.g., generating titles or tags) to ensure
        these independent calls aren't negatively affected by user chat settings.
        """
        known_body_keys = {
            "stream",
            "model",
            "messages",
            "files",
            "options",
            "stream_options",
        }
        merged_params = {
            key: value for key, value in body.items() if key not in known_body_keys
        }
        log.debug("Model page parameters extracted from body:", payload=merged_params)

        if __metadata__.get("task"):
            log.debug(
                f"Task model detected (task: {__metadata__.get('task')}). Ignoring chat control parameters."
            )
            return merged_params

        chat_control_params = __metadata__.get("chat_control_params", {})
        if chat_control_params:
            log.debug(
                "Chat control parameters extracted from metadata:",
                payload=chat_control_params,
            )
            merged_params.update(chat_control_params)

        return merged_params

    @staticmethod
    async def _get_toggleable_feature_status(
        filter_id: str,
        __metadata__: "Metadata",
    ) -> tuple[bool, bool]:
        """
        Checks the complete status of a toggleable filter (function).

        This function performs a series of checks to determine if a feature
        is available for use and if the user has activated it.

        1. Checks if the filter is installed.
        2. Checks if the filter's master toggle is active in the Functions dashboard.
        3. Checks if the filter is enabled for the current model (or is global).
        4. Checks if the user has toggled the feature ON for the current request.

        Args:
            filter_id: The ID of the filter to check.
            __metadata__: The metadata object for the current request.

        Returns:
            A tuple (is_available: bool, is_toggled_on: bool).
            - is_available: True if the filter is installed, active, and configured for the model.
            - is_toggled_on: True if the user has the toggle ON in the UI for this request.
        """
        # 1. Check if the filter is installed
        f = await Functions.get_function_by_id(filter_id)
        if not f:
            log.warning(
                f"The '{filter_id}' filter is not installed. "
                "Install it to use the corresponding front-end toggle."
            )
            return (False, False)

        # 2. Check if the master toggle is active
        if not f.is_active:
            log.warning(
                f"The '{filter_id}' filter is installed but is currently disabled in the "
                "Functions dashboard (master toggle is off). Enable it to make it available."
            )
            return (False, False)

        # 3. Check if the filter is enabled for the model or is global
        model_info = __metadata__.get("model", {}).get("info", {})
        model_filter_ids = model_info.get("meta", {}).get("filterIds", [])
        is_enabled_for_model = filter_id in model_filter_ids or f.is_global

        log.debug(
            f"Checking model enablement for '{filter_id}': in_model_filters={filter_id in model_filter_ids}, "
            f"is_global={f.is_global} -> is_enabled={is_enabled_for_model}"
        )

        if not is_enabled_for_model:
            # This is a configuration issue, not a user-facing warning. Debug is appropriate.
            model_id = __metadata__.get("model", {}).get("id", "Unknown")
            log.debug(f"Filter '{filter_id}' is not enabled for model '{model_id}' and is not global.")
            return (False, False)

        # 4. Check if the user has toggled the feature ON for this request
        user_toggled_ids = __metadata__.get("filter_ids", [])
        is_toggled_on = filter_id in user_toggled_ids

        if is_toggled_on:
            log.info(
                f"Feature '{filter_id}' is available and enabled by the front-end toggle for this request."
            )
        else:
            log.debug(
                f"Feature '{filter_id}' is available but not enabled by the front-end toggle for this request."
            )

        return (True, is_toggled_on)

    @staticmethod
    def _get_merged_valves(
        default_valves: "Pipe.Valves",
        user_valves: "Pipe.UserValves | None",
        user_email: str,
    ) -> "Pipe.Valves":
        """
        Merges UserValves into a base Valves configuration.

        The general rule is that if a field in UserValves is not None or an empty
        string, it overrides the corresponding field in the default_valves.
        Otherwise, the default_valves field value is used.

        Exceptions:
        - If default_valves.USER_MUST_PROVIDE_AUTH_CONFIG is True and the user is
          not on the AUTH_WHITELIST, then GEMINI_FREE_API_KEY and
          GEMINI_PAID_API_KEY in the merged result will be taken directly from
          user_valves (even if they are None), and Vertex AI usage is disabled.

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
            log.info(
                f"User '{user_email}' is required to provide their own authentication credentials due to USER_MUST_PROVIDE_AUTH_CONFIG=True."
                " Admin-provided API keys and Vertex AI settings will not be used."
            )
            # If USER_MUST_PROVIDE_AUTH_CONFIG is True and user is not in the whitelist,
            # they must provide their own API keys.
            # They are also disallowed from using the admin's Vertex AI configuration.
            merged_data["GEMINI_FREE_API_KEY"] = user_valves.GEMINI_FREE_API_KEY
            merged_data["GEMINI_PAID_API_KEY"] = user_valves.GEMINI_PAID_API_KEY
            merged_data["VERTEX_PROJECT"] = None
            merged_data["USE_VERTEX_AI"] = False

        # Create a new Valves instance with the merged data.
        # Pydantic will validate the data against the Valves model definition during instantiation.
        return Pipe.Valves(**merged_data)

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

    def _check_companion_filter_version(self, features: "Features | dict") -> None:
        """
        Checks for the presence and version compatibility of the Gemini Manifold Companion filter.
        Logs warnings if the filter is missing or outdated.
        """
        companion_version = features.get("gemini_manifold_companion_version")

        if companion_version is None:
            log.warning(
                "Gemini Manifold Companion filter not detected. "
                "Since v2.0.0, this pipe requires the companion filter to be installed and active. "
                "Please install the companion filter or ensure it is active "
                "for this model (or activate it globally)."
            )
        else:
            # Comparing tuples of integers is a robust way to handle versions like '1.10.0' vs '1.2.0'.
            try:
                companion_v_tuple = tuple(map(int, companion_version.split(".")))
                recommended_v_tuple = tuple(
                    map(int, RECOMMENDED_COMPANION_VERSION.split("."))
                )

                if companion_v_tuple < recommended_v_tuple:
                    log.warning(
                        f"The installed Gemini Manifold Companion filter version ({companion_version}) is older than "
                        f"the recommended version ({RECOMMENDED_COMPANION_VERSION}). "
                        "Some features may not work as expected. Please update the filter."
                    )
                else:
                    log.debug(
                        f"Gemini Manifold Companion filter detected with version: {companion_version}"
                    )
            except (ValueError, TypeError):
                # This handles cases where the version string is malformed (e.g., '1.a.0').
                log.error(
                    f"Could not parse companion version string: '{companion_version}'. Version check skipped."
                )

    # endregion 1.7 Utility helpers

    # endregion 1. Helper methods inside the Pipe class
