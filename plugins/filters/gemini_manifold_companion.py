"""
title: Gemini Manifold Companion
id: gemini_manifold_companion
description: A companion filter for "Gemini Manifold google_genai" pipe providing enhanced functionality.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 2.1.0
"""

VERSION = '2.1.0'

# This filter can detect that a feature like web search or code execution is enabled in the front-end,
# set the feature back to False so Open WebUI does not run it's own logic and then
# pass custom values to "Gemini Manifold google_genai" that signal which feature was enabled and intercepted.

# FIXME: Add `from __future__ import annotations` at the top of this file (repo
# convention, see AGENTS.md). Until then the file relies on string annotations.

import asyncio
import functools
import importlib.metadata
import sys
import time
import urllib.request
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final, Literal, cast, get_args, get_origin

import aiohttp
import yaml
from fastapi import Request
from fastapi.datastructures import State
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, create_model


def _unload_stale_modules() -> None:
    """
    Open WebUI pip installs frontmatter `requirements` into the already running server, which can replace
    packages on disk while their old versions are still in `sys.modules`. Importing `google.genai` would then
    mix modules from both versions and fail, e.g. with `cannot import name 'OP_BINARY' from 'websockets.frames'`
    (`google-genai` requires `websockets<17`, so pip downgrades it) or with
    `module 'google.genai.types' has no attribute ...` (`google-genai` itself got upgraded).
    Dropping the stale modules makes the import load a consistent set of modules from disk.
    Code that already imported the old modules (e.g. uvicorn) keeps its own references to them.
    """
    unloaded = False
    # Distribution name, package name and the module whose `__version__` tells which version is loaded.
    for distribution, package, version_module in (
        ('websockets', 'websockets', 'websockets'),
        ('google-genai', 'google.genai', 'google.genai.version'),
    ):
        loaded = sys.modules.get(version_module)
        if loaded is None:
            continue
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        if getattr(loaded, '__version__', None) == installed_version:
            continue
        for name in [n for n in sys.modules if n == package or n.startswith(f'{package}.')]:
            del sys.modules[name]
        # `from google import genai` would otherwise still return the old module set on the parent package.
        parent_name, _, child = package.rpartition('.')
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, child):
            delattr(parent, child)
        unloaded = True
    if unloaded:
        importlib.invalidate_caches()


# Must run before `google.genai` is imported.
_unload_stale_modules()

from google.genai import types

if TYPE_CHECKING:
    # My personal types in a separate file for more robustness.
    from utils.manifold_types import (
        Body,
        ChatCompletionEvent,
        CitationEvent,
        Event,
        Features,
        Metadata,
        NotificationEvent,
        Source,
        SourceMetadata,
        StatusEvent,
        TextContent,
        UserData,
    )

# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main log.
log = logger.bind(auditable=False)


class EventEmitter:
    """
    A unified, thread-safe event emitter for Open WebUI plugins.
    Uses internal queues to guarantee ordered, non-blocking delivery of websocket events.
    Includes an idle timeout to prevent memory leaks from orphaned instances.
    """

    def __init__(
        self,
        event_emitter: Callable[['Event'], Awaitable[None]] | None,
        *,
        status_mode: str = 'visible',
        idle_timeout: float = 3600.0,
    ):
        self._emitter = event_emitter
        self.status_mode = status_mode
        self.start_time = time.monotonic()

        # Used by external garbage collection to detect dead instances
        self.is_abandoned: bool = False
        self._idle_timeout = idle_timeout

        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._toast_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        self._worker_task: asyncio.Task | None = None
        self._toast_worker_task: asyncio.Task | None = None

        if self._emitter is not None:
            self._worker_task = asyncio.create_task(self._process_queue(self._queue))
            self._toast_worker_task = asyncio.create_task(self._process_queue(self._toast_queue))

    async def _process_queue(self, queue: asyncio.Queue['Event | None']) -> None:
        """
        A generic consumer for event queues.
        Processes items sequentially until a None poison pill is encountered
        or the idle timeout is reached.
        """
        while True:
            try:
                # The timeout only applies to the waiting period for new events.
                # If an event takes a long time to process below, it won't trigger this.
                event = await asyncio.wait_for(queue.get(), timeout=self._idle_timeout)
            except TimeoutError:
                # If no events arrive within the timeout window, assume the parent
                # request was unexpectedly dropped. Set the flag for external cleanup.
                self.is_abandoned = True
                break

            if event is None:
                queue.task_done()
                break

            if self._emitter:
                try:
                    await self._emitter(event)
                except Exception:
                    log.exception('Error in EventEmitter background worker')

            queue.task_done()

    def _enqueue(self, event: 'Event', is_toast: bool = False) -> None:
        """Pushes a new event into the appropriate queue without blocking."""
        if self._emitter is None:
            return

        target_queue = self._toast_queue if is_toast else self._queue
        target_queue.put_nowait(event)

    async def flush(self) -> None:
        """Blocks until all currently queued events across all queues have been processed."""
        await asyncio.gather(self._queue.join(), self._toast_queue.join())

    async def shutdown(self) -> None:
        """Sends the poison pill to all active workers and waits for them to finish."""
        tasks_to_await = []

        if self._worker_task and not self._worker_task.done():
            self._queue.put_nowait(None)
            tasks_to_await.append(self._worker_task)

        if self._toast_worker_task and not self._toast_worker_task.done():
            self._toast_queue.put_nowait(None)
            tasks_to_await.append(self._toast_worker_task)

        if tasks_to_await:
            await asyncio.gather(*tasks_to_await)

    def emit_toast(
        self,
        msg: str,
        type: Literal['info', 'success', 'warning', 'error'] = 'info',
    ) -> None:
        event: NotificationEvent = {
            'type': 'notification',
            'data': {'type': type, 'content': msg},
        }
        self._enqueue(event, is_toast=True)

    def emit_status(
        self,
        description: str,
        done: bool = False,
        hidden: bool = False,
        *,
        is_successful_finish: bool = False,
        is_thought: bool = False,
        indent_level: int = 0,
    ) -> None:
        if self.status_mode == 'disable':
            return
        if self.status_mode == 'hidden_compact' and is_thought:
            return

        if 'visible_timed' in self.status_mode:
            elapsed = time.monotonic() - self.start_time
            description = f'{description} (+{elapsed:.2f}s)'

        final_hidden = hidden or (self.status_mode in ('hidden_compact', 'hidden_detailed') and is_successful_finish)

        if not final_hidden and indent_level > 0:
            description = f'{"- " * indent_level}{description}'

        event: StatusEvent = {
            'type': 'status',
            'data': {'description': description, 'done': done, 'hidden': final_hidden},
        }
        self._enqueue(event)

    def emit_completion(
        self,
        content: str | None = None,
        done: bool = False,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        data: dict[str, Any] = {'done': done}
        if content is not None:
            data['content'] = content
        if error is not None:
            data['error'] = {'detail': error}
        if usage is not None:
            data['usage'] = usage

        event: ChatCompletionEvent = {
            'type': 'chat:completion',
            'data': cast(Any, data),
        }
        self._enqueue(event)

    def emit_sources(self, source_data: 'Source') -> None:
        event: CitationEvent = {
            'type': 'source',
            'data': {
                'source': source_data['source'],
                'document': source_data['document'],
                'metadata': source_data['metadata'],
            },
        }
        self._enqueue(event)

    def emit_error(self, error_msg: str, exception: bool = True) -> None:
        log.opt(depth=1, exception=exception).error(error_msg)
        self.emit_completion(error=f'\n{error_msg}', done=True)

    def emit_grounding_queries(self, queries: list[str]) -> None:
        if not queries:
            return
        event: StatusEvent = {
            'type': 'status',
            'data': {
                'action': 'web_search_queries_generated',
                'queries': queries,
                'done': False,
            },
        }
        self._enqueue(event)


_VALVE_DESCRIPTIONS: Final[dict[str, str]] = {
    'USE_PERMISSIVE_SAFETY': ('Whether to request relaxed safety filtering for Gemini models.'),
    'BYPASS_BACKEND_RAG': (
        "Bypass Open WebUI's built-in RAG processing and pass documents directly to the Gemini API.\n\n"
        '*Note: Temporary chats (`local`) cannot bypass RAG and will fallback to default RAG.*'
    ),
    'MODEL_CONFIG_PATH': (
        'Publicly accessible URL (`http://` or `https://`) to the YAML file containing model definitions and capabilities.'
    ),
    'URL_RESOLVE_TIMEOUT': ('Timeout in seconds for resolving grounding source web URLs.'),
    'URL_RESOLVE_MAX_RETRIES': ('Maximum number of retry attempts to resolve grounding URLs before giving up.'),
    'URL_RESOLVE_BASE_DELAY': (
        'Initial delay in seconds between retries when resolving grounding URLs (uses exponential backoff).'
    ),
    'STATUS_EMISSION_BEHAVIOR': (
        'Controls status message visibility and detail level in the chat interface:\n'
        '- `disable`: Suppress all status messages.\n'
        '- `hidden_compact`: Hide final completion status; hide thinking details.\n'
        '- `hidden_detailed`: Hide final completion status; include detailed thinking steps.\n'
        '- `visible`: Show all status messages.\n'
        '- `visible_timed`: Show all status messages with execution timestamps.'
    ),
}


def _format_valve_desc(text: str, default: Any = None, is_user: bool = False) -> str:
    """Formats Markdown descriptions for Valves and UserValves fields."""
    text = text.strip()
    sep = '\n\n---\n\n'
    if is_user:
        return f"{text}\n\n*If not set, the admin's setting is used.*{sep}"
    formatted_default = f'`{default}`' if default is not None else '`None`'
    return f'{text}\n\n**Default:** {formatted_default}{sep}'


def _admin_field(name: str, default: Any = None, admin_section_start: bool = False) -> Any:
    """Helper to construct a Pydantic Field with default value and formatted admin description."""
    raw_desc = _VALVE_DESCRIPTIONS.get(name, '')
    desc = _format_valve_desc(raw_desc, default=default)
    if admin_section_start:
        desc = f'{desc}### Admin-Only Options'
    return Field(default=default, description=desc)


class _SharedValves(BaseModel):
    """Base model holding user-configurable options shared between Valves and UserValves."""

    USE_PERMISSIVE_SAFETY: bool = _admin_field('USE_PERMISSIVE_SAFETY', False)
    BYPASS_BACKEND_RAG: bool = _admin_field('BYPASS_BACKEND_RAG', True)
    MODEL_CONFIG_PATH: str = _admin_field(
        'MODEL_CONFIG_PATH',
        'https://raw.githubusercontent.com/suurt8ll/open_webui_functions/master/plugins/pipes/gemini_models.yaml',
    )
    URL_RESOLVE_TIMEOUT: int = _admin_field('URL_RESOLVE_TIMEOUT', 10)
    URL_RESOLVE_MAX_RETRIES: int = _admin_field('URL_RESOLVE_MAX_RETRIES', 3)
    URL_RESOLVE_BASE_DELAY: float = _admin_field('URL_RESOLVE_BASE_DELAY', 0.5)
    STATUS_EMISSION_BEHAVIOR: Literal[
        'disable',
        'hidden_compact',
        'hidden_detailed',
        'visible',
        'visible_timed',
    ] = _admin_field('STATUS_EMISSION_BEHAVIOR', 'hidden_detailed')


def _generate_user_valves(shared_cls: type[BaseModel]) -> type[BaseModel]:
    """Generates Filter.UserValves model from _SharedValves with Optional fields and user descriptions."""
    fields: dict[str, Any] = {}
    for name, field_info in shared_cls.model_fields.items():
        raw_desc = _VALVE_DESCRIPTIONS.get(name, '')
        user_desc = _format_valve_desc(raw_desc, is_user=True)

        ann = field_info.annotation
        if get_origin(ann) is Literal:
            args = get_args(ann)
            user_ann = (
                Literal[(*args, '')] | None if '' not in args else ann | None  # type: ignore[operator]
            )
        elif ann is not None:
            user_ann = ann | None
        else:
            user_ann = Any

        fields[name] = (user_ann, Field(default=None, description=user_desc))

    return create_model('UserValves', __base__=shared_cls, **fields)  # type: ignore[call-overload]


class Filter:
    class Valves(_SharedValves):
        """Admin-level configuration for the filter.

        Currently adds no extra fields, it only inherits the shared options from
        `_SharedValves`. The subclass exists so that admin-only options (which
        user valves and request parameters cannot override) can be added later
        without reshaping the class hierarchy.
        """

    UserValves = _generate_user_valves(_SharedValves)

    def __init__(self):
        self.valves = self.Valves()
        log.success('Function has been initialized.')

    def inlet(
        self,
        body: 'Body',
        __request__: Request,
        __metadata__: 'Metadata',
        __event_emitter__: Callable[['Event'], Awaitable[None]],
        __user__: 'UserData',
    ) -> 'Body':
        """Modifies the incoming request payload before it's sent to the LLM. Operates on the `form_data` dictionary."""
        log.debug(f'inlet method has been called. Gemini Manifold Companion version is {VERSION}')

        canonical_model_name, is_manifold = self._get_model_name(body)

        # Exit early if this is not a model that the companion is intended to be tied to.
        if not is_manifold:
            log.debug('Returning the original body object because the request is not for a Gemini Manifold pipe model.')
            return body

        # Resolve the final option values from admin valves, user valves and any
        # parameters that Open WebUI has already merged into the request body.
        user_valves = __user__.get('valves') if isinstance(__user__, dict) else None
        valves = _resolve_options(self.valves, user_valves, body)

        app_state: State = __request__.app.state

        # Perform housekeeping before creating new state objects.
        # This ensures that even if a pipe/filter pair crashes or hangs,
        # the memory footprint doesn't grow indefinitely over time.
        self._cleanup_event_emitters(app_state)

        emitter = EventEmitter(__event_emitter__, status_mode=valves.STATUS_EMISSION_BEHAVIOR)
        # The outlet receives a body that OWUI has rebuilt from scratch, without any
        # request parameters. Ferry the inlet-resolved options (which already include
        # model page/chat parameter overrides) through the shared app state instead.
        self._store_data_in_state(
            app_state,
            __metadata__,
            {
                'gemini_event_emitter': emitter,
                'gemini_resolved_valves': valves,
            },
        )
        app_state._state['gemini_dummy_event_emitter'] = EventEmitter(None)

        # Load and store model configuration in app state
        log.debug('Loading model configuration...')
        model_config = self._load_model_config(valves.MODEL_CONFIG_PATH)
        app_state._state['gemini_model_config'] = model_config
        log.debug(f'Stored model config in app state with {len(model_config)} model(s).')

        # Check if the model supports grounding or code execution using YAML config
        is_grounding_model = self._check_model_capability(canonical_model_name, model_config, 'search_grounding')
        is_code_exec_model = self._check_model_capability(canonical_model_name, model_config, 'code_execution')
        log.debug(f'{is_grounding_model=}, {is_code_exec_model=}')

        # Open WebUI snapshots `body['features']` into `metadata['features']` BEFORE
        # filters run, so when the request has no `features` field the snapshot is a
        # detached `{}` and anything written only to `metadata['features']` would be
        # dropped when Open WebUI later updates `metadata` with `body['features']`.
        # Assigning a single shared dict to both guarantees the flags below survive.
        # The pipe reads `__metadata__['features']`, while Open WebUI itself pops
        # `body['features']` after the filters to decide native web_search and
        # code_interpreter handling.
        features = body.get('features')
        if not isinstance(features, dict):
            features = cast('Features', {})
            body['features'] = features
        __metadata__['features'] = features
        log.debug('body.features:', payload=features)

        # Add the companion version to the payload for the pipe to consume.
        features['gemini_manifold_companion_version'] = VERSION

        if is_grounding_model:
            web_search_enabled = features.get('web_search', False)
            if web_search_enabled:
                log.info('Search feature is enabled, disabling it and adding custom feature called google_search_tool.')
                # Disable web_search
                features['web_search'] = False
                features['google_search_tool'] = True
        if is_code_exec_model:
            code_execution_enabled = features.get('code_interpreter', False)
            if code_execution_enabled:
                log.info(
                    'Code interpreter feature is enabled, disabling it and adding custom feature called google_code_execution.'
                )
                # Disable code_interpreter
                features['code_interpreter'] = False
                features['google_code_execution'] = True
        if valves.USE_PERMISSIVE_SAFETY:
            log.info('Adding permissive safety settings to `__metadata__`.')
            __metadata__['safety_settings'] = self._get_permissive_safety_settings(canonical_model_name)
        if valves.BYPASS_BACKEND_RAG:
            chat_id = __metadata__.get('chat_id')
            chat_id_str = chat_id if isinstance(chat_id, str) else ''
            is_temp_chat = not chat_id_str or 'temporary' in chat_id_str or 'local' in chat_id_str
            if is_temp_chat:
                # TODO toast notification
                log.warning(
                    "Bypassing Open WebUI's RAG is not possible for temporary chats. "
                    'The Manifold pipe requires a database entry to access uploaded files, '
                    "which temporary chats do not have. Falling back to Open WebUI's RAG."
                )
                features['upload_documents'] = False
            else:
                log.info(
                    'BYPASS_BACKEND_RAG is enabled, bypassing Open WebUI RAG to let the Manifold pipe handle documents.'
                )
                if files := body.get('files'):
                    log.info(f'Removing {len(files)} files from the Open WebUI RAG pipeline.')
                    body['files'] = []
                features['upload_documents'] = True
        else:
            log.info("BYPASS_BACKEND_RAG is disabled. Open WebUI's RAG will be used if applicable.")
            features['upload_documents'] = False

        # TODO: Filter out the citation markers here.

        log.debug('inlet method has finished.')
        return body

    def stream(self, event: dict) -> dict:
        """Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data."""
        return event

    async def outlet(
        self,
        body: 'Body',
        __request__: Request,
        __metadata__: 'Metadata',
        __event_emitter__: Callable[['Event'], Awaitable[None]],
        __user__: 'UserData',
    ) -> 'Body':
        """Modifies the complete response payload after it's received from the LLM. Operates on the final `body` dictionary."""

        log.debug('outlet method has been called.')

        chat_id = __metadata__.get('chat_id', '')
        message_id = __metadata__.get('message_id', '')
        app_state: State = __request__.app.state

        # Prefer the inlet-resolved options (which already include model page/chat
        # parameter overrides) ferried through the app state. Falls back to resolving
        # admin + user valves only for requests where the inlet did not store state
        # (task models, missing ids, stale state).
        stored_valves = (
            self._get_and_clear_data_from_state(
                app_state,
                chat_id,
                message_id,
                key_suffix='gemini_resolved_valves',
                clear_after_read=False,
            )
            if chat_id and message_id
            else None
        )
        if isinstance(stored_valves, ResolvedOptions):
            valves: Filter.Valves = stored_valves
        else:
            log.debug('No resolved valves found in state for this request. Resolving from admin and user valves only.')
            user_valves = __user__.get('valves') if isinstance(__user__, dict) else None
            valves = _resolve_options(self.valves, user_valves, None)

        log.debug(f'Checking for attributes for message {message_id} in request state.')
        stored_metadata: types.GroundingMetadata | None = (
            self._get_and_clear_data_from_state(app_state, chat_id, message_id, 'grounding', True)
            if chat_id and message_id
            else None
        )

        event_emitter: EventEmitter | None = (
            self._get_and_clear_data_from_state(
                app_state,
                chat_id,
                message_id,
                key_suffix='gemini_event_emitter',
                clear_after_read=True,
            )
            if chat_id and message_id
            else None
        )
        if not event_emitter:
            log.debug(
                "No event emitter found in state for this request. Companion filter's inlet did not run? Event emissions will be disabled for this request."
            )
            # TODO: any better way how to do this that does not require a dummy empty emitter?
            event_emitter = app_state._state.get('gemini_dummy_event_emitter')
        if not event_emitter:
            raise ValueError(
                'No event emitter available. This is unexpected as the dummy event emitter should always be present.'
            )

        if stored_metadata:
            log.info('Found grounding metadata, processing citations.')
            log.trace('Stored grounding metadata:', payload=stored_metadata)

            current_content = body['messages'][-1]['content']
            if isinstance(current_content, list):
                text_to_use = ''
                for item in current_content:
                    if item.get('type') == 'text':
                        item = cast('TextContent', item)
                        text_to_use = item['text']
                        break
            else:
                text_to_use = current_content

            # Insert citation markers into the response text
            cited_text = self._get_text_w_citation_markers(
                stored_metadata,
                text_to_use,
            )

            if cited_text:
                target_msg = body['messages'][-1]
                content = target_msg.get('content')

                # 1. Update message content
                if isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            item = cast('TextContent', item)
                            item['text'] = cited_text
                            break
                else:
                    target_msg['content'] = cited_text

                # 2. Update message output array if present (used by UI for reasoning/structured models)
                if 'output' in target_msg and isinstance(target_msg['output'], list):
                    for out_item in target_msg['output']:
                        if isinstance(out_item, dict) and out_item.get('type') == 'message':
                            out_content = out_item.get('content')
                            if isinstance(out_content, list):
                                for sub_item in out_content:
                                    if isinstance(sub_item, dict) and sub_item.get('type') in ('output_text', 'text'):
                                        sub_item['text'] = cited_text

            # Emit status event with search queries before resolving URLs
            if stored_metadata.web_search_queries:
                event_emitter.emit_grounding_queries(stored_metadata.web_search_queries)
            else:
                log.debug('Grounding metadata does not contain any search queries.')

            # Emit sources to the front-end.
            gs_supports = stored_metadata.grounding_supports
            gs_chunks = stored_metadata.grounding_chunks
            if gs_supports and gs_chunks:
                await self._resolve_and_emit_sources(
                    grounding_chunks=gs_chunks,
                    supports=gs_supports,
                    emitter=event_emitter,
                    valves=valves,
                )
                event_emitter.emit_status('This response was grounded with a Google tool', done=True)
            else:
                msg = "Grounding metadata was found but it's missing grounding supports or chunks. The response is likely not grounded."
                log.info(msg)
                event_emitter.emit_status(msg, done=True)
        else:
            log.info('No grounding metadata found in request state.')

        log.debug('outlet method has finished.')
        return body

    # region 1. Helper methods inside the Filter class

    # region 1.1 Add citations

    def _get_text_w_citation_markers(
        self,
        grounding_metadata: types.GroundingMetadata,
        raw_str: str,
    ) -> str | None:
        """
        Returns the model response with citation markers.
        Thoughts, if present as THOUGHT_START_TAG...THOUGHT_END_TAG at the beginning of raw_str,
        are preserved but excluded from the citation indexing process.
        Everything up to the *last* THOUGHT_END_TAG tag is considered part of the thought.
        """

        supports = grounding_metadata.grounding_supports
        grounding_chunks = grounding_metadata.grounding_chunks
        if not supports or not grounding_chunks:
            log.info(
                "Grounding metadata missing supports or chunks, can't insert citation markers. "
                'Response was probably just not grounded.'
            )
            return None

        log.trace('raw_str:', payload=raw_str, _log_truncation_enabled=False)

        thought_prefix = ''
        content_for_citation_processing = raw_str

        THOUGHT_START_TAG = '<details'
        THOUGHT_END_TAG = '</details>\n'

        if raw_str.startswith(THOUGHT_START_TAG):
            last_end_thought_tag_idx = raw_str.rfind(THOUGHT_END_TAG)
            if last_end_thought_tag_idx != -1 and last_end_thought_tag_idx >= len(THOUGHT_START_TAG) - 1:
                thought_block_end_offset = last_end_thought_tag_idx + len(THOUGHT_END_TAG)
                thought_prefix = raw_str[:thought_block_end_offset]
                content_for_citation_processing = raw_str[thought_block_end_offset:]
                log.info(
                    'Model thoughts detected at the beginning of the response. '
                    'Citations will be processed on the content following the last thought block.'
                )
            else:
                log.warning(
                    'Detected THOUGHT_START_TAG at the start of raw_str without a subsequent closing THOUGHT_END_TAG '
                    'or a malformed thought block. The entire raw_str will be processed for citations. '
                    'This might lead to incorrect marker placement if thoughts were intended and indices '
                    'are relative to content after thoughts.'
                )

        processed_content_part_with_markers = content_for_citation_processing

        if content_for_citation_processing:
            try:
                modified_content_bytes = bytearray(content_for_citation_processing.encode('utf-8'))
                for support in reversed(supports):
                    segment = support.segment
                    indices = support.grounding_chunk_indices
                    if not (indices is not None and segment and segment.end_index is not None):
                        log.debug(f'Skipping support due to missing data: {support}')
                        continue
                    end_pos = segment.end_index
                    if not (0 <= end_pos <= len(modified_content_bytes)):
                        log.warning(
                            f'Support segment end_index ({end_pos}) is out of bounds for the processable content '
                            f'(length {len(modified_content_bytes)} bytes after potential thought stripping). '
                            f"Content (first 50 chars): '{content_for_citation_processing[:50]}...'. Skipping this support. Support: {support}"
                        )
                        continue
                    citation_markers = ''.join(f'[{index + 1}]' for index in indices)
                    encoded_citation_markers = citation_markers.encode('utf-8')
                    modified_content_bytes[end_pos:end_pos] = encoded_citation_markers
                processed_content_part_with_markers = modified_content_bytes.decode('utf-8')
            except Exception as e:
                log.error(
                    f'Error injecting citation markers into content: {e}. '
                    f'Using content part (after potential thought stripping) without new markers.'
                )
        else:
            if raw_str and not content_for_citation_processing:
                log.info(
                    'Content for citation processing is empty (e.g., raw_str contained only thoughts). '
                    'No citation markers will be injected.'
                )
            elif not raw_str:
                log.warning('Raw string is empty, cannot inject citation markers.')

        final_result_str = thought_prefix + processed_content_part_with_markers
        log.trace('final_result_str:', payload=final_result_str, _log_truncation_enabled=False)
        return final_result_str

    async def _resolve_url(self, session: aiohttp.ClientSession, url: str, valves: 'Filter.Valves') -> tuple[str, bool]:
        """
        Resolves a given URL using values from Valves.
        Returns the final URL and a boolean indicating success.
        """
        if not url:
            return '', False

        timeout = aiohttp.ClientTimeout(total=valves.URL_RESOLVE_TIMEOUT)
        max_retries = valves.URL_RESOLVE_MAX_RETRIES
        base_delay = valves.URL_RESOLVE_BASE_DELAY

        for attempt in range(max_retries + 1):
            try:
                async with session.get(
                    url,
                    allow_redirects=True,
                    timeout=timeout,
                ) as response:
                    final_url = str(response.url)
                    log.debug(f"Resolved URL '{url}' to '{final_url}' after {attempt} retries")
                    return final_url, True
            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt == max_retries:
                    log.error(f"Failed to resolve URL '{url}' after {max_retries + 1} attempts: {e}")
                    return url, False
                else:
                    delay = min(base_delay * (2**attempt), 10.0)
                    log.warning(f"Retry {attempt + 1}/{max_retries + 1} for URL '{url}': {e}. Waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
            except Exception as e:
                log.error(f"Unexpected error resolving URL '{url}': {e}")
                return url, False
        return url, False

    async def _resolve_and_emit_sources(
        self,
        grounding_chunks: list[types.GroundingChunk],
        supports: list[types.GroundingSupport],
        emitter: EventEmitter,
        valves: 'Filter.Valves',
    ):
        """
        Resolves URLs in the background and emits a chat completion event
        containing only the source information, along with status updates.
        """
        initial_metadatas: list[tuple[int, str]] = []
        for i, g_c in enumerate(grounding_chunks):
            uri = None
            if (web_info := g_c.web) and web_info.uri:
                uri = web_info.uri
            elif (maps_info := g_c.maps) and maps_info.uri:
                uri = maps_info.uri

            if uri:
                initial_metadatas.append((i, uri))

        if not initial_metadatas:
            log.info('No source URIs found, skipping source emission.')
            return

        urls_to_resolve = [
            uri
            for _, uri in initial_metadatas
            if uri.startswith('https://vertexaisearch.cloud.google.com/grounding-api-redirect/')
        ]
        resolved_uris_map = {}

        if urls_to_resolve:
            num_urls = len(urls_to_resolve)
            emitter.emit_status(f'Resolving {num_urls} source URLs...')

            try:
                log.info(f'Resolving {num_urls} source URLs...')
                async with aiohttp.ClientSession() as session:
                    tasks = [self._resolve_url(session, url, valves) for url in urls_to_resolve]
                    results = await asyncio.gather(*tasks)
                log.info('URL resolution completed.')

                resolved_uris = [res[0] for res in results]
                resolved_uris_map = dict(zip(urls_to_resolve, resolved_uris))

                success_count = sum(1 for _, success in results if success)
                final_status_msg = (
                    'URL resolution complete'
                    if success_count == num_urls
                    else f'Resolved {success_count}/{num_urls} URLs'
                )
                emitter.emit_status(final_status_msg, done=True)

            except Exception as e:
                log.error(f'Error during URL resolution: {e}')
                resolved_uris_map = {url: url for url in urls_to_resolve}
                emitter.emit_status('URL resolution failed', done=True)

        source_metadatas_template: list[SourceMetadata] = [
            {'source': None, 'original_url': None, 'supports': []} for _ in grounding_chunks
        ]
        populated_metadatas = [m.copy() for m in source_metadatas_template]

        for chunk_index, original_uri in initial_metadatas:
            final_uri = resolved_uris_map.get(original_uri, original_uri)
            if 0 <= chunk_index < len(populated_metadatas):
                populated_metadatas[chunk_index]['original_url'] = original_uri
                populated_metadatas[chunk_index]['source'] = final_uri
            else:
                log.warning(f'Chunk index {chunk_index} out of bounds when populating resolved URLs.')

        # Create a mapping from each chunk index to the text segments it supports.
        chunk_index_to_segments: dict[int, list[types.Segment]] = {}
        for support in supports:
            segment = support.segment
            indices = support.grounding_chunk_indices
            if not (segment and segment.text and indices is not None):
                continue

            for index in indices:
                if index not in chunk_index_to_segments:
                    chunk_index_to_segments[index] = []
                chunk_index_to_segments[index].append(segment)
                populated_metadatas[index]['supports'].append(support.model_dump())  # type: ignore

        valid_source_metadatas: list[SourceMetadata] = []
        doc_list: list[str] = []

        for i, meta in enumerate(populated_metadatas):
            if meta.get('original_url') is not None:
                valid_source_metadatas.append(meta)

                content_parts: list[str] = []
                chunk = grounding_chunks[i]

                if maps_info := chunk.maps:
                    title = maps_info.title or 'N/A'
                    place_id = maps_info.place_id or 'N/A'
                    content_parts.append(f'Title: {title}\nPlace ID: {place_id}')

                supported_segments = chunk_index_to_segments.get(i)
                if supported_segments:
                    if content_parts:
                        content_parts.append('')  # Add a blank line for separation

                    # Use a set to show each unique snippet only once per source.
                    unique_snippets = {
                        (seg.text, seg.start_index, seg.end_index) for seg in supported_segments if seg.text is not None
                    }

                    # Sort snippets by their appearance in the text.
                    sorted_snippets = sorted(unique_snippets, key=lambda s: s[1] or 0)

                    snippet_strs = [f'- "{text}" (Indices: {start}-{end})' for text, start, end in sorted_snippets]
                    content_parts.append('Supported text snippets:')
                    content_parts.extend(snippet_strs)

                doc_list.append('\n'.join(content_parts))

        sources_list: list[Source] = []
        if valid_source_metadatas:
            sources_list.append(
                {
                    'source': {'name': 'web_search'},
                    'document': doc_list,
                    'metadata': valid_source_metadatas,
                }
            )

        # TODO: emit sources as they come in, real time
        for source in sources_list:
            emitter.emit_sources(source)

    # endregion 1.1 Add citations

    # region 1.2 Remove citation markers
    # TODO: Remove citation markers from model input.
    # endregion 1.2 Remove citation markers

    # region 1.3 Get permissive safety settings

    def _get_permissive_safety_settings(self, model_name: str) -> list[types.SafetySetting]:
        """Get safety settings based on model name and permissive setting."""

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
            'gemini-1.5-pro-001',
            'gemini-1.5-flash-001',
            'gemini-1.5-flash-8b-exp-0827',
            'gemini-1.5-flash-8b-exp-0924',
            'gemini-pro',
            'gemini-1.0-pro',
            'gemini-1.0-pro-001',
        ]:
            for category in category_threshold_map:
                category_threshold_map[category] = types.HarmBlockThreshold.BLOCK_NONE

        # Gemini 2.0 Flash supports CIVIC_INTEGRITY OFF
        if model_name in [
            'gemini-2.0-flash',
            'gemini-2.0-flash-001',
            'gemini-2.0-flash-exp',
        ]:
            category_threshold_map[types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = types.HarmBlockThreshold.OFF

        log.debug(f'Safety settings: {str({k.value: v.value for k, v in category_threshold_map.items()})}')

        safety_settings = [
            types.SafetySetting(category=category, threshold=threshold)
            for category, threshold in category_threshold_map.items()
        ]
        return safety_settings

    # endregion 1.3 Get permissive safety settings

    # region 1.4 Configuration loading

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _load_model_config(config_path: str) -> dict:
        """Loads the model configuration from a URL.

        Uses LRU cache to avoid reloading the same configuration repeatedly.
        Cache is tied to the config_path argument.
        """
        if not config_path:
            log.warning('MODEL_CONFIG_PATH is empty, returning empty config.')
            return {}

        try:
            if not (config_path.startswith('http://') or config_path.startswith('https://')):
                log.error(f'MODEL_CONFIG_PATH must be a URL (http:// or https://), got: {config_path}')
                return {}

            log.debug(f'Loading model configuration from: {config_path}')
            with urllib.request.urlopen(config_path) as response:
                config = yaml.safe_load(response.read())
                log.success(f'Successfully loaded model configuration with {len(config)} model(s).')
                return config
        except Exception as e:
            log.error(f'Failed to load model config from {config_path}: {e}')
            return {}

    # endregion 1.4 Configuration loading

    # region 1.5 Model capability checks

    @staticmethod
    def _check_model_capability(model_id: str, config: dict, capability: str) -> bool:
        """Check if a model supports a specific capability based on YAML config.

        Args:
            model_id: The canonical model id (without prefixes)
            config: The loaded YAML configuration dict
            capability: The capability to check (e.g., "search_grounding", "code_execution")

        Returns:
            True if the model supports the capability, False otherwise
        """
        if model_id not in config:
            log.debug(f"Model '{model_id}' not found in config, capability '{capability}' check returns False.")
            return False

        model_config = config[model_id]
        capabilities = model_config.get('capabilities', {})
        result = capabilities.get(capability, False)

        log.debug(f"Model '{model_id}' capability '{capability}' check: {result}")
        return result

    # endregion 1.5 Model capability checks

    # region 1.6 Utility helpers

    @staticmethod
    def _cleanup_event_emitters(app_state: State) -> None:
        """
        Scans the FastAPI app state for abandoned EventEmitter instances and removes them.
        This acts as a garbage collector for orphaned state data.
        """
        # We iterate over a copy of keys to avoid "dictionary changed size during iteration" errors.
        # We specifically look for the namespaced emitter keys (gemini_event_emitter_<chat>_<msg>).
        abandoned_keys = [
            key
            for key, value in app_state._state.items()
            if key.startswith('gemini_event_emitter_') and isinstance(value, EventEmitter) and value.is_abandoned
        ]

        for key in abandoned_keys:
            log.warning(f'Garbage Collector: Removing abandoned EventEmitter from app state: {key}')
            # Removing the reference allows the Python GC to reclaim the EventEmitter instance
            # and its internal queue resources.
            del app_state._state[key]

    @staticmethod
    def _store_data_in_state(
        app_state: State,
        __metadata__: 'Metadata',
        data: dict[str, Any],
    ):
        """
        Stores multiple values in the app state, namespaced by chat and message ID.
        Exits early if this is a task model (e.g. title generation) to prevent
        state bloat and interference with the main chat's filter logic.
        """
        if __metadata__.get('task'):
            return

        chat_id = __metadata__.get('chat_id')
        message_id = __metadata__.get('message_id')

        if not chat_id or not message_id:
            log.warning('Skipping state storage: chat_id or message_id missing from metadata.')
            return

        for key_suffix, value in data.items():
            key = f'{key_suffix}_{chat_id}_{message_id}'
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
        key = f'{key_suffix}_{chat_id}_{message_id}'
        value = getattr(app_state, key, None)
        if value is None:
            return None

        if clear_after_read:
            log.debug(f"Retrieved and cleared data from app state for key '{key}'.")
            try:
                delattr(app_state, key)
            except AttributeError:
                # This case is unlikely but handles a race condition where the attribute might already be gone.
                log.warning(f"State key '{key}' was already gone before deletion attempt.")
        else:
            log.debug(f"Retrieved data from app state for key '{key}' without clearing it.")
        return value

    def _get_first_candidate(self, candidates: list[types.Candidate] | None) -> types.Candidate | None:
        """Selects the first candidate, logging a warning if multiple exist."""
        if not candidates:
            log.warning('Received chunk with no candidates, skipping processing.')
            return None
        if len(candidates) > 1:
            log.warning('Multiple candidates found, defaulting to first candidate.')
        return candidates[0]

    @staticmethod
    def _get_model_name(body: 'Body') -> tuple[str, bool]:
        """
        Extracts the effective and canonical model name from the request body.

        Handles standard model names and custom workspace models by prioritizing
        the base_model_id found in metadata.

        Args:
            body: The request body dictionary.

        Returns:
            A tuple containing:
            - The canonical model name (prefix removed).
            - A boolean indicating if the effective model name contained the
              'gemini_manifold_google_genai.' prefix.
        """
        # 1. Get the initially requested model name from the top level
        effective_model_name: str = body.get('model', '')
        initial_model_name = effective_model_name
        base_model_name = None

        # 2. Check for a base model ID in the metadata for custom models
        # If metadata exists, attempt to extract the base_model_id
        if metadata := body.get('metadata'):
            # Safely navigate the nested structure: metadata -> model -> info -> base_model_id
            base_model_name = metadata.get('model', {}).get('info', {}).get('base_model_id', None)
            # If a base model ID is found, it overrides the initially requested name
            if base_model_name:
                effective_model_name = base_model_name

        # 3. Determine if the effective model name contains the manifold prefix.
        # This flag indicates if the model (after considering base_model_id)
        # appears to be one defined or routed via the manifold pipe function.
        is_manifold_model = 'gemini_manifold_google_genai.' in effective_model_name

        # 4. Create the canonical model name by removing the manifold prefix
        # from the effective model name.
        canonical_model_name = effective_model_name.replace('gemini_manifold_google_genai.', '')

        # 5. Log the relevant names for debugging purposes
        log.debug(
            f"Model Name Extraction: initial='{initial_model_name}', "
            f"base='{base_model_name}', effective='{effective_model_name}', "
            f"canonical='{canonical_model_name}', is_manifold={is_manifold_model}"
        )

        # 6. Return the canonical name and the manifold flag
        return canonical_model_name, is_manifold_model

    # endregion 1.4 Utility helpers

    # endregion 1. Helper methods inside the Filter class


# region 2. Option resolution

# TODO: Share the logic with Gemini Manifold Pipe somehow?
# (the single-file constraint of Open WebUI Functions makes literal code sharing impossible,
# so this region mirrors the pipe's option resolution machinery.)

# Synonyms map alternative (user-facing) spellings to canonical valve field names.
OPTION_SYNONYMS: Final[dict[str, str]] = {}

# Dynamic sets distinguishing overridable vs admin-only valve options
USER_OVERRIDABLE_VALVE_FIELDS: Final[set[str]] = set(_SharedValves.model_fields.keys())
ADMIN_ONLY_VALVE_FIELDS: Final[set[str]] = set(Filter.Valves.model_fields.keys()) - USER_OVERRIDABLE_VALVE_FIELDS

# Pre-computed case-insensitive lookup dictionaries
_VALVE_FIELD_LOOKUP: Final[dict[str, str]] = {
    field_name.lower(): field_name for field_name in Filter.Valves.model_fields.keys()
}

_SYNONYM_LOOKUP: Final[dict[str, str]] = {
    synonym.lower(): canonical_target for synonym, canonical_target in OPTION_SYNONYMS.items()
}


def canonicalize_option_key(key: str) -> str:
    """
    Normalizes an option key using case-insensitive lookup against synonyms
    and known valve fields. Returns the canonical key name or the stripped key if custom.
    """
    key_clean = key.strip()
    key_lower = key_clean.lower()

    if canonical := _SYNONYM_LOOKUP.get(key_lower):
        return canonical

    if canonical := _VALVE_FIELD_LOOKUP.get(key_lower):
        return canonical

    return key_clean


class ResolvedOptions(Filter.Valves):
    """
    Consolidated configuration combining Admin Valves, User Valves,
    and request parameters according to priority.

    Allows extra dynamic options defined at model or chat levels.
    """

    model_config = ConfigDict(extra='allow')

    def get_custom_params(self) -> dict[str, Any]:
        """Returns extra parameters not defined in Filter.Valves."""
        valve_fields = set(Filter.Valves.model_fields.keys())
        return {k: v for k, v in self.model_dump().items() if k not in valve_fields}


def _resolve_options(
    admin_valves: 'Filter.Valves',
    user_valves: Filter.UserValves | None,
    body: 'Body | None',
) -> ResolvedOptions:
    """
    Resolves the option hierarchy for the filter:

    admin valves < user valves < model page params < chat params.

    Since Open WebUI v0.12 the backend merges model page params and chat params
    into the request body before filter inlets run (body-level keys winning over
    model-side writes), so anything unknown in `body` is simply adopted as an
    option override. `body=None` (the outlet fallback path) ignores the body
    entirely and resolves from valves only.

    Admin-only options cannot be overridden by user-level sources.
    Option keys and synonyms are normalized case-insensitively.

    TODO (security tradeoff): allowing model page/chat parameters to override any
    user-overridable valve means a chat user can control options like
    `MODEL_CONFIG_PATH` from the chat level. `MODEL_CONFIG_PATH` can point the
    filter at an arbitrary YAML config, and the loaded config is cached in the
    *shared* app state (`gemini_model_config`), so one user's override affects
    every request until it is overwritten. This is not a new footgun (UserValves
    could already do the same), but consider demoting `MODEL_CONFIG_PATH` (and
    anything similarly sensitive) to an admin-only field before opening this up
    to less trusted audiences.
    """
    # Priority 1: Base options from Admin Valves
    merged_data = admin_valves.model_dump()

    # Priority 2: User Valves (user-overridable fields only)
    if user_valves is not None:
        for field_name in USER_OVERRIDABLE_VALVE_FIELDS:
            user_val = getattr(user_valves, field_name, None)
            if user_val is not None and user_val != '':
                merged_data[field_name] = user_val

    # Priority 3: Anything that the backend merged into the body is treated as
    # an option override. Since Open WebUI v0.12 the backend itself resolves
    # the priority between admin defaults, model page params and chat params
    # before the filters run, so no per-source distinction is needed here.
    if body is not None:
        known_body_keys = {
            'stream',
            'model',
            'messages',
            'files',
            'options',
            'stream_options',
            'features',
            'metadata',
            'params',
            'tool_ids',
            'tools',
            'tool_choice',
            'variables',
            'skill_ids',
            'terminal_id',
            'chat_id',
        }
        body_options = {k: v for k, v in body.items() if k not in known_body_keys}
        if isinstance(body.get('options'), dict):
            for opt_k, opt_v in body['options'].items():
                if opt_k not in body_options:
                    body_options[opt_k] = opt_v

        for raw_key, val in body_options.items():
            if val is None or val == '':
                continue
            canonical_key = canonicalize_option_key(raw_key)
            if canonical_key in ADMIN_ONLY_VALVE_FIELDS:
                log.warning(
                    f"Body parameter '{raw_key}' attempts to override admin-only valve '{canonical_key}'. Ignoring override."
                )
                continue
            merged_data[canonical_key] = val

    return ResolvedOptions(**merged_data)


# endregion Option resolution
