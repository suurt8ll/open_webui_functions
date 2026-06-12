"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.11.0
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.

# TODO: Use another LLM model to generate the image prompt?
# TODO: Negative prompts
# TODO: Upscaling

import copy
import inspect
import io
import json
import mimetypes
import os
import sys
import time
import asyncio
import uuid
import aiohttp
import base64
from collections.abc import Awaitable, Callable
from typing import (
    Any,
    Literal,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field, ValidationError
from fastapi import Request
import pydantic_core
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler  # type: ignore
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


class VeniceStepsConstraint(BaseModel):
    """Represents min/max iteration bounds for image generation."""

    default: int
    max: int


class VeniceConstraints(BaseModel):
    """
    Validation schema mapping Venice's constraints.
    Some models omit resolutions or aspect ratios; these are mapped to optional list fields
    to capture the physical capabilities of each model dynamically.
    """

    promptCharacterLimit: int
    steps: VeniceStepsConstraint
    widthHeightDivisor: int
    aspectRatios: list[str] | None = None
    defaultAspectRatio: str | None = None
    resolutions: list[str] | None = None
    defaultResolution: str | None = None


class VeniceModelSpec(BaseModel):
    """Houses human-readable metadata and operational bounds for Venice models."""

    name: str
    constraints: VeniceConstraints


class VeniceImageModel(BaseModel):
    """
    Root validator matching Venice's payload schema.
    Strictly asserts object type criteria to filter out non-image/non-model endpoints.
    """

    id: str
    object: Literal["model"]
    type: Literal["image"]
    model_spec: VeniceModelSpec


class EventEmitter:
    """
    An asynchronous queue-based event emitter tailored for image generation models.
    Guarantees in-order, non-blocking delivery of websocket status events to the Open WebUI frontend,
    allowing tasks to dispatch events instantly without yielding or halting network cycles.
    """

    def __init__(
        self,
        event_emitter: Callable[["Event"], Awaitable[None]] | None,
        verbosity: Literal["disabled", "visible", "visible_timed"] = "visible_timed",
    ):
        self._emitter = event_emitter
        self.verbosity = verbosity
        self.start_time = time.monotonic()

        self._queue: asyncio.Queue["Event | None"] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

        if self._emitter is not None:
            self._worker_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """
        Sequentially consumes and processes events until a None poison pill is encountered.
        This design guarantees that status updates and error details are delivered chronologically.
        """
        while True:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break

            if event is None:
                self._queue.task_done()
                break

            if self._emitter:
                try:
                    await self._emitter(event)
                except Exception:
                    log.exception("Error processing event in emitter background worker")

            self._queue.task_done()

    def _enqueue(self, event: "Event") -> None:
        if self._emitter is None:
            return
        self._queue.put_nowait(event)

    async def shutdown(self) -> None:
        """Gracefully halts the worker and waits for remaining events to drain."""
        if self._worker_task and not self._worker_task.done():
            self._queue.put_nowait(None)
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def emit_status(
        self, description: str, done: bool = False, hidden: bool = False
    ) -> None:
        """
        Sends status notifications to the frontend UI.
        Supports disabled output, raw messages, and timed increments showing total execution length.
        """
        if self.verbosity == "disabled":
            return

        if self.verbosity == "visible_timed":
            elapsed = time.monotonic() - self.start_time
            description = f"{description} (+{elapsed:.2f}s)"

        event: "StatusEvent" = {
            "type": "status",
            "data": {
                "description": description,
                "done": done,
                "hidden": hidden,
            },
        }
        self._enqueue(event)

    def emit_completion_error(self, error_msg: str) -> None:
        """
        Directly delivers raw error messages into chat interface outputs to ensure
        frontend rendering remains responsive and correctly terminates processing state.
        """
        event: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {
                "done": True,
                "error": {"detail": f"\n{error_msg}"},
            },
        }
        self._enqueue(event)


class Pipe:
    # TODO: UserValves
    class Valves(BaseModel):
        VENICE_API_TOKEN: str | None = Field(
            default=None, description="Venice.ai API Token"
        )
        HEIGHT: int | None = Field(
            default=None,
            description="Image height (ignored for models supporting aspect ratios)",
        )
        WIDTH: int | None = Field(
            default=None,
            description="Image width (ignored for models supporting aspect ratios)",
        )
        STEPS: int | None = Field(
            default=None,
            description="Image generation steps. If specified, values are strictly clamped to the model's supported limit.",
        )
        CFG_SCALE: int | None = Field(
            default=None, description="Image generation scale (CFG)"
        )
        ASPECT_RATIO: str | None = Field(
            default=None,
            description="Optional aspect ratio configuration for supported models (e.g. '1:1', '16:9').",
        )
        RESOLUTION: str | None = Field(
            default=None,
            description="Optional resolution tier for supported models (e.g. '1K', '2K').",
        )
        SAFE_MODE: bool | None = Field(
            default=None,
            description="Enables content blurring for adult material generated by models.",
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description="Whether to request models only on first load.",
        )
        EMISSION_VERBOSITY: Literal["disabled", "visible", "visible_timed"] = Field(
            default="visible_timed",
            description="Control websocket emission verbosity.",
        )
        USE_FILES_API: bool = Field(
            title="Use Files API",
            default=True,
            description="Save the image files using Open WebUI's API for files.",
        )
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.log_level = self.valves.LOG_LEVEL
        self._add_log_handler()
        self.models: list[VeniceImageModel] = []

        log.success("Function has been initialized.")
        log.trace("Full self object:", payload=self.__dict__)

    async def pipes(self) -> list["ModelData"]:
        if self.log_level != self.valves.LOG_LEVEL:
            log.info(
                f"Detected log level change: {self.log_level=} and {self.valves.LOG_LEVEL=}. "
                "Running the logging setup again."
            )
            self._add_log_handler()

        # Cache check targeting the strict internal model schema
        if self.models and self.valves.CACHE_MODELS:
            log.info("Models are already initialized. Returning mapped list.")
            return [{"id": m.id, "name": m.model_spec.name} for m in self.models]

        try:
            self.models = await self._get_models()
            # Mapped response output strictly contains only what OWUI needs, omitting description payloads
            return [{"id": m.id, "name": m.model_spec.name} for m in self.models]
        except Exception as e:
            log.exception("Retrieval of Venice image models failed.")
            return [self._return_error_model(str(e))]

    async def pipe(
        self,
        body: dict,
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __task__: str,
        __metadata__: dict[str, Any],
    ) -> str | None:

        emitter = EventEmitter(
            __event_emitter__, verbosity=self.valves.EMISSION_VERBOSITY
        )

        try:
            if "error" in __metadata__["model"]["id"]:
                error_msg = f'There has been an error during model retrieval phase: {str(__metadata__["model"])}'
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            if not self.valves.VENICE_API_TOKEN:
                error_msg = "Missing VENICE_API_TOKEN in valves configuration."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            selected_model_id = body.get("model", "").split(".", 1)[-1]

            # Fail fast if the pipeline is triggered without a validated cache
            if not self.models:
                error_msg = "Image generation blocked: Venice models cache has not been populated."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            # Lookup target model's validated Pydantic definition
            model_spec = next(
                (m for m in self.models if m.id == selected_model_id), None
            )
            if not model_spec:
                error_msg = f"Requested model '{selected_model_id}' was not found in the initialized Venice models list."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            prompt = next(
                (
                    msg["content"]
                    for msg in reversed(body["messages"])
                    if msg["role"] == "user"
                ),
                "",
            )

            if not prompt:
                error_msg = "No prompt found in user message."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            if __task__ == "title_generation":
                log.warning(
                    "Detected title generation task! I do not know how to handle this so I'm returning something generic."
                )
                return '{"title": "🖼️ Image Generation"}'
            if __task__ == "tags_generation":
                log.warning(
                    "Detected tag generation task! I do not know how to handle this so I'm returning an empty list."
                )
                return '{"tags": []}'

            log.debug(f"Target model parsed: {model_spec.id}, Prompt: {prompt}")

            # 1. Functional preparation (Clamping, Downscaling, Divisor Alignments)
            emitter.emit_status("Preparing image parameters...", done=False)
            payload = self._prepare_generation_payload(model_spec, prompt)

            # 2. Asynchronous API Execution
            emitter.emit_status("Generating image...", done=False)
            image_data = await self._generate_image(payload, emitter)

            success = image_data and image_data.get("images")

            status_text = f"Image {'generated' if success else 'generation failed'}"
            emitter.emit_status(status_text, done=True)

            if not success:
                return None

            log.info("Image generated successfully!")
            base64_image = image_data["images"][0]  # type: ignore

            if self.valves.USE_FILES_API:
                image_data_bytes = base64.b64decode(base64_image)
                image_url = await self._upload_image(
                    image_data_bytes,
                    "image/png",
                    model_spec.id,
                    prompt,
                    __user__["id"],
                    __request__,
                )
                return f"![Generated Image]({image_url})" if image_url else None
            else:
                return f"![Generated Image](data:image/png;base64,{base64_image})"

        finally:
            await emitter.shutdown()

    # region 1. Helper methods inside the Pipe class

    # region 1.1 Model retrieval

    async def _get_models(self) -> list[VeniceImageModel]:
        """
        Retrieves and parses image models strictly utilizing Pydantic.
        Invalid schemas are skipped and reported to prevent runtime errors in generation steps.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.venice.ai/api/v1/models?type=image",
                headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
            ) as response:
                response.raise_for_status()
                raw_models_data = await response.json()
                raw_models = raw_models_data.get("data", [])

                if not raw_models:
                    log.warning("Venice API returned no models.")
                    return []

                valid_models: list[VeniceImageModel] = []
                for model_dict in raw_models:
                    try:
                        validated = VeniceImageModel.model_validate(model_dict)
                        valid_models.append(validated)
                    except ValidationError as e:
                        log.warning(
                            f"Skipping malformed Venice model object ID '{model_dict.get('id', 'UNKNOWN')}'. "
                            f"Validation failed: {e.errors()}"
                        )

                return valid_models

    def _return_error_model(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> "ModelData":
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        return {
            "id": "error",
            "name": "[venice_manifold] " + error_msg,
            "description": error_msg,
        }

    # endregion 1.1 Model retrieval

    # region 1.2 Image generation

    def _prepare_generation_payload(
        self, model_spec: VeniceImageModel, prompt: str
    ) -> dict[str, Any]:
        """
        Builds and validates the JSON payload for Venice.ai's image generation API.
        Separates deterministic transformation and clamping logic from network operations.

        Guarantees dimensions never exceed 1280x1280 by performing aspect-ratio preserving
        downscaling first, and then performing divisor alignments that safely round downwards
        if adjustments threaten to breach the 1280 ceiling.
        """
        constraints = model_spec.model_spec.constraints

        payload: dict[str, Any] = {
            "model": model_spec.id,
            "prompt": prompt,
            "hide_watermark": True,
            "return_binary": False,
            "format": "png",
        }

        # 1. CFG Scale (direct integration)
        if self.valves.CFG_SCALE is not None:
            payload["cfg_scale"] = self.valves.CFG_SCALE

        # 2. Safe Mode (direct integration)
        if self.valves.SAFE_MODE is not None:
            payload["safe_mode"] = self.valves.SAFE_MODE

        # 3. Steps processing with constraint boundary checks
        if self.valves.STEPS is not None:
            max_steps = constraints.steps.max
            if self.valves.STEPS > max_steps:
                # TODO: emit toast messages for warnings like this
                log.warning(
                    f"Steps setting ({self.valves.STEPS}) exceeds maximum for model ({max_steps}). Clamping."
                )
                payload["steps"] = max_steps
            elif self.valves.STEPS < 1:
                log.warning(
                    f"Steps setting ({self.valves.STEPS}) must be 1 or higher. Clamping."
                )
                payload["steps"] = 1
            else:
                payload["steps"] = self.valves.STEPS
        else:
            payload["steps"] = constraints.steps.default

        # 4. Dimension & Aspect Ratio Resolution Pipeline
        supports_aspect_ratio = bool(constraints.aspectRatios)

        if supports_aspect_ratio:
            log.debug(
                f"Model '{model_spec.id}' processes aspect ratios. Ignoring static width/height parameters."
            )

            # Aspect Ratio check
            selected_ar = self.valves.ASPECT_RATIO
            if selected_ar:
                if selected_ar in constraints.aspectRatios: # type: ignore
                    payload["aspect_ratio"] = selected_ar
                else:
                    fallback_ar = constraints.defaultAspectRatio or "1:1"
                    log.warning(
                        f"Requested aspect ratio '{selected_ar}' is unsupported by '{model_spec.id}'. "
                        f"Reverting to default: '{fallback_ar}'."
                    )
                    payload["aspect_ratio"] = fallback_ar
            else:
                if constraints.defaultAspectRatio:
                    payload["aspect_ratio"] = constraints.defaultAspectRatio

            # Resolution Tier check
            if constraints.resolutions:
                selected_res = self.valves.RESOLUTION
                if selected_res:
                    if selected_res in constraints.resolutions:
                        payload["resolution"] = selected_res
                    else:
                        fallback_res = (
                            constraints.defaultResolution or constraints.resolutions[0]
                        )
                        log.warning(
                            f"Requested resolution tier '{selected_res}' is unsupported. "
                            f"Reverting to default: '{fallback_res}'."
                        )
                        payload["resolution"] = fallback_res
                else:
                    if constraints.defaultResolution:
                        payload["resolution"] = constraints.defaultResolution

        else:
            # Traditional Width / Height Path (e.g. Venice SD35)
            log.debug(f"Model '{model_spec.id}' utilizes raw dimensions.")

            user_width = self.valves.WIDTH if self.valves.WIDTH is not None else 1024
            user_height = self.valves.HEIGHT if self.valves.HEIGHT is not None else 1024
            max_limit = 1280

            # Step A: Scale down preserving the aspect ratio if either dimension breaches Venice's hard upper bound
            if user_width > max_limit or user_height > max_limit:
                scale = min(max_limit / user_width, max_limit / user_height)
                original_width, original_height = user_width, user_height
                user_width = max(1, int(user_width * scale))
                user_height = max(1, int(user_height * scale))
                log.warning(
                    f"Requested dimensions {original_width}x{original_height} exceed limit of {max_limit}. "
                    f"Preserved aspect ratio and downscaled parameters to {user_width}x{user_height}."
                )

            # Step B: Align with constraints divisor
            divisor = constraints.widthHeightDivisor
            if divisor > 1:
                adjusted_width = int(round(user_width / divisor) * divisor)
                adjusted_height = int(round(user_height / divisor) * divisor)

                # If divisor-rounding pushes a dimension above 1280, force a round-down instead
                if adjusted_width > max_limit:
                    adjusted_width = int((user_width // divisor) * divisor)
                if adjusted_height > max_limit:
                    adjusted_height = int((user_height // divisor) * divisor)

                # Ensure math limits do not accidentally drop resolutions below the minimum divisibility threshold
                adjusted_width = max(divisor, adjusted_width)
                adjusted_height = max(divisor, adjusted_height)

                if adjusted_width != user_width or adjusted_height != user_height:
                    log.warning(
                        f"Adjusted dimensions to conform with divisor {divisor}: "
                        f"{user_width}x{user_height} -> {adjusted_width}x{adjusted_height}."
                    )
                payload["width"] = adjusted_width
                payload["height"] = adjusted_height
            else:
                payload["width"] = user_width
                payload["height"] = user_height

        log.debug(f"Deterministic Venice payload built successfully: {payload}")
        return payload

    async def _generate_image(
        self, payload: dict[str, Any], emitter: EventEmitter
    ) -> dict | None:
        """
        Dispatches the prepared payload directly to the Venice.ai endpoint.
        Handles status-reporting side-effects and standardizes connection errors.
        """
        try:
            async with aiohttp.ClientSession() as session:
                log.info(
                    f"Sending image generation request to Venice.ai for model: {payload.get('model')}"
                )
                async with session.post(
                    "https://api.venice.ai/api/v1/image/generate",
                    headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
                    json=payload,
                ) as response:
                    log.info(
                        f"Received response from Venice.ai with status: {response.status}"
                    )
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientResponseError as e:
            error_msg = f"Image generation failed: {str(e)}"
            log.error(error_msg)
            emitter.emit_completion_error(error_msg)
            return None
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            log.error(error_msg)
            emitter.emit_completion_error(error_msg)
            return None

    async def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        model: str,
        prompt: str,
        user_id: str,
        __request__: Request,
    ) -> str | None:
        image_format = mimetypes.guess_extension(mime_type)
        id = str(uuid.uuid4())
        name = os.path.basename(f"generated-image{image_format}")
        imagename = f"{id}_{name}"
        image = io.BytesIO(image_data)
        image_metadata = {
            "model": model,
            "prompt": prompt,
        }

        log.info("Uploading the model generated image to Open WebUI backend.")
        log.debug("Uploading to the configured storage provider.")
        try:
            sig = inspect.signature(Storage.upload_file)
            has_tags = "tags" in sig.parameters
        except Exception as e:
            log.error(f"Error checking Storage.upload_file signature: {e}")
            has_tags = False

        try:
            if has_tags:
                contents, image_path = await asyncio.to_thread(
                    Storage.upload_file, image, imagename, tags={}
                )
            else:
                contents, image_path = await asyncio.to_thread(
                    Storage.upload_file, image, imagename  # type: ignore
                )
        except Exception:
            error_msg = "Error occurred during upload to the storage provider."
            log.exception(error_msg)
            return None

        log.info("Adding the image file to Open WebUI files database.")
        file_item = await Files.insert_new_file(
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
            log.warning("Files.insert_new_file did not return anything.")
            return None

        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        return image_url

    # endregion 1.2 Image generation

    # region 1.3 Logging
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

    def _add_log_handler(self):
        """
        Adds or updates the loguru handler specifically for this plugin.
        Includes logic for serializing and truncating extra data.
        """

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__

        # Get the desired level name and number
        desired_level_name = self.valves.LOG_LEVEL
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
            self.log_level = desired_level_name
            log.add(
                sys.stdout,
                level=desired_level_name,
                format=self.plugin_stdout_format,
                filter=plugin_filter,
            )
            log.debug(
                f"Added new handler to loguru for {__name__} with level {desired_level_name}."
            )

    # endregion 1.3 Logging

    # endregion 1. Helper methods inside the Pipe class
