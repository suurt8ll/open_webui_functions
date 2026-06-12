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
    from loguru._handler import Handler
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


class VeniceStepsConstraint(BaseModel):
    """Represents min/max iteration bounds for image generation."""

    default: int
    max: int


class VeniceImageConstraints(BaseModel):
    """
    Validation schema mapping standard text-to-image constraints.
    Some models omit resolutions or aspect ratios; these are mapped to optional fields.
    """

    promptCharacterLimit: int
    steps: VeniceStepsConstraint
    widthHeightDivisor: int
    aspectRatios: list[str] | None = None
    defaultAspectRatio: str | None = None
    resolutions: list[str] | None = None
    defaultResolution: str | None = None


class VeniceImageModelSpec(BaseModel):
    """Houses human-readable metadata and operational bounds for text-to-image models."""

    name: str
    constraints: VeniceImageConstraints


class VeniceImageModel(BaseModel):
    """
    Root validator matching Venice's standard text-to-image metadata schema.
    We strictly assert the type criteria to enforce standard pipeline routing.
    """

    id: str
    object: Literal["model"]
    type: Literal["image"]
    model_spec: VeniceImageModelSpec


class VeniceInpaintConstraints(BaseModel):
    """
    Validation schema mapping image editing/inpainting constraints.
    Edit models process tasks differently and require aspect ratio support without raw step limits.
    """

    promptCharacterLimit: int
    aspectRatios: list[str]  # Enforced as mandatory for inpaint models
    defaultAspectRatio: str | None = None
    resolutions: list[str] | None = None
    defaultResolution: str | None = None
    combineImages: bool | None = None
    singleImageAspectRatio: bool | None = None


class VeniceInpaintModelSpec(BaseModel):
    """Houses human-readable metadata and operational bounds for image editing/inpainting models."""

    name: str
    constraints: VeniceInpaintConstraints


class VeniceInpaintModel(BaseModel):
    """
    Root validator matching Venice's image editing metadata schema.
    Asserts the type criteria to enforce editing pipeline routing.
    """

    id: str
    object: Literal["model"]
    type: Literal["inpaint"]
    model_spec: VeniceInpaintModelSpec


# A Type Alias representing either a standard or edit model.
# Enables clean union handling across validation and payload steps.
VeniceModel = VeniceImageModel | VeniceInpaintModel


class VeniceAPIError(Exception):
    """
    Custom exception representing transport or operational failures when communicating
    with the Venice.ai API. Decouples client exceptions from Open WebUI websocket handlers.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class VeniceParams(BaseModel):
    """
    Unified transfer schema containing generation and rendering configurations.
    Protects the API client from direct dependencies on Open WebUI Valve variables.
    """

    cfg_scale: int | None = None
    safe_mode: bool | None = None
    steps: int | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None
    width: int | None = None
    height: int | None = None
    scale: float | None = None
    replication: float | None = None


class VeniceAPIClient:
    """
    Dedicated client managing transport payloads, validation checks, endpoints,
    and model mapping tasks directly targeting the Venice.ai REST endpoints.
    """

    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.venice.ai/api/v1"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    async def get_models(self) -> list[VeniceModel]:
        """
        Retrieves and parses image and inpaint models strictly utilizing Pydantic.
        Invalid schemas are skipped and reported to prevent runtime errors in subsequent steps.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models?type=all",
                    headers=self._headers(),
                ) as response:
                    if response.status != 200:
                        raise VeniceAPIError(
                            f"Failed to fetch models: HTTP {response.status}"
                        )

                    raw_models_data = await response.json()
                    raw_models = raw_models_data.get("data", [])

                    if not raw_models:
                        log.warning("Venice API returned no models.")
                        return []

                    valid_models: list[VeniceModel] = []
                    for model_dict in raw_models:
                        m_type = model_dict.get("type")
                        if m_type == "image":
                            try:
                                validated = VeniceImageModel.model_validate(model_dict)
                                valid_models.append(validated)
                            except ValidationError as e:
                                log.warning(
                                    f"Skipping malformed image model '{model_dict.get('id', 'UNKNOWN')}'. "
                                    f"Validation failed: {e.errors()}"
                                )
                        elif m_type == "inpaint":
                            try:
                                constraints = model_dict.get("model_spec", {}).get(
                                    "constraints", {}
                                )
                                if (
                                    "aspectRatios" not in constraints
                                    or not constraints["aspectRatios"]
                                ):
                                    log.warning(
                                        f"Skipping inpaint model '{model_dict.get('id')}': Missing or empty aspectRatios constraint."
                                    )
                                    continue

                                validated = VeniceInpaintModel.model_validate(
                                    model_dict
                                )
                                valid_models.append(validated)
                            except ValidationError as e:
                                log.warning(
                                    f"Skipping malformed inpaint model '{model_dict.get('id', 'UNKNOWN')}'. "
                                    f"Validation failed: {e.errors()}"
                                )
                    return valid_models
        except Exception as e:
            if not isinstance(e, VeniceAPIError):
                raise VeniceAPIError(f"Failed to retrieve models: {str(e)}") from e
            raise

    async def generate_image(
        self,
        model_spec: VeniceImageModel,
        prompt: str,
        params: VeniceParams,
    ) -> bytes:
        """Processes text-to-image queries and returns the raw output file byte stream."""
        payload = self._prepare_generation_payload(model_spec, prompt, params)
        return await self._request_image("/image/generate", payload)

    async def edit_image(
        self,
        model_spec: VeniceInpaintModel,
        prompt: str,
        image_url: str,
        params: VeniceParams,
    ) -> bytes:
        """Processes editing/inpainting queries and returns the raw output file byte stream."""
        payload = self._prepare_edit_payload(model_spec, prompt, image_url, params)
        return await self._request_image("/image/edit", payload)

    async def upscale_image(
        self,
        image_url: str,
        params: VeniceParams,
    ) -> bytes:
        """
        Upscales or enhances an image based on the supplied parameters.
        Enforces strict validation bounding checks to protect against payload rejections.
        """
        # Range validations conforming directly with Venice's specification
        scale = params.scale if params.scale is not None else 2.0
        scale = max(1.0, min(4.0, scale))

        replication = params.replication if params.replication is not None else 0.35
        replication = max(0.0, min(1.0, replication))

        payload: dict[str, Any] = {
            "image": self._clean_image_payload(image_url),
            "enhance": False,  # Enhancer disabled statically as requested
            "scale": scale,
            "replication": replication,
        }

        return await self._request_image("/image/upscale", payload)

    def _prepare_generation_payload(
        self, model_spec: VeniceImageModel, prompt: str, params: VeniceParams
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

        if params.cfg_scale is not None:
            payload["cfg_scale"] = params.cfg_scale

        if params.safe_mode is not None:
            payload["safe_mode"] = params.safe_mode

        if params.steps is not None:
            max_steps = constraints.steps.max
            if params.steps > max_steps:
                log.warning(
                    f"Steps setting ({params.steps}) exceeds maximum for model ({max_steps}). Clamping."
                )
                payload["steps"] = max_steps
            elif params.steps < 1:
                log.warning(
                    f"Steps setting ({params.steps}) must be 1 or higher. Clamping."
                )
                payload["steps"] = 1
            else:
                payload["steps"] = params.steps
        else:
            payload["steps"] = constraints.steps.default

        supports_aspect_ratio = bool(constraints.aspectRatios)

        if supports_aspect_ratio:
            log.debug(
                f"Model '{model_spec.id}' processes aspect ratios. Ignoring static width/height parameters."
            )

            selected_ar = params.aspect_ratio
            if selected_ar:
                if selected_ar in constraints.aspectRatios:
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

            if constraints.resolutions:
                selected_res = params.resolution
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
            log.debug(f"Model '{model_spec.id}' utilizes raw dimensions.")

            user_width = params.width if params.width is not None else 1024
            user_height = params.height if params.height is not None else 1024
            max_limit = 1280

            if user_width > max_limit or user_height > max_limit:
                scale = min(max_limit / user_width, max_limit / user_height)
                original_width, original_height = user_width, user_height
                user_width = max(1, int(user_width * scale))
                user_height = max(1, int(user_height * scale))
                log.warning(
                    f"Requested dimensions {original_width}x{original_height} exceed limit of {max_limit}. "
                    f"Preserved aspect ratio and downscaled parameters to {user_width}x{user_height}."
                )

            divisor = constraints.widthHeightDivisor
            if divisor > 1:
                adjusted_width = int(round(user_width / divisor) * divisor)
                adjusted_height = int(round(user_height / divisor) * divisor)

                if adjusted_width > max_limit:
                    adjusted_width = int((user_width // divisor) * divisor)
                if adjusted_height > max_limit:
                    adjusted_height = int((user_height // divisor) * divisor)

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

        log.debug("Deterministic Venice payload built successfully:", payload=payload)
        return payload

    def _clean_image_payload(self, image_url: str) -> str:
        """
        Strips the data URI scheme prefix (e.g., 'data:image/png;base64,') from the input URL,
        leaving only the raw base64 data. Decoupled helper ready for conditional invocation.
        """
        if image_url.startswith("data:"):
            if "," in image_url:
                return image_url.split(",", 1)[1]
        return image_url

    def _prepare_edit_payload(
        self,
        model_spec: VeniceInpaintModel,
        prompt: str,
        image_url: str,
        params: VeniceParams,
    ) -> dict[str, Any]:
        """
        Builds and validates the JSON payload for Venice.ai's image editing API (/image/edit).
        Ensures proper aspect ratio mapping and model specific configuration bounds.
        """
        constraints = model_spec.model_spec.constraints

        payload: dict[str, Any] = {
            "model": model_spec.id,
            "prompt": prompt,
            "image": self._clean_image_payload(image_url),
            "output_format": "png",
        }

        if params.safe_mode is not None:
            payload["safe_mode"] = params.safe_mode

        selected_ar = params.aspect_ratio
        if selected_ar:
            if selected_ar in constraints.aspectRatios:
                payload["aspect_ratio"] = selected_ar
            else:
                fallback_ar = constraints.defaultAspectRatio or "auto"
                log.warning(
                    f"Requested aspect ratio '{selected_ar}' is unsupported by '{model_spec.id}'. "
                    f"Reverting to default/fallback: '{fallback_ar}'."
                )
                payload["aspect_ratio"] = fallback_ar
        else:
            if constraints.defaultAspectRatio:
                payload["aspect_ratio"] = constraints.defaultAspectRatio
            else:
                payload["aspect_ratio"] = "auto"

        if constraints.resolutions:
            selected_res = params.resolution
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

        log.debug(
            "Deterministic Venice edit payload built successfully:", payload=payload
        )
        return payload

    async def _request_image(self, endpoint: str, payload: dict[str, Any]) -> bytes:
        """
        Executes raw POST requests and simplifies data streams. Unifies both API JSON outputs
        (extracting base64 image strings) and direct image binary streams into clean raw bytes.
        """
        try:
            async with aiohttp.ClientSession() as session:
                log.info(
                    f"Sending request to Venice.ai {endpoint} for model: {payload.get('model')}"
                )
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    log.info(
                        f"Received response from Venice.ai with status: {response.status}"
                    )
                    if response.status != 200:
                        try:
                            err_body = await response.json()
                            error_detail = err_body.get("error", {}).get(
                                "message", "Unknown Venice API error."
                            )
                        except Exception:
                            error_detail = await response.text()
                        raise VeniceAPIError(
                            f"Venice API Error ({response.status}): {error_detail}",
                            status_code=response.status,
                        )

                    content_type = response.headers.get("Content-Type", "")

                    if content_type.startswith("image/"):
                        return await response.read()

                    # Venice default fallback parsing
                    json_data = await response.json()
                    images = json_data.get("images")
                    if not images:
                        raise VeniceAPIError(
                            "Venice API response did not contain any images."
                        )

                    return base64.b64decode(images[0])

        except aiohttp.ClientResponseError as e:
            raise VeniceAPIError(f"API request failed: {str(e)}") from e
        except Exception as e:
            if not isinstance(e, VeniceAPIError):
                raise VeniceAPIError(f"Network or execution error: {str(e)}") from e
            raise


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
        UPSCALER_SCALE: float = Field(
            default=2.0,
            description="The upscaling size multiplier. Real range values are strictly restricted between 1.0 and 4.0.",
        )
        UPSCALER_REPLICATION: float = Field(
            default=0.35,
            description="Controls how closely lines and patterns from the source are preserved (0.0 to 1.0).",
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
        self.models: list[VeniceModel] = []

        log.success("Function has been initialized.")
        log.trace("Full self object:", payload=self.__dict__)

    async def pipes(self) -> list["ModelData"]:
        if self.log_level != self.valves.LOG_LEVEL:
            log.info(
                f"Detected log level change: {self.log_level=} and {self.valves.LOG_LEVEL=}. "
                "Running the logging setup again."
            )
            self._add_log_handler()

        if self.models and self.valves.CACHE_MODELS:
            log.info("Models are already initialized. Returning mapped list.")
            mapped = [{"id": m.id, "name": m.model_spec.name} for m in self.models]
            # Manual append for the virtual upscaling utility
            mapped.append(
                {"id": "upscaler", "name": "Venice Image Upscaler & Enhancer"}
            )
            return mapped

        if not self.valves.VENICE_API_TOKEN:
            return [
                self._return_error_model(
                    "Missing VENICE_API_TOKEN in valves configuration."
                )
            ]

        try:
            client = VeniceAPIClient(self.valves.VENICE_API_TOKEN)
            self.models = await client.get_models()
            mapped = [{"id": m.id, "name": m.model_spec.name} for m in self.models]
            mapped.append(
                {"id": "upscaler", "name": "Venice Image Upscaler & Enhancer"}
            )
            return mapped
        except Exception as e:
            log.exception("Retrieval of Venice image/inpaint models failed.")
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

            # Upscaler behaves as a virtual model and can skip model catalog cache checks
            is_upscale_task = selected_model_id == "upscaler"

            if not is_upscale_task and not self.models:
                error_msg = "Image generation blocked: Venice models cache has not been populated."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            model_spec = None
            if not is_upscale_task:
                model_spec = next(
                    (m for m in self.models if m.id == selected_model_id), None
                )
                if not model_spec:
                    error_msg = f"Requested model '{selected_model_id}' was not found in the initialized Venice models list."
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

            last_user_message = next(
                (
                    msg
                    for msg in reversed(body.get("messages", []))
                    if msg.get("role") == "user"
                ),
                None,
            )

            if not last_user_message:
                error_msg = "No user message found to process."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            content = last_user_message.get("content")
            if not content:
                error_msg = "User message has empty or missing content."
                log.error(error_msg)
                emitter.emit_completion_error(error_msg)
                return

            prompt, image_url = self._extract_media_from_message(content)

            # Verify prompt or assign placeholder values if processing pure upscaling tasks
            if is_upscale_task:
                if not image_url:
                    error_msg = "Requested upscaling requires an attached image, but none was provided."
                    log.error(error_msg)
                    emitter.emit_completion_error(error_msg)
                    return
                # Provide a generic visual target prompt to prevent blank parameters down the line
                prompt = prompt or "Upscaled Image"
            else:
                if not prompt:
                    error_msg = "No valid text prompt found in the user's message."
                    log.error(error_msg)
                    emitter.emit_completion_error(error_msg)
                    return

            is_edit_model = model_spec.type == "inpaint" if model_spec else False

            if is_upscale_task:
                log.debug("Target pipeline isolated: Image Upscale")
            elif is_edit_model:
                if not image_url:
                    error_msg = f"Requested editing model '{selected_model_id}' requires an attached image, but none was provided."
                    log.error(error_msg)
                    emitter.emit_completion_error(error_msg)
                    return
                log.debug(
                    f"Target edit model parsed: {model_spec.id if model_spec else ''}, Prompt: {prompt}, Image: [Present]"
                )
            else:
                log.debug(
                    f"Target generation model parsed: {model_spec.id if model_spec else ''}, Prompt: {prompt}"
                )

            # Construct configuration overrides
            params = VeniceParams(
                cfg_scale=self.valves.CFG_SCALE,
                safe_mode=self.valves.SAFE_MODE,
                steps=self.valves.STEPS,
                aspect_ratio=self.valves.ASPECT_RATIO,
                resolution=self.valves.RESOLUTION,
                width=self.valves.WIDTH,
                height=self.valves.HEIGHT,
                scale=self.valves.UPSCALER_SCALE,
                replication=self.valves.UPSCALER_REPLICATION,
            )

            client = VeniceAPIClient(self.valves.VENICE_API_TOKEN)

            emitter.emit_status("Preparing image parameters...", done=False)

            if is_upscale_task:
                emitter.emit_status("Processing upscale...", done=False)
            else:
                emitter.emit_status(
                    f"Processing {'edit' if is_edit_model else 'image'}...", done=False
                )

            try:
                if is_upscale_task:
                    image_bytes = await client.upscale_image(image_url, params)
                elif is_edit_model:
                    image_bytes = await client.edit_image(
                        model_spec, prompt, image_url, params
                    )
                else:
                    image_bytes = await client.generate_image(
                        model_spec, prompt, params
                    )
                success = True
            except VeniceAPIError as e:
                log.error(str(e))
                emitter.emit_completion_error(str(e))
                success = False

            status_text = f"Image {'processed' if success else 'failed'}"
            emitter.emit_status(status_text, done=True)

            if not success:
                return None

            log.info("Image request completed successfully!")

            output_model_id = (
                "upscaler"
                if is_upscale_task
                else (model_spec.id if model_spec else "unknown")
            )

            if self.valves.USE_FILES_API:
                uploaded_url = await self._upload_image(
                    image_bytes,
                    "image/png",
                    output_model_id,
                    prompt,
                    __user__["id"],
                    __request__,
                )
                return f"![Generated Image]({uploaded_url})" if uploaded_url else None
            else:
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                return f"![Generated Image](data:image/png;base64,{base64_image})"

        finally:
            await emitter.shutdown()

    # region 1. Helper methods inside the Pipe class

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

    def _extract_media_from_message(
        self, content: Any
    ) -> tuple[str | None, str | None]:
        """
        Parses text prompts and base64-encoded image components from user messages.
        Supports standard raw string payloads and complex multimedia structure arrays.
        """
        if isinstance(content, str):
            return content.strip() or None, None

        if not isinstance(content, list):
            return None, None

        texts: list[str] = []
        images: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_val = block.get("text", "")
                if isinstance(text_val, str) and text_val.strip():
                    texts.append(text_val.strip())
            elif block_type == "image_url":
                img_url_obj = block.get("image_url")
                if isinstance(img_url_obj, dict):
                    img_url = img_url_obj.get("url")
                    if isinstance(img_url, str) and img_url.strip():
                        images.append(img_url.strip())

        primary_text = texts[-1] if texts else None
        primary_image = images[-1] if images else None

        if len(texts) > 1 or len(images) > 1:
            log.warning(
                f"Detected multiple text/image objects in the last user message. "
                f"Using the last available blocks (Texts: {len(texts)}, Images: {len(images)})."
            )

        return primary_text, primary_image

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
                    Storage.upload_file, image, imagename
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
        handlers: dict[int, "Handler"] = log._core.handlers
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
