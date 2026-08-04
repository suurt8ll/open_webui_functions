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

import inspect
import io
import mimetypes
import os
import time
import asyncio
import uuid
import aiohttp
import base64
from collections.abc import Awaitable, Callable
from typing import (
    Literal,
    Any,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field, ValidationError
from fastapi import Request
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from loguru import logger

if TYPE_CHECKING:
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


_SHARED_VALVE_DESCS = {
    "HEIGHT": "Height of generated images in pixels.",
    "WIDTH": "Width of generated images in pixels.",
    "STEPS": "Number of inference and generation steps.",
    "CFG_SCALE": "Classifier-Free Guidance (CFG) scale determining how closely the generation follows the prompt.",
    "ASPECT_RATIO": "Optional aspect ratio configuration for supported models (e.g. '1:1', '16:9').",
    "RESOLUTION": "Optional resolution tier for supported models (e.g. '1K', '2K').",
    "SAFE_MODE": "Enables content blurring for adult material generated by models.",
    "UPSCALER_SCALE": "The upscaling size multiplier. Real range values are strictly restricted between 1.0 and 4.0.",
    "UPSCALER_REPLICATION": "Controls how closely lines and patterns from the source are preserved (0.0 to 1.0).",
    "EMISSION_VERBOSITY": "Control websocket emission verbosity.",
}

_ADMIN_VALVE_DESCS = {
    "VENICE_API_TOKEN": "Venice.ai API Token.",
    "CACHE_MODELS": "Whether to request and cache available models only on initial load.",
    "USE_FILES_API": "Save generated image files using Open WebUI's file storage API.",
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
    class Valves(BaseModel):
        VENICE_API_TOKEN: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["VENICE_API_TOKEN"], default=None
            ),
        )
        HEIGHT: int = Field(
            default=1024,
            description=_format_valve_desc(_SHARED_VALVE_DESCS["HEIGHT"], default=1024),
        )
        WIDTH: int = Field(
            default=1024,
            description=_format_valve_desc(_SHARED_VALVE_DESCS["WIDTH"], default=1024),
        )
        STEPS: int = Field(
            default=16,
            description=_format_valve_desc(_SHARED_VALVE_DESCS["STEPS"], default=16),
        )
        CFG_SCALE: float = Field(
            default=4.0,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["CFG_SCALE"], default=4.0
            ),
        )
        ASPECT_RATIO: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["ASPECT_RATIO"], default=None
            ),
        )
        RESOLUTION: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["RESOLUTION"], default=None
            ),
        )
        SAFE_MODE: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["SAFE_MODE"], default=None
            ),
        )
        UPSCALER_SCALE: float = Field(
            default=2.0,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["UPSCALER_SCALE"], default=2.0
            ),
        )
        UPSCALER_REPLICATION: float = Field(
            default=0.35,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["UPSCALER_REPLICATION"], default=0.35
            ),
        )
        CACHE_MODELS: bool = Field(
            default=True,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["CACHE_MODELS"], default=True
            ),
        )
        EMISSION_VERBOSITY: Literal["disabled", "visible", "visible_timed"] = Field(
            default="visible_timed",
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["EMISSION_VERBOSITY"], default="visible_timed"
            ),
        )
        USE_FILES_API: bool = Field(
            default=True,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["USE_FILES_API"], default=True
            ),
        )

    class UserValves(BaseModel):
        HEIGHT: int | None = Field(
            default=None,
            description=_format_valve_desc(_SHARED_VALVE_DESCS["HEIGHT"], is_user=True),
        )
        WIDTH: int | None = Field(
            default=None,
            description=_format_valve_desc(_SHARED_VALVE_DESCS["WIDTH"], is_user=True),
        )
        STEPS: int | None = Field(
            default=None,
            description=_format_valve_desc(_SHARED_VALVE_DESCS["STEPS"], is_user=True),
        )
        CFG_SCALE: float | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["CFG_SCALE"], is_user=True
            ),
        )
        ASPECT_RATIO: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["ASPECT_RATIO"], is_user=True
            ),
        )
        RESOLUTION: str | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["RESOLUTION"], is_user=True
            ),
        )
        SAFE_MODE: bool | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["SAFE_MODE"], is_user=True
            ),
        )
        UPSCALER_SCALE: float | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["UPSCALER_SCALE"], is_user=True
            ),
        )
        UPSCALER_REPLICATION: float | None = Field(
            default=None,
            description=_format_valve_desc(
                _SHARED_VALVE_DESCS["UPSCALER_REPLICATION"], is_user=True
            ),
        )
        EMISSION_VERBOSITY: Literal["disabled", "visible", "visible_timed"] | None = (
            Field(
                default=None,
                description=_format_valve_desc(
                    _SHARED_VALVE_DESCS["EMISSION_VERBOSITY"], is_user=True
                ),
            )
        )

    @staticmethod
    def _get_merged_valves(
        default_valves: "Pipe.Valves",
        user_valves: "Pipe.UserValves | dict[str, Any] | None",
    ) -> "Pipe.Valves":
        """Merges UserValves into a base Valves configuration.

        If a field in UserValves is not None or an empty string, it overrides
        the corresponding field in default_valves.
        """
        if user_valves is None:
            return default_valves.model_copy(deep=True)

        merged_data = default_valves.model_dump()

        if isinstance(user_valves, dict):
            for field_name, user_value in user_valves.items():
                if user_value is not None and user_value != "":
                    if field_name in merged_data:
                        merged_data[field_name] = user_value
        else:
            for field_name in Pipe.UserValves.model_fields:
                user_value = getattr(user_valves, field_name)
                if user_value is not None and user_value != "":
                    if field_name in merged_data:
                        merged_data[field_name] = user_value

        return Pipe.Valves(**merged_data)

    def __init__(self):
        self.valves = self.Valves()
        self.models: list[VeniceModel] = []
        log.success("Function has been initialized.")

    async def pipes(self) -> list["ModelData"]:
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

        user_valves = __user__.get("valves") if isinstance(__user__, dict) else None
        valves = self._get_merged_valves(self.valves, user_valves)

        emitter = EventEmitter(__event_emitter__, verbosity=valves.EMISSION_VERBOSITY)

        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrieval phase: {str(__metadata__["model"])}'
            log.error(error_msg)
            emitter.emit_completion_error(error_msg)
            return

        if not valves.VENICE_API_TOKEN:
            error_msg = "Missing VENICE_API_TOKEN in valves configuration."
            log.error(error_msg)
            emitter.emit_completion_error(error_msg)
            return

        selected_model_id = body.get("model", "").split(".", 1)[-1]

        # Upscaler behaves as a virtual model and can skip model catalog cache checks
        is_upscale_task = selected_model_id == "upscaler"

        if not is_upscale_task and not self.models:
            error_msg = (
                "Image generation blocked: Venice models cache has not been populated."
            )
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
            cfg_scale=valves.CFG_SCALE,
            safe_mode=valves.SAFE_MODE,
            steps=valves.STEPS,
            aspect_ratio=valves.ASPECT_RATIO,
            resolution=valves.RESOLUTION,
            width=valves.WIDTH,
            height=valves.HEIGHT,
            scale=valves.UPSCALER_SCALE,
            replication=valves.UPSCALER_REPLICATION,
        )

        client = VeniceAPIClient(valves.VENICE_API_TOKEN)

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
                image_bytes = await client.generate_image(model_spec, prompt, params)
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

        if valves.USE_FILES_API:
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

    # endregion 1.2 Image generation

    # endregion 1. Helper methods inside the Pipe class
