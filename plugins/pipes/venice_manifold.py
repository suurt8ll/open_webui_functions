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

import io
import mimetypes
import os
import time
import asyncio
import uuid
import aiohttp
import base64
from loguru import logger
from fastapi import Request
from collections.abc import Awaitable, Callable
from pydantic import BaseModel, Field, ValidationError, ConfigDict, create_model
from typing import (
    Literal,
    Any,
    Final,
    TYPE_CHECKING,
    get_args,
    get_origin,
)

from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


# region Pydantic Models


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
    negative_prompt: str | None = None
    variants: int | None = None


# endregion Pydantic Models


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
    ) -> list[bytes]:
        """Processes text-to-image queries and returns a list of raw output image byte streams."""
        payload = self._prepare_generation_payload(model_spec, prompt, params)
        return await self._request_image("/image/generate", payload)

    async def edit_image(
        self,
        model_spec: VeniceInpaintModel,
        prompt: str,
        image_url: str,
        params: VeniceParams,
    ) -> list[bytes]:
        """Processes editing/inpainting queries and returns a list of raw output image byte streams."""
        payload = self._prepare_edit_payload(model_spec, prompt, image_url, params)
        return await self._request_image("/image/edit", payload)

    async def upscale_image(
        self,
        image_url: str,
        params: VeniceParams,
    ) -> list[bytes]:
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

    def _derive_dimensions_from_aspect_ratio(self, aspect_ratio: str) -> tuple[int, int]:
        """
            Converts an aspect ratio string (e.g. '16:9') into pixel dimensions
            with the longer side fixed at 1280. Used for models that only accept
            raw width/height but the user configured an aspect ratio.
            """
        try:
            w_str, h_str = aspect_ratio.split(":")
            ratio_w, ratio_h = int(w_str), int(h_str)
            if ratio_w <= 0 or ratio_h <= 0:
                raise ValueError
        except (ValueError, AttributeError):
            log.warning(
                    f"Could not parse aspect ratio '{aspect_ratio}' for dimension derivation. "
                    f"Falling back to 1:1 (1280x1280)."
                )
            return 1280, 1280

        base = 1280
        if ratio_w >= ratio_h:
            width = base
            height = round(base * ratio_h / ratio_w)
        else:
            height = base
            width = round(base * ratio_w / ratio_h)

        return width, height

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
            "mbed_exif_metadata": True,
        }

        if params.negative_prompt is not None:
            payload["negative_prompt"] = params.negative_prompt
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

        if params.variants is not None:
            variants = max(1, min(4, params.variants))
            if variants != params.variants:
                log.warning(
                    f"Variants setting ({params.variants}) clamped to {variants}. Must be between 1 and 4."
                )
            payload["variants"] = variants

        if constraints.aspectRatios:
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

            if params.width is not None or params.height is not None:
                user_width = params.width if params.width is not None else 1024
                user_height = params.height if params.height is not None else 1024
            elif params.aspect_ratio is not None:
                user_width, user_height = self._derive_dimensions_from_aspect_ratio(
                    params.aspect_ratio
                )
                log.info(
                    f"Model '{model_spec.id}' does not support aspect ratios natively. "
                    f"Derived dimensions {user_width}x{user_height} from aspect ratio '{params.aspect_ratio}'."
                )
            else:
                user_width, user_height = 1024, 1024

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

    async def _request_image(self, endpoint: str, payload: dict[str, Any]) -> list[bytes]:
        """
        Executes raw POST requests and simplifies data streams. Unifies both API JSON outputs
        (extracting base64 image strings) and direct image binary streams into clean raw bytes.
        Returns a list of decoded images; binary responses yield a single-element list.
        """
        try:
            async with aiohttp.ClientSession() as session:
                log.info(
                    f"Sending request to Venice.ai {endpoint} for model: {payload.get('model')}"
                )
                log.debug("Request payload:", payload=payload)
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    log.info(
                        f"Received response from Venice.ai with status: {response.status}"
                    )
                    log.debug("Response object:", payload=response)
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
                        return [await response.read()]

                    # Venice default fallback parsing
                    json_data = await response.json()
                    log.debug("Response json:", payload=json_data)
                    images = json_data.get("images")
                    if not images:
                        raise VeniceAPIError(
                            "Venice API response did not contain any images."
                        )

                    return [base64.b64decode(img) for img in images]

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
            log.debug("Shutting down EventEmitter worker task.")
            self._queue.put_nowait(None)
            try:
                await self._worker_task
                log.debug("EventEmitter worker task has been shut down successfully.")
            except asyncio.CancelledError:
                log.debug("EventEmitter worker task was cancelled during shutdown.")

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


_VALVE_DESCRIPTIONS: Final[dict[str, str]] = {
    # Admin-Only
    "VENICE_API_TOKEN": "Venice.ai API Token.",
    "CACHE_MODELS": "Whether to request and cache available models only on initial load.",
    "USE_FILES_API": "Save generated image files using Open WebUI's file storage API.",
    # User-Configurable
    "HEIGHT": "Height of generated images in pixels.",
    "WIDTH": "Width of generated images in pixels.",
    "STEPS": "Number of inference and generation steps.",
    "CFG_SCALE": "Classifier-Free Guidance (CFG) scale determining how closely the generation follows the prompt.",
    "ASPECT_RATIO": "Optional aspect ratio configuration for supported models (e.g. '1:1', '16:9').",
    "RESOLUTION": "Optional resolution tier for supported models (e.g. '1K', '2K').",
    "NEGATIVE PROMPT": "Optional negative prompt to steer the generation away from undesired features.",
    "VARIANTS": "Number of image variants to generate per request (1-4). Only supported by text-to-image models, not inpaint or upscale.",
    "SAFE_MODE": "Enables content blurring for adult material generated by models.",
    "UPSCALER_SCALE": "The upscaling size multiplier. Real range values are strictly restricted between 1.0 and 4.0.",
    "UPSCALER_REPLICATION": "Controls how closely lines and patterns from the source are preserved (0.0 to 1.0).",
    "EMISSION_VERBOSITY": "Control websocket emission verbosity.",
}


def _format_valve_desc(text: str, default: Any = None, is_user: bool = False) -> str:
    """Formats Markdown descriptions for Valves and UserValves fields."""
    text = text.strip()
    sep = "\n\n---\n\n"
    if is_user:
        return f"{text}\n\n*If not set, the admin's setting is used.*{sep}"
    formatted_default = f"`{default}`" if default is not None else "`None`"
    return f"{text}\n\n**Default:** {formatted_default}{sep}"


def _admin_field(
    name: str, default: Any = None, admin_section_start: bool = False
) -> Any:
    """Helper to construct a Pydantic Field with default value and formatted admin description."""
    raw_desc = _VALVE_DESCRIPTIONS.get(name, "")
    desc = _format_valve_desc(raw_desc, default=default)
    if admin_section_start:
        desc = f"{desc}### Admin-Only Options"
    return Field(default=default, description=desc)


class _SharedValves(BaseModel):
    """Base model holding user-configurable options and shared field validators."""

    HEIGHT: int | None = _admin_field("HEIGHT")
    WIDTH: int | None = _admin_field("WIDTH")
    STEPS: int | None = _admin_field("STEPS")
    CFG_SCALE: int | None = _admin_field("CFG_SCALE")
    ASPECT_RATIO: str | None = _admin_field("ASPECT_RATIO")
    RESOLUTION: str | None = _admin_field("RESOLUTION")
    SAFE_MODE: bool | None = _admin_field("SAFE_MODE")
    NEGATIVE_PROMPT: str | None = _admin_field("NEGATIVE_PROMPT")
    VARIANTS: int = _admin_field("VARIANTS", 1)
    UPSCALER_SCALE: float | None = _admin_field("UPSCALER_SCALE")
    UPSCALER_REPLICATION: float | None = _admin_field("UPSCALER_REPLICATION")
    EMISSION_VERBOSITY: Literal["disabled", "visible", "visible_timed"] = _admin_field(
        "EMISSION_VERBOSITY", "visible_timed", admin_section_start=True
    )


def _generate_user_valves(shared_cls: type[BaseModel]) -> type[BaseModel]:
    """Generates Pipe.UserValves model from _SharedValves with Optional fields and user descriptions."""
    fields: dict[str, Any] = {}
    for name, field_info in shared_cls.model_fields.items():
        raw_desc = _VALVE_DESCRIPTIONS.get(name, "")
        user_desc = _format_valve_desc(raw_desc, is_user=True)

        ann = field_info.annotation
        if get_origin(ann) is Literal:
            args = get_args(ann)
            user_ann = (
                Literal[(*args, "")] | None if "" not in args else ann | None  # type: ignore[operator]
            )
        elif ann is not None:
            user_ann = ann | None
        else:
            user_ann = Any

        fields[name] = (user_ann, Field(default=None, description=user_desc))

    return create_model("UserValves", __base__=shared_cls, **fields)  # type: ignore[call-overload]


class Pipe:
    class Valves(_SharedValves):
        VENICE_API_TOKEN: str | None = _admin_field("VENICE_API_TOKEN")
        CACHE_MODELS: bool = _admin_field("CACHE_MODELS", True)
        USE_FILES_API: bool = _admin_field("USE_FILES_API", True)

    UserValves = _generate_user_valves(_SharedValves)

    def __init__(self):
        self.valves = self.Valves()
        self.models: list[VeniceModel] = []
        log.success("Function has been initialized.")

    async def pipes(self) -> list[dict[str, str]]:
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
        body: "Body",
        __user__: "UserData",
        __request__: Request,
        __metadata__: "Metadata",
        __event_emitter__: Callable[["Event"], Awaitable[None]],
    ) -> str:

        options = _resolve_options(
            self.valves,
            __user__.get("valves") if isinstance(__user__, dict) else None,
            __user__.get("email", "") if isinstance(__user__, dict) else "",
            body,
            __metadata__,
        )

        emitter = EventEmitter(__event_emitter__, verbosity=options.EMISSION_VERBOSITY)

        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrieval phase: {str(__metadata__["model"])}'
            raise RuntimeError(error_msg)

        if not options.VENICE_API_TOKEN:
            error_msg = "Missing VENICE_API_TOKEN in valves configuration."
            raise RuntimeError(error_msg)

        selected_model_id = body.get("model", "").split(".", 1)[-1]

        # Upscaler behaves as a virtual model and can skip model catalog cache checks
        is_upscale_task = selected_model_id == "upscaler"

        if not is_upscale_task and not self.models:
            error_msg = (
                "Image generation blocked: Venice models cache has not been populated."
            )
            raise RuntimeError(error_msg)

        model_spec = None
        if not is_upscale_task:
            model_spec = next(
                (m for m in self.models if m.id == selected_model_id), None
            )
            if not model_spec:
                error_msg = f"Requested model '{selected_model_id}' was not found in the initialized Venice models list."
                raise RuntimeError(error_msg)

        task = __metadata__.get("task")
        if task == "title_generation":
            log.warning(
                "Detected title generation task! I do not know how to handle this so I'm returning something generic."
            )
            return '{"title": "🖼️ Image Generation"}'
        if task == "tags_generation":
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
            raise ValueError(error_msg)

        content = last_user_message.get("content")
        if not content:
            error_msg = "User message has empty or missing content."
            raise ValueError(error_msg)

        prompt, image_url = self._extract_media_from_message(content)

        # Verify prompt or assign placeholder values if processing pure upscaling tasks
        if is_upscale_task:
            if not image_url:
                error_msg = "Requested upscaling requires an attached image, but none was provided."
                raise ValueError(error_msg)
            # Provide a generic visual target prompt to prevent blank parameters down the line
            prompt = prompt or "Upscaled Image"
        else:
            if not prompt:
                error_msg = "No valid text prompt found in the user's message."
                raise ValueError(error_msg)

        is_edit_model = model_spec.type == "inpaint" if model_spec else False

        if is_upscale_task:
            log.debug("Target pipeline isolated: Image Upscale")
        elif is_edit_model:
            if not image_url:
                error_msg = f"Requested editing model '{selected_model_id}' requires an attached image, but none was provided."
                raise ValueError(error_msg)
            log.debug(
                f"Target edit model parsed: {model_spec.id if model_spec else ''}, Prompt: {prompt}, Image: [Present]"
            )
        else:
            log.debug(
                f"Target generation model parsed: {model_spec.id if model_spec else ''}, Prompt: {prompt}"
            )

        emitter.emit_status("Preparing image parameters...", done=False)
        # Construct configuration overrides
        params = VeniceParams(
            cfg_scale=options.CFG_SCALE,
            safe_mode=options.SAFE_MODE,
            steps=options.STEPS,
            aspect_ratio=options.ASPECT_RATIO,
            resolution=options.RESOLUTION,
            width=options.WIDTH,
            height=options.HEIGHT,
            scale=options.UPSCALER_SCALE,
            replication=options.UPSCALER_REPLICATION,
            negative_prompt=options.NEGATIVE_PROMPT,
            variants=options.VARIANTS,
        )

        client = VeniceAPIClient(options.VENICE_API_TOKEN)

        if is_upscale_task:
            emitter.emit_status("Processing upscale...", done=False)
        elif is_edit_model:
            emitter.emit_status("Processing edit...", done=False)
        else:
            num_variants = params.variants or 1
            if num_variants > 1:
                emitter.emit_status(f"Generating {num_variants} images...", done=False)
            else:
                emitter.emit_status("Generating image...", done=False)

        image_bytes_list: list[bytes] = []
        try:
            if is_upscale_task:
                assert image_url
                image_bytes_list = await client.upscale_image(image_url, params)
            elif is_edit_model:
                assert image_url and isinstance(model_spec, VeniceInpaintModel)
                image_bytes_list = await client.edit_image(
                    model_spec, prompt, image_url, params
                )
            else:
                assert isinstance(model_spec, VeniceImageModel)
                image_bytes_list = await client.generate_image(model_spec, prompt, params)
        except VeniceAPIError as e:
            emitter.emit_status("Image generation failed", done=True)
            await emitter.shutdown()
            raise e

        num_images = len(image_bytes_list)
        log.info(f"Image request completed successfully! ({num_images} image(s) generated)")
        if num_images > 1:
            emitter.emit_status(f"Image generation successful ({num_images} images)", done=True)
        else:
            emitter.emit_status("Image generation successful", done=True)

        output_model_id = (
            "upscaler"
            if is_upscale_task
            else (model_spec.id if model_spec else "unknown")
        )

        if options.USE_FILES_API:
            uploaded_urls = []
            for img_bytes in image_bytes_list:
                # TODO: catch errors and fall-back to bytes?
                url = await self._upload_image(
                    img_bytes,
                    "image/png",
                    output_model_id,
                    prompt,
                    __user__["id"],
                    __request__,
                )
                uploaded_urls.append(url)
            response = "\n\n".join(f"![Generated Image]({url})" for url in uploaded_urls)
        else:
            parts = []
            for img_bytes in image_bytes_list:
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                parts.append(f"![Generated Image](data:image/png;base64,{base64_image})")
            response = "\n\n".join(parts)
        await emitter.shutdown()
        return response

    # region 1. Helper methods inside the Pipe class

    def _return_error_model(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> dict[str, str]:
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
    ) -> str:
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
            contents, image_path = await asyncio.to_thread(
                Storage.upload_file, image, imagename, tags={}
            )
        except Exception as e:
            raise Exception(f"Failed to upload the image to storage: {str(e)}") from e

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
            raise ValueError(
                "Failed to insert new file. Files.insert_new_file did not return anything."
            )

        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        return image_url

    # endregion 1.2 Image generation

    # endregion 1. Helper methods inside the Pipe class


# region Option resolution

OPTION_SYNONYMS: Final[dict[str, str]] = {
    "scale": "UPSCALER_SCALE",
    "replication": "UPSCALER_REPLICATION",
}

USER_OVERRIDABLE_VALVE_FIELDS: Final[set[str]] = set(_SharedValves.model_fields.keys())
ADMIN_ONLY_VALVE_FIELDS: Final[set[str]] = (
    set(Pipe.Valves.model_fields.keys()) - USER_OVERRIDABLE_VALVE_FIELDS
)

_VALVE_FIELD_LOOKUP: Final[dict[str, str]] = {
    field_name.lower(): field_name for field_name in Pipe.Valves.model_fields.keys()
}

_SYNONYM_LOOKUP: Final[dict[str, str]] = {
    synonym.lower(): canonical_target
    for synonym, canonical_target in OPTION_SYNONYMS.items()
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


class ResolvedOptions(Pipe.Valves):
    """
    Consolidated configuration combining Admin Valves, User Valves,
    Model Advanced Params, and Chat Advanced Params according to priority.

    Allows extra dynamic options defined at model or chat levels.
    """

    model_config = ConfigDict(extra="allow")

    def get_custom_params(self) -> dict[str, Any]:
        """Returns extra parameters not defined in Pipe.Valves."""
        valve_fields = set(Pipe.Valves.model_fields.keys())
        return {k: v for k, v in self.model_dump().items() if k not in valve_fields}


def _resolve_options(
    admin_valves: Pipe.Valves,
    user_valves: _SharedValves | None,
    user_email: str,
    body: "Body",
    metadata: "Metadata",
) -> ResolvedOptions:
    """
    Hierarchically resolves configuration options from 4 priority sources:
    1. Admin Valves (Lowest priority)
    2. User Valves
    3. Model Page Advanced Params
    4. Chat Side-Panel Advanced Params (Highest priority)

    Admin-only options cannot be overridden by user-level sources.
    Option keys and synonyms are normalized case-insensitively.
    """
    # Priority 1: Base options from Admin Valves
    merged_data = admin_valves.model_dump()

    # Priority 2: User Valves (user-overridable fields only)
    if user_valves is not None:
        for field_name in USER_OVERRIDABLE_VALVE_FIELDS:
            user_val = getattr(user_valves, field_name, None)
            if user_val is not None and user_val != "":
                merged_data[field_name] = user_val

    # Priority 3: Model Page Advanced Params
    known_body_keys = {
        "stream",
        "model",
        "messages",
        "files",
        "options",
        "stream_options",
    }
    model_params = {k: v for k, v in body.items() if k not in known_body_keys}
    if isinstance(body.get("options"), dict):
        for opt_k, opt_v in body["options"].items():  # type: ignore[reportTypedDictNotRequiredAccess]
            if opt_k not in model_params:
                model_params[opt_k] = opt_v

    for raw_key, val in model_params.items():
        if val is None or val == "":
            continue
        canonical_key = canonicalize_option_key(raw_key)
        if canonical_key in ADMIN_ONLY_VALVE_FIELDS:
            log.warning(
                f"Model parameter '{raw_key}' attempts to override admin-only valve '{canonical_key}'. Ignoring override."
            )
            continue
        merged_data[canonical_key] = val

    # Priority 4: Chat Side-Panel Advanced Params (Skipped for task models)
    if not metadata.get("task"):
        chat_params = metadata.get("chat_control_params", {})
        if isinstance(chat_params, dict):
            for raw_key, val in chat_params.items():
                if val is None or val == "":
                    continue
                canonical_key = canonicalize_option_key(raw_key)
                if canonical_key in ADMIN_ONLY_VALVE_FIELDS:
                    log.warning(
                        f"Chat parameter '{raw_key}' attempts to override admin-only valve '{canonical_key}'. Ignoring override."
                    )
                    continue
                merged_data[canonical_key] = val
    else:
        log.debug(
            f"Task model detected ('{metadata.get('task')}'). Chat side-panel parameters ignored."
        )

    return ResolvedOptions(**merged_data)


# endregion Option resolution
