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
from pydantic import BaseModel, Field
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


_SHARED_VALVE_DESCS = {
    "HEIGHT": "Height of generated images in pixels.",
    "WIDTH": "Width of generated images in pixels.",
    "STEPS": "Number of inference and generation steps.",
    "CFG_SCALE": "Classifier-Free Guidance (CFG) scale determining how closely the generation follows the prompt.",
}

_ADMIN_VALVE_DESCS = {
    "VENICE_API_TOKEN": "Venice.ai API Token.",
    "CACHE_MODELS": "Whether to request and cache available models only on initial load.",
    "LOG_LEVEL": "Select logging verbosity level. View logs using `docker logs -f open-webui`.",
    "USE_FILES_API": "Save generated image files using Open WebUI's file storage API.",
}


def _format_valve_desc(text: str, default: Any = None, is_user: bool = False) -> str:
    """Formats Markdown descriptions for Valves and UserValves fields."""
    text = text.strip()
    if is_user:
        return f"{text}\n\n*If not set, the admin's setting is used.*"
    formatted_default = f"`{default}`" if default is not None else "`None`"
    return f"{text}\n\n**Default:** {formatted_default}"


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
        CACHE_MODELS: bool = Field(
            default=True,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["CACHE_MODELS"], default=True
            ),
        )
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["LOG_LEVEL"], default="INFO"
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
        self.log_level = self.valves.LOG_LEVEL
        self._add_log_handler()

        self.models: list["ModelData"] = []

        log.success("Function has been initialized.")
        log.trace("Full self object:", payload=self.__dict__)

    async def pipes(self) -> list["ModelData"]:
        if self.log_level != self.valves.LOG_LEVEL:
            log.info(
                f"Detected log level change: {self.log_level=} and {self.valves.LOG_LEVEL=}. "
                "Running the logging setup again."
            )
            self.log_level = self.valves.LOG_LEVEL
            self._add_log_handler()

        if (
            self.models
            and self.valves.CACHE_MODELS
            and not any(model["id"] == "error" for model in self.models)
        ):
            log.info("Models are already initialized. Returning the cached list.")
            return self.models

        self.models = await self._get_models()
        return self.models

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

        # FIXME: Bad idea, every chat turn has it's own unique emitter and using old one would lead to very weird behaviour.
        self.__event_emitter__ = __event_emitter__

        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrival phase: {str(__metadata__["model"])}'
            await self._emit_error(error_msg, exception=False)
            return

        if not valves.VENICE_API_TOKEN:
            error_msg = "Missing VENICE_API_TOKEN in valves configuration."
            await self._emit_error(error_msg, exception=False)
            return

        model = body.get("model", "").split(".", 1)[-1]
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
            await self._emit_error(error_msg, exception=False)
            return

        # FIXME move these to the beginning.
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

        log.debug(f"Model: {model}, Prompt: {prompt}")

        # FIXME [refac] Move it out of pipe for cleaner code?
        async def timer_task(start_time: float):
            """Counts up and emits status updates."""
            try:
                while True:
                    elapsed_time = time.time() - start_time
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Generating image... Time elapsed: {elapsed_time:.2f}s",
                                "done": False,
                                "hidden": False,
                            },
                        }
                    )
                    await asyncio.sleep(1)  # Update every second
            except asyncio.CancelledError:
                log.debug("Timer task cancelled.")

        start_time = time.time()
        timer = asyncio.create_task(timer_task(start_time))

        image_data = await self._generate_image(model, prompt, valves)

        timer.cancel()
        try:
            await timer  # Ensure timer is fully cleaned up
        except asyncio.CancelledError:
            pass  # Expected, already handled

        total_time = time.time() - start_time
        success = image_data and image_data.get("images")
        status_text = f"Image {'generated' if success else 'generation failed'} after {total_time:.2f}s"

        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": status_text,
                    "done": True,
                    "hidden": False,
                },
            }
        )
        if not success:
            return None

        log.info("Image generated successfully!")
        base64_image = image_data["images"][0]  # type: ignore

        if valves.USE_FILES_API:
            # Decode the base64 image data
            image_data = base64.b64decode(base64_image)
            # FIXME make mime type dynamic
            image_url = await self._upload_image(
                image_data, "image/png", model, prompt, __user__["id"], __request__
            )
            return f"![Generated Image]({image_url})" if image_url else None
        else:
            return f"![Generated Image](data:image/png;base64,{base64_image})"

    # region 1. Helper methods inside the Pipe class

    # region 1.1 Model retrieval

    async def _get_models(self) -> list["ModelData"]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.venice.ai/api/v1/models?type=image",
                    headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
                ) as response:
                    response.raise_for_status()
                    raw_models = await response.json()
                    raw_models = raw_models.get("data", [])
                    if not raw_models:
                        log.warning("Venice API returned no models.")
                    return [
                        {"id": model["id"], "name": model["id"], "description": None}
                        for model in raw_models
                    ]
        except aiohttp.ClientResponseError as e:
            error_msg = f"Error getting models: {str(e)}"
            return [self._return_error_model(error_msg)]
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            return [self._return_error_model(error_msg)]

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
            "name": "[venice_manifold] " + error_msg,
            "description": error_msg,
        }

    # endregion 1.1 Model retrieval

    # region 1.2 Image generation

    async def _generate_image(self, model: str, prompt: str, valves: "Pipe.Valves") -> dict | None:
        try:
            async with aiohttp.ClientSession() as session:
                log.info(
                    f"Sending image generation request to Venice.ai for model: {model}"
                )
                async with session.post(
                    "https://api.venice.ai/api/v1/image/generate",
                    headers={"Authorization": f"Bearer {valves.VENICE_API_TOKEN}"},
                    json={
                        "model": model,
                        "prompt": prompt,
                        "width": valves.WIDTH,
                        "height": valves.HEIGHT,
                        "steps": valves.STEPS,
                        "hide_watermark": True,
                        "return_binary": False,
                        "cfg_scale": valves.CFG_SCALE,
                        "safe_mode": False,
                    },
                ) as response:
                    log.info(
                        f"Received response from Venice.ai with status: {response.status}"
                    )
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientResponseError as e:
            error_msg = f"Image generation failed: {str(e)}"
            await self._emit_error(error_msg)
            return
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            await self._emit_error(error_msg)
            return

    async def _upload_image(
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

        # `Storage.upload_file` remains synchronous upstream; run it in a thread so it
        # doesn't block the event loop now that this method is async.
        try:
            # TODO: Remove this in the future.
            if has_tags:
                # New version with tags support >=v0.6.6
                contents, image_path = await asyncio.to_thread(
                    Storage.upload_file, image, imagename, tags={}
                )
            else:
                # Old version without tags <v0.6.5
                contents, image_path = await asyncio.to_thread(
                    Storage.upload_file, image, imagename  # type: ignore
                )
        except Exception:
            error_msg = "Error occurred during upload to the storage provider."
            log.exception(error_msg)
            return None
        # Add the image file to files database.
        # Open WebUI >= 0.9.0 made `Files.insert_new_file` async.
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
        # Get the image url.
        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        return image_url

    # endregion 1.2 Image generation

    # region 1.3 Event emissions

    async def _emit_error(
        self, error_msg: str, warning: bool = False, exception: bool = True
    ) -> None:
        """Emits an event to the front-end that causes it to display a nice red error message."""
        error: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {
                "done": True,
                "error": {"detail": "\n" + error_msg},
            },
        }
        if warning:
            log.opt(depth=1, exception=False).warning(error_msg)
        else:
            log.opt(depth=1, exception=exception).error(error_msg)
        await self.__event_emitter__(error)

    # endregion 1.3 Event emissions

    # region 1.4 Logging
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

    # endregion 1.4 Logging

    # endregion 1. Helper methods inside the Pipe class
