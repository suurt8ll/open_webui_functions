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
    Any,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from fastapi import Request
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from loguru import logger

if TYPE_CHECKING:
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
        CACHE_MODELS: bool = Field(
            default=True,
            description=_format_valve_desc(
                _ADMIN_VALVE_DESCS["CACHE_MODELS"], default=True
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
        self.models: list["ModelData"] = []
        log.success("Function has been initialized.")

    async def pipes(self) -> list["ModelData"]:
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

    # endregion 1. Helper methods inside the Pipe class
