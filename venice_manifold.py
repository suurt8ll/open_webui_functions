"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.9.3
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.

# TODO: Use another LLM model to generate the image prompt?
# TODO: Negative prompts
# TODO: Upscaling

import io
import mimetypes
import os
import sys
import time
import asyncio
import uuid
import aiohttp
import base64
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Iterator,
    Literal,
    Callable,
    Awaitable,
    Optional,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
from open_webui.models.files import Files, FileForm
from open_webui.utils.logger import stdout_format
from open_webui.storage.provider import Storage
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler
    from manifold_types import *  # My personal types in a separate file for more robustness.


# Setting auditable=False avoids duplicate output for log levels that would be printed out by the main logger.
log = logger.bind(auditable=False)


class Pipe:
    class Valves(BaseModel):
        VENICE_API_TOKEN: str = Field(default="", description="Venice.ai API Token")
        HEIGHT: int = Field(default=1024, description="Image height")
        WIDTH: int = Field(default=1024, description="Image width")
        STEPS: int = Field(default=16, description="Image generation steps")
        CFG_SCALE: int = Field(default=4, description="Image generation scale")
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )
        USE_FILES_API: bool = Field(
            title="Use Files API",
            default=True,
            description="Save the image files using Open WebUI's API for files.",
        )

    def __init__(self):
        self.valves = self.Valves()
        print("[venice_manifold] Function has been initialized!")

    async def pipes(self) -> list["ModelData"]:
        # I'm adding the handler here because LOG_LEVEL is not set inside __init__ sadly.
        self._add_log_handler()
        return await self._get_models()

    async def pipe(
        self,
        body: dict,
        __user__: "UserData",
        __request__: Request,
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        __task__: str,
        __metadata__: dict[str, Any],
    ) -> str | dict | StreamingResponse | Iterator | AsyncGenerator | Generator | None:

        self.__event_emitter__ = __event_emitter__

        if "error" in __metadata__["model"]["id"]:
            error_msg = f'There has been an error during model retrival phase: {str(__metadata__["model"])}'
            await self._emit_error(error_msg, exception=False)
            return

        if not self.valves.VENICE_API_TOKEN:
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

        image_data = await self._generate_image(model, prompt)

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

        if self.valves.USE_FILES_API:
            # Decode the base64 image data
            image_data = base64.b64decode(base64_image)
            # FIXME make mime type dynamic
            image_url = self._upload_image(
                image_data, "image/png", model, prompt, __user__["id"], __request__
            )
            return f"![Generated Image]({image_url})" if image_url else None
        else:
            return f"![Generated Image](data:image/png;base64,{base64_image})"

    """
    ---------- Helper functions inside the Pipe class. ----------
    """

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
                        {"id": model["id"], "name": model["id"]} for model in raw_models
                    ]
        except aiohttp.ClientResponseError as e:
            error_msg = f"Error getting models: {str(e)}"
            return [self._return_error_model(error_msg)]
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            return [self._return_error_model(error_msg)]

    async def _generate_image(self, model: str, prompt: str) -> Optional[dict]:
        try:
            async with aiohttp.ClientSession() as session:
                log.info(
                    f"Sending image generation request to Venice.ai for model: {model}"
                )
                async with session.post(
                    "https://api.venice.ai/api/v1/image/generate",
                    headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
                    json={
                        "model": model,
                        "prompt": prompt,
                        "width": self.valves.WIDTH,
                        "height": self.valves.HEIGHT,
                        "steps": self.valves.STEPS,
                        "hide_watermark": True,
                        "return_binary": False,
                        "cfg_scale": self.valves.CFG_SCALE,
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
        log.info("Uploading the image to the configured storage provider.")
        try:
            contents, image_path = Storage.upload_file(image, imagename)
        except Exception:
            error_msg = f"Error occurred during upload to the storage provider."
            log.exception(error_msg)
            return None
        # Add the image file to files database.
        log.info("Adding the image file to Open WebUI files database.")
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
            log.warning("Files.insert_new_file did not return anything.")
            return None
        # Get the image url.
        image_url: str = __request__.app.url_path_for(
            "get_file_content_by_id", id=file_item.id
        )
        return image_url

    def _add_log_handler(self):
        """Adds handler to the root loguru instance for this plugin if one does not exist already."""

        def plugin_filter(record: "Record"):
            """Filter function to only allow logs from this plugin (based on module name)."""
            return record["name"] == __name__  # Filter by module name

        # Access the internal state of the logger
        handlers: dict[int, "Handler"] = logger._core.handlers  # type: ignore
        for key, handler in handlers.items():
            existing_filter = handler._filter
            if (
                hasattr(existing_filter, "__name__")
                and existing_filter.__name__ == plugin_filter.__name__
                and hasattr(existing_filter, "__module__")
                and existing_filter.__module__ == plugin_filter.__module__
            ):
                log.debug("Handler for this plugin is already present!")
                return

        logger.add(
            sys.stdout,
            level=self.valves.LOG_LEVEL,
            format=stdout_format,
            filter=plugin_filter,
        )
        log.info(
            f"Added new handler to loguru with level {self.valves.LOG_LEVEL} and filter {__name__}."
        )

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
        }
