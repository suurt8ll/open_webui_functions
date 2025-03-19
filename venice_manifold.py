"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.7.0
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.

# TODO Use another LLM model to generate the image prompt?

import asyncio
import sys
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Iterator,
    NotRequired,
    TypedDict,
    Literal,
    Callable,
    Awaitable,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
import requests
import aiohttp
import time
import base64
from open_webui.routers.images import upload_image
from open_webui.models.users import Users
from open_webui.utils.logger import stdout_format
from loguru import logger

if TYPE_CHECKING:
    from loguru import Record
    from loguru._handler import Handler


class StatusEventData(TypedDict):
    description: str
    done: bool
    hidden: bool


class ChatEventData(TypedDict):
    type: Literal["status"]
    data: StatusEventData


class UserData(TypedDict):
    id: str
    email: str
    name: str
    role: Literal["admin", "user", "pending"]
    valves: NotRequired[Any]  # object of type UserValves


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

    # FIXME Make it async.
    def pipes(self) -> list[dict]:
        self._add_log_handler()
        try:
            models = self._get_models()
            log.debug("Got models:", models=models)
            return [{"id": model["id"], "name": model["id"]} for model in models]
        except Exception:
            error_msg = "Error getting models:"
            log.exception(error_msg)
            return []

    async def pipe(
        self,
        body: dict,
        __user__: UserData,
        __request__: Request,
        __event_emitter__: Callable[[ChatEventData], Awaitable[None]],
        __task__: str,
    ) -> str | dict | StreamingResponse | Iterator | AsyncGenerator | Generator:

        if not self.valves.VENICE_API_TOKEN:
            return "Error: Missing VENICE_API_TOKEN in valves configuration"

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
            return "Error: No prompt found in user message"

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

        try:
            image_data = await self._generate_image(model, prompt)
        finally:
            timer.cancel()  # Always cancel the timer, even if _generate_image fails
            try:
                await timer  # Ensure timer is fully cleaned up
            except asyncio.CancelledError:
                pass  # Expected, already handled

        total_time = time.time() - start_time
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Image generated in {total_time:.2f}s",
                    "done": True,
                    "hidden": False,
                },
            }
        )

        if image_data and image_data.get("images"):
            log.info("Image generated successfully!")
            base64_image = image_data["images"][0]

            if self.valves.USE_FILES_API:
                # Decode the base64 image data
                image_data = base64.b64decode(base64_image)
                # FIXME make mime type dynamic
                image_url = self._upload_image(
                    image_data, "image/png", model, prompt, __user__, __request__
                )
                return f"![Generated Image]({image_url})"
            else:
                return f"![Generated Image](data:image/png;base64,{base64_image})"

        log.error("Image generation failed.")

        total_time = time.time() - start_time
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Image generation failed after {total_time:.2f}s",
                    "done": True,
                    "hidden": False,
                },
            }
        )
        return "Error: Failed to generate image"

    """Helper functions inside the Pipe class."""

    def _get_models(self) -> list[dict[str, Any]]:
        try:
            response = requests.get(
                "https://api.venice.ai/api/v1/models?type=image",
                headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json().get("data", [])
        except requests.exceptions.RequestException:
            error_msg = "Error getting models:"
            log.exception(error_msg)
            return []
        except Exception:
            error_msg = "An unexpected error occurred:"
            log.exception(error_msg)
            return []

    async def _generate_image(self, model: str, prompt: str) -> dict | None:
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

        except aiohttp.ClientResponseError:
            error_msg = "Image generation failed:"
            log.exception(error_msg)
            return None
        except Exception:
            error_msg = "Generation error:"
            log.exception(error_msg)
            return None

    def _upload_image(
        self,
        image_data: bytes,
        mime_type: str,
        model: str,
        prompt: str,
        __user__: UserData,
        __request__: Request,
    ) -> str:

        # Create metadata for the image
        image_metadata = {
            "model": model,
            "prompt": prompt,
        }

        # Get the *full* user object from the database
        user = Users.get_user_by_id(__user__["id"])
        if user is None:
            return "Error: User not found"

        # Upload the image using the imported function
        image_url: str = upload_image(
            request=__request__,
            image_metadata=image_metadata,
            image_data=image_data,
            content_type=mime_type,
            user=user,
        )
        log.info(f"Image uploaded. URL: {image_url}")
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
