"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.6.0
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.

# TODO Use another LLM model to generate the image prompt?
# TODO Option to save the generated images onto disk, bypassing database?

import asyncio
import json
import traceback
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
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
import requests
import aiohttp
import time
import inspect

COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "RESET": "\033[0m",
}


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


class Pipe:
    class Valves(BaseModel):
        VENICE_API_TOKEN: str = Field(default="", description="Venice.ai API Token")
        HEIGHT: int = Field(default=1024, description="Image height")
        WIDTH: int = Field(default=1024, description="Image width")
        STEPS: int = Field(default=16, description="Image generation steps")
        CFG_SCALE: int = Field(default=4, description="Image generation scale")
        LOG_LEVEL: Literal["INFO", "WARNING", "ERROR", "DEBUG", "OFF"] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    def __init__(self):
        self.valves = self.Valves()

    # FIXME Make it async.
    def pipes(self) -> list[dict]:
        try:
            models = self._get_models()
            self._print_colored("Got models:", "DEBUG")
            if self.valves.LOG_LEVEL == "DEBUG":
                print(json.dumps(models, indent=2, default=str))
            return [{"id": model["id"], "name": model["id"]} for model in models]
        except Exception as e:
            error_msg = f"Error getting models: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
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
            self._print_colored(
                "Detected title generation task! I do not know how to handle this so I'm returning something generic.",
                "WARNING",
            )
            return '{"title": "🖼️ Image Generation"}'
        if __task__ == "tags_generation":
            self._print_colored(
                "Detected tag generation task! I do not know how to handle this so I'm returning an empty list.",
                "WARNING",
            )
            return '{"tags": []}'

        self._print_colored(f"Model: {model}, Prompt: {prompt}", "DEBUG")

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
                self._print_colored("Timer task cancelled.", "DEBUG")

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

        if image_data and image_data.get("images"):
            self._print_colored("Image generated successfully", "INFO")
            base64_image = image_data["images"][0]

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

            return f"![Generated Image](data:image/png;base64,{base64_image})"

        self._print_colored("Image generation failed.", "ERROR")

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
        except requests.exceptions.RequestException as e:
            error_msg = f"Error getting models: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return []
        except Exception as e:
            error_msg = (
                f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
            )
            self._print_colored(error_msg, "ERROR")
            return []

    async def _generate_image(self, model: str, prompt: str) -> dict | None:
        try:
            async with aiohttp.ClientSession() as session:
                self._print_colored(
                    f"Sending image generation request to Venice.ai for model: {model}",
                    "INFO",
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
                    self._print_colored(
                        f"Received response from Venice.ai with status: {response.status}",
                        "INFO",
                    )
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientResponseError as e:
            error_msg = f"Image generation failed with status: {str(e.status)}. Error: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return None
        except Exception as e:
            error_msg = f"Generation error: {str(e)}\n{traceback.format_exc()}"
            self._print_colored(error_msg, "ERROR")
            return None

    def _print_colored(self, message: str, level: str = "INFO") -> None:
        """
        Prints a colored log message to the console, respecting the configured log level.
        """
        if not hasattr(self, "valves") or self.valves.LOG_LEVEL == "OFF":
            return

        # Define log level hierarchy
        level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

        # Only print if message level is >= configured level
        if level_priority.get(level, 0) >= level_priority.get(self.valves.LOG_LEVEL, 0):
            color_map = {
                "INFO": COLORS["GREEN"],
                "WARNING": COLORS["YELLOW"],
                "ERROR": COLORS["RED"],
                "DEBUG": COLORS["BLUE"],
            }
            color = color_map.get(level, COLORS["WHITE"])
            frame = inspect.currentframe()
            if frame:
                frame = frame.f_back
            method_name = frame.f_code.co_name if frame else "<unknown>"
            print(
                f"{color}[{level}][venice_manifold][{method_name}]{COLORS['RESET']} {message}"
            )
