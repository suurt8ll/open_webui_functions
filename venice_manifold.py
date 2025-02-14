"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.4.0
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.

# TODO Improve logging by using something better than print statements.
# TODO Use another LLM model to generate the image prompt?
# TODO Option to save the generated images onto disk, bypassing database?

import asyncio
from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Union,
    TypedDict,
    Literal,
    Callable,
    Awaitable,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
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


def print_colored(message: str, level: str = "INFO") -> None:
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
        f"{color}[{level}][venice_image_generation][{method_name}]{COLORS['RESET']} {message}"
    )


class StatusEventData(TypedDict):
    description: str
    done: bool
    hidden: bool


class ChatEventData(TypedDict):
    type: Literal["status"]
    data: StatusEventData


class Pipe:
    class Valves(BaseModel):
        VENICE_API_TOKEN: str = Field(default="", description="Venice.ai API Token")
        HEIGHT: int = Field(default=1024, description="Image height")
        WIDTH: int = Field(default=1024, description="Image width")
        STEPS: int = Field(default=16, description="Image generation steps")
        CFG_SCALE: int = Field(default=4, description="Image generation scale")
        DEBUG: bool = Field(default=False, description="Enable debug logging")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        try:
            models = self._get_models()
            return [{"id": model["id"], "name": model["id"]} for model in models]
        except Exception as e:
            print_colored(f"Error getting models: {e}", "ERROR")
            return []

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[ChatEventData], Awaitable[None]],
    ) -> Union[str, dict, StreamingResponse, Iterator, AsyncGenerator, Generator]:
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

        if self.valves.DEBUG:
            print_colored(f"Model: {model}, Prompt: {prompt}", "DEBUG")

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
                print_colored("Timer task cancelled.", "DEBUG")

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
            if self.valves.DEBUG:
                print_colored("Image generated successfully", "INFO")
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

        if self.valves.DEBUG:
            print_colored("Image generation failed.", "ERROR")

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

    def _get_models(self) -> list:
        try:
            response = requests.get(
                "https://api.venice.ai/api/v1/models?type=image",
                headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print_colored(f"Error getting models: {e}", "ERROR")
            return []
        except Exception as e:
            print_colored(f"An unexpected error occurred: {e}", "ERROR")
            return []

    async def _generate_image(self, model: str, prompt: str) -> Union[dict, None]:
        try:
            async with aiohttp.ClientSession() as session:
                print_colored(
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
                    print_colored(
                        f"Received response from Venice.ai with status: {response.status}",
                        "INFO",
                    )
                    response.raise_for_status()
                    return await response.json()

        except aiohttp.ClientResponseError as e:
            print_colored(
                f"Image generation failed with status: {e.status}. Error: {e}", "ERROR"
            )
            return None
        except Exception as e:
            print_colored(f"Generation error: {str(e)}", "ERROR")
            return None
