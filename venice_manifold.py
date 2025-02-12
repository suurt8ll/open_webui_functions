"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.3.0
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.

# TODO Improve logging by using something better than print statements.
# TODO Use another LLM model to generate the image prompt?

from typing import AsyncGenerator, Generator, Iterator, Union
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import requests
import aiohttp
import time


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
        models = self._get_models()
        return [{"id": model["id"], "name": model["id"]} for model in models]

    async def pipe(
        self, body: dict
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
            print(f"[pipe] Model: {model}, Prompt: {prompt}")

        image_data = await self._generate_image(model, prompt)

        if image_data and image_data.get("images"):
            if self.valves.DEBUG:
                print(f"[pipe] Image generated successfully")
            base64_image = image_data["images"][0]
            return f"![Generated Image](data:image/png;base64,{base64_image})"

        if self.valves.DEBUG:
            print(f"[pipe] Image generation failed.")
        return "Error: Failed to generate image"

    def _get_models(self) -> list:
        try:
            response = requests.get(
                "https://api.venice.ai/api/v1/models?type=image",
                headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
            )
            if response.status_code == 200:
                return response.json().get("data", [])
            return []
        except Exception:
            return []

    async def _generate_image(self, model: str, prompt: str) -> Union[dict, None]:
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                print(
                    f"[_generate_image] Sending image generation request to Venice.ai for model: {model}"
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
                    print(
                        f"[_generate_image] Received response from Venice.ai with status: {response.status}"
                    )
                    if response.status == 200:
                        json_response = await response.json()
                        end_time = time.time()
                        generation_time = end_time - start_time
                        print(
                            f"[_generate_image] Image generation successful. Time taken: {generation_time:.2f} seconds"
                        )
                        return json_response
                    else:
                        end_time = time.time()
                        generation_time = end_time - start_time
                        print(
                            f"[_generate_image] Image generation failed with status: {response.status}. Time taken: {generation_time:.2f} seconds"
                        )
                        return None
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            print(
                f"[_generate_image] Generation error: {str(e)}. Time taken: {generation_time:.2f} seconds"
            )
            return None
