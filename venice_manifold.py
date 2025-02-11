"""
title: Venice Image Generation
id: venice_image_generation
description: Generate images using Venice.ai's API.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.1.0
"""

# NB! This is work in progress and not yet fully featured.
# Feel free to contribute to the development of this function in my GitHub repository!
# Currently it takes the last user message as prompt and generates an image using the selected model and returns it as a markdown image.
# Hard-coded parameters are used for image generation, but these will be configurable in the future.
# Parameters: width=512, height=512, hide_watermark=True, steps=16, cfg_scale=4, safe_mode=False

from typing import AsyncGenerator, Generator, Iterator, Union
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import json
import requests


class Pipe:
    class Valves(BaseModel):
        # TODO Allow user to set image generation parameters here.
        VENICE_API_TOKEN: str = Field(default="", description="Venice.ai API Token")
        DEBUG: bool = Field(default=False, description="Enable debug logging")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        models = self._get_models()
        return [{"id": model["id"], "name": model["id"]} for model in models]

    def pipe(
        self, body: dict
    ) -> Union[str, dict, StreamingResponse, Iterator, AsyncGenerator, Generator]:
        if not self.valves.VENICE_API_TOKEN:
            return "Error: Missing VENICE_API_TOKEN in valves configuration"

        model = body.get("model", "").split(".")[-1]
        # TODO Allow piping the prompt to another LLM that generates the full image generation prompt?
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

        image_data = self._generate_image(model, prompt)

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
                image_models = response.json().get("data", [])

                if self.valves.DEBUG:
                    print(
                        f"[_get_models] Retrieved {len(image_models)} image models from API"
                    )
                    print(
                        f"[_get_models] Raw model data: {json.dumps(image_models, indent=2)}"
                    )

                return image_models

            if self.valves.DEBUG:
                print(
                    f"[_get_models] Model retrieval failed: {response.status_code} - {response.text}"
                )
            return []

        except Exception as e:
            if self.valves.DEBUG:
                print(f"[_get_models] Model retrieval error: {str(e)}")
            return []

    def _generate_image(self, model: str, prompt: str) -> Union[dict, None]:
        try:
            response = requests.post(
                "https://api.venice.ai/api/v1/image/generate",
                headers={"Authorization": f"Bearer {self.valves.VENICE_API_TOKEN}"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "width": 512,
                    "height": 512,
                    "steps": 16,
                    "hide_watermark": True,
                    "return_binary": False,
                    "cfg_scale": 4,
                    "safe_mode": False,
                },
            )

            if self.valves.DEBUG:
                print(
                    f"[_generate_image] Generation response: {response.status_code} - {response.text[:200]}..."
                )

            return response.json() if response.status_code == 200 else None

        except Exception as e:
            if self.valves.DEBUG:
                print(f"[_generate_image] Generation error: {str(e)}")
            return None
