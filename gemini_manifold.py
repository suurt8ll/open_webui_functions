"""
title: Gemini Manifold (google-genai)
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.1.0
"""

import os
import re
from typing import Iterator, List, Union
from google import genai
from google.genai import _api_client
from pydantic import BaseModel, Field

DEBUG = True


class Pipe:

    class Valves(BaseModel):
        GEMINI_API_KEY: str = Field(default="")
        MODEL_WHITELIST: str = Field(
            default="", description="Comma-separated list of allowed model names"
        )

    def __init__(self):
        try:
            self.valves = self.Valves()
            if DEBUG:
                print("[INIT] Initialized Pipe with Valves configuration.")
        except Exception as e:
            if DEBUG:
                print(f"[INIT] Error during initialization: {e}")
        finally:
            if DEBUG:
                print("[INIT] Initialization complete.")

    def __get_google_models(self):
        """Retrieve Google models with prefix stripping."""
        try:
            whitelist = (
                self.valves.MODEL_WHITELIST.split(",")
                if self.valves.MODEL_WHITELIST
                else []
            )
            models = self.client.models.list(config={"query_base": True})
            if DEBUG:
                print(
                    f"[get_google_models] Retrieved {len(models)} models from Gemini Developer API."
                )
            return [
                {
                    "id": self.__strip_prefix(model.name),
                    "name": model.display_name,
                }
                for model in models
                if not whitelist or model.name in [f"models/{w}" for w in whitelist]
                if model.supported_actions
                and "generateContent" in model.supported_actions
                if model.name and model.name.startswith("models/")
            ]
        except Exception as e:
            if DEBUG:
                print(f"[get_google_models] Error retrieving models: {e}")
            return [
                {
                    "id": "error",
                    "name": "Error retrieving models. Please check the logs.",
                }
            ]

    def __strip_prefix(self, model_name: str) -> str:
        """
        Strip any prefix from the model name up to and including the first '.' or '/'.
        This makes the method generic and adaptable to varying prefixes.
        """
        try:
            # Use non-greedy regex to remove everything up to and including the first '.' or '/'
            stripped = re.sub(r"^.*?[./]", "", model_name)
            if DEBUG:
                print(
                    f"[strip_prefix] Stripped prefix: '{stripped}' from '{model_name}'"
                )
            return stripped
        except Exception as e:
            if DEBUG:
                print(f"[strip_prefix] Error stripping prefix: {e}")
            return model_name  # Return original if stripping fails
        finally:
            if DEBUG:
                print("[strip_prefix] Completed prefix stripping.")

    def pipes(self) -> List[dict]:
        """Register all available Google models."""
        try:
            if not self.valves.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set.")
            # GEMINI_API_KEY is not available inside __init__ for whatever reason so we initialize the client here
            self.client = genai.Client(
                api_key=self.valves.GEMINI_API_KEY,
                http_options=_api_client.HttpOptions(api_version="v1alpha"),
            )
            models = self.__get_google_models()
            if DEBUG:
                print(f"[pipes] Registered models: {models}")
            return models
        except Exception as e:
            if DEBUG:
                print(f"[pipes] Error in pipes method: {e}")
            return []
        finally:
            if DEBUG:
                print("[pipes] Completed pipes method.")

    async def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        """Main pipe method to process incoming requests."""
        if DEBUG:
            print(f"[pipe] Received request: {body}")
        return "AMOGUS"
