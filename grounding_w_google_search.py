"""
title: Grounding with Google Search
id: grounding_with_google_search
description: Filter function that gives Gemini models search ablities. Must be used with "Gemini Manifold google_genai" pipe function!
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.2.0
"""

import json
from pydantic import BaseModel, Field


class Filter:

    class Valves(BaseModel):
        SET_TEMP_TO_ZERO: bool = Field(
            default=False,
            description="Decide if you want to set the temperature to 0 for grounded answers, Google reccomends it in their docs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        print("[grounding_w_google_search] Filter function has been initialized!")

    def inlet(self, body: dict) -> dict:
        """Modifies the incoming request payload before it's sent to the LLM. Operates on the `form_data` dictionary."""
        features = body.get("features", {})
        web_search_enabled = (
            features.get("web_search", False) if isinstance(features, dict) else False
        )

        if web_search_enabled:
            print(
                "Search feature is enabled, disabling it and adding custom feature called grounding_w_google_search."
            )
            # Disable web_search
            features["web_search"] = False
            # Ensure metadata structure exists and add new feature
            metadata = body.setdefault("metadata", {})
            metadata_features = metadata.setdefault("features", {})
            metadata_features["grounding_w_google_search"] = True
            # Google suggest setting temperature to 0 if using grounding:
            # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-with-google-search#:~:text=For%20ideal%20results%2C%20use%20a%20temperature%20of%200.0.
            if self.valves.SET_TEMP_TO_ZERO:
                body["temperature"] = 0

        print(f"Returning body:\n{json.dumps(body, indent=2, default=str)}")
        return body

    def stream(self, event: dict) -> dict:
        """Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data."""
        return event

    def outlet(self, body: dict) -> dict:
        """Modifies the complete response payload after it's received from the LLM. Operates on the final `body` dictionary."""
        # TODO: Filter out the citation markers here.
        return body
