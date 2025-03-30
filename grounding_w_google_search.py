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


class Filter:

    def __init__(self):
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

        return body

    def stream(self, event: dict) -> dict:
        """Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data."""
        return event

    def outlet(self, body: dict) -> dict:
        """Modifies the complete response payload after it's received from the LLM. Operates on the final `body` dictionary."""
        # TODO: Filter out the citation markers here.
        return body
