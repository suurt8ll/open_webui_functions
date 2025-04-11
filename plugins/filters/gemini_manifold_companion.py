"""
title: Gemini Manifold Companion
id: gemini_manifold_companion
description: A companion filter for "Gemini Manifold google_genai" pipe providing enhanced functionality.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

# This filter can detect that a feature like web search or code execution is enabled in the front-end,
# set the feature back to False so Open WebUI does not run it's own logic and then
# pass custom values to "Gemini Manifold google_genai" that signal which feature was enabled and intercepted.

from pydantic import BaseModel, Field

# according to https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-gemini
ALLOWED_GROUNDING_MODELS = [
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-001",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro",
]

# according to https://ai.google.dev/gemini-api/docs/code-execution
ALLOWED_CODE_EXECUTION_MODELS = [
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-001",
]


class Filter:

    class Valves(BaseModel):
        SET_TEMP_TO_ZERO: bool = Field(
            default=False,
            description="""Decide if you want to set the temperature to 0 for grounded answers, 
            Google reccomends it in their docs.""",
        )
        GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD: float | None = Field(
            default=None,
            description="""See https://ai.google.dev/gemini-api/docs/grounding?lang=python#dynamic-threshold for more information.
            Only supported for 1.0 and 1.5 models""",
        )

    def __init__(self):
        self.valves = self.Valves()
        print("[gemini_manifold_companion] Filter function has been initialized!")

    def inlet(self, body: dict) -> dict:
        """Modifies the incoming request payload before it's sent to the LLM. Operates on the `form_data` dictionary."""

        # Exit early if we are filtering an unsupported model.
        model_name: str = body.get("model", "")

        # Extract and use base model name in case of custom Workspace models
        metadata = body.get("metadata", {})
        base_model_name = (
            metadata.get("model", {}).get("info", {}).get("base_model_id", None)
        )
        if base_model_name:
            model_name = base_model_name

        canonical_model_name = model_name.replace("gemini_manifold_google_genai.", "")

        if (
            "gemini_manifold_google_genai." not in model_name
            or canonical_model_name not in ALLOWED_GROUNDING_MODELS
            or canonical_model_name not in ALLOWED_CODE_EXECUTION_MODELS
        ):
            return body

        features = body.get("features", {})
        print(f"!!!!!!! Features: {features}")

        # Ensure metadata structure exists and add new feature
        metadata = body.setdefault("metadata", {})
        metadata_features = metadata.setdefault("features", {})

        if canonical_model_name in ALLOWED_GROUNDING_MODELS:
            web_search_enabled = (
                features.get("web_search", False)
                if isinstance(features, dict)
                else False
            )
            if web_search_enabled:
                print(
                    "[gemini_manifold_companion] Search feature is enabled, disabling it and adding custom feature called grounding_w_google_search."
                )
                # Disable web_search
                features["web_search"] = False
                # Use "Google Search Retrieval" for 1.0 and 1.5 models and "Google Search as a Tool for >=2.0 models".
                if "1.0" in model_name or "1.5" in model_name:
                    metadata_features["google_search_retrieval"] = True
                    metadata_features["google_search_retrieval_threshold"] = (
                        self.valves.GROUNDING_DYNAMIC_RETRIEVAL_THRESHOLD
                    )
                else:
                    metadata_features["google_search_tool"] = True
                # Google suggest setting temperature to 0 if using grounding:
                # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/ground-with-google-search#:~:text=For%20ideal%20results%2C%20use%20a%20temperature%20of%200.0.
                if self.valves.SET_TEMP_TO_ZERO:
                    print("[gemini_manifold_companion] Setting temperature to 0.")
                    body["temperature"] = 0

        if canonical_model_name in ALLOWED_CODE_EXECUTION_MODELS:
            code_execution_enabled = (
                features.get("code_interpreter", False)
                if isinstance(features, dict)
                else False
            )
            if code_execution_enabled:
                print(
                    "[gemini_manifold_companion] Code interpreter feature is enabled, disabling it and adding custom feature called google_code_execution."
                )
                # Disable code_interpreter
                features["code_interpreter"] = False
                # Use "Google Search Retrieval" for 1.0 and 1.5 models and "Google Search as a Tool for >=2.0 models".
                metadata_features["google_code_execution"] = True

        return body

    def stream(self, event: dict) -> dict:
        """Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data."""
        return event

    def outlet(self, body: dict) -> dict:
        """Modifies the complete response payload after it's received from the LLM. Operates on the final `body` dictionary."""
        # TODO: Filter out the citation markers here.
        return body
