"""
title: Google Maps Grounding
id: gemini_maps_grounding_toggle
description: Ground the model's response in Google Maps data
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.0.0
"""

from typing import TYPE_CHECKING, cast
from pydantic import BaseModel

# This block is skipped at runtime.
if TYPE_CHECKING:
    # Imports custom type definitions (TypedDicts) for static analysis purposes (mypy/pylance).
    from utils.manifold_types import *


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Makes the filter toggleable in the front-end.
        self.toggle = True
        # Icon from https://icon-sets.iconify.design/line-md/map-marker/
        self.icon = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjkiIHI9IjIuNSIgZmlsbD0iY3VycmVudENvbG9yIi8+PHBhdGggZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgc3Ryb2tlLXdpZHRoPSIyIiBkPSJNMTIgMjAuNWMwIDAgLTYgLTcgLTYgLTExLjVjMCAtMy4zMSAyLjY5IC02IDYgLTZjMy4zMSAwIDYgMi42OSA2IDZjMCA0LjUgLTYgMTEuNSAtNiAxMS41WiIvPjwvc3ZnPg=="

    async def inlet(
        self,
        body: "Body",
    ) -> "Body":
        # Signal downstream Gemini Manifold pipe that Maps grounding is enabled.

        # Ensure features field exists
        metadata = body.get("metadata")
        metadata_features = metadata.get("features")
        if metadata_features is None:
            metadata_features = cast(Features, {})
            metadata["features"] = metadata_features

        metadata_features["google_maps_grounding"] = True
        return body
