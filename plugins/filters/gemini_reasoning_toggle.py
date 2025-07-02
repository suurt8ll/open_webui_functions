"""
title: Reason
id: gemini_reasoning_toggle
description: Reason before answering
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 1.1.0
"""

# Shoutout to jrkropp for making me aware of Filter.toggle!

from typing import TYPE_CHECKING, cast
from pydantic import BaseModel

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Makes the filter toggleable in the front-end.
        self.toggle = True
        # Lamp bulb icon
        self.icon = "data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgY2xhc3M9ImgtWzE4cHldIHctWzE4cHldIj48cGF0aCBkPSJtMTIgM2MtMy41ODUgMC02LjUgMi45MjI1LTYuNSA2LjUzODUgMCAyLjI4MjYgMS4xNjIgNC4yOTEzIDIuOTI0OCA1LjQ2MTVoNy4xNTA0YzEuNzYyOC0xLjE3MDIgMi45MjQ4LTMuMTc4OSAyLjkyNDgtNS40NjE1IDAtMy42MTU5LTIuOTE1LTYuNTM4NS02LjUtNi41Mzg1em0yLjg2NTMgMTRoLTUuNzMwNnYxaDUuNzMwNnYtMXptLTEuMTMyOSAzSC03LjQ2NDhjMC4zNDU4IDAuNTk3OCAwLjk5MjEgMSAxLjczMjQgMXMxLjM4NjYtMC40MDIyIDEuNzMyNC0xem0tNS42MDY0IDBjMC40NDQwMyAxLjcyNTIgMi4wMTAxIDMgMy44NzQgM3MzLjQzLTEuMjc0OCAzLjg3NC0zYzAuNTQ4My0wLjAwNDcgMC45OTEzLTAuNDUwNiAwLjk5MTMtMXYtMi40NTkzYzIuMTk2OS0xLjU0MzEgMy42MzQ3LTQuMTA0NSAzLjYzNDctNy4wMDIyIDAtNC43MTA4LTMuODAwOC04LjUzODUtOC41LTguNTM4NS00LjY5OTIgMC04LjUgMy44Mjc2LTguNSA4LjUzODUgMCAyLjg5NzcgMS40Mzc4IDUuNDU5MSAzLjYzNDcgNy4wMDIydjIuNDU5M2MwIDAuNTQ5NCAwLjQ0MzAxIDAuOTk1MyAwLjk5MTI4IDF6IiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGZpbGw9ImN1cnJlbnRDb2xvciIgZmlsbC1ydWxlPSJldmVub2RkIj48L3BhdGg+PC9zdmc+"

    async def inlet(
        self,
        body: "Body",
    ) -> "Body":
        # Signal downstream Gemini Manifold pipe that reasoning is enabled.

        # Ensure features field exists
        metadata = body.get("metadata")
        metadata_features = metadata.get("features")
        if metadata_features is None:
            metadata_features = cast(Features, {})
            metadata["features"] = metadata_features

        metadata_features["reason"] = True
        return body
