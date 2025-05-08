"""
title: Bypass Files Processing
description: Test filter to see how to bypass Open WebUI's internal file processing pipeline.
id: bypass_files_processing
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.0.0
"""

import json
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Filter:

    def __init__(self):
        print(f"[{__name__}] Function has been initialized.")

    def inlet(
        self, body: "Body", __event_emitter__: Callable[["Event"], Awaitable[None]]
    ) -> "Body":

        print(f"[{__name__}] --- Inlet Filter ---")
        print(f"[{__name__}] Original Request Body:")
        print(json.dumps(body, indent=2, default=str))

        print(f"[{__name__}] Setting body.files to empty list.")
        # This bypasses backend's RAG completely.
        body["files"] = []

        print(f"[{__name__}] Modified Request Body (before sending to LLM):")
        print(json.dumps(body, indent=2, default=str))

        return body

    async def stream(self, event: dict[str, Any]) -> dict[str, Any]:
        return event

    async def outlet(
        self, body: "Body", __event_emitter__: Callable[["Event"], Awaitable[None]]
    ) -> "Body":

        print(f"[{__name__}] --- Outlet Filter ---")
        print(f"[{__name__}] Response Body:")
        print(json.dumps(body, indent=2, default=str))
        return body
