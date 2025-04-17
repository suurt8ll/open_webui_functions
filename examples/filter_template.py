"""
title: Filter Template
description: Very basic filter function.
id: filter_template
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.0.0
"""

import json
from pydantic import BaseModel
from typing import Any, Awaitable, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Filter:

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        print(f"{[__name__]} Function has been initialized.")

    def inlet(self, body: "Body", **kwargs) -> "Body":

        print(f"\n--- Inlet Filter ---")
        print(f"{[__name__]} Original Request Body:")
        print(json.dumps(body, indent=2, default=str))
        # print(f"{[__name__]} Original __metadata__:")
        # print(json.dumps(__metadata__, indent=2, default=str))

        # body["files"] = []
        body["messages"][-1]["content"] = "This was injected by Filter inlet method."

        print(f"{[__name__]} Modified Request Body (before sending to LLM):")
        print(json.dumps(body, indent=2, default=str))

        return body

    async def stream(self, event: dict[str, Any]) -> dict[str, Any]:
        # print(f"\n--- Stream Filter ---")
        # print("Event Object:")
        # print(json.dumps(event, indent=2, default=str))
        return event

    async def outlet(
        self, body: "Body", __event_emitter__: Callable[["Event"], Awaitable[None]]
    ) -> "Body":

        print(f"\n--- Outlet Filter ---")
        print(f"{[__name__]} Original Response Body:")
        print(json.dumps(body, indent=2, default=str))

        body["messages"][-1]["content"] = "This was injected by Filter outlet method."

        print(f"{[__name__]} Modified Response Body:")
        print(json.dumps(body, indent=2, default=str))

        return body

    # region ----- Helper methods inside the Pipe class -----

    # endregion
