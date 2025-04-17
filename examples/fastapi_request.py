"""
title: FastAPI Request
description: Exploration of what the __request__ object actually is.
id: fastapi_request
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.0.0
"""

import json
from fastapi import Request
from pydantic import BaseModel
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Filter:

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self._log(f"Function has been initialized.", "__init__")

    def inlet(self, body: "Body", **kwargs) -> "Body":
        """This allows modifying the input LLM model gets."""

        self._log("Original Request Body:", "inlet")
        print(json.dumps(body, indent=2, default=str))

        # body["files"] = []
        body["messages"][-1]["content"] = "This was injected by Filter inlet method."

        self._log("Modified Request Body:", "inlet")
        print(json.dumps(body, indent=2, default=str))

        return body

    async def stream(self, event: dict[str, Any]) -> dict[str, Any]:
        """This allows modifying each stream chunk."""
        # self._log("Event Object:", "stream")
        # print(json.dumps(event, indent=2, default=str))
        return event

    async def outlet(self, body: "Body", __request__: Request, **kwargs) -> "Body":
        """This allows modifying the LLM final output."""

        self._log("__request__.state._state:", "outlet")
        print(json.dumps(__request__.state._state, indent=2, default=str))

        self._log("__request__.app.state._state:", "outlet")
        print(json.dumps(__request__.app.state._state, indent=2, default=str))

        self._log("Original Response Body:", "outlet")
        print(json.dumps(body, indent=2, default=str))

        body["messages"][-1]["content"] = "This was injected by Filter outlet method."

        self._log("Modified Response Body:", "outlet")
        print(json.dumps(body, indent=2, default=str))

        return body

    # region ----- Helper methods inside the Pipe class -----

    def _log(self, msg: str, method_name: str = ""):
        """Simple helper method for more informative logging."""
        if method_name:
            method_name = f" | {method_name}"
        print(f"{__name__}{method_name} | {msg}")

    # endregion
