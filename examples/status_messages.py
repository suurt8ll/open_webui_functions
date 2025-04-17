"""
title: Status Messages
id: status_messages
description: Example Pipe that shows how to emit status messages to the front-end.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

import asyncio
from typing import (
    Any,
    Awaitable,
    Callable,
    TYPE_CHECKING,
)
from pydantic import BaseModel

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Pipe:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        print("[status_messages] Function has been initialized!")

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Callable[["Event"], Awaitable[None]],
        **kwargs,
    ) -> str:

        asyncio.create_task(self.countdown(__event_emitter__))
        return "Look at this cool count-down!"

    # region Helper methods inside the Pipe class

    async def countdown(self, event_emitter: Callable[["Event"], Awaitable[None]]):
        """Displays a countdown timer in the front-end with loading icon."""
        for i in range(5, 0, -1):
            status_count: StatusEvent = {
                "type": "status",
                "data": {
                    "description": f"Time remaining: {i}s",
                    "done": False,
                    "hidden": False,
                },
            }
            await event_emitter(status_count)
            await asyncio.sleep(1)

        status_finish: StatusEvent = {
            "type": "status",
            "data": {
                "description": "Process complete!",
                "done": True,
                "hidden": False,
            },
        }
        await event_emitter(status_finish)

    # endregion
