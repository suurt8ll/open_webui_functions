"""
title: Filter Function Template
description: Very basic filter function.
id: filter_template
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
version: 0.0.0
"""

import datetime as dt
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import Request
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # My personal types in a separate file for more robustness.
    from utils.manifold_types import (
        Body,
        Event,
        Metadata,
        UserData,
    )


class Filter:
    class Valves(BaseModel):
        EXAMPLE_STRING: str = Field(default='', title='Admin String', description='String configurable by admin')

    class UserValves(BaseModel):
        EXAMPLE_STRING_USER: str = Field(default='', title='User String', description='String configurable by user')

    def __init__(self):
        self.valves = self.Valves()
        self._log('Function has been initialized!')

    def _log(self, message: str):
        timestamp = dt.datetime.now().isoformat()
        caller_name = inspect.stack()[1].function
        print(f'[{timestamp}] [{__name__}.{caller_name}] {message}')

    async def inlet(
        self,
        body: 'Body',
        __id__: str,
        __metadata__: 'Metadata',
        __user__: 'UserData',
        __request__: Request,
        __model__: dict,
        __event_emitter__: Callable[['Event'], Awaitable[None]],
        __event_call__: Callable[['Event'], Awaitable[Any]],
    ) -> 'Body':

        message_content = f'Returning all local variables as JSON:\n{json.dumps(locals(), indent=2, default=str)}'
        self._log(message_content)

        return body

    async def stream(
        self,
        event: dict[str, Any],
        __id__: str,
        __metadata__: 'Metadata',
        __user__: 'UserData',
        __request__: Request,
        __model__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __event_call__: Callable[[dict], Awaitable[Any]],
    ) -> dict | None:

        message_content = f'Returning all local variables as JSON:\n{json.dumps(locals(), indent=2, default=str)}'
        self._log(message_content)

        return event

    async def outlet(
        self,
        body: 'Body',
        __id__: str,
        __metadata__: 'Metadata',
        __user__: 'UserData',
        __request__: Request,
        __model__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __event_call__: Callable[[dict], Awaitable[Any]],
    ) -> 'Body':

        message_content = f'Returning all local variables as JSON:\n{json.dumps(locals(), indent=2, default=str)}'
        self._log(message_content)

        return body


# region ----- Helper methods inside the Pipe class -----

# endregion
