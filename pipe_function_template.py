"""
title: This Is The Function Title
id: pipe_function_template
description: Good starting point for creating new pipe functions for Open WebUI.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
"""

from typing import (
    AsyncGenerator,
    Generator,
    Iterator,
    Union,
    Callable,
    Awaitable,
    Dict,
    Any,
    TypedDict,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.requests import Request
import json


# Define TypedDict for each possible structure of event_data
class StatusEventData(TypedDict):
    type: str  # Required: "status"
    data: Dict[
        str, Union[int, float, str, None]
    ]  # Status-specific data (e.g., progress)


class MessageEventData(TypedDict):
    type: str  # Required: "message"
    data: Dict[str, str]  # Message-specific data (e.g., content)


class ReplaceEventData(TypedDict):
    type: str  # Required: "replace"
    data: Dict[str, str]  # Replace-specific data (e.g., content)


# Define a general EventData type that is a Union of all the specific types
EventData = Union[StatusEventData, MessageEventData, ReplaceEventData]


# Type alias for the event emitter
AsyncEventCallback = Callable[[EventData], Awaitable[None]]

# Type alias for the event caller
AsyncEventCaller = Callable[[EventData], Awaitable[Any]]


class Pipe:
    # Values here appear in the front-end as user input fields.
    class Valves(BaseModel):
        EXAMPLE_STRING: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()

    # This function is not required, but can be used to create more than one model in the front-end.
    def pipes(self) -> list[dict]:
        print("[pipes] Returning models")
        # This will register 2 models in the front-end. Their full model names will be [function_id].[model_id].
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: AsyncEventCallback,
        __event_call__: AsyncEventCaller,
        __task__: str,
        __task_body__: dict,
        __files__: list,
        __metadata__: dict,
        __tools__: list,
    ) -> Union[str, dict, StreamingResponse, Iterator, AsyncGenerator, Generator]:
        """
        Pipe function that captures all possible injected parameters and returns them as a JSON string.
        """

        # Create a dictionary to hold all parameters (including body, __user__, etc.)
        all_params = {
            "body": body,
            "__user__": __user__,
            "__request__": __request__,
            "__event_emitter__": __event_emitter__,
            "__event_call__": __event_call__,
            "__task__": __task__,
            "__task_body__": __task_body__,
            "__files__": __files__,
            "__metadata__": __metadata__,
            "__tools__": __tools__,
        }

        # Convert the dictionary to a JSON string
        all_params_json = json.dumps(
            all_params, indent=2, default=str
        )  # Using default=str for non-serializable objects

        print("[pipe] Returning all parameters as JSON")
        print(all_params_json)

        return "Hello from pipe function!"
