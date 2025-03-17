"""
title: Pipe Function Skeletion
id: pipe_function_template
description: Good starting point for creating new pipe functions for Open WebUI.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

from typing import AsyncGenerator, Awaitable, Generator, Iterator, Callable, Any
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from fastapi import Request
import json

# You can import any module from the Open WebUI backend itself.  However, be mindful of security implications and code stability.
from open_webui.models.files import Files


class Pipe:
    """
    A skeleton Pipe function that demonstrates all possible methods and attributes.
    """

    class Valves(BaseModel):
        """
        Defines input parameters configurable by admins.
        Use pydantic.Field for descriptions, titles, and constraints.
        """

        EXAMPLE_STRING: str = Field(
            default="", title="Admin String", description="String configurable by admin"
        )

    class UserValves(BaseModel):
        """
        Defines input parameters configurable by individual users.
        Use pydantic.Field for descriptions, titles, and constraints.
        """

        EXAMPLE_STRING_USER: str = Field(
            default="", title="User String", description="String configurable by user"
        )

    def __init__(self):
        """
        Initializes the Pipe function.
        """
        self.valves = self.Valves()

    def pipes(self) -> list[dict]:
        """
        Defines multiple models (sub-pipes) for this Pipe function (manifold function).
        Returns a list of dictionaries, each with "id" and "name" keys.
        """
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
        ]

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any],
        __request__: Request,
        # FIXME: Figure out how to type hint the event emitter and event call. See Open WebUI documentation for more information.
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]],
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]],
        __task__: str,
        __task_body__: dict[str, Any],
        __files__: list[dict[str, Any]],
        __metadata__: dict[str, Any],
        __tools__: list[Any],
    ) -> (
        str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator
    ):
        """
        The core logic of the Pipe function.

        Args:
            body (dict[str, Any]): The main request payload.
            __user__ (dict[str, Any]): User information (ID, email, name, role) and user-specific valves ( __user__["valves"]).
            __request__: The FastAPI request object.
            __event_emitter__:  Event emitter for sending events.
            __event_call__: Event call object.
            __task__ (str): Task ID.
            __task_body__ (dict[str, Any]): Task body.
            __files__ (list[dict[str, Any]]): A list of uploaded files.
            __metadata__ (dict[str, Any]): Metadata associated with the chat session or message.
            __tools__ (list[Any]): A list of available tools.

        Returns:
            str | dict[str, Any] | StreamingResponse | Iterator | AsyncGenerator | Generator: The response from the pipe function.
        """
        try:
            string_from_valve = self.valves.EXAMPLE_STRING
            string_from_user_valve = __user__["valves"].EXAMPLE_STRING_USER

            print("[pipe] String from valve: ", string_from_valve)
            if string_from_user_valve:
                print("[pipe] String from user valve: ", string_from_user_valve)

            stored_files = Files.get_files()
            print("[pipe] Stored files: ", stored_files)

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

            all_params_json = json.dumps(all_params, indent=2, default=str)

            print("[pipe] Returning all parameters as JSON:")
            print(all_params_json)

            return "Hello from pipe function!"

        except Exception as e:
            print(f"An error occurred: {e}")
            return f"Error: {e}"  # Or return a more structured error response
