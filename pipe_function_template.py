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

from typing import AsyncGenerator, Generator, Iterator, Union
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import json


class Pipe:
    # Values here appear in the front-end as user input fields.
    class Valves(BaseModel):
        EXAMPLE_STRING: str = Field(default="")

    def __init__(self):
        self.valves = self.Valves()

    # This function is not required, but can be used to create more than one model in the front-end.
    def pipes(self) -> list[dict]:
        print("[pipes] Returning models")
        # This will register 3 models in the front-end. Their full model names will be [function_id].[model_id].
        return [
            {"id": "model_id_1", "name": "model_1"},
            {"id": "model_id_2", "name": "model_2"},
            {"id": "model_id_3", "name": "model_3"},
        ]

    def pipe(
        self, body: dict
    ) -> Union[str, dict, StreamingResponse, Iterator, AsyncGenerator, Generator]:
        """
        Example of body:
        {
            'stream': true,
            'model': 'pipe_function_template.model_id_1',
            'messages': [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    'role': 'user',
                    'content': 'what’s up?'
                },
                {
                    'role': 'assistant',
                    'content': 'Hello, how can I help you?'
                },
                {
                    'role': 'user',
                    'content': 'Help me refactor this code.'
                }
            ],
            "temperature": 0.5
        }
        """
        print("[pipe] body object:")
        print(json.dumps(body, indent=4))
        model = body.get("model", "")
        # Simple example of how to respond to user, this will make the assistant respond with a blue square.
        response = f"{model}\n\n![Hi](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAaUlEQVR4nOzPUQkAIQDA0OMwkZnNZwZD+PEQ9hJsY679vezXAbca0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0BrQGtAa0E4AAAD//wN6Akf5tCRQAAAAAElFTkSuQmCC)"
        return response
