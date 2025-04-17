"""
title: Task Detection
id: task_detection
description: Example Pipe that shows how to detect if the model is begin used for a task.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

from typing import (
    Any,
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
        print(f"{[__name__]} Function has been initialized!")

    async def pipe(self, body: dict[str, Any], __task__: str, **kwargs) -> str:

        if __task__ == "title_generation":
            print(f"{[__name__]} Detected title generation task!")
            return '{"title": "Example Title"}'

        if __task__ == "tags_generation":
            print(f"{[__name__]} Detected tag generation task!")
            return '{"tags": ["tag1", "tag2", "tag3"]}'

        return "This model is used for regular chat."
