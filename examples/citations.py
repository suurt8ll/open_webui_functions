"""
title: Citations Test
id: citations_test
description: Test function for citations.
author: suurt8ll
author_url: https://github.com/suurt8ll
funding_url: https://github.com/suurt8ll/open_webui_functions
license: MIT
version: 0.0.0
requirements:
"""

from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterator,
    Callable,
    Literal,
    TYPE_CHECKING,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

if TYPE_CHECKING:
    from utils.manifold_types import *  # My personal types in a separate file for more robustness.


class Pipe:
    class Valves(BaseModel):
        LOG_LEVEL: Literal[
            "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
        ] = Field(
            default="INFO",
            description="Select logging level. Use `docker logs -f open-webui` to view logs.",
        )

    def __init__(self):
        self.valves = self.Valves()
        print("[citations_test] Function has been initialized.")

    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Callable[["Event"], Awaitable[None]],
    ) -> (
        str
        | dict[str, Any]
        | StreamingResponse
        | Iterator
        | AsyncGenerator
        | Generator
        | None
    ):

        self.__event_emitter__ = __event_emitter__

        sources: list["Source"] = [
            {
                "source": {"name": "example search query"},
                "document": ["", ""],
                "metadata": [
                    {"source": "https://example1.com/source1"},
                    {"source": "https://subdomain.example2.com/source2"},
                ],
            },
            {
                "source": {"name": "example search query2"},
                "document": [
                    "This is an example document coming from source3 in second query.",
                    "This is another example document coming from source4 in second query.",
                ],
                "metadata": [
                    {"title": "source3", "source": "https://example3.com/source3"},
                    {
                        "title": "source4",
                        "source": "https://example4.com/source4",
                    },
                ],
            },
        ]

        sources_test: "ChatCompletionEvent" = {
            "type": "chat:completion",
            "data": {
                "content": "This is an example response [1][2]. This sentence has citations from the second query [3][4].",
                "done": True,
                "sources": sources,
            },
        }
        await __event_emitter__(sources_test)

        return None
