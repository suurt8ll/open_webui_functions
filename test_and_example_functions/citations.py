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
    NotRequired,
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterator,
    Callable,
    Literal,
    Optional,
    TypedDict,
)
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse


class SourceSource(TypedDict):
    name: str


class SourceMetadata(TypedDict, total=False):
    source: str


class Source(TypedDict):
    source: SourceSource
    document: list[str]
    metadata: list[SourceMetadata]


class ErrorData(TypedDict):
    detail: str


class ChatCompletionEventData(TypedDict):
    content: Optional[str]
    done: bool
    sources: NotRequired[list[Source]]
    error: NotRequired[ErrorData]


class ChatCompletionEvent(TypedDict):
    type: Literal["chat:completion"]
    data: ChatCompletionEventData


Event = ChatCompletionEvent


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
        __event_emitter__: Callable[[Event], Awaitable[None]],
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

        sources: list[Source] = [
            {
                "source": {"name": "sigma"},
                "document": ["Sigma sigma boy sigma boy.", "sigma boy brainrot"],
                "metadata": [
                    {"source": "https://en.wikipedia.org/wiki/Sigma_Boy"},
                    {"source": "https://en.wikipedia.org/wiki/Brain_rot"},
                ],
            }
        ]

        sources_test: ChatCompletionEvent = {
            "type": "chat:completion",
            "data": {
                "content": "This is a heartmoving song [0].",
                "done": True,
                "sources": sources,
            },
        }
        await __event_emitter__(sources_test)

        return None
