from typing import Any, Literal, NotRequired, Optional, TypedDict
from uuid import UUID

from pydantic import BaseModel, Field


class FileInfo(TypedDict):
    type: str
    file: dict[str, Any]
    id: str
    url: str
    name: str
    status: str
    size: int
    error: str
    itemId: str


class Message(BaseModel):
    id: UUID
    parentId: Optional[UUID] = None
    childrenIds: list[UUID] = Field(default_factory=list)
    role: Literal["user", "assistant"]
    content: str
    files: Optional[list[FileInfo]]
    timestamp: int
    models: list[str] = Field(default_factory=list)
    model: Optional[str] = None  # Only for assistant role
    modelName: Optional[str] = None  # Only for assistant role
    modelIdx: Optional[int] = None  # Only for assistant role
    userContext: Optional[Any] = None  # Only for assistant role
    sources: Optional[list[dict[str, Any]]]  # Only for assistant role
    done: Optional[bool]  # Only for assistant role


class UserData(TypedDict):
    """
    This is how `__user__` `dict` looks like.
    """

    id: str
    email: str
    name: str
    role: Literal["admin", "user", "pending"]
    valves: NotRequired[Any]  # object of type UserValves


class ModelData(TypedDict):
    """
    This is how the `pipes` function expects the `dict` to look like.
    """

    id: str
    name: str


class SourceSource(TypedDict):
    docs: NotRequired[list[dict]]
    name: str  # the search query used
    type: NotRequired[Literal["web_search"]]
    urls: NotRequired[list[str]]


class SourceMetadata(TypedDict, total=False):
    source: str  # url
    title: NotRequired[str]  # website title
    description: NotRequired[str]  # website description
    language: NotRequired[str]  # website language


class Source(TypedDict):
    source: SourceSource
    document: list[str]
    metadata: list[SourceMetadata]
    distances: NotRequired[list[float]]


class ErrorData(TypedDict):
    detail: str


class ChatCompletionEventData(TypedDict):
    content: NotRequired[str]
    done: NotRequired[bool]
    error: NotRequired[ErrorData]
    sources: NotRequired[list[Source]]


class ChatCompletionEvent(TypedDict):
    type: Literal["chat:completion"]
    data: ChatCompletionEventData


class StatusEventData(TypedDict):
    action: NotRequired[Literal["web_search", "knowledge_search"]]
    description: str
    done: NotRequired[bool]
    query: NotRequired[str]  # knowledge_search
    urls: NotRequired[list[str]]  # web_search
    hidden: NotRequired[bool]


class StatusEvent(TypedDict):
    type: Literal["status"]
    data: StatusEventData


Event = ChatCompletionEvent | StatusEvent
