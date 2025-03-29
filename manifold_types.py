from typing import Any, Literal, NotRequired, Optional, TypedDict
from uuid import UUID
from datetime import datetime
from open_webui.models.files import FileModelResponse


class FileInfo(TypedDict):
    type: str  # could be "file"
    file: FileModelResponse
    id: str
    url: str
    name: str
    collection_name: Optional[str]
    status: str
    size: int
    error: str
    itemId: str


class SourceSource(TypedDict):
    docs: NotRequired[list[dict]]
    name: str  # the search query used
    type: NotRequired[Literal["web_search", "file"]]
    file: NotRequired[FileInfo]
    urls: NotRequired[list[str]]


class SourceMetadata(TypedDict):
    source: str  # url
    title: NotRequired[str]  # website title
    description: NotRequired[str]  # website description
    language: NotRequired[str]  # website language
    # These keys are not used by Open WebUI front-end, they for my plugin only.
    original_url: NotRequired[str]  # original, unresolved url
    supports: list[dict]  # needs to be serialized from genai.types.GroundingSupport


class Source(TypedDict):
    source: SourceSource
    document: list[str]
    metadata: list[SourceMetadata]
    distances: NotRequired[list[float]]


class MessageModel(TypedDict):
    id: UUID
    parentId: Optional[UUID]
    childrenIds: list[UUID]
    content: str
    timestamp: datetime


class UserMessageModel(MessageModel):
    role: Literal["user"]
    files: NotRequired[list[FileInfo]]
    models: list[str]


class AssistantMessageModel(MessageModel):
    role: Literal["assistant"]
    model: str
    modelName: str
    modelIdx: int
    userContext: Any
    sources: NotRequired[list[Source]]
    done: NotRequired[bool]


class ChatChatModel(TypedDict):
    """Type for the `ChatModel.chat` variable"""

    id: str  # Or UUID if appropriate
    title: str
    models: list[str]
    params: dict[str, Any]  # Use a more specific type if you know the structure
    history: dict[str, Any]
    messages: list[MessageModel]
    tags: list[str]
    timestamp: datetime
    files: list[FileInfo]  # Or a more specific type if you have file objects


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
