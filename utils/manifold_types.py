from typing import Any, NotRequired, Literal, TypedDict
from uuid import UUID
from datetime import datetime
from open_webui.models.files import FileModelResponse

# ---------- File Structures ----------
# Define the nested structures found within the 'files' list items
# and also within metadata.files


class FileData(TypedDict):
    """Represents the data extracted from a file."""

    content: str  # Example shows text content


class FileMeta(TypedDict):
    """Represents metadata about a file."""

    name: str
    content_type: str
    size: int
    data: dict[
        str, Any
    ]  # Example shows empty dict, but could potentially hold other meta
    collection_name: str


class FileDetails(TypedDict):
    """Represents detailed information about a file."""

    id: str  # UUID
    user_id: str  # UUID
    hash: str
    filename: str
    data: FileData
    meta: FileMeta
    created_at: int  # Unix timestamp
    updated_at: int  # Unix timestamp


class FileInfo(TypedDict):
    """Represents an item in the top-level 'files' list or metadata.files."""

    type: Literal["file"]  # The type of the item, always "file" for file uploads
    file: FileDetails  # Detailed file information
    id: str  # UUID (seems to duplicate file.id)
    url: str  # API endpoint for the file
    name: str  # (seems to duplicate file.filename and file.meta.name)
    collection_name: str  # (seems to duplicate file.meta.collection_name)
    status: str  # e.g., "uploaded", "processing", "error"
    size: int  # (seems to duplicate file.meta.size)
    error: str  # Error message if status is "error"
    itemId: str  # UUID (seems to be a unique ID for this specific file *usage* in the message)


# ---------- __event_emitter__ ----------


class SourceSource(TypedDict):
    docs: NotRequired[list[dict]]
    name: str | None  # the search query used
    type: NotRequired[Literal["web_search", "file"]]
    file: NotRequired[FileInfo]
    urls: NotRequired[list[str]]


class SourceMetadata(TypedDict):
    source: str | None  # url
    # ^ if None then front-end seems to use SourceSource.name instead.
    title: NotRequired[str]  # website title
    description: NotRequired[str]  # website description
    language: NotRequired[str]  # website language
    # These keys are not used by Open WebUI front-end, they for my plugin only.
    original_url: NotRequired[str | None]  # original, unresolved url
    supports: NotRequired[list[dict]]


class Source(TypedDict):
    source: SourceSource
    document: list[str]
    metadata: list[SourceMetadata]
    distances: NotRequired[list[float]]


class ErrorData(TypedDict):
    detail: str


class NotificationEventData(TypedDict):
    type: Literal["info", "success", "warning", "error"]
    content: str


class NotificationEvent(TypedDict):
    type: Literal["notification"]
    data: NotificationEventData


class ChatCompletionEventData(TypedDict):
    content: NotRequired[str]
    done: NotRequired[bool]
    error: NotRequired[ErrorData]
    sources: NotRequired[list[Source]]
    usage: NotRequired[dict[str, Any]]


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


Event = ChatCompletionEvent | StatusEvent | NotificationEvent

# ---------- Message Content ----------
# These seem fine and cover multimodal cases, even if the example only shows text.


class TextContent(TypedDict):
    """Represents text content within a message."""

    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    """Represents an image URL within a message."""

    url: str  # e.g., data:image/png;base64,iVBw0KGgoAAAA.... or a standard URL


class ImageContent(TypedDict):
    """Represents image content within a message."""

    type: Literal["image_url"]
    image_url: ImageURL


Content = TextContent | ImageContent  # Union of possible content types

# ---------- Messages ----------
# These also seem fine and cover different roles and multimodal content.


class UserMessage(TypedDict):
    """Represents a message from the user."""

    role: Literal["user"]
    content: (
        str | list[Content]
    )  # Content can be a simple string or a list of Content blocks


class AssistantMessage(TypedDict):
    """Represents a message from the assistant."""

    role: Literal["assistant"]
    content: str  # Assistant messages typically have string content


class SystemMessage(TypedDict):
    """Represents a system message."""

    role: Literal["system"]
    content: str


Message = (
    UserMessage | AssistantMessage | SystemMessage
)  # Union of possible message types

# ---------- Features ----------
# Define the specific structure of the 'features' object


class Features(TypedDict):
    """Represents the enabled/disabled features for the request."""

    image_generation: bool
    code_interpreter: bool
    web_search: bool


# ---------- Metadata Structures ----------
# Define the detailed structure of the 'metadata' object


class ModelDetails(TypedDict):
    """Details about the model within Ollama metadata."""

    parent_model: str
    format: str
    family: str
    families: list[str]
    parameter_size: str
    quantization_level: str


class OllamaDetails(TypedDict):
    """Ollama specific details for the model."""

    name: str
    model: str
    modified_at: str  # ISO 8601 datetime string
    size: int
    digest: str
    details: ModelDetails
    urls: list[
        int
    ]  # Example shows [0], type might be more complex? List[Any] is safer if unsure.


class MetadataModel(TypedDict):
    """Represents the model information within metadata."""

    id: str
    name: str
    object: Literal["model"]
    created: int  # Unix timestamp
    owned_by: str
    ollama: OllamaDetails
    tags: list[str]
    actions: list[Any]  # Structure of actions is not clear from example


class MetadataVariables(TypedDict):
    """Represents variables used in the prompt/request."""

    # Keys are variable names (e.g., "{{USER_NAME}}"), values are strings
    __dict__: dict[str, str]


class Metadata(TypedDict):
    """Represents the metadata object in the request body."""

    user_id: str  # UUID
    chat_id: str  # UUID
    message_id: str  # UUID
    session_id: str
    tool_ids: list[str] | None  # Can be a list of strings or null
    tool_servers: list[
        dict[str, Any]
    ]  # Example is empty list, assuming list of objects
    files: list[FileInfo]  # List of files, using the same FileInfo structure
    features: Features  # Using the specific Features TypedDict
    variables: MetadataVariables  # Using the specific MetadataVariables TypedDict
    model: MetadataModel  # Using the specific MetadataModel TypedDict
    direct: bool


# ---------- Options ----------


class Options(TypedDict):
    """Represents optional parameters for the model request."""

    temperature: NotRequired[float]
    top_p: NotRequired[float]
    min_p: NotRequired[float]
    top_k: NotRequired[float]
    # Add other potential options if known, e.g., num_predict, stop, etc.
    # Using NotRequired as the example shows an empty object {}


# ---------- Main Body ----------


class Body(TypedDict):
    """Represents the main request body structure."""

    stream: bool
    model: str
    messages: list[Message]
    files: NotRequired[list[FileInfo]]  # Optional list of files
    features: NotRequired[Features]  # Optional features object
    metadata: NotRequired[Metadata]  # Optional metadata object
    options: NotRequired[Options]  # Optional options object


# ---------- Chats.ChatModel ----------


class MessageModel(TypedDict):
    id: UUID
    parentId: UUID | None
    childrenIds: list[UUID]
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime


class UserMessageModel(MessageModel):
    files: NotRequired[list[FileInfo]]
    models: list[str]


class AssistantMessageModel(MessageModel):
    model: str
    modelName: str
    modelIdx: int
    userContext: Any
    sources: NotRequired[list[Source]]
    done: NotRequired[bool]


class ChatParams(TypedDict):
    system: NotRequired[str]
    temperature: NotRequired[float]


class ChatChatModel(TypedDict):
    """Type for the `ChatModel.chat` variable"""

    id: str  # Or UUID if appropriate
    title: str
    models: list[str]
    params: ChatParams
    history: dict[str, Any]
    messages: list[MessageModel]
    tags: list[str]
    timestamp: datetime
    files: list[FileInfo]  # Or a more specific type if you have file objects


# ---------- __user__ ----------


class UserData(TypedDict):
    """
    This is how `__user__` `dict` looks like.
    """

    id: str
    email: str
    name: str
    role: Literal["admin", "user", "pending"]
    valves: NotRequired[Any]  # object of type UserValves


# ---------- pipes return dict ----------


class ModelData(TypedDict):
    """
    This is how the `pipes` function expects the `dict` to look like.
    """

    id: str
    name: str
    # My own variables, these do not have any effect on Open WebUI's behaviour.
    description: NotRequired[str | None]
