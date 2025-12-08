from typing import Any, NotRequired, Literal, TypedDict
from google.genai import types


# region `__files__` and `__metadata__.files`
class FileContentDataTD(TypedDict):
    content: str


class FileMetadataTD(TypedDict):
    name: str
    content_type: str
    size: int
    data: dict[str, Any]  # Assuming this is always a dict, even if empty
    collection_name: str


class InnerFileDetailTD(TypedDict):
    id: str
    user_id: str
    hash: str
    filename: str
    data: FileContentDataTD
    meta: FileMetadataTD
    created_at: int
    updated_at: int


class FileAttachmentTD(TypedDict):
    type: str
    file: InnerFileDetailTD
    id: str
    url: str
    name: str
    collection_name: str
    status: str
    size: int
    error: str
    itemId: str


# endregion `__files__` and `__metadata__.files`


# region source object
class SourceSource(TypedDict):
    docs: NotRequired[list[dict]]
    name: str | None  # the search query used
    type: NotRequired[Literal["web_search", "file"]]
    file: NotRequired[FileAttachmentTD]
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


# endregion source object


# region __event_emitter__
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
# endregion __event_emitter__


# region `__metadata__`

# Ollama-specific model details. Not present for pipe models.
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


# Nested types for `MetadataModel.info`
class ModelInfoMetaCapabilities(TypedDict):
    vision: bool
    file_upload: bool
    web_search: bool
    image_generation: bool
    code_interpreter: bool
    citations: bool
    status_updates: bool
    usage: bool


class ModelInfoMeta(TypedDict):
    profile_image_url: str
    description: str | None
    capabilities: ModelInfoMetaCapabilities
    suggestion_prompts: Any | None
    tags: list[str]
    filterIds: list[str]
    defaultFilterIds: list[str]


class AccessControlPermissions(TypedDict):
    group_ids: list[str]
    user_ids: list[str]


class ModelInfoAccessControl(TypedDict):
    read: AccessControlPermissions
    write: AccessControlPermissions


class ModelInfo(TypedDict):
    id: str
    user_id: str
    base_model_id: str | None
    name: str
    params: dict[str, Any]
    meta: ModelInfoMeta
    access_control: ModelInfoAccessControl
    is_active: bool
    updated_at: int
    created_at: int


class ModelFilter(TypedDict):
    """Represents a filter associated with a model in the metadata."""

    id: str
    name: str
    description: str
    icon: str
    has_user_valves: bool


class ModelPipe(TypedDict):
    """Represents the pipe information for a model."""

    type: Literal["pipe"]


class MetadataModel(TypedDict):
    """Represents the model information within metadata."""

    id: str
    name: str
    object: Literal["model"]
    created: int  # Unix timestamp
    owned_by: str
    actions: list[Any]
    tags: list[str]

    # Pipe-model specific fields
    pipe: NotRequired[ModelPipe]
    has_user_valves: NotRequired[bool]
    info: NotRequired[ModelInfo]
    filters: NotRequired[list[ModelFilter]]

    # Ollama-model specific fields
    ollama: NotRequired[OllamaDetails]


class MetadataVariables(TypedDict):
    """Represents variables used in the prompt/request."""

    # Keys are variable names (e.g., "{{USER_NAME}}"), values are strings
    __dict__: dict[str, str]


class Features(TypedDict):
    """Represents the enabled/disabled features for the request."""

    image_generation: bool
    code_interpreter: bool
    web_search: bool

    # These are my own custom fields, not used by Open WebUI.
    google_search_retrieval: NotRequired[bool]
    google_search_retrieval_threshold: NotRequired[float | None]
    google_search_tool: NotRequired[bool]
    google_code_execution: NotRequired[bool]
    upload_documents: NotRequired[bool]
    reason: NotRequired[bool]
    url_context: NotRequired[bool]
    google_maps_grounding: NotRequired[bool]
    stream: NotRequired[bool]
    gemini_manifold_companion_version: NotRequired[str]


class MetadataParams(TypedDict):
    """Represents the 'params' object within metadata."""

    stream_delta_chunk_size: int | None
    reasoning_tags: Any | None
    function_calling: Literal["default", "native"]


class Metadata(TypedDict):
    """Represents the metadata object in the request body."""

    user_id: str  # UUID
    chat_id: str  # UUID
    message_id: str  # UUID
    session_id: str
    filter_ids: list[str]
    tool_ids: list[str] | None
    tool_servers: list[Any]
    files: list[FileAttachmentTD] | None
    features: Features | None
    variables: MetadataVariables
    model: MetadataModel
    direct: bool
    params: MetadataParams

    # Optional/Context-dependent keys
    task: NotRequired[str | None]
    task_body: NotRequired[dict[str, Any] | None]

    # These are my own added custom keys, not used by Open WebUI.
    safety_settings: list[types.SafetySetting]  # Added in `Filter.inlet`
    chat_control_params: dict[str, Any]  # Added in `Filter.inlet`
    merged_custom_params: dict[str, Any]  # Added in `Pipe.pipe`
    is_paid_api: NotRequired[bool]  # Added in `Pipe.pipe`
    is_vertex_ai: NotRequired[bool]  # Added in `Pipe.pipe`
    canonical_model_id: NotRequired[str] # Added in `Filter.inlet`
    cumulative_tokens: NotRequired[int | None]  # Added in `Pipe.pipe`
    cumulative_cost: NotRequired[float | None]  # Added in `Pipe.pipe`


# endregion `__metadata__`


# region `body.messages`


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
    # This custom key is added by the Gemini Manifold companion filter to store the
    # raw structured response parts from the Gemini API for potential future use.
    # It is not part of the standard Open WebUI message format and will be ignored by the core system.
    gemini_parts: NotRequired[list[dict[str, Any]]]
    # This custom key stores the original, unmodified text content generated by the model.
    # This serves as a ground truth, preserving the model's output before any user edits.
    original_content: NotRequired[str]


class SystemMessage(TypedDict):
    """Represents a system message."""

    role: Literal["system"]
    content: str


Message = UserMessage | AssistantMessage | SystemMessage
# endregion `body.messages`


# region `body` dict
class Options(TypedDict):
    """Represents optional parameters for the model request."""

    temperature: NotRequired[float]
    top_p: NotRequired[float]
    min_p: NotRequired[float]
    top_k: NotRequired[float]
    # Add other potential options if known, e.g., num_predict, stop, etc.
    # Using NotRequired as the example shows an empty object {}


class Body(TypedDict):
    """
    Represents the main request body structure. 
    This differs between `Filter.inlet`, `Pipe.pipe`, and `Filter.outlet`.
    """

    stream: bool
    model: str
    messages: list[Message]
    files: NotRequired[list[FileAttachmentTD]]
    features: NotRequired[Features]  # Only present in `Filter.inlet`
    metadata: Metadata # Only present in `Filter.inlet`
    options: NotRequired[Options]
# endregion `body` dict


# region `ChatModel.chat`
class ChatMessageTD(TypedDict):
    # Required fields for all messages
    id: str
    parentId: str | None  # Can be null for the root message
    childrenIds: list[str]
    role: str  # "user" or "assistant"
    content: str
    timestamp: int

    # Fields that are not always present (use NotRequired)
    # Primarily for user messages
    files: NotRequired[list[FileAttachmentTD]]
    models: NotRequired[list[str]]  # e.g. ["associate_messages_to_files"] for user

    # Primarily for assistant messages
    model: NotRequired[str]  # e.g. "associate_messages_to_files" for assistant
    modelName: NotRequired[str]  # e.g. "Associate Messages to Files"
    modelIdx: NotRequired[int]
    userContext: NotRequired[
        Any | None
    ]  # Can be null if present, or not present at all
    sources: NotRequired[
        list[Source]
    ]  # Present in history.messages for assistant, not in top-level messages list
    statusHistory: NotRequired[list[StatusEventData]] # assistant messages only
    usage: NotRequired[dict[str, Any]]  # assistant messages only

    # Custom keys added by Gemini Manifold plugin
    gemini_parts: NotRequired[list[dict[str, Any]]]
    original_content: NotRequired[str]


class ChatHistoryTD(TypedDict):
    messages: dict[str, ChatMessageTD]  # Key is message ID
    currentId: str


class ChatObjectDataTD(TypedDict):
    id: str
    title: str
    models: list[str]  # e.g. ["associate_messages_to_files"]
    params: dict[str, Any]  # Empty in example, but structure is a dict
    history: ChatHistoryTD
    messages: list[ChatMessageTD]  # A list of messages
    tags: list[Any]  # Empty in example, could be list[str] if always strings
    timestamp: int  # Milliseconds timestamp
    files: list[FileAttachmentTD]  # List of files associated with the chat overall


# endregion `ChatModel.chat`


# region `__user__`
class UserData(TypedDict):
    """
    This is how `__user__` `dict` looks like.
    """

    id: str
    email: str
    name: str
    role: Literal["admin", "user", "pending"]
    valves: NotRequired[Any]  # object of type UserValves


# endregion `__user__`


# region dict returned by `Pipe.pipes()`
class ModelData(TypedDict):
    """
    This is how the `pipes` function expects the `dict` to look like.
    """

    id: str
    name: str
    # My own variables, these do not have any effect on Open WebUI's behaviour.
    description: NotRequired[str | None]


# endregion dict returned by `Pipe.pipes()`
