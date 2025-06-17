"""
requirements: pydantic,matplotlib
version: 1.0.0
funding_url: https://github.com/sponsors/your-username
"""

# Open WebUI - Pipe Function Template
#
# Target: Python >= 3.11
#
# This template is the result of a full-stack analysis of the Open WebUI codebase.
# It includes all discovered features and best practices for creating Pipe functions.
#
#
# --- FILE STRUCTURE OVERVIEW ---
#
# 1. FRONTMATTER (The """...""" block above)
#    - MUST be the very first thing in the file. The backend parser checks only the first line.
#    - Contains key-value pairs read by the backend.
#    - IMPORTANT: No comments or blank lines are allowed inside the frontmatter block.
#
# 2. DOCUMENTATION & METADATA (This comment block)
#    - Explains the difference between metadata set in the UI and frontmatter in this file.
#
# 3. PYTHON CODE (The rest of the file)
#    - Your `Pipe` class and any other necessary logic.
#
#
# --- METADATA (Set in the Open WebUI Admin Panel) ---
#
# The following properties are configured in the WebUI when you create or edit the function.
# They are NOT set in the frontmatter block.
#
# - id: A unique identifier for the pipe (e.g., "my_awesome_pipe").
# - name: The display name for the pipe (e.g., "My Awesome Pipe").
# - description: A short description of what the pipe does.
#
#
# --- FRONTMATTER (Defined in the block at the top of the file) ---
#
# The backend explicitly reads and acts on the following keys:
#
# - requirements: A comma-separated list of Python packages to install via pip.
#                 (e.g., "pydantic, requests, beautifulsoup4==4.12.3")
#
# The front-end reads the following keys from the manifest to alter the UI:
#
# - version: A version string (e.g., "1.0.0"). Displayed as a badge in the UI.
# - funding_url: A URL to a donation page (e.g., GitHub Sponsors). Enables a "Support" button.
#

# Imports should come AFTER the documentation.
from collections.abc import Iterator, AsyncGenerator, Generator, Callable, Awaitable
from pydantic import BaseModel, Field

# You may need to import Request for type hinting `__request__`
from fastapi import Request
from starlette.responses import StreamingResponse

# --- Importing from the Open WebUI Backend ---
# You can import and use modules directly from the Open WebUI backend.
# These are just examples; explore the backend codebase for more.
from open_webui.utils.misc import get_last_user_message
from open_webui.models.users import UserModel


class Pipe:
    """
    This is the main class for an Open WebUI Pipe function.
    The backend discovers and executes this class.
    """

    class Valves(BaseModel):
        """Defines admin-only settings for the pipe."""

        api_key: str = Field(
            default="",
            title="API Key",
            description="An API key for an external service.",
        )

    class UserValves(BaseModel):
        """
        Defines settings that each user can configure individually for this pipe.

        Unlike the admin `Valves`, this class is NOT initialized in `__init__`.
        This is because `__init__` runs only once when the pipe is loaded, but
        these settings are specific to each user and must be fetched for every request.

        The backend automatically creates an instance of this `UserValves` class
        with the current user's saved settings and injects it into the `__user__`
        dictionary under the key "valves" before passing it to your `pipe` method.
        """

        language: str = Field(
            default="en-US",
            title="Output Language",
            description="The language for the response.",
        )

    def __init__(self) -> None:
        """
        The __init__ method is called once when the pipe is first loaded.
        `self.valves` is initialized here and later overwritten by the system
        with the admin's saved settings before `pipe()` is called.
        """
        self.valves = self.Valves()

    # The `pipes` attribute is OPTIONAL. It enables "manifold" mode.
    # - If defined, this pipe exposes multiple tools as separate selectable "models" in the UI.
    # - If this attribute is omitted or set to None, the pipe acts as a standard, single function.
    #
    # It can be defined in three ways:
    # 1. A static list of dictionaries.
    # 2. A synchronous method that returns a list of dictionaries.
    # 3. An asynchronous method that returns a list of dictionaries.
    async def pipes(self) -> list[dict]:
        # You could fetch this list from an external API, a config file, etc.
        return [
            {"id": "dynamic_tool_1", "name": "Dynamic Tool 1"},
            {"id": "dynamic_tool_2", "name": "Dynamic Tool 2"},
        ]

    async def pipe(
        self,
        # You must explicitly name all the parameters you want to receive.
        body: dict,
        __user__: dict,
        __files__: list | None = None,
        __chat_id__: str | None = None,
        __request__: Request | None = None,
        __tools__: dict[str, dict] | None = None,
    ) -> (
        str
        | dict
        | BaseModel
        | StreamingResponse
        | Generator
        | AsyncGenerator
        | Iterator
    ):
        """
        This is the main entry point for the pipe's logic. It is called on every request.
        This method can be `async def` (as shown) or a regular `def`.

        --- DYNAMIC PARAMETER INJECTION ---

        The Open WebUI backend dynamically inspects this method's signature. It will only
        pass the parameters that you explicitly list in the function definition.
        For example, if you want to access the request object, you must add
        `__request__: Request` to the signature.

        --- AVAILABLE PARAMETERS ---

        Below is a list of all injectable parameters, their types, and descriptions.

        - `body: dict`: The raw OpenAI-compatible request body. Includes `model`, `messages`, `stream`, etc. Guaranteed to be present.
        - `__user__: dict`: A dictionary representing the current user (`UserModel`). Also contains a `valves` key with an instance of your `UserValves` class. Guaranteed.
        - `__tools__: dict[str, dict]`: A dictionary mapping available tool function names to their definitions. Guaranteed.
        - `__request__: Request`: The raw FastAPI/Starlette request object. Gives access to headers, client IP, etc. Guaranteed.
        - `__metadata__: dict`: A dictionary containing various metadata from the request. Guaranteed.
        - `__files__: list | None`: A list of file objects attached to the user's message. Can be `None`.
        - `__chat_id__: str | None`: The unique identifier for the current chat session.
        - `__session_id__: str | None`: The unique identifier for the user's browser session.
        - `__message_id__: str | None`: The unique identifier for the current message being processed.
        - `__task__: str | None`: Identifies an internal system task (e.g., "title_generation"). `None` during normal chats.
        - `__task_body__: dict | None`: The original request body associated with the `__task__`. `None` during normal chats.
        - `__event_emitter__: Callable[[dict], Awaitable[None]] | None`: An async function to "fire-and-forget" an event to the UI.
        - `__event_call__: Callable[[dict], Awaitable[Any]] | None`: An async function to send an event to the UI and wait for a response.

        --- ADVANCED UI INTERACTION: EVENT EMITTER & CALLER ---

        For a pipe to interact with the UI in ways beyond sending text, it must use the
        `__event_emitter__` and `__event_call__` parameters. These are the officially supported
        hooks that allow your Python code to trigger built-in frontend capabilities, such as
        showing confirmation dialogs, requesting user input, or running client-side code. They
        are the primary mechanism for creating rich, interactive experiences, as the standard
        UI is specifically designed to listen for and respond to the events they send.

        - `__event_emitter__`: This function sends a "fire-and-forget" event to the UI. Your
          pipe's execution continues immediately without waiting for a response. It is ideal for
          sending non-critical status updates or notifications to the user.

        - `__event_call__`: This function sends an event to the UI and **pauses** your pipe's
          execution until it receives a response from the user or the client-side task. This
          creates a powerful request-response cycle, enabling you to build interactive workflows
          that depend on user input or the results of client-side operations.

        --- RETURN TYPES & UI BEHAVIOR ---

        The frontend UI is designed to render responses that follow the standard OpenAI format.
        Returning or yielding custom data structures will NOT be rendered in the chat window.

        --- A. Non-Streaming Mode (`stream: false`) ---
        The goal is to return a single, complete response object.

        - `return str`:
            The simplest and most common return type. The backend wraps your string in a
            standard OpenAI Chat Completion Object, and the UI displays it as the final message.
            The message is automatically saved to the chat history.

        - `return dict` or `return pydantic.BaseModel`:
            This is an advanced option. The returned object MUST conform to the OpenAI
            Chat Completion Object structure (i.e., have a `choices[0].message.content` path).
            The backend will use it as-is and save the content to the database. Returning any
            other structure will result in no message being displayed in the UI.

        --- B. Streaming Mode (`stream: true`) ---
        The goal is to yield a sequence of chunks that the UI can render in real-time.

        - `yield str`:
            The standard way to stream text. The backend wraps each string in an OpenAI
            "Chat Completion Chunk" and sends it to the UI. This is the easiest way to
            show Chain of Thought by embedding special tags directly in your string:
            `yield "<​thinking>I am reasoning about the query...<​/thinking>"`
            The UI will render this within a collapsible "Thinking" panel.

        - `yield dict` or `yield pydantic.BaseModel`:
            This provides granular control over the stream and is necessary for advanced
            features like tool calling. Each yielded object MUST conform to the OpenAI
            Chat Completion Chunk structure. The UI specifically looks for the `delta`
            object within each chunk and processes the following keys:
            - `content`: Streams the final, user-visible answer.
            - `reasoning` (or `thinking`): Streams text to the "Thinking" panel.
            - `tool_calls`: Streams a request to call a tool, triggering the agentic loop.
            Example: `yield {"choices": [{"delta": {"reasoning": "I should call a tool."}}]}`
            Yielding any other dictionary structure will be ignored by the UI.

        - `return Generator`, `AsyncGenerator`, or `Iterator`:
            This is the required return type when using `yield`. The backend iterates through
            the object and processes each yielded item as described above.

        - `return starlette.responses.StreamingResponse`:
            An advanced override. This gives you 100% raw control over the response stream.
            You are responsible for formatting all Server-Sent Events (SSE), including
            `data: ` prefixes and the final `data: [DONE]` message. This is only recommended
            if you have a deep understanding of the SSE protocol and OpenAI's format.
        """

        # Use the improted utility function to get the last user message.
        user_message = get_last_user_message(body["messages"])

        # Access user settings (safe to access `__user__` directly)
        user_valves: Pipe.UserValves = __user__.get("valves", self.UserValves())
        user_lang = user_valves.language

        # Access files (must handle the None case)
        # A robust way to handle this is to default to an empty list.
        files = __files__ or []
        file_info = f"{len(files)} file(s) attached."

        # Access chat_id (must check for None)
        chat_id_info = (
            f"Chat ID: {__chat_id__}" if __chat_id__ else "No Chat ID provided."
        )

        # Access tools
        tools = __tools__ or {}
        tool_info = f"{len(tools)} tool(s) available."

        if body.get("stream", False):

            async def stream_generator():
                yield f"Streaming for user: {__user__.get('name', 'Anonymous')}.\n"
                yield f"{chat_id_info}\n"
                yield f"{file_info}\n"
                yield f"{tool_info}\n"
                yield f"Processing message in '{user_lang}': '{user_message}'.\n"

            return stream_generator()
        else:
            # For non-streaming, you can return a simple string or a dictionary.
            # The backend will wrap it in the appropriate OpenAI-compatible format.
            return (
                f"Completed for user: {__user__.get('name', 'Anonymous')}.\n"
                f"{chat_id_info}\n"
                f"{file_info}\n"
                f"{tool_info}\n"
                f"Processed message in '{user_lang}': '{user_message}'."
            )
