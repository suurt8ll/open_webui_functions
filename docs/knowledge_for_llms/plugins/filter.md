# Filter Plugin Reference

This document provides a detailed reference for Filter plugins in the Open WebUI plugin system. Filter plugins modify request and response data at various stages of the LLM interaction.

## Filter Plugin Methods

The behavior of Filter plugins is determined by the methods it defines within its `Filter` class:

*   **`inlet(body: dict) -> dict`**: Modifies the incoming request payload *before* it's sent to the LLM. Operates on the `form_data` dictionary.
*   **`stream(event: dict) -> dict`**: Modifies the streaming response from the LLM in real-time. Operates on individual chunks of data.
*   **`outlet(body: dict) -> dict`**: Modifies the complete response payload *after* it's received from the LLM. Operates on the final `body` dictionary.

## Technical Implementation (Filter Plugins)

*   **Method Invocation:**
    *   The `process_filter_functions` function in `backend/open_webui/utils/filter.py` retrieves a list of filter IDs, loads the corresponding plugin code, and calls the appropriate method (`inlet`, `stream`, or `outlet`) within the plugin's `Filter` class based on the `filter_type`.

## Plugin Structure (Filter Plugins)

A Filter plugin consists of the following:

*   A class named `Filter`.
*   One or more of the following methods within the `Filter` class: `inlet`, `stream`, `outlet`

## Data Structures

**1. `inlet`'s `body` (dict):**

```python
{
    "stream": bool,  # Whether streaming is enabled
    "model": str,  # Model ID (e.g., "llama3.2:1b")
    "messages": list[dict],  # List of messages in the conversation
    [
        {
            "role": str,  # "user" or "system"
            "content": str,  # Message text
        },
        # ... more messages ...
    ],
    "features": dict[str, bool],  # Enabled features
    {
        "image_generation": bool,
        "code_interpreter": bool,
        "web_search": bool,
    },
    "metadata": dict,  # Metadata about the request
    {
        "user_id": str,  # User ID
        "chat_id": str,  # Chat session ID
        "message_id": str,  # Unique message ID
        "session_id": str,  # Session ID
        "tool_ids": list[str] | None, # List of tool IDs, if any
        "files": list[dict] | None, # List of file metadata, if any
        "features": dict[str, bool], # (Redundant) Enabled features
        "variables": dict[str, str], # User-defined variables
        "model": dict, # (Redundant) Model details
        {
            "id": str, # Model ID
            "name": str, # Model Name
            # ... other model details ...
        },
        "direct": bool, # Direct connection flag
    },
    "options": dict,  # (Often empty) Additional options
    {}
}
```

**2. `stream`'s `event` (dict):**

```python
{
    "id": str,  # Unique ID for the chunk
    "created": int,  # Timestamp
    "model": str,  # Model ID
    "choices": list[dict],  # List of choices (usually one)
    [
        {
            "index": int,  # Choice index (usually 0)
            "logprobs": None,  # Log probabilities (often None)
            "finish_reason": str | None,  # "stop" or None
            "delta": dict,  # Incremental changes to the response
            {
                "content": str,  # Text chunk
            },
        },
        # ... more choices (rare) ...
    ],
    "object": str,  # "chat.completion.chunk"
    "usage": dict | None,  # Usage statistics (only in the last chunk)
    {
        "response_token/s": float,
        "prompt_token/s": float,
        "total_duration": int,
        # ... other usage details ...
    },
}
```

**3. `outlet`'s `body` (dict):**

```python
{
    "model": str,  # Model ID
    "messages": list[dict],  # List of messages in the conversation
    [
        {
            "id": str,  # Message ID
            "role": str,  # "user" or "assistant"
            "content": str,  # Message text
            "timestamp": int,  # Timestamp
            "usage": dict | None, # Usage statistics (only for assistant messages)
            {
                "response_token/s": float,
                "prompt_token/s": float,
                "total_duration": int,
                # ... other usage details ...
            },
        },
        # ... more messages ...
    ],
    "chat_id": str,  # Chat session ID
    "session_id": str,  # Session ID
    "id": str,  # ID of the last message
}
```

**Key Points:**

*   **Type Hints:** The data structures are represented using Python type hints for clarity.
*   **Optional Fields:** Some fields are marked as `Optional` (using `| None`) to indicate that they may not always be present.
*   **List Structure:** The `messages` and `choices` fields are lists of dictionaries.
*   **Dynamic Nature:** The exact structure of these dictionaries can vary depending on the LLM backend, features enabled, and other factors. It's always a good idea to inspect the actual data being passed to your filter plugins to ensure that you're accessing the correct fields.