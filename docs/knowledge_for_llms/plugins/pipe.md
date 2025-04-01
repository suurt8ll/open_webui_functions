# Pipe Plugin Reference

This document provides a detailed reference for Pipe plugins in the Open WebUI plugin system. Pipe plugins allow you to integrate custom models and behaviors into Open WebUI. They are defined in Python and can be created, updated, and executed via the API.

## Pipe Plugin Structure

A Pipe plugin consists of the following:

*   **Code File:** A Python file containing the plugin's logic.
*   **Class Definition:** A class named `Pipe` that encapsulates the plugin's functionality.
*   **`pipe()` Method:** The main execution logic of the plugin. This method is called when a user interacts with a model associated with the plugin.
*   **`pipes()` Method (Optional):** A method that defines multiple model variants exposed by the plugin.

## The `pipe()` Method

The `pipe()` method is the heart of a Pipe plugin. It contains the core logic for processing input data and returning output. It acts as a conduit (or "pipe") for data transformation and processing.

**Purpose:**

The `pipe()` function is responsible for processing input data and generating output in a structured manner. It acts as a conduit (or "pipe") for data transformation and processing.

**Signature:**

```python
async def pipe(self, body: dict, **kwargs) -> str | Generator | AsyncGenerator | dict | BaseModel | StreamingResponse:
```

**Parameters:**

*   `body` (dict): The request payload, typically containing:
    *   `messages`: A list of chat messages.
    *   `model`: The selected model identifier.
    *   `stream`: A boolean indicating whether to stream the response.
    *   ... other model-specific parameters.
*   `**kwargs`: Additional keyword arguments that may include:
    *   `__user__`: User-specific metadata (e.g., ID, role, valves).
    *   `__tools__`: Available tools.
    *   `__files__`: Available files.
    *   `__metadata__`: Additional metadata.

**Returns:**

The `pipe()` method can return several types of objects, each handled differently by the backend:

*   `str`: A simple text response. The backend will wrap this in an OpenAI-like chat message format.
*   `dict`: A structured response with key-value pairs (e.g., JSON). The backend will send this directly to the client as JSON.
*   `BaseModel` (from `pydantic`): A Pydantic model object, used to enforce a schema on the data. The backend will serialize this to JSON using `model_dump()`.
*   `StreamingResponse` (from `starlette.responses`): Used for streaming data back to the client. The backend will stream the `body_iterator` directly to the client.
*   `Generator`: A Python generator that yields data incrementally. The backend will convert the generator output into a single concatenated string or stream it to the client.
*   `AsyncGenerator`: An asynchronous generator that yields data incrementally. The backend will process each chunk asynchronously and stream it to the client.

**Example:**

```python
async def pipe(self, body: dict, **kwargs) -> str:
    """
    Main execution logic of the function.
    """
    messages = body.get("messages", [])
    # Implement your custom logic here...
    result = "This is a response from MyFunction"
    return result
```

## The `pipes()` Method

The `pipes()` method is an optional method that defines multiple model variants exposed by the plugin. This allows a single plugin to expose multiple models with different configurations or behaviors.

**Signature:**

```python
async def pipes(self) -> list:
```

**Returns:**

*   `list`: A list of dictionaries, each representing a model:
    *   `id`: A unique identifier for the model.
    *   `name`: A user-friendly name for the model.

**Example:**

```python
async def pipes(self) -> list:
    """
    (Optional) Defines multiple model variants exposed by this function.
    """
    return [
        {"id": "model_variant_1", "name": "Model Variant 1"},
        {"id": "model_variant_2", "name": "Model Variant 2"},
    ]
```

## Pipe Plugin Creation Example

Here's an example of a basic Pipe plugin:

```python
from pydantic import BaseModel, Field

class Pipe:
    async def pipes(self) -> list:
        """
        (Optional) Defines multiple model variants exposed by this function.

        Returns:
            list: A list of dictionaries, each representing a model:
                - id: A unique identifier for the model.
                - name: A user-friendly name for the model.
        """
        return [
            {"id": "model_variant_1", "name": "Model Variant 1"},
            {"id": "model_variant_2", "name": "Model Variant 2"},
        ]

    async def pipe(self, body: dict, **kwargs) -> str | Generator | AsyncGenerator:
        """
        Main execution logic of the function.

        Args:
            body (dict): The request payload, typically containing:
                - messages: A list of chat messages.
                - model: The selected model identifier.
                - stream: A boolean indicating whether to stream the response.
                - ... other model-specific parameters.
            **kwargs: Additional keyword arguments (e.g., user information, tools).

        Returns:
            str | Generator | AsyncGenerator: The function's response.
        """
        # Access valve settings:
        # param_value = self.valves.parameter_1 #Access values like this.

        # Access request
        messages = body.get("messages", [])

        # Implement your custom logic here...
        result = "This is a response from Pipe"

        return result
```

## Best Practices for Pipe Plugins

*   **Asynchronous Operations:** Utilize `async def` for I/O-bound operations within the `pipe()` method to ensure responsiveness.
*   **Error Handling:** Implement robust error handling within the `pipe()` method to gracefully handle unexpected situations and provide informative error messages. *(More details needed on specific error handling mechanisms)*
*   **Valve Validation:** Leverage Pydantic's validation features (e.g., `ge`, `le`, `...` for required fields) to ensure the integrity of valve configurations.
*   **Resource Management:** Properly manage external resources (e.g., API connections, database connections) using `async with` context managers to prevent leaks.
*   **Security:** Sanitize user inputs and validate data received from external sources to mitigate security risks.
*   **Modularity:** Design functions with a clear separation of concerns, making them reusable and maintainable.
*   **Documentation:** Provide comprehensive docstrings for functions and methods to facilitate understanding and collaboration.

## Security Considerations Specific to Pipe Plugins

*(More details needed on specific security considerations for Pipe plugins, especially regarding the execution of custom code and access to system resources.)*

## Execution Context and Parameters

When a `pipe()` function is executed, it receives a `body` dictionary containing the user's input and other relevant data. It also receives additional keyword arguments (`**kwargs`) that provide context and access to various system resources.

Here's a breakdown of the typical parameters available in the `pipe()` function's execution context:

*   **`body` (dict):** The main data payload, typically containing:
    *   `messages`: A list of chat messages.
    *   `model`: The selected model identifier.
    *   `stream`: A boolean indicating whether to stream the response.
    *   ... other model-specific parameters.
*   **`__user__` (object):** An object representing the current user, containing:
    *   `id`: The user's ID.
    *   `role`: The user's role (e.g., "admin", "user").
    *   `valves`: User-specific valve settings (if any).
*   **`__tools__` (list):** A list of available tools that the plugin can use.
*   **`__files__` (list):** A list of available files that the plugin can access.
*   **`__metadata__` (dict):** Additional metadata about the request.

## Handling Different Return Types

The `pipe()` function can return a variety of data types, and the backend handles each type differently to ensure seamless integration with the client.

Here's a summary of how the backend processes each return type:

| Return Type            | Backend Handling                                                                                              | Example Use Case                              |
|------------------------|-------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| **`str`**              | Wrapped in OpenAI-like response format and sent to the client.                                              | Simple text responses.                       |
| **`dict`**             | Sent as-is (JSON encoded) to the client.                                                                    | Structured API responses.                    |
| **`BaseModel`**        | Converted to JSON using `model_dump()` and sent to the client.                                              | Enforcing response schema.                   |
| **`StreamingResponse`**| The `body_iterator` is streamed directly to the client.                                                    | Real-time or large data streaming.           |
| **`Generator`**        | Data is concatenated or streamed to the client, line by line.                                               | Dynamic or iterative data generation.        |
| **`AsyncGenerator`**   | Data is streamed asynchronously to the client.                                                             | Streaming large or asynchronous operations.  |

## Error Handling in Pipe Plugins

When implementing a Pipe plugin, it's crucial to handle errors gracefully to prevent unexpected behavior and provide informative error messages to the user.

*(More details needed on specific error handling mechanisms and best practices. Should plugins raise exceptions? Return error codes? Log errors?)*

## Example Scenarios for Pipe Plugins

Here are a few example scenarios where Pipe plugins might be used:

1.  **Chat Completion Pipeline:** The `pipe()` function processes user messages and generates completions using a model.
2.  **Document Processing:** The `pipe()` function ingests a document, performs operations (e.g., summarization), and returns the result.
3.  **Custom Logic:** A developer defines a custom `pipe()` function to implement business logic (e.g., filtering, validation).
4.  **Integrating with External APIs:** The `pipe()` function calls an external API and returns the results.

## Key Functions for Handling Responses

The backend uses several key functions to handle the responses from Pipe plugins:

*   **`openai_chat_chunk_message_template()`:** Wraps non-streaming responses into an OpenAI-compatible format.
*   **`process_line()`:** Processes individual chunks of data (strings, dictionaries, etc.) into the appropriate format for streaming.