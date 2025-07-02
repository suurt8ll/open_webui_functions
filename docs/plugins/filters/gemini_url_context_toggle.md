# `gemini_url_context_toggle.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_url_context_toggle.py` Open WebUI filter.

## Description

`gemini_url_context_toggle.py` is a companion filter for the [`gemini_manifold.py`](../pipes/gemini_manifold.md) pipe. Its primary function is to add a "URL Context" toggle button to the chat interface. When this toggle is enabled, it instructs compatible Gemini models to fetch and analyze the content of any URLs provided in the user's prompt, using that content as context to generate a more informed response.

This feature is particularly useful for tasks like summarizing articles, answering questions about web page content, or comparing information from multiple online sources without needing to manually copy and paste the text.

## Features

*   **UI Toggle:** Adds a "URL Context" button with a link icon to the chat input area.
*   **Dynamic Control:** Allows users to enable or disable URL fetching on a per-request basis.
*   **Seamless Integration:** Works directly with the `gemini_manifold.py` pipe to enable Google's native URL context tool.

## How It Looks

The filter enhances the user interface by adding a dedicated toggle button for the URL Context feature.

**With Filter Enabled:** A "URL Context" toggle appears alongside other chat options.

**Without Filter:** The "URL Context" toggle is not present. The feature can only be controlled globally via the `gemini_manifold.py` pipe's settings.

## Installation

1.  **Prerequisite:** Ensure you have the [`gemini_manifold.py`](../pipes/gemini_manifold.md) pipe installed and configured, as this filter depends on it.
2.  Install this filter from the Open WebUI community page.
3.  Navigate to the Open WebUI settings (`Admin Settings` > `Models`).
4.  Apply the `URL Context (gemini_url_context_toggle)` filter to the specific Gemini models for which you want to enable this functionality. Note that only certain models support this tool (e.g., Gemini 2.5 Pro, Gemini 2.5 Flash).

## Usage

1.  In the Open WebUI chat view, select a compatible Gemini model that has this filter applied.
2.  You will see a "URL Context" toggle button with a link icon appear above the message input field.
3.  Click the button to enable the feature for your next message.
4.  Include one or more URLs directly in your prompt (e.g., "Summarize this article for me: https://...").
5.  The model will retrieve the content from the URL(s) and use it to formulate its response.

## How It Works

This filter is a simple but powerful signaling mechanism for the main `gemini_manifold.py` pipe.

*   **`self.toggle = True`**: This property tells the Open WebUI frontend to render a toggleable button. The `title` ("URL Context") and `icon` (a link symbol) define the button's appearance.
*   **`inlet(body: "Body")`**: When the "URL Context" toggle is active, this function intercepts the outgoing request. It adds a `url_context: true` flag to the `features` dictionary within the request's metadata: `metadata["features"]["url_context"] = True`.
*   The `gemini_manifold.py` pipe is designed to check for this `url_context` flag. If it's present and `True`, the pipe activates Google's `UrlContext` tool in the API call, assuming the selected model is compatible.

### What if this filter is not installed?

The `gemini_manifold.py` pipe is designed to function without this filter. If the filter is not installed or not active for a model:
*   The pipe will log a warning to the console indicating the filter is missing.
*   It will fall back to its own internal configuration valve, `ENABLE_URL_CONTEXT_TOOL`.
*   By default, this valve is `False`, meaning the feature will be **disabled** unless an administrator explicitly enables it in the pipe's settings.

This filter provides **user-level, per-request control**, while the pipe's valve provides **admin-level, global control**.

## Dependencies

This filter is **strictly a companion** to the [`gemini_manifold.py`](../pipes/gemini_manifold.md) pipe. It will have no effect if used with other model pipes, as they would not be programmed to recognize the `url_context: true` flag.

## License

MIT License. See the `LICENSE` file for details.