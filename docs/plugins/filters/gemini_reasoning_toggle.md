# `gemini_reasoning_toggle.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_reasoning_toggle.py` Open WebUI filter.

## Description

`gemini_reasoning_toggle.py` is a simple yet effective filter for Open WebUI. Its sole purpose is to add a "Reason" toggle button to the chat interface for specific models. When activated, this toggle signals the [`gemini_manifold.py`](../pipes/gemini_manifold.md) pipe to first output its "thinking" or reasoning process before providing the final answer. This is useful for understanding the model's thought process and for debugging complex prompts.

## Features

*   **UI Toggle:** Adds a "Reason" button with a lightbulb icon directly into the chat input area.
*   **Conditional Display:** The button only appears when a model that has this filter applied is selected.
*   **Simple Signaling:** When enabled, it adds a `reason: true` flag to the request sent to the backend pipe.

## How It Looks

The filter modifies the user interface by adding a new toggle button. The difference is clear when comparing a model with the filter to one without it.

**With Filter Enabled:** A "Reason" toggle appears next to other options like "Web Search".


**Without Filter:** The "Reason" toggle is absent.


## Installation

1.  Ensure you have the [`gemini_manifold.py`](../pipes/gemini_manifold.md) pipe installed and configured, as this filter is a companion to it.
2.  Install this filter to your Open WebUI instance.
3.  Navigate to the Open WebUI settings (`Admin Settings` > `Models`).
4.  Apply the `Reason (gemini_reasoning_toggle)` filter to the specific Gemini models (provided by `gemini_manifold.py`) for which you want to enable this functionality. You can do this by editing a model and checking the filter from in the filter list.

## Usage

1.  In the Open WebUI chat view, select a Gemini model that you have applied the filter to.
2.  You will see a "Reason" toggle button with a lightbulb icon appear above the message input field.
3.  Click this button to toggle the reasoning feature on or off for your next message.
4.  When enabled, the model will first stream its thought process, followed by the final answer.

## How It Works

This filter leverages Open WebUI's built-in filter capabilities in a straightforward manner:

*   **`self.toggle = True`**: This property in the filter's code tells the Open WebUI frontend to render a toggleable button. The `title` ("Reason") and `icon` (a lightbulb) properties define the button's appearance.
*   **`inlet(body: dict)`**: This is the core function. When the user enables the "Reason" toggle in the UI, this function intercepts the outgoing request. It simply adds a new key to the request body: `body["reason"] = True`.
*   The modified request is then passed to the `gemini_manifold.py` pipe, which is programmed to look for this `reason` flag and adjust its behavior accordingly.

## Dependencies

This filter is **strictly a companion** to the [`gemini_manifold.py`](../pipes/gemini_manifold.md) pipe. It serves no purpose on its own and will have no effect if used with other model pipes, as they would not know how to interpret the `reason: true` flag.

## License

MIT License. See the `LICENSE` file for details.