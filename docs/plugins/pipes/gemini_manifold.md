# `gemini_manifold.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_manifold.py` Open WebUI plugin.

## Description

This is a manifold pipe function that adds support for Google's Gemini Studio API into Open WebUI.

## Features

Here's a breakdown of implemented and planned features for the Gemini Manifold plugin:

**Implemented Features:**

-   [x] Native image generation (image output), use "gemini-2.0-flash-exp-image-generation"
-   [x] Document understanding (PDF and plaintext files). (Gemini Manifold Companion >= 1.4.0 required)
-   [x] Display citations in the front-end.
-   [x] Image input
-   [x] YouTube video input (automatically detects youtube.com and youtu.be URLs in messages)
-   [x] Streaming
-   [x] Grounding with Google Search (requires installing the "Gemini Manifold Companion" >= 1.2.0 filter - see [it's doc](../filters/gemini_manifold_companion.md))
-   [x] Permissive safety settings (Gemini Manifold Companion >= 1.3.0 required)
-   [x] Each user can decide to use their own API key.
-   [x] White- and blacklist based model retrieval and registration.
-   [x] Display usage statistics (token counts)
-   [x] Code execution tool. ("Gemini Manifold Companion" >= 1.1.0 required)
-   [x] URL context tool (allows the model to fetch and use content from provided URLs for grounding). (Gemini Manifold Companion not required for this specific tool, but model compatibility is necessary).

**Planned Features:**

-   [ ] Audio input support.
-   [ ] Video input support (other than YouTube URLs).

## Installation

To install this plugin, navigate to the [Open WebUI Community page for Gemini Manifold](https://openwebui.com/f/suurt8ll/gemini_manifold_google_genai) and click the white "Get" button.

## Configuration

After installation, click the gear icon next to the `gemini_manifold_google_genai` function within Open WebUI. At a minimum, you must enter your Google Gemini API key. Other configurable options are also available on that settings page.

## URL Context Tool

The URL Context Tool enhances Gemini's capabilities by allowing it to fetch and incorporate content from web pages directly into its context when generating responses. This is useful for grounding answers with specific, up-to-date information from the web.

**Enabling the Tool:**

To use this feature, you need to enable it via the `ENABLE_URL_CONTEXT_TOOL` valve in the `Pipe.Valves` configuration. Set this valve to `True`. By default, it is `False`.

```python
class Pipe:
    class Valves(BaseModel):
        # ... other valves
        ENABLE_URL_CONTEXT_TOOL: bool = Field(
            default=False,
            description="Enable the URL context tool to allow the model to fetch and use content from provided URLs. This tool is only compatible with specific models.",
        )
        # ... other valves
```

**Supported Models:**

This tool is only available for the following models:
*   `gemini-2.5-pro-preview-05-06`
*   `gemini-2.5-flash-preview-05-20`
*   `gemini-2.0-flash`
*   `gemini-2.0-flash-live-001`

If the tool is enabled but an unsupported model is selected, it will be automatically skipped.

**Important Notes:**

*   This tool is intended for general URLs provided in your prompt. It is separate from the automatic handling of YouTube URLs (from `youtube.com` or `youtu.be`), which are processed differently by the manifold.
*   When the tool successfully retrieves content from URLs, these URLs will be listed in the chat interface, providing transparency about the information sources used by the model. (This requires front-end support for the `chat:url_context` event).

## Usage

If the whitelist and blacklist are configured to allow models, those models will appear in the Open WebUI model selection list. To use a Gemini model, simply select it from the list and begin your chat.

## Troubleshooting

If you encounter issues, check the Open WebUI logs for error messages. The logs contain detailed information that should help pinpoint the problem. If you need further assistance, please open a new issue in this repository.

## Contributing

Contribution guidelines are the same as described in the main `README.md` of this repository. For this plugin, I've included several ideas in `TODO` comments within the code. These comments can serve as a starting point for contributions, but feel free to propose completely new features as well!

## License

MIT License. See the `LICENSE` file for details.
