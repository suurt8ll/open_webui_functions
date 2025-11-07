# `gemini_manifold.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_manifold.py` Open WebUI plugin.

## Description

This is a manifold pipe function that adds support for Google's Gemini Studio API and Vertex AI into Open WebUI using `google-genai` SDK.

## Features

Here's a breakdown of implemented and planned features for the Gemini Manifold plugin:

**Implemented Features:**

-   [x] Display thinking summary
-   [x] Thinking budget
-   [x] Reasoning toggle (Reason filter function required, see [it's doc](../filters/gemini_reasoning_toggle.md))
-   [x] Native image generation and editing (image output)
-   [x] Document understanding (PDF and plaintext files). (Gemini Manifold Companion >= 1.4.0 filter required, see [it's doc](../filters/gemini_manifold_companion.md))
-   [x] Image input
-   [x] YouTube video input (automatically detects youtube.com and youtu.be URLs in messages)
-   [x] Video input support (other than YouTube URLs)
-   [x] Audio input support
-   [x] Google Files API
-   [x] Grounding with Google Search (Gemini Manifold Companion >= 1.2.0 required)
-   [x] Grounding with Google Maps (Gemini Manifold Companion >= 1.7.0 required). If you want to toggle this just like reasoning then install [Google Maps Grounding](../../../plugins/filters/gemini_map_grounding_toggle.py) filter function.
-   [x] Display citations in the front-end. (Gemini Manifold Companion >= 1.5.0 required)
-   [x] Permissive safety settings (Gemini Manifold Companion >= 1.3.0 required)
-   [x] Each user can decide to use their own API key.
-   [x] Token usage data
-   [x] Code execution tool. (Gemini Manifold Companion >= 1.1.0 required)
-   [x] URL context tool (Gemini Manifold Companion >= 1.5.0 required if you want to see citations in the front-end). If you want to toggle this then install [URL Context](../../../plugins/filters/gemini_url_context_toggle.py) filter function.
-   [x] Streaming and non-streaming responses.

**Planned Features:**

-   [ ] Native tool calling
-   [ ] Ability to easily switch between paid and free API

## Installation

To install this plugin, navigate to the [Open WebUI Community page for Gemini Manifold](https://openwebui.com/f/suurt8ll/gemini_manifold_google_genai) and click the white "Get" button.

## Configuration

After installation, click the gear icon next to the `gemini_manifold_google_genai` function within Open WebUI. At a minimum, you must enter your Google Gemini API key. Other configurable options are also available on that settings page.

## Usage

If the whitelist and blacklist are configured to allow models, those models will appear in the Open WebUI model selection list. If valid credentials for both Gemini Developer API and Vertex AI are provided then the models get fetched from both sources and merged together. To use a Gemini model, simply select it from the list and begin your chat.

## Troubleshooting

If you encounter issues, check the Open WebUI logs for error messages. The logs contain detailed information that should help pinpoint the problem. If you need further assistance, please open a new issue in this repository.

## Contributing

See `CONTRIBUTING.md`. For this plugin, I've also included several ideas in `TODO` comments within the code. These comments can serve as a starting point for contributions, but feel free to propose completely new features as well!

## License

MIT License. See the `LICENSE` file for details.