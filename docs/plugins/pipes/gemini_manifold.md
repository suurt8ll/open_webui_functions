# `gemini_manifold.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_manifold.py` Open WebUI plugin.

## Description

This is a manifold pipe function that adds support for Google's Gemini Studio API into Open WebUI.

## Features

Here's a breakdown of implemented and planned features for the Gemini Manifold plugin:

**Implemented Features:**

-   [x] Native image generation (image output), use "gemini-2.0-flash-exp-image-generation"
-   [x] Display citations in the front-end.
-   [x] Image input
-   [x] Streaming
-   [x] Grounding with Google Search (requires installing the "Gemini Manifold Companion" >= 1.2.0 filter - see [it's doc](../filters/gemini_manifold_companion.md))
-   [x] Safety settings
-   [x] Each user can decide to use their own API key.
-   [x] White- and blacklist based model retrieval and registration.
-   [x] Display usage statistics (token counts)
-   [x] Code execution tool. ("Gemini Manifold Companion" >= 1.1.0 required)

**Planned Features:**

-   [ ] Audio input support.
-   [ ] Video input support.
-   [ ] PDF (other documents?) input support, `__files__` param passed to the `pipe()` function can be used for this.

## Installation

To install this plugin, navigate to the [Open WebUI Community page for Gemini Manifold](https://openwebui.com/f/suurt8ll/gemini_manifold_google_genai) and click the white "Get" button.

## Configuration

After installation, click the gear icon next to the `gemini_manifold_google_genai` function within Open WebUI. At a minimum, you must enter your Google Gemini API key. Other configurable options are also available on that settings page.

## Usage

If the whitelist and blacklist are configured to allow models, those models will appear in the Open WebUI model selection list. To use a Gemini model, simply select it from the list and begin your chat.

## Troubleshooting

If you encounter issues, check the Open WebUI logs for error messages. The logs contain detailed information that should help pinpoint the problem. If you need further assistance, please open a new issue in this repository.

## Contributing

Contribution guidelines are the same as described in the main `README.md` of this repository. For this plugin, I've included several ideas in `TODO` comments within the code. These comments can serve as a starting point for contributions, but feel free to propose completely new features as well!

## License

MIT License. See the `LICENSE` file for details.
