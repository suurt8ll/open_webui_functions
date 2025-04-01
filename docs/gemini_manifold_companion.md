# `gemini_manifold_companion.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_manifold_companion.py` Open WebUI filter, designed as a companion to the `gemini_manifold.py` pipe.

## Description

The `gemini_manifold_companion.py` filter enhances the functionality of the "Gemini Manifold google_genai" pipe within Open WebUI. Its primary function is to intercept and modify requests to enable Google Search grounding with compatible Gemini models.  Future functionality may include hijacking code execution requests and citation marker removal. It's a separate, complementary component in the Open WebUI plugin system.

## Features

**Implemented Features:**

-   **Google Search Grounding:** Enables Google Search grounding for supported Gemini models.  Intercepts the Open WebUI's built-in search feature, disables it, and passes custom values to `gemini_manifold.py` to trigger grounding.
-   **Configurable Valves:** Offers configurable settings for grounding behavior.

**Planned Features:**

-   Code execution request hijacking and processing.
-   Citation marker removal.

## Installation

1.  Install and configure `gemini_manifold.py`, see [it's docs](docs/gemini_manifold.md).
2.  Install the filter within Open WebUI. You can get it from [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold_companion).
3.  Enable the filter globally, or on specific models that support grounding with Google Search. This can be done through Open WebUI's plugin management interface.

## Configuration

The filter is configured via a "Valves" settings menu accessible by clicking the gear icon associated with the filter in Open WebUI. The following settings are available:

*   **Set Temp To Zero:** *(Enabled/Disabled)* When enabled, the filter will override the temperature setting and set it to `0` for grounded answers, as recommended by Google.
*   **Grounding Dynamic Retrieval Threshold:** *(Default/Custom)* This setting controls the dynamic retrieval threshold for Google Search grounding.
    *   **Default:** Uses Google's default threshold value. Refer to the [Google Gemini API documentation on grounding](https://ai.google.dev/gemini-api/docs/grounding?lang=python#dynamic-threshold) for more information.
    *   **Custom:** Allows specifying a custom threshold value.  Consult the Google documentation for information on acceptable values and their impact. This setting is only supported for Gemini 1.0 and 1.5 models.

## Usage

Once installed and configured, the filter automatically modifies requests for models registered by `gemini_manifold.py` and listed in `ALLOWED_GROUNDING_MODELS` (defined in the filter's code) when the user enables web search in the Open WebUI chat interface.

The filter checks for the `gemini_manifold_google_genai.` prefix in the model name to ensure it's interacting with a model managed by the `gemini_manifold.py` pipe.

## Data Pipeline

The filter's operation can be summarized as follows:

1.  **`inlet(body: dict) -> dict`:** This function modifies the incoming request payload (`body`) *before* it's sent to the `gemini_manifold.py` pipe.
    *   It checks if the model is a supported Gemini model and if web search is enabled in the request's `features` dictionary.
    *   If both conditions are met, it disables the default `web_search` feature in the `features` dictionary.
    *   It adds custom features to the `metadata` dictionary within the `body` to signal to the `gemini_manifold.py` pipe that Google Search grounding should be enabled.  It uses `"google_search_retrieval"` for Gemini 1.0 and 1.5 models and `"google_search_tool"` for later models.  These flags are added to `body["metadata"]["features"]`.
    *   If the "Set Temp To Zero" valve is enabled, it overwrites the `temperature` value in the `body` with `0`.
2.  **`stream(event: dict) -> dict`:** This function currently performs no modifications on the streaming response. It's included for potential future use.
3.  **`outlet(body: dict) -> dict`:** This function is intended for post-processing the complete response, but currently only contains a `TODO` for citation marker removal.

## Request Body Modification Example

The following demonstrates how the `inlet()` function modifies the request body when web search is enabled:

**Before:**

```json
{
  "model": "gemini_manifold_google_genai.gemini-2.0-flash",
  "prompt": "What is the capital of France?",
  "features": {
    "web_search": true
  },
  "temperature": 0.7
}
```

**After:**

```json
{
  "model": "gemini_manifold_google_genai.gemini-2.0-flash",
  "prompt": "What is the capital of France?",
  "features": {
    "web_search": false
  },
  "temperature": 0,
  "metadata": {
    "features": {
      "google_search_tool": true
    }
  }
}
```

## Dependencies

This filter is designed to work exclusively with the `gemini_manifold.py` pipe. It has no external dependencies. The `gemini_manifold.py` function handles the actual interaction with the Google Gemini API. The filter is designed to fail gracefully if the "Gemini Manifold google\_genai" pipe function is not properly configured or is unavailable, by skipping all modifications.

## Troubleshooting

If grounding is not working as expected, check the following:

*   Ensure the filter is enabled globally or for the specific model being used.
*   Verify that the model name includes the `gemini_manifold_google_genai.` prefix and is present in the `ALLOWED_GROUNDING_MODELS` list within the filter's code.
*   Consult the Open WebUI logs for any error messages related to the filter.

## License

MIT License. See the `LICENSE` file for details.