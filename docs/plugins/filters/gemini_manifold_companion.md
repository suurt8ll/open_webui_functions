# `gemini_manifold_companion.py` - Detailed Documentation

This document provides a comprehensive overview of the `gemini_manifold_companion.py` Open WebUI filter, designed as a companion to the `gemini_manifold.py` pipe.

## Description

The `gemini_manifold_companion.py` filter enhances the functionality of the "Gemini Manifold google_genai" pipe within Open WebUI. Its primary function is to intercept and modify requests to enable **Google Search grounding**, **code execution**, and **backend RAG bypass** with compatible Gemini models. It also **adds citation markers and source information** to grounded responses. It's a separate, complementary component in the Open WebUI plugin system.

## Features

**Implemented Features:**

-   **Google Search Grounding:** Enables Google Search grounding for supported Gemini models. Intercepts the Open WebUI's built-in search feature, disables it, and passes custom values to `gemini_manifold.py` to trigger grounding.
-   **Google Code Execution:** Intercepts Open WebUI's code interpreter feature, disables it, and signals `gemini_manifold.py` to use Google's code execution tool via the Gemini API.
-   **Backend RAG Bypass:** Allows bypassing Open WebUI's built-in RAG pipeline and sending uploaded documents directly to the Gemini API via `gemini_manifold.py` for native processing.
-   **Citation Marker Addition & Source Emission:** Adds `[index]` style citation markers to the model's response text based on grounding metadata and emits separate events for the UI to display source information (resolved URLs) and search queries.
-   **Configurable Valves:** Offers configurable settings to control grounding behavior, RAG bypass, permissive safety settings, and logging verbosity.

**Planned Features:**

-   Citation marker removal (from input prompt).

## Installation

1.  Install and configure `gemini_manifold.py`, see [it's docs](../pipes/gemini_manifold.md).
2.  Install the filter within Open WebUI. You can get it from [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold_companion).
3.  Enable the filter globally, or on specific models that support the desired features (grounding, code execution, RAG bypass). This can be done through Open WebUI's plugin management interface.

## Configuration

The filter is configured via a "Valves" settings menu accessible by clicking the gear icon associated with the filter in Open WebUI. The following settings are available:

*   **Set Temp To Zero:** *(Enabled/Disabled)* When enabled, the filter will override the temperature setting and set it to `0` for grounded answers, as recommended by Google.
*   **Grounding Dynamic Retrieval Threshold:** *(Default/Custom)* This setting controls the dynamic retrieval threshold for Google Search grounding.
    *   **Default:** Uses Google's default threshold value. Refer to the [Google Gemini API documentation on grounding](https://ai.google.dev/gemini-api/docs/grounding?lang=python#dynamic-threshold) for more information.
    *   **Custom:** Allows specifying a custom threshold value. Consult the Google documentation for information on acceptable values and their impact. This setting is only supported for Gemini 1.0 and 1.5 models.
*   **USE_PERMISSIVE_SAFETY:** *(Enabled/Disabled)* When enabled, the filter will add permissive safety settings to the request payload, potentially allowing the model to generate responses that might otherwise be blocked by default safety filters. Refer to Google's safety settings documentation for details.
*   **BYPASS_BACKEND_RAG:** *(Enabled/Disabled)* When enabled, the filter intercepts document uploads, prevents Open WebUI's backend RAG from processing them, and signals the `gemini_manifold.py` pipe to handle the raw documents directly with the Gemini API. **Note: This feature is not supported in temporary ('local') chats.**
*   **LOG_LEVEL:** *(TRACE/DEBUG/INFO/SUCCESS/WARNING/ERROR/CRITICAL)* Controls the logging verbosity of this specific filter. Use `docker logs -f open-webui` to view logs.

## Usage

Once installed and configured, the filter automatically modifies requests for models registered by `gemini_manifold.py` and listed in `ALLOWED_GROUNDING_MODELS` or `ALLOWED_CODE_EXECUTION_MODELS` (defined in the filter's code) when the user enables **web search** or **code interpreter** in the Open WebUI chat interface, or when **documents are uploaded** to a persistent chat (and `BYPASS_BACKEND_RAG` is enabled).

The filter checks for the `gemini_manifold_google_genai.` prefix in the model name (or its base model ID) to ensure it's interacting with a model managed by the `gemini_manifold.py` pipe.

## Data Pipeline

The filter's operation can be summarized as follows:

1.  **`inlet(body: dict, __metadata__: dict[str, Any]) -> dict`:** This function modifies the incoming request payload (`body`) *before* it's sent to the `gemini_manifold.py` pipe.
    *   It checks if the model is a supported Gemini model for grounding or code execution using the `gemini_manifold_google_genai.` prefix and internal model lists (`ALLOWED_GROUNDING_MODELS`, `ALLOWED_CODE_EXECUTION_MODELS`).
    *   If **web search** is enabled in the request's `features` dictionary and the model supports grounding, it disables the default `web_search` feature. It then adds custom features (`google_search_retrieval` for 1.0/1.5 models, `google_search_tool` for >=2.0 models) to `body["metadata"]["features"]` to signal the pipe to use Google Search grounding. If the "Set Temp To Zero" valve is enabled, it overwrites the `temperature` value with `0`.
    *   If **code interpreter** is enabled in the request's `features` dictionary and the model supports code execution, it disables the default `code_interpreter` feature. It then adds the custom feature `google_code_execution` to `body["metadata"]["features"]` to signal the pipe to use Google's code execution tool.
    *   If the **`BYPASS_BACKEND_RAG`** valve is enabled and documents are present in `body["files"]` (and the chat is not temporary), it clears the `body["files"]` list to prevent Open WebUI's backend RAG from processing them. It then adds `upload_documents: true` to `body["metadata"]["features"]` to signal the pipe to handle the raw documents directly with the Gemini API. If the valve is disabled or it's a temporary chat, `upload_documents: false` is added.
    *   If the **`USE_PERMISSIVE_SAFETY`** valve is enabled, it adds permissive safety settings to `body["metadata"]["safety_settings"]`.
    *   It ensures the necessary `metadata` and `metadata.features` structures exist in the `body`.
2.  **`stream(event: dict) -> dict`:** This function currently performs no modifications on the streaming response from the LLM. It's included for potential future use.
3.  **`outlet(body: dict, __request__: Request, __metadata__: dict[str, Any], __event_emitter__: Callable[["Event"], Awaitable[None]], **kwargs) -> dict`:** This function processes the complete response payload *after* it's received from the `gemini_manifold.py` pipe.
    *   It checks for `GroundingMetadata` that the pipe may have stored in the request state (`__request__.app.state`).
    *   If grounding metadata is found (indicating the response was grounded), it processes the model's response text to **add `[index]` style citation markers** based on the `grounding_supports` and `grounding_chunks` information provided by the API. The response text in `body["messages"][-1]["content"]` is updated with these markers.
    *   It initiates a background task to **resolve the original source URLs** (e.g., following redirects) found in the grounding metadata. Once resolved, it **emits a `chat:completion` event containing the resolved source information** for display in the UI's source panel.
    *   It **emits a `status` event** containing the actual Google search queries used by the model, often including links to the search results pages.
    *   The original `TODO` regarding citation marker *removal* from the *input* prompt remains a separate potential future feature.

## Dependencies

This filter is designed to work exclusively with the `gemini_manifold.py` pipe. It has no external dependencies beyond the standard Open WebUI environment and the `google-genai` library (which is a dependency of the pipe). The `gemini_manifold.py` function handles the actual interaction with the Google Gemini API. The filter is designed to fail gracefully if the "Gemini Manifold google\_genai" pipe function is not properly configured or is unavailable, by skipping all modifications.

## Troubleshooting

If features are not working as expected, check the following:

*   Ensure the filter is enabled globally or for the specific model being used.
*   Verify that the model name includes the `gemini_manifold_google_genai.` prefix and is present in the relevant `ALLOWED_GROUNDING_MODELS` or `ALLOWED_CODE_EXECUTION_MODELS` list within the filter's code.
*   **For Grounding:** Ensure web search is enabled in the Open WebUI chat interface.
*   **For Code Execution:** Ensure the code interpreter feature is enabled in the Open WebUI chat interface.
*   **For RAG Bypass:** Ensure the `BYPASS_BACKEND_RAG` valve is enabled for the filter. Verify you are using a persistent chat (not a temporary 'local' chat). Check if the documents are actually being included in the initial request payload sent by the UI.
*   Consult the Open WebUI logs (using `docker logs -f open-webui`) for any error messages related to the filter. Adjust the `LOG_LEVEL` valve for more detailed output if needed.

## License

MIT License. See the `LICENSE` file for details.