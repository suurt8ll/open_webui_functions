# Open WebUI Functions

This repository contains a collection of helper scripts and enhancement functions designed to extend the capabilities of Open WebUI. These tools streamline development and provide advanced features, including integration with Google's Gemini API.

## `gemini_manifold.py`

[Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold)

### Description

`gemini_manifold.py` is a comprehensive tool for interacting with Google's Gemini Studio API. It supports:

-   Dynamic retrieval and registration of Google models.
-   Text and image content processing.
-   Streaming and non-streaming content generation.
-   Special handling for thinking models, including thought encapsulation.
-   Model name prefix stripping for generic adaptability.

**Important Note:** `gemini_manifold.py` requires the `google-genai` library to be installed in the Open WebUI environment. This library is not currently included in the Open WebUI repository, so users must install it manually. See [this discussion](https://github.com/open-webui/open-webui/discussions/8951) that is related to this.

**Installation Instructions for `google-genai`:**

-   **Python Environment:**
    ```bash
    pip install google-genai
    ```
-   **Docker Container:**
    1. Enter the running Open WebUI container:
        ```bash
        docker exec -it open-webui /bin/bash
        ```
    2. Install the library inside the container:
        ```bash
        pip install google-genai
        ```

**Warning:** Currently, Open WebUI updates may overwrite the manual installation of `google-genai`. You may need to reinstall it after each update.

### Usage

Requires Open WebUI v0.5.5 or later. The script automatically handles various MIME types and manages model-specific configurations.

### Debugging

Set `DEBUG = True` for detailed logging output, useful for development and troubleshooting. Use `docker logs -f open-webui` to view logs.

## Additional Scripts

-   `thinking_gemini.py`: [Open WebUI Community](https://openwebui.com/f/suurt8ll/thinking_gemini) An outdated script, initially developed for `gemini-2.0-flash-thinking-exp` model interactions. Superseded by `gemini_manifold.py`.
-   `function_updater.py`: Automates updating functions on a server via a REST API, streamlining development.

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, commit your changes, and open a pull request.

## License

MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the Open WebUI team and all contributors. Special thanks to Matthew for the groundwork on `thinking_gemini.py`.
