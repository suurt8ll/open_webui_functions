# Open WebUI Functions

This repository contains a collection of helper scripts and enhancement functions designed to extend the capabilities of Open WebUI. These tools streamline development and provide advanced features, including integration with Google's Gemini API.

## `gemini_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold)

### Description

This is a manifold pipe function that adds support for Google's Gemini Studio API.

### Features

-   Whitelist based model retrieval and registration.
-   Support for text and image input.
-   Streaming and non-streaming content generation.
-   Special handling for thinking models, including thought encapsulation.

### Usage

Requires Open WebUI v0.5.5 or later. **NB!** `gemini_manifold.py` requires the `google-genai` library to be installed in the Open WebUI environment. This library is not currently included in the Open WebUI repository, so users must install it manually. See [this discussion](https://github.com/open-webui/open-webui/discussions/8951) that is related to this.

### Installation Instructions for `google-genai`:

-   **Python Environment:**
    ```bash
    pip install google-genai==0.7.0
    ```
-   **Docker Container:**
    1. Enter the running Open WebUI container:
        ```bash
        docker exec -it open-webui /bin/bash
        ```
    2. Install the library inside the container:
        ```bash
        pip install google-genai==0.7.0
        ```

**Warning:** Currently, Open WebUI updates may overwrite the manual installation of `google-genai`. You may need to reinstall it after each update.

## Additional Scripts

-   `thinking_gemini.py`: [Open WebUI Community](https://openwebui.com/f/suurt8ll/thinking_gemini) An outdated script, initially developed for `gemini-2.0-flash-thinking-exp` model interactions. Superseded by `gemini_manifold.py`.
-   `function_updater.py`: Automates updating functions on a server via a REST API, streamlining development.
-   `smallthinker_filter.py`: At first I wanted to code a simple filter that makes output of the `smallthinker:3b` nicer but now I'm thinkig of turning it into more general reasoning model formatter.
-   `system_prompt_injector.py`: The idea is to allow chaning chat options like system prompt and temperature from the chatbox. It would pair nicely with Prompts feature Open WebUI offers.

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, commit your changes, and open a pull request.

## License

MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the Open WebUI team and all contributors. Special thanks to Matthew for the groundwork on `thinking_gemini.py`.
