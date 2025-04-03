# Open WebUI Functions

This repository contains a collection of Open WebUI plugins I have personally coded for my own use and find useful.

## Plugins and Scripts

-   `gemini_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold_google_genai) | This plugin provides support for Google's Gemini Studio API. See the [Detailed Documentation](docs/gemini_manifold.md) for more information.
-   `gemini_manifold_companion.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold_companion) | A companion filter for "Gemini Manifold google_genai" pipe providing enhanced functionality, such as Google Search grounding. See the [Detailed Documentation](docs/gemini_manifold_companion.md) for more information.
-   `venice_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/venice_image_generation) | Enables image creation by using any diffusion model offered by Venice.ai API.
-   `system_prompt_injector.py` | The idea is to allow changing chat options like system prompt and temperature from the chatbox. It would pair nicely with Prompts feature Open WebUI offers.
-   `function_updater.py` | Monitors specified Python files (defined in the `.env` file). When a file change is detected, it automatically updates the corresponding function in the Open WebUI backend using the REST API. Requires function metadata (id, title, description) to be defined in the file's docstring.
-   `archived_functions/` | This directory contains old functions that I'm not developing anymore.
-   `test_and_example_functions/` | This directory contains different test and example plugins I usually code up when exploring what is possible.

## Contributing

Contributions are welcome! Here's how to contribute:

1.  **Fork the repository** to your own GitHub account.
2.  **Create a feature branch** based on the `master` branch.  Name your branch something descriptive, like `feature/add-new-functionality` or `fix/bug-description`.

    ```bash
    git checkout master
    git checkout -b feature/your-feature-name
    ```

3.  **Commit your changes** with clear and informative commit messages.
4.  **Push your feature branch** to your forked repository.

    ```bash
    git push origin feature/your-feature-name
    ```

5.  **Open a pull request** on GitHub, targeting the **`master` branch** of the main repository.

## License

MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the Open WebUI team and all contributors. Special thanks to Matthew for the groundwork on `thinking_gemini.py`.
