# Open WebUI Functions

This repository contains a collection of Open WebUI plugins I personally use and find useful.

## Plugins and Scripts

-   `gemini_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold) | This plugin provides support for Google's Gemini Studio API. See the [Detailed Documentation](docs/gemini_manifold.md) for more information.
-   `venice_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/venice_image_generation) | Enables image creation by using any diffusion model offered by Venice.ai API.
-   `system_prompt_injector.py`: The idea is to allow changing chat options like system prompt and temperature from the chatbox. It would pair nicely with Prompts feature Open WebUI offers.
-   `function_updater.py`: Listens for file changes on selected files in `.env` file and automatically updates the functions in the backend with REST API if change is detected.
-   `smallthinker_filter.py`: At first I wanted to code a simple filter that makes output of the `smallthinker:3b` nicer but now I'm thinking of turning it into more general reasoning model formatter.
-   `thinking_gemini.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/thinking_gemini) | An outdated script, initially developed for `gemini-2.0-flash-thinking-exp` model interactions. Superseded by `gemini_manifold.py`.
-   `test_and_example_functions/`: This directory contains different test and example plugins I usually code up when exploring what is possible.

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