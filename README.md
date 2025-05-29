# Open WebUI Functions

This repository contains a collection of Open WebUI plugins I have personally coded for my own use and find useful.

## Plugins

The plugins are organized by type within the `plugins/` directory:

-   `plugins/pipes/`: Contains pipe plugins that integrate custom models and behaviors.
    -   `gemini_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold_google_genai) | This plugin provides support for Google's Gemini Studio API and Vertex AI. See the [Detailed Documentation](docs/plugins/pipes/gemini_manifold.md) for more information.
    -   `venice_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/venice_image_generation) | Enables image creation by using any diffusion model offered by Venice.ai API.
-   `plugins/filters/`: Contains filter plugins that modify request and response data.
    -   `gemini_manifold_companion.py`: A companion filter for "Gemini Manifold google_genai" pipe providing enhanced functionality, such as Google Search grounding. See the [Detailed Documentation](docs/plugins/filters/gemini_manifold_companion.md) for more information.
    -   `system_prompt_injector.py`: Allows changing chat options like system prompt and temperature from the chatbox. It would pair nicely with Prompts feature Open WebUI offers.

## Utilities

The `utils/` directory contains shared code and utility modules used by the plugins:

-   `manifold_types.py`: Defines shared data types and structures related to manifold plugins.

## Examples

The `examples/` directory contains example plugins and test scripts demonstrating various functionalities and plugin capabilities.

## Development Environment

The `dev/` directory contains tools and configuration files used for developing and testing the Open WebUI plugins:

*   `function_updater.py`: Monitors plugin files and automatically updates the corresponding Open WebUI functions via the REST API.
*   `dev.sh`: A shell script that sets up the development environment.
*   `.env.example`: An example environment file for configuring development settings. Copy this to `.env` and modify as needed.

## Archived Functions

The `plugins/archived/` directory contains old functions that are no longer actively developed.

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the Open WebUI team and all contributors. Special thanks to Matthew for the groundwork on `thinking_gemini.py`.