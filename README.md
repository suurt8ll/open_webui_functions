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

## Installation/Updating

### Option 1: Manual (Copy-Paste)
1. Open your Open WebUI instance and navigate to **Admin Panel** -> **Functions** -> **New Function** or click on a single one if you want to update it.
2. Copy the entire content of the desired `.py` file from this repo and paste it into the editor.
3. **CRITICAL:** Ensure the `id` field in the Open WebUI interface matches the `id` defined in the file's frontmatter (docstring). Some logic within these plugins depends on these IDs being exact.
   
   *Example from `gemini_manifold.py`:*
   ```python
   """
   title: Gemini Manifold google_genai
   id: gemini_manifold_google_genai
   ...
   """
   ```

### Option 2: Automated (One-Time Sync)
Use the included utility script to automatically create/update multiple functions at once via the Open WebUI API.

1. **Clone the repo:** `git clone https://github.com/suurt8ll/open_webui_functions.git`
2. **Configure:** Copy `dev/.env.example` to `dev/.env.install`.
   - Set `ONE_TIME_RUN=true`.
   - Set `API_KEY`, [this doc page](https://docs.openwebui.com/reference/monitoring/#authentication-setup-for-api-key-) explains how to get it.
   - List the files you want to install in `FILEPATHS`.
   - Set `HOST`, `PORT` etc.
3. **Setup Environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Run Sync:**
   ```bash
   python dev/function_updater.py --env dev/.env.install
   ```
The script will wait for a connection, sync the functions, and exit.

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