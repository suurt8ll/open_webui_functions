# Open WebUI Functions

This repository contains a collection of Open WebUI plugins I have personally coded for my own use. I’m sharing them here in case they are useful to others, but please keep in mind that these are developed primarily to satisfy my own needs and workflows.

> [!CAUTION]
> **Use at your own risk.** Running third-party plugins in your Open WebUI instance involves executing code that can access your environment and APIs. I am not a security expert, and I cannot guarantee the absolute safety or stability of these scripts. Always review the code before installing.

## Project Philosophy & Stability

- **Personal Use First:** I develop these plugins as my own motivation and needs arise. As such, I don't always prioritize making them perfectly intuitive for third parties. You may find some "quirks" that make sense for my setup but require adjustment for yours.
- **Master vs. Tags:** The `master` branch is my active development area and should be considered "Canary" or "Dev" software—it is **not guaranteed to work** at any given moment. 
- **Stable Versions:** For a more reliable experience, please use the [latest Version Tags](https://github.com/suurt8ll/open_webui_functions/tags) to find checkpoints I consider stable.
- **Feedback:** Questions, Issues, and Pull Requests are very welcome! I am happy to help or collaborate, though I will respond to them only as my personal time allows.

## Plugins

The plugins are organized by type within the `plugins/` directory:

-   `plugins/pipes/`: Pipe plugins that integrate custom models and behaviors.
    -   `gemini_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/gemini_manifold_google_genai) | Provides support for Google's Gemini Studio API and Vertex AI. See the [Detailed Documentation](docs/plugins/pipes/gemini_manifold.md).
    -   `venice_manifold.py` | [Open WebUI Community](https://openwebui.com/f/suurt8ll/venice_image_generation) | Enables image creation using any diffusion model offered by Venice.ai API.
-   `plugins/filters/`: Filter plugins that modify request and response data.
    -   `gemini_manifold_companion.py`: A companion filter for the Gemini Manifold pipe providing enhanced functionality like Google Search grounding. See the [Detailed Documentation](docs/plugins/filters/gemini_manifold_companion.md).
    -   `system_prompt_injector.py`: Allows changing chat options like system prompt and temperature directly from the chatbox. Pairs well with the Open WebUI "Prompts" feature.

## Installation/Updating

### Option 1: Manual (Copy-Paste)
1. Open your Open WebUI instance and navigate to **Admin Panel** -> **Functions** -> **New Function** (or click an existing one to update).
2. Copy the entire content of the desired `.py` file (ideally from a **Tag**, not Master) and paste it into the editor.
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
2. **Configure:** Copy `dev/.env.install.example` to `dev/.env.install`.
   - Set `ONE_TIME_RUN=true`.
   - Set `API_KEY` (see [this doc page](https://docs.openwebui.com/reference/monitoring/#authentication-setup-for-api-key-) for help).
   - List the files you want to install in `FILEPATHS`.
   - Set `HOST`, `PORT`, etc.
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

## Utilities & Development

- **`utils/`**: Contains shared code like `manifold_types.py` used by multiple plugins.
- **`examples/`**: Test scripts and example plugins demonstrating specific capabilities.
- **`dev/`**: Tools for active development, including `function_updater.py` for monitoring file changes and `dev.sh` for environment setup.

## Archived Functions

The `plugins/archived/` directory contains old functions that are no longer actively developed or maintained.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. As this is a personal project, please be patient with response times.

## License

MIT License. See the `LICENSE` file for details.

## Acknowledgements

Thanks to the Open WebUI team and all contributors. Special thanks to Matthew for the groundwork on `thinking_gemini.py`.