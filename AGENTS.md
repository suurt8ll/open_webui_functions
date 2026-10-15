# AGENTS.md

Guidance for coding agents working in this repository.

## What this repo is

A personal collection of [Open WebUI](https://github.com/open-webui/open-webui) plugins (Functions), written in Python. Plugins are developed primarily for the author's own use; contributions and refactors are still welcome.

## Repository layout

- `plugins/pipes/` — Pipe functions (custom "models" exposed in the OWUI chat UI). e.g. `gemini_manifold.py`, `venice_manifold.py`, `dynamic_plugin_log_manager.py`.
- `plugins/filters/` — Filter functions that modify request/response data. Many are companions/toggles for the manifold pipes.
- `plugins/archived/` — Dead code, kept for reference only. Don't develop here.
- `utils/` — Shared code (e.g. `manifold_types.py` for typing only, `gemini_yaml_wizard.py`).
- `examples/` — Example plugins and small scripts demonstrating OWUI plugin APIs. Good starting points for learning how a feature works.
- `tests/` — pytest tests. Currently sparse and unmaintained; see Testing below.
- `docs/plugins/` — Per-plugin documentation, linked from the README.
- `dev/` — Development tooling: `dev.sh` (headless env setup + backend/updater process manager: start/stop/status/logs), `function_updater.py` (live-sync of plugin files into a running OWUI instance via its API), `README.md` (quickstart + headless/agent notes), `release.sh`.
- `submodules/open-webui` (symlinked as `open_webui/`) — The actual Open WebUI source as a git submodule. **The authoritative reference for how the plugin system works.** When in doubt about plugin APIs (`Pipe`, `Filter`, `Action`, event payloads, request/response formats), read the OWUI source here instead of guessing. Don't modify the submodule.
- `.venv/` — Python 3.11 virtualenv for this repo's tooling.

## Plugin conventions

### Single-file constraint

Each plugin must be **one self-contained `.py` file**. Open WebUI Functions are installed by copy-pasting the file into the OWUI UI (or syncing it via API), so a plugin cannot import from sibling files in this repo at runtime. Everything the plugin needs must live in that one file (or come from pip dependencies / the network).

- `utils/manifold_types.py` is an exception used **only for linting/type checking**; plugins are written so they don't require it at runtime.
- `plugins/pipes/gemini_models.yaml` is fetched over the network by the plugin at runtime and cached in memory — it is not bundled.

### Frontmatter

Every plugin starts with a docstring frontmatter. OWUI-specific fields (`title`, `description`, `requirements`, ...) and custom fields coexist; the important ones:

- `id` — must exactly match the function's `id` in the OWUI instance. Some plugins depend on this match; never change it casually.
- `version` — bump when making a release. Some pipes also define a module-level `VERSION` and `RECOMMENDED_COMPANION_VERSION` constant that must be kept in sync with the frontmatter (these exist to avoid merge conflicts during PRs; don't bump them in a PR unless asked).

### OWUI plugin system

- Classes must be named `Pipe`, `Filter`, or `Action` per the plugin type.
- Prefer `loguru`'s `logger` for logging (see `dynamic_plugin_log_manager.py` for the author's logging setup).
- `from __future__ import annotations` at the top of plugin files.
- Type hints are appreciated but not forced everywhere; use judgment.
- Pydantic models are the norm for structured data.

## Development workflow

The repo ships tooling for live development against a running OWUI instance. It is fully headless — see `dev/README.md` for details:

1. `bash dev/dev.sh start` — sets up venvs/deps (only when stale or unhealthy), then launches the OWUI backend and the function updater as detached background processes with file logs, and waits for `/health` to pass.
2. Configure `dev/.env.dev` (copy from `dev/.env.example`) with `HOST`, `PORT`, `API_KEY`, and the `FILEPATHS` to sync. An explicitly passed env file always wins over pre-exported environment variables.
3. Edit tracked plugin files — they are pushed into the OWUI instance in real time via its API. `ONE_TIME_RUN=true` does a single sync and exits (useful for installs).
4. Verify from a script: `bash dev/dev.sh status` (processes, port health, per-file sync state) and `bash dev/dev.sh logs backend|updater [-f]`; stop with `bash dev/dev.sh stop`. Do not attach to any terminal — there isn't one.

Agents **may run a local OWUI instance** and interact with it through its API to test changes — nothing in this repo prevents it. Keep credentials in `dev/.env*` files (gitignored); never commit real API keys.

If you find a better agent-friendly development loop (e.g. more automated verification), you are encouraged to propose or build it — don't force-fit the existing manual-ish pipeline.

## Testing

- Run `pytest` (config in `pytest.ini`) before pushing to remote. CI (`.github/workflows/python-ci.yml`) runs on push/PR to `master` and a red X pings the author — don't let it break.
- The test suite is sparse and not rigorously maintained. You may add tests for code you work on, but it is not mandated.
- Deep verification of plugin behavior against a live instance stays human-in-the-loop for now.

## Style & formatting

- Use **ruff** for linting and formatting so that hand edits and agent edits stay consistent. (`pip install ruff` into `.venv`; there is no committed config yet — default ruleset is fine, or propose one.)
- PEP 8, clear names, docstrings and comments where they add value.

## Autonomy expectations

- You are **not** boxed into existing patterns. If you spot a bad pattern, drastic refactors and reworks are allowed — say what you're doing and why.
- Keep the human in the loop for anything destructive or hard to reverse (deleting large amounts of code, changing `id` fields, touching the submodule).

## Documentation upkeep

When a change affects user-facing behavior, update the relevant docs in the same change:

- `README.md` (plugin list, install instructions if affected)
- `docs/plugins/**` (the per-plugin page for whatever you touched)
