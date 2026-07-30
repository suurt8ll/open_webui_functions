# `dynamic_plugin_log_manager.py` - Detailed Documentation

A centralised Loguru handler injector for Open WebUI plugins. It eliminates the need to copy-paste custom logging logic into every plugin you write.

## How it works

1. **Triggered on every model list refresh** — OWUI calls `pipes()` on every enabled `Pipe` when the front-end saves or refreshes; the log handler sync runs inside `pipes()`.

2. **Reads a JSON config from Valves** — each key is a *function ID* (as set in the target plugin's `Valves` → `id`), each value is a `loguru` level (`TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

3. **Injects / updates Loguru handlers** — for each configured function ID it creates a dedicated handler that:
   - only captures logs from `function_<id>` (via filter on `record["name"]`)
   - uses the compact, truncating format shown below
   - respects `auditable=False` (the standard OWUI trick to avoid double-printing, see OWUI `start_logger`)

4. **Removes stale handlers** — if you remove a function from the config or change its level, the old handler is cleaned up on the next refresh.

## Configuration

Open the Pipe's Valves and set `FUNCTIONS_LOG_LEVELS` to a JSON object like:

```json
{
  "gemini_manifold_companion": "DEBUG",
  "my_custom_filter": "INFO"
}
```

The ID is the string you gave in the target plugin's docstring header (`id: gemini_manifold_companion`). The plugin internally prepends `function_` if missing.

## Using the format in your own plugin

> If you **don't** want the custom format, you don't need to do anything — the handler still works and your logs appear with normal OWUI formatting.  
> The extras below are **optional** and only needed if you want structured payload serialisation.

In the target plugin (e.g. `gemini_manifold_companion`), get a bound logger:

```python
from loguru import logger

log = logger.bind(auditable=False)
```

Now use `log.info`, `log.debug`, etc. as usual.

### Structured payload logging (optional)

To attach serialisable data to a log line:

```python
log.info("Request finished", payload={
    "status": 200, "model": "gemini-2.0-flash", "tokens": 145
})
```

The plugin's format function will:
- serialise the payload via `pydantic_core.to_jsonable_python`
- flatten compact JSON for flat dicts (single line), pretty‑print nested ones
- truncate long strings to 256 chars by default (can be overridden via extras – see below)

### Controlling truncation per call

Add these to `extra` on the log call (or bind them once on the logger):

| Extra key | Type | Default | Behaviour |
|-----------|------|---------|-----------|
| `_log_truncation_enabled` | `bool` | `True` | Set `False` to disable truncation |
| `_log_max_length` | `int` | `256` | Truncation max length |
| `_log_truncation_marker` | `str` | `[...]` | Suffix appended when truncated |
| `_log_exclude_none` | `bool` | `True` | Determines if keys with `None` values will be shown in the dump. |

Example:

```python
log.warning("Large response", payload=data,
            _log_truncation_enabled=True, _log_max_length=512)
```

## Caveats / prerequisites

1. **OWUI must call `pipes()` on save.** This is true for current OWUI versions – every front-end model list refresh triggers `pipes()` on all enabled Pipes. If this behaviour changes upstream the plugin will stop working silently (log handlers won't be injected).

2. **The `auditable=False` bind trick** assumes OWUI's built-in `stdout_format` / `_json_sink` filters out entries whose `extra` contains `'auditable'` (by default it does). If a future OWUI version changes this filter, logs may appear duplicated.

3. **The plugin itself is a Pipe** — it must be **enabled** in the admin panel for the handlers to be active. Disabling it removes all managed handlers on the next refresh cycle.

4. **Manual refresh** – if you change the JSON config while OWUI is running and want immediate effect without waiting for a save event, open the Valves panel and click **Save** — this triggers the refresh.