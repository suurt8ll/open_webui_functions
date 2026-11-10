# `dev/` — development tooling

Headless development environment for Open WebUI plugin work: it sets up the
Python environments, runs the OWUI backend and the function-updater as
**detached background processes with file logs**, and gives you simple commands
to inspect and stop them. No TTY or terminal multiplexer required — everything
is scriptable, which makes it equally usable by humans and coding agents.

## Files

| File | Purpose |
| --- | --- |
| `dev.sh` | Environment setup + process manager (start/stop/status/logs) |
| `function_updater.py` | Live-syncs plugin files into a running OWUI instance via its API |
| `release.sh` | Interactive helper for tagging plugin releases (unchanged) |
| `.env.dev` | Your local env (gitignored): target host/port, API key, tracked files |
| `.env.example` / `.env.install.example` | Templates: dev loop / one-time install |
| `logs/` | Backend & updater logs (gitignored), `.log` current, `.log.1` previous run |
| `.dev_state/` | Venv/build hashes, sync-state hashes, pid files (gitignored) |

## Quickstart

```bash
bash dev/dev.sh start                 # uses dev/.env.dev by default
bash dev/dev.sh status                # is everything up? is everything in sync?
bash dev/dev.sh logs -f               # follow both logs
bash dev/dev.sh stop
```

`start` is idempotent and cheap when warm: venvs are reused unless unhealthy,
dependencies are installed only when `requirements.txt` changes, and the
frontend build is skipped when prebuilt assets exist. Explicitly provided env
files always win over pre-exported environment variables.

### Other commands

```bash
bash dev/dev.sh [env-file] restart        # stop, then start
bash dev/dev.sh [env-file] setup          # setup only
bash dev/dev.sh setup --recreate          # wipe & rebuild both venvs
bash dev/dev.sh setup --frontend          # force npm install + frontend build
bash dev/dev.sh logs backend|updater      # one log instead of both
```

## The dev loop

1. `dev.sh start` — backend on `$HOST:$PORT`, updater watching your tracked
   plugin files.
2. Edit any tracked file — the updater pushes it into OWUI within ~1s
   (inotify) and keeps a hash state of what was last synced successfully.
3. Verify with `dev.sh status` (per-file `IN SYNC` / `MODIFIED (sync pending)`)
   or query the API directly:

   ```bash
   curl -s -H "Authorization: Bearer $API_KEY" \
        "$PROTOCOL://$HOST:$PORT/api/v1/functions/id/<function_id>" | jq -r .content
   ```

4. The updater also exposes this state standalone (no OWUI needed):

   ```bash
   python dev/function_updater.py --status --env dev/.env.dev
   ```

### Disposable / second instance

To run a throwaway instance (fresh database, different port) alongside or
instead of the main dev instance, copy `.env.example` somewhere (e.g.
`/tmp/.env.owui-test`), give it its own `PORT`, `API_KEY` and a fresh
`DATA_DIR`, then:

```bash
bash dev/dev.sh /tmp/.env.owui-test start
```

`DATA_DIR` is created if it does not exist, so the fresh boot "just works".
Keep in mind the pid files in `dev/.dev_state/` track only the most recent
`start`, so `dev.sh stop` stops the last instance you launched.

## Notes for agents (and headless humans)

- **Never attach to a terminal.** There is nothing to attach to. Read
  `dev/logs/*.log`, probe `GET /health`, and use `dev.sh status`.
- `dev.sh start` waits until the backend answers `/health` (default up to
  120s, override with `BACKEND_TIMEOUT=<seconds>`), then returns. A non-zero
  exit means something actually failed; the backend's last 30 log lines are
  printed to stderr.
- Process identity is verified via `/proc/<pid>/cmdline` before any signal is
  sent, so stale pid files can't make you kill an unrelated process.
- `stop` sends SIGTERM first and only escalates to SIGKILL after 10s.
- API credentials live in the env file (gitignored). The `API_KEY` in
  `.env.dev` is for the local instance only; never commit real keys.
- **Fresh-instance caveat:** `start` uses whatever database the backend is
  configured with (default `submodules/open-webui/backend/data/webui.db`).
  Booting a *second, empty* instance used to fail in alembic migrations
  (`sqlite3.OperationalError: unable to open database file`): when run from
  source, OWUI never creates a custom `DATA_DIR`, and SQLite cannot create the
  database file inside a missing directory. `dev.sh` now supports a `DATA_DIR`
  env-file variable (resolved and created automatically), so a disposable
  instance with a fresh database is just a second env file away — see
  [Disposable / second instance](#disposable--second-instance).

## Env file reference

See `.env.example`. Key variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PYTHON_VERSION` | `3.11` | Interpreter for both venvs (created via `uv`) |
| `PROTOCOL`/`HOST`/`PORT` | `http`/`127.0.0.1`/`8080` | Where the backend listens |
| `API_KEY` | — | OWUI API key the updater authenticates with |
| `POLLING_INTERVAL` | `60` | Fallback poll period (s); file-watcher events are instant |
| `ONE_TIME_RUN` | `false` | Sync all files once and exit (install mode) |
| `FILEPATHS` | — | Comma-separated plugin paths; relative ones resolve against the env file's directory |
| `DATA_DIR` | — | Override the backend's `DATA_DIR` (SQLite DB, uploads, vector store). Created automatically if missing; relative paths resolve against the env file's directory |
| `WEBUI_SECRET_KEY` | — | Passed through to the OWUI backend |
