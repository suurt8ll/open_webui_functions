#!/bin/bash
#
# dev.sh — headless dev-environment manager for Open WebUI plugin development.
#
# Sets up the environment (venvs, deps, optionally the frontend build) and runs
# the OWUI backend + the function updater as detached background processes,
# logging to files. Built for both humans and agents: everything is scriptable,
# nothing requires a TTY or a terminal multiplexer.
#
# Usage:
#   dev.sh [env-file] start      Set up (if needed) + launch backend & updater
#   dev.sh [env-file] restart    stop, then start
#   dev.sh stop                  Stop backend & updater
#   dev.sh status                Show process/port state + updater sync state
#   dev.sh logs [name] [-f]      Show logs (backend|updater|all), -f to follow
#   dev.sh [env-file] setup      Environment setup only (--recreate to rebuild
#                                venvs, --frontend to force an npm build)
#   dev.sh help
#
# If the first argument is not a known command but an existing file, it is
# treated as the env file for `start` (backwards compatible with the old
# `dev.sh dev/.env.dev` invocation).
#
# Env file (defaults to dev/.env.dev when present) may set:
#   PYTHON_VERSION, PROTOCOL, HOST, PORT, API_KEY, POLLING_INTERVAL,
#   ONE_TIME_RUN, FILEPATHS, WEBUI_SECRET_KEY, ...
# See .env.example for documentation. The explicit env file always wins over
# ambient environment variables (it is sourced with allexport after parsing).

set -euo pipefail

# --- Paths ---

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
FRONTEND_DIR="$PROJECT_ROOT/submodules/open-webui"
BACKEND_DIR="$FRONTEND_DIR/backend"
BACKEND_VENV="$BACKEND_DIR/.venv"
UPDATER_VENV="$PROJECT_ROOT/.venv"
STATE_DIR="$SCRIPT_DIR/.dev_state"
LOG_DIR="$SCRIPT_DIR/logs"
BACKEND_LOG="$LOG_DIR/backend.log"
UPDATER_LOG="$LOG_DIR/updater.log"
BACKEND_PID_FILE="$STATE_DIR/backend.pid"
UPDATER_PID_FILE="$STATE_DIR/updater.pid"
RUNTIME_STATE_FILE="$STATE_DIR/runtime.env"

# Pin uv's managed Python install + cache inside the repo so venvs created
# here also work inside the pi-fluid dev container (and vice versa).
# shellcheck source=uv-shared-env.sh
source "$SCRIPT_DIR/uv-shared-env.sh"

# --- Pretty output ---

log()  { printf '\033[1;34m[dev]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dev]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[dev]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# --- Config loading ---

ENV_FILE=""
SETUP_RECREATE=false
SETUP_FRONTEND=false

# Extracts setup-related flags so they don't get mistaken for an env file
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --recreate) SETUP_RECREATE=true ;;
        --frontend) SETUP_FRONTEND=true ;;
        *) ARGS+=("$arg") ;;
    esac
done

COMMAND="${ARGS[0]:-help}"
if ! [[ "$COMMAND" =~ ^(start|stop|restart|status|logs|setup|help)$ ]]; then
    if [ -f "$COMMAND" ]; then
        # Backwards compatibility: first arg was the env file
        ENV_FILE="$COMMAND"
        COMMAND="${ARGS[1]:-start}"
        [ "${#ARGS[@]}" -le 2 ] || die "Unexpected extra arguments: ${ARGS[*]:2}"
    else
        die "Unknown command '$COMMAND'. Run 'dev.sh help'."
    fi
elif [ "${#ARGS[@]}" -gt 1 ]; then
    if [ "$COMMAND" = "logs" ]; then
        # logs takes its own args, never an env file
        LOGS_ARGS=("${ARGS[@]:1}")
    elif [ -f "${ARGS[1]}" ]; then
        ENV_FILE="${ARGS[1]}"
        [ "${#ARGS[@]}" -le 2 ] || die "Unexpected extra arguments: ${ARGS[*]:2}"
    else
        die "'$ARGS[1]' is not a file."
    fi
fi

load_env() {
    if [ -z "$ENV_FILE" ]; then
        if [ -f "$SCRIPT_DIR/.env.dev" ]; then
            ENV_FILE="$SCRIPT_DIR/.env.dev"
        else
            log "No env file given and no .env.dev found; using built-in defaults."
            return
        fi
    fi
    [ -f "$ENV_FILE" ] || die "Env file '$ENV_FILE' not found."
    ENV_FILE=$(realpath "$ENV_FILE")
    # 'set -a' exports everything the file defines; the file is the source of
    # truth and intentionally wins over any pre-exported variables.
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
    log "Config: $ENV_FILE"
}

load_env

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROTOCOL="${PROTOCOL:-http}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
API_URL="$PROTOCOL://$HOST:$PORT"

# --- Small helpers ---

ensure_dir() {
    mkdir -p "$1"
    printf '*\n' > "$1/.gitignore"
}

pid_alive() {
    local pid="$1" state
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
    # kill -0 also succeeds for zombies; in containers without a reaping init,
    # an orphaned process that exited can stay defunct indefinitely. Treat
    # zombie state as dead so stop() doesn't needlessly escalate to SIGKILL.
    state=$(awk '/^State:/{print $2}' "/proc/$pid/status" 2>/dev/null || true)
    [ -z "$state" ] || [ "$state" != "Z" ]
}

# A PID file may be stale and the PID recycled by an unrelated process; verify
# the command line matches what we expect before trusting it.
pid_matches() {
    local pid="$1" needle="$2"
    pid_alive "$pid" || return 1
    if [ -r "/proc/$pid/cmdline" ]; then
        tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "$needle"
    else
        return 1
    fi
}

read_pid() {
    local file="$1" needle="$2"
    [ -f "$file" ] || return 1
    local pid
    pid=$(cat "$file" 2>/dev/null) || return 1
    if pid_matches "$pid" "$needle"; then
        printf '%s' "$pid"
        return 0
    fi
    rm -f "$file"   # stale or recycled PID
    return 1
}

health_ok() {
    curl -sf -o /dev/null --max-time 3 "$API_URL/health"
}

write_runtime_state() {
    cat > "$RUNTIME_STATE_FILE" <<EOF
ENV_FILE=$ENV_FILE
PROTOCOL=$PROTOCOL
HOST=$HOST
PORT=$PORT
EOF
}

# --- Process management ---

stop_pid_file() {
    local file="$1" needle="$2" name="$3"
    local pid
    if ! pid=$(read_pid "$file" "$needle"); then
        log "$name: not running"
        return 0
    fi
    log "$name: stopping (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 100); do
        pid_alive "$pid" || break
        sleep 0.1
    done
    if pid_alive "$pid"; then
        warn "$name: did not exit after SIGTERM, sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
    log "$name: stopped"
}

do_stop() {
    stop_pid_file "$UPDATER_PID_FILE" "function_updater" "updater"
    stop_pid_file "$BACKEND_PID_FILE" "uvicorn" "backend"
    rm -f "$RUNTIME_STATE_FILE"
}

wait_for_health() {
    local timeout="${1:-90}"
    log "Waiting up to ${timeout}s for backend at $API_URL ..."
    local waited=0 pid
    while ! health_ok; do
        # read_pid only succeeds while the recorded PID is alive and still is
        # the uvicorn process (avoids the kill -0 0 process-group pitfall)
        if ! pid=$(read_pid "$BACKEND_PID_FILE" "uvicorn"); then
            warn "Backend process died during startup. Last log lines:"
            tail -n 30 "$BACKEND_LOG" >&2 || true
            die "Backend failed to start. Full log: $BACKEND_LOG"
        fi
        if [ "$waited" -ge "$timeout" ]; then
            warn "Backend not healthy after ${timeout}s. Last log lines:"
            tail -n 30 "$BACKEND_LOG" >&2 || true
            die "Backend health check timed out. Full log: $BACKEND_LOG"
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

rotate_log() {
    local file="$1"
    [ -f "$file" ] && mv "$file" "$file.1"
    : > "$file"
}

# --- Environment setup ---

check_uv() {
    command -v uv &> /dev/null || die "'uv' is required but not installed (https://docs.astral.sh/uv/)."
}

# Hash of the requirements files that drive a dependency install
deps_hash() {
    local files=()
    local f
    for f in "$@"; do
        [ -f "$f" ] && files+=("$f")
    done
    [ ${#files[@]} -eq 0 ] && { echo none; return; }
    cat "${files[@]}" | md5sum | awk '{print $1}'
}

# A hash alone can't tell a healthy venv from one whose interpreter vanished
# (e.g. after a container/base-image change). Actually run the interpreter and
# import a sentinel package before trusting the venv.
venv_healthy() {
    local venv="$1" sentinel="$2"
    [ -x "$venv/bin/python" ] || return 1
    "$venv/bin/python" -c "import $sentinel" &> /dev/null
}

setup_venv() {
    local name="$1" venv="$2" requirements="$3" sentinel="$4"
    local hash_file="$STATE_DIR/${name}_pip_deps.hash"

    if $SETUP_RECREATE; then
        log "$name: --recreate given, wiping venv..."
        rm -rf "$venv"
        rm -f "$hash_file"
    fi

    if ! venv_healthy "$venv" "$sentinel"; then
        if [ -d "$venv" ]; then
            warn "$name: venv exists but is unhealthy (broken interpreter or missing core deps). Recreating..."
            rm -rf "$venv"
            rm -f "$hash_file"
        fi
        log "$name: creating venv (Python $PYTHON_VERSION)..."
        uv venv --python "$PYTHON_VERSION" --seed "$venv"
        rm -f "$hash_file"   # fresh venv, force install below
    fi

    if [ ! -f "$requirements" ]; then
        warn "$name: requirements file '$requirements' not found, skipping deps."
        return 0
    fi

    local current stored
    current=$(deps_hash "$requirements")
    stored=$(cat "$hash_file" 2>/dev/null || echo nostate)

    if [ "$current" != "$stored" ] || ! venv_healthy "$venv" "$sentinel"; then
        log "$name: installing dependencies..."
        uv pip install --python "$venv/bin/python" -r "$requirements" \
            || die "$name: dependency install failed."
    else
        log "$name: dependencies up to date."
    fi

    venv_healthy "$venv" "$sentinel" || die "$name: venv still unhealthy after install. $venv"
    echo "$current" > "$hash_file"
}

setup_frontend() {
    local build_dir="$FRONTEND_DIR/build"
    local npm_hash_file="$STATE_DIR/open-webui_npm_deps.hash"
    local build_hash_file="$STATE_DIR/frontend_build.hash"

    if ! $SETUP_FRONTEND && [ -d "$build_dir" ] && [ -n "$(ls -A "$build_dir" 2>/dev/null)" ]; then
        log "Frontend: prebuilt assets found (pass --frontend to force a rebuild)."
        return 0
    fi

    command -v npm &> /dev/null || die "npm is required for the frontend build but not installed."
    log "Frontend: installing npm dependencies..."
    ( cd "$FRONTEND_DIR" && npm install ) || die "npm install failed."
    echo "$(deps_hash "$FRONTEND_DIR/package-lock.json")" > "$npm_hash_file"

    log "Frontend: building static assets (this can take several minutes)..."
    ( cd "$FRONTEND_DIR" && npm run build ) || die "frontend build failed."
    echo "$(deps_hash "$FRONTEND_DIR/package.json")" > "$build_hash_file"
}

do_setup() {
    check_uv
    ensure_dir "$STATE_DIR"
    ensure_dir "$LOG_DIR"

    setup_frontend
    setup_venv "backend" "$BACKEND_VENV" "$BACKEND_DIR/requirements.txt" "uvicorn, fastapi"
    setup_venv "updater" "$UPDATER_VENV" "$PROJECT_ROOT/requirements.txt" "aiohttp, watchfiles, dotenv, loguru"
    log "Setup complete."
}

# --- Commands ---

do_start() {
    do_stop   # clean slate: also clears stale pid files
    do_setup

    log "Launching backend ($API_URL)..."
    rotate_log "$BACKEND_LOG"
    (
        set -a
        # shellcheck disable=SC1090
        [ -n "$ENV_FILE" ] && . "$ENV_FILE"
        set +a
        cd "$BACKEND_DIR"
        exec "$BACKEND_VENV/bin/uvicorn" open_webui.main:app \
            --host "$HOST" --port "$PORT"
    ) < /dev/null >> "$BACKEND_LOG" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"

    log "Launching function updater..."
    rotate_log "$UPDATER_LOG"
    (
        cd "$SCRIPT_DIR"
        env_arg=""
        [ -n "$ENV_FILE" ] && env_arg="--env $ENV_FILE"
        exec "$UPDATER_VENV/bin/python" function_updater.py $env_arg
    ) < /dev/null >> "$UPDATER_LOG" 2>&1 &
    echo $! > "$UPDATER_PID_FILE"

    write_runtime_state

    # catch instant-death failures (bad args, missing interpreter, port taken)
    sleep 2
    pid_matches "$(cat "$BACKEND_PID_FILE")" "uvicorn" || { tail -n 20 "$BACKEND_LOG" >&2; die "Backend exited immediately. See $BACKEND_LOG"; }
    pid_matches "$(cat "$UPDATER_PID_FILE")" "function_updater" || { tail -n 20 "$UPDATER_LOG" >&2; die "Updater exited immediately. See $UPDATER_LOG"; }

    wait_for_health "${BACKEND_TIMEOUT:-120}"

    log "Running:"
    log "  backend : pid $(cat "$BACKEND_PID_FILE")  $API_URL  (log: $BACKEND_LOG)"
    log "  updater : pid $(cat "$UPDATER_PID_FILE")  (log: $UPDATER_LOG)"
    log "Stop with 'dev.sh stop'. Watch with 'dev.sh logs -f'. Status with 'dev.sh status'."
}

do_status() {
    local backend_pid updater_pid
    if backend_pid=$(read_pid "$BACKEND_PID_FILE" "uvicorn" 2>/dev/null); then
        if health_ok; then
            log "backend : RUNNING (pid $backend_pid), healthy at $API_URL"
        else
            warn "backend : RUNNING (pid $backend_pid) but NOT healthy at $API_URL/health"
        fi
    else
        log "backend : stopped"
    fi

    if updater_pid=$(read_pid "$UPDATER_PID_FILE" "function_updater" 2>/dev/null); then
        log "updater : RUNNING (pid $updater_pid)"
    else
        log "updater : stopped"
    fi

    # Updater sync state: delegate to the updater's --status mode when possible
    if [ -x "$UPDATER_VENV/bin/python" ]; then
        local env_arg=""
        [ -n "$ENV_FILE" ] && env_arg="--env $ENV_FILE"
        # shellcheck disable=SC2086
        "$UPDATER_VENV/bin/python" "$SCRIPT_DIR/function_updater.py" --status $env_arg \
            2>/dev/null || warn "updater: could not determine sync state (see warning above)"
    fi
}

do_logs() {
    local name="all" follow=false
    for arg in ${LOGS_ARGS[@]+"${LOGS_ARGS[@]}"}; do
        case "$arg" in
            -f|--follow) follow=true ;;
            backend|updater|all) name="$arg" ;;
            *) die "logs: unknown argument '$arg' (use backend|updater|all, -f)" ;;
        esac
    done

    show_one() {
        local file="$1" label="$2"
        [ -f "$file" ] || { warn "$label: no log file yet ($file)"; return 0; }
        echo "==> $label ($file) <=="
        if $follow; then
            tail -F "$file"
        else
            tail -n 50 "$file"
        fi
        echo
    }

    case "$name" in
        backend) show_one "$BACKEND_LOG" "backend" ;;
        updater) show_one "$UPDATER_LOG" "updater" ;;
        all)
            show_one "$BACKEND_LOG" "backend"
            show_one "$UPDATER_LOG" "updater"
            ;;
    esac
}

case "$COMMAND" in
    start)   do_start ;;
    restart) do_stop; do_start ;;
    stop)    do_stop ;;
    status)  do_status ;;
    logs)    do_logs ;;
    setup)   do_setup ;;
    help|*)  usage ;;
esac
