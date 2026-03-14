#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# Prevent errors in a pipeline from being masked.
set -euo pipefail

# --- Configuration ---
PYTHON_VERSION="3.11"  # Set your desired Python version here
PORT="${PORT:-8080}"   # Default port, easily overridden via env var (e.g., PORT=8081 ./dev.sh)

# Project structure definition
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
FRONTEND_DIR="$PROJECT_ROOT/submodules/open-webui"
BACKEND_DIR="$FRONTEND_DIR/backend"

# Virtual environment paths
BACKEND_VENV_DIR="$BACKEND_DIR/.venv"
UPDATER_VENV_DIR="$PROJECT_ROOT/.venv"

# Single State directory
STATE_DIR="$SCRIPT_DIR/.dev_state"

# --- Helper Functions ---

ensure_state_dir() {
    local dir="$1"
    mkdir -p "$dir"
    echo "*" > "$dir/.gitignore"
}

check_openssl() {
    if ! command -v openssl &> /dev/null; then
        echo "ERROR: openssl is required for hashing but is not installed." >&2
        exit 1
    fi
}

check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "ERROR: 'uv' is required for Python package management but is not installed." >&2
        echo "Please install uv (e.g., curl -LsSf https://astral.sh/uv/install.sh | sh)." >&2
        exit 1
    fi
}

get_hash() {
    check_openssl
    local files_to_hash=()
    for f in "$@"; do
        if [[ -f "$f" ]]; then
            files_to_hash+=("$f")
        fi
    done
    if [[ ${#files_to_hash[@]} -eq 0 ]]; then
        echo "d41d8cd98f00b204e9800998ecf8427e" # MD5 of empty string
        return
    fi
    # shellcheck disable=SC2002
    cat "${files_to_hash[@]}" | openssl md5 | awk '{print $2}'
}

read_state() {
    local state_file="$1"
    if [ -f "$state_file" ]; then
        cat "$state_file"
    else
        echo "nostate"
    fi
}

write_state() {
    local state_file="$1"
    local hash="$2"
    echo "$hash" > "$state_file"
}

# --- Environment Setup Functions ---

check_and_create_frontend_env() {
    local frontend_dir="$1"
    local env_file="$frontend_dir/.env"
    local example_env_file="$frontend_dir/.env.example"

    echo "Checking for frontend .env file..."
    if [ ! -f "$env_file" ]; then
        if [ -f "$example_env_file" ]; then
            echo "Frontend .env file not found. Copying from .env.example."
            cp "$example_env_file" "$env_file"
        else
            echo "Warning: $env_file and $example_env_file not found."
        fi
    fi
}

check_and_install_npm() {
    local target_dir="$1"
    local component_name="$2"
    local state_file="$STATE_DIR/${component_name}_npm_deps.hash"
    local needs_install=false

    echo "Checking frontend dependencies in '$target_dir'..."
    ensure_state_dir "$STATE_DIR"

    (
        cd "$target_dir" || exit 1

        local lock_file="package-lock.json"
        local pkg_file="package.json"
        local dep_file_to_hash=""

        if [ -f "$lock_file" ]; then
            dep_file_to_hash="$lock_file"
        elif [ -f "$pkg_file" ]; then
            dep_file_to_hash="$pkg_file"
        else
            echo "Warning: Neither package.json nor package-lock.json found. Skipping npm install."
            exit 0
        fi

        local current_hash
        local stored_hash
        current_hash=$(get_hash "$dep_file_to_hash")
        stored_hash=$(read_state "$state_file")

        if [ ! -d "node_modules" ] || [ "$current_hash" != "$stored_hash" ]; then
            needs_install=true
        fi

        if $needs_install; then
            echo "Installing frontend dependencies..."
            if npm install; then
                write_state "$state_file" "$current_hash"
            else
                echo "Warning: 'npm install' failed. Retrying with --force..."
                if npm install --force; then
                    write_state "$state_file" "$current_hash"
                else
                    echo "ERROR: npm install failed." >&2
                    exit 1
                fi
            fi
        else
            echo "Frontend dependencies are up to date."
        fi
    )
}

build_frontend() {
    local target_dir="$1"
    local build_dir="$target_dir/build"
    local state_file="$STATE_DIR/frontend_build.hash"
    local needs_build=false

    echo "Checking if frontend requires building..."
    ensure_state_dir "$STATE_DIR"

    (
        cd "$target_dir" || exit 1
        
        # We hash package.json as a proxy to detect if the OWUI submodule was updated
        local current_hash
        local stored_hash
        current_hash=$(get_hash "package.json")
        stored_hash=$(read_state "$state_file")

        if [ ! -d "$build_dir" ] || [ "$current_hash" != "$stored_hash" ]; then
            needs_build=true
        fi

        if $needs_build; then
            echo "Building frontend static assets (npm run build)..."
            if npm run build; then
                echo "Frontend built successfully."
                write_state "$state_file" "$current_hash"
            else
                echo "ERROR: Frontend build failed." >&2
                exit 1
            fi
        else
            echo "Frontend build is up to date."
        fi
    )
}

setup_pip_venv() {
    local target_desc="$1"
    local component_name="$2"
    local target_dir="$3"
    local venv_dir="$4"
    local requirements_file="$5"
    local state_file="$STATE_DIR/${component_name}_pip_deps.hash"
    local needs_install=false

    echo "Checking $target_desc Python environment in '$target_dir'..."
    ensure_state_dir "$STATE_DIR"

    if [ ! -d "$venv_dir" ] || [ ! -f "$venv_dir/bin/activate" ]; then
        echo "$target_desc virtual environment not found. Creating it with 'uv' (Python $PYTHON_VERSION)..."
        # --seed ensures pip is available for OWUI's internal plugin installer
        uv venv --python "$PYTHON_VERSION" --seed "$venv_dir"
        needs_install=true
    fi

    if [ ! -f "$requirements_file" ]; then
        echo "Warning: $requirements_file not found. Skipping dependency check."
        rm -f "$state_file"
        return
    fi

    local current_hash
    local stored_hash
    current_hash=$(get_hash "$requirements_file")
    stored_hash=$(read_state "$state_file")

    if [ "$current_hash" != "$stored_hash" ]; then
        needs_install=true
    fi

    if $needs_install; then
        echo "Installing/updating $target_desc dependencies using uv..."
        if uv pip install --python "$venv_dir" -r "$requirements_file"; then
            echo "$target_desc dependencies installed/updated successfully."
            write_state "$state_file" "$current_hash"
        else
            echo "ERROR: uv pip install failed for $target_desc" >&2
            exit 1
        fi
    else
        echo "$target_desc dependencies are up to date."
    fi
}

# --- Main Setup ---

echo "Starting development environment setup..."

check_uv

# 1. Setup & Build Frontend
check_and_create_frontend_env "$FRONTEND_DIR"
check_and_install_npm "$FRONTEND_DIR" "open-webui"
build_frontend "$FRONTEND_DIR"

# 2. Setup Backend Environment
setup_pip_venv "backend" "backend" "$BACKEND_DIR" "$BACKEND_VENV_DIR" "$BACKEND_DIR/requirements.txt"

# 3. Setup Updater Environment
setup_pip_venv "updater" "updater" "$PROJECT_ROOT" "$UPDATER_VENV_DIR" "$PROJECT_ROOT/requirements.txt"


echo "Setup complete. Launching tmux session..."

# --- tmux Launch ---

if [ -n "${TMUX:-}" ]; then
    echo "Already inside a tmux session. Skipping new session creation."
    exit 0
fi

SESSION_NAME="openwebui_dev"

# --- Define Commands for Each tmux Window ---

# 1. Unified OWUI Server Command (Backend serving built Frontend)
# Note: OWUI automatically serves ../build relative to the backend directory
BACKEND_START_COMMAND="uvicorn open_webui.main:app --port $PORT --host 127.0.0.1"
BACKEND_COMMAND="cd '$BACKEND_DIR' && \
                   echo 'Activating backend venv...' && \
                   . '$BACKEND_VENV_DIR/bin/activate' && \
                   echo '--- Starting Open WebUI Server (Port: $PORT) ---' && \
                   exec $BACKEND_START_COMMAND"

# 2. Function Updater Command
UPDATER_COMMAND="cd '$SCRIPT_DIR' && \
                   echo 'Activating updater venv...' && \
                   . '$UPDATER_VENV_DIR/bin/activate' && \
                   echo '--- Starting Function Updater ---' && \
                   exec python function_updater.py"

# --- Create and Configure tmux Session ---

# Create a new detached tmux session with two windows
tmux new-session -d -s "$SESSION_NAME" -n "open-webui" "$BACKEND_COMMAND" \; \
     new-window -n "function-updater" "$UPDATER_COMMAND"

# Select the first window to be active when attaching
tmux select-window -t "$SESSION_NAME:open-webui"

# Enable mouse mode for convenience
tmux set-option -g mouse on

# Attach to the created tmux session
echo "Attaching to tmux session '$SESSION_NAME'..."
echo "Windows created: [open-webui], [function-updater]"
echo "Use 'Ctrl-b n' to cycle to the next window, 'Ctrl-b p' for the previous."
tmux attach-session -t "$SESSION_NAME"

echo "Tmux session '$SESSION_NAME' ended."
