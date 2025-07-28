#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# Prevent errors in a pipeline from being masked.
set -euo pipefail

# --- Configuration ---

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

# Ensure state directory exists
ensure_state_dir() {
    local dir="$1"
    mkdir -p "$dir"
    # Ignore this dir
    echo "*" > "$dir/.gitignore"
}

# Check if openssl is available
check_openssl() {
    if ! command -v openssl &> /dev/null; then
        echo "ERROR: openssl is required for hashing but is not installed." >&2
        echo "Please install openssl (e.g., apt install openssl or brew install openssl)." >&2
        exit 1
    fi
}

# Get hash of file(s) using openssl
get_hash() {
    check_openssl # Ensure openssl is available
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
    cat "${files_to_hash[@]}" | openssl md5 | awk '{print $2}'
}

# Read stored hash from state file
read_state() {
    local state_file="$1"
    if [ -f "$state_file" ]; then
        cat "$state_file"
    else
        echo "nostate" # Return a value indicating no previous state
    fi
}

# Write current hash to state file
write_state() {
    local state_file="$1"
    local hash="$2"
    echo "$hash" > "$state_file"
}

# --- Dependency & Environment Setup Functions ---

# Check and create frontend .env file if it doesn't exist
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
            echo "Warning: $env_file and $example_env_file not found. Frontend may not be configured correctly."
        fi
    else
        echo "Frontend .env file already exists."
    fi
}


check_and_install_npm() {
    local target_dir="$1"
    local component_name="$2" # e.g., "open-webui"
    local state_file="$STATE_DIR/${component_name}_npm_deps.hash"
    local needs_install=false

    echo "Checking frontend dependencies in '$target_dir'..."
    ensure_state_dir "$STATE_DIR"
    cd "$target_dir"

    local lock_file="package-lock.json"
    local pkg_file="package.json"
    local current_hash=""
    local dep_file_to_hash=""

    # Prefer package-lock.json if it exists
    if [ -f "$lock_file" ]; then
        dep_file_to_hash="$lock_file"
    elif [ -f "$pkg_file" ]; then
        dep_file_to_hash="$pkg_file"
    else
        echo "Warning: Neither package.json nor package-lock.json found in $target_dir. Skipping npm install check."
        cd "$PROJECT_ROOT"
        return
    fi

    current_hash=$(get_hash "$dep_file_to_hash")
    stored_hash=$(read_state "$state_file")

    if [ ! -d "node_modules" ]; then
        echo "Node modules directory not found."
        needs_install=true
    elif [ "$current_hash" != "$stored_hash" ]; then
        echo "Dependency file ($dep_file_to_hash) hash changed ('$current_hash' vs '$stored_hash')."
        needs_install=true
    fi

    if $needs_install; then
        echo "Installing frontend dependencies (based on $dep_file_to_hash)..."
        
        # First, try a standard 'npm install'
        if npm install; then
            echo "Frontend dependencies installed successfully."
            write_state "$state_file" "$current_hash"
        else
            # If the first attempt fails, try again with --force
            echo "Warning: 'npm install' failed. Retrying with --force..."
            echo "This can resolve peer dependency conflicts and is often safe for development."
            
            if npm install --force; then
                echo "Frontend dependencies installed successfully using --force."
                write_state "$state_file" "$current_hash"
            else
                # If the --force attempt also fails, give up.
                echo "ERROR: Both 'npm install' and 'npm install --force' failed in $target_dir." >&2
                echo "Please review the npm error logs above to diagnose the issue." >&2
                cd "$PROJECT_ROOT"
                exit 1
            fi
        fi
        # --- End of new logic ---
    else
        echo "Frontend dependencies are up to date (hash: $current_hash)."
    fi

    cd "$PROJECT_ROOT" # Return to original root dir
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

    if [ ! -f "$venv_dir/bin/activate" ]; then
        echo "ERROR: $target_desc virtual environment not found at '$venv_dir'." >&2
        echo "Please create it first (e.g., python -m venv $venv_dir)" >&2
        exit 1
    fi

    if [ ! -f "$requirements_file" ]; then
        echo "Warning: $requirements_file not found. Skipping dependency check for $target_desc."
        rm -f "$state_file"
        return
    fi

    local current_hash=$(get_hash "$requirements_file")
    local stored_hash=$(read_state "$state_file")

    if [ "$current_hash" != "$stored_hash" ]; then
        echo "$requirements_file hash changed ('$current_hash' vs '$stored_hash')."
        needs_install=true
    fi

    if $needs_install; then
        echo "Installing/updating $target_desc dependencies from $requirements_file..."
        (
            # shellcheck source=/dev/null
            source "$venv_dir/bin/activate"
            if pip install -r "$requirements_file"; then
                echo "$target_desc dependencies installed/updated successfully."
            else
                echo "ERROR: pip install failed for $target_desc" >&2
                exit 1
            fi
        )
        local pip_status=$?
        if [ $pip_status -eq 0 ]; then
             write_state "$state_file" "$current_hash"
        else
            exit 1 # Propagate the error
        fi
    else
        echo "$target_desc dependencies are up to date (hash: $current_hash)."
    fi
}


# --- Main Setup ---

echo "Starting development environment setup..."

# 1. Setup Frontend Environment
check_and_create_frontend_env "$FRONTEND_DIR"
check_and_install_npm "$FRONTEND_DIR" "open-webui"

# 2. Setup Backend Environment
setup_pip_venv "backend" "backend" "$BACKEND_DIR" "$BACKEND_VENV_DIR" "$BACKEND_DIR/requirements.txt"

# 3. Setup Updater Environment
setup_pip_venv "updater" "updater" "$PROJECT_ROOT" "$UPDATER_VENV_DIR" "$PROJECT_ROOT/requirements.txt"


echo "Setup complete. Launching tmux session..."

# --- tmux Launch ---

# Check if already inside tmux
set +u # Temporarily allow unset variables
if [ -n "$TMUX" ]; then
    echo "Already inside a tmux session. Skipping new session creation."
    exit 0
fi
set -u # Re-enable strict unset variable checking

SESSION_NAME="openwebui_dev"

# --- Define Commands for Each tmux Window ---

# 1. Frontend Dev Server Command
# This will run the Vite dev server with Hot Module Replacement (HMR)
FRONTEND_START_COMMAND="npm run dev"
FRONTEND_COMMAND="cd '$FRONTEND_DIR' && \
                    echo '--- Starting Frontend Dev Server (npm run dev) ---' && \
                    exec $FRONTEND_START_COMMAND"

# 2. Backend API Server Command
PORT="${PORT:-8080}"
BACKEND_START_COMMAND="uvicorn open_webui.main:app --port $PORT --host 127.0.0.1 --forwarded-allow-ips '*' --reload"
BACKEND_COMMAND="cd '$BACKEND_DIR' && \
                   echo 'Activating backend venv...' && \
                   . '$BACKEND_VENV_DIR/bin/activate' && \
                   echo '--- Starting Backend API Server ---' && \
                   exec $BACKEND_START_COMMAND"

# 3. Function Updater Command
UPDATER_COMMAND="cd '$SCRIPT_DIR' && \
                   echo 'Activating updater venv...' && \
                   . '$UPDATER_VENV_DIR/bin/activate' && \
                   echo '--- Starting Function Updater ---' && \
                   exec python function_updater.py"

# --- Create and Configure tmux Session ---

# Create a new detached tmux session with three windows, one for each process
tmux new-session -d -s "$SESSION_NAME" -n "backend" "$BACKEND_COMMAND" \; \
     new-window -n "frontend" "$FRONTEND_COMMAND" \; \
     new-window -n "function-updater" "$UPDATER_COMMAND"

# Select the first window (backend) to be active when attaching
tmux select-window -t "$SESSION_NAME:backend"

# Enable mouse mode for convenience
tmux set-option -g mouse on

# Attach to the created tmux session
echo "Attaching to tmux session '$SESSION_NAME'..."
echo "Windows created: [backend], [frontend], [function-updater]"
echo "Use 'Ctrl-b n' to cycle to the next window, 'Ctrl-b p' for the previous."
tmux attach-session -t "$SESSION_NAME"

echo "Tmux session '$SESSION_NAME' ended."