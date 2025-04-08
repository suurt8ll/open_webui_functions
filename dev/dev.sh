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
# Usage: get_hash <file1> [file2 ...]
get_hash() {
    check_openssl # Ensure openssl is available

    # Concatenate files and calculate hash
    # Handle non-existent files gracefully (treat as empty)
    local files_to_hash=()
    for f in "$@"; do
        if [[ -f "$f" ]]; then
            files_to_hash+=("$f")
        fi
    done

    if [[ ${#files_to_hash[@]} -eq 0 ]]; then
        # If no files exist, return a consistent hash for "empty"
        echo "d41d8cd98f00b204e9800998ecf8427e" # MD5 of empty string
        return
    fi

    cat "${files_to_hash[@]}" | openssl md5 | awk '{print $2}'
}

# Get hash of directory contents (files only) using openssl
# Usage: get_dir_hash <directory>
get_dir_hash() {
    check_openssl # Ensure openssl is available

    local dir="$1"

    if [ ! -d "$dir" ]; then
        echo "d41d8cd98f00b204e9800998ecf8427e" # MD5 of empty string
        return
    fi

    # Find all files, sort them, hash their contents, then hash the list of hashes
    find "$dir" -type f -print0 | sort -z | xargs -0 openssl md5 | openssl md5 | awk '{print $2}'
}

# Read stored hash from state file
# Usage: read_state <state_file>
read_state() {
    local state_file="$1"
    if [ -f "$state_file" ]; then
        cat "$state_file"
    else
        echo "nostate" # Return a value indicating no previous state
    fi
}

# Write current hash to state file
# Usage: write_state <state_file> <hash>
write_state() {
    local state_file="$1"
    local hash="$2"
    echo "$hash" > "$state_file"
}

# --- Improved Dependency Checks ---

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
        if npm install; then
            echo "Frontend dependencies installed successfully."
            write_state "$state_file" "$current_hash"
        else
            echo "ERROR: npm install failed in $target_dir" >&2
            # Optional: remove state file on failure?
            # rm -f "$state_file"
            cd "$PROJECT_ROOT"
            exit 1
        fi
    else
        echo "Frontend dependencies are up to date (hash: $current_hash)."
    fi

    cd "$PROJECT_ROOT" # Return to original root dir
}

check_and_build_npm() {
    local target_dir="$1"
    local component_name="$2" # e.g., "open-webui"
    local build_dir="$target_dir/build" # Standard build output dir
    local src_dir="$target_dir/src"
    local state_file="$STATE_DIR/${component_name}_npm_build.hash"
    local needs_rebuild=false

    echo "Checking frontend build status in '$target_dir'..."
    ensure_state_dir "$STATE_DIR"
    cd "$target_dir"

    # --- Define inputs that trigger a rebuild ---
    # 1. Source code
    local src_hash=$(get_dir_hash "$src_dir")
    # 2. Dependency state (use the hash file we created earlier)
    local deps_hash=$(read_state "$STATE_DIR/${component_name}_npm_deps.hash")
    # 3. Key configuration files (add others if needed)
    local config_files=("package.json" "vite.config.js" "vite.config.ts" "webpack.config.js" "tsconfig.json" ".env") # Add relevant config files
    local config_hash=$(get_hash "${config_files[@]}")
    # 4. Node/NPM version (optional but recommended for robustness)
    # local node_version_hash=$(echo "$(node -v)-$(npm -v)" | openssl md5 | awk '{print $2}') # Example

    # Combine all input hashes into a single current hash
    local current_build_inputs_hash=$(echo "$src_hash:$deps_hash:$config_hash" | openssl md5 | awk '{print $2}') # Add $node_version_hash if using
    local stored_hash=$(read_state "$state_file")

    if [ ! -d "$build_dir" ]; then
        echo "Build directory '$build_dir' not found."
        needs_rebuild=true
    elif [ "$current_build_inputs_hash" != "$stored_hash" ]; then
        echo "Build inputs changed (src, deps, or config). Hash: '$current_build_inputs_hash' vs '$stored_hash'."
        needs_rebuild=true
    fi

    if $needs_rebuild; then
        echo "Building OpenWebUI frontend..."
        if npm run build; then
            echo "Frontend build complete."
            # Update the state file ONLY on successful build
            write_state "$state_file" "$current_build_inputs_hash"
        else
            echo "ERROR: npm run build failed in $target_dir" >&2
            # Optional: remove state file on failure?
            # rm -f "$state_file"
            cd "$PROJECT_ROOT"
            exit 1
        fi
    else
        echo "OpenWebUI frontend build is up to date (hash: $current_build_inputs_hash)."
    fi

    cd "$PROJECT_ROOT" # Return to original root dir
}

setup_pip_venv() {
    local target_desc="$1" # e.g., "backend" or "updater"
    local component_name="$2" # e.g., "backend", "updater"
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
        # Ensure state file is removed if requirements are gone
        rm -f "$state_file"
        return
    fi

    local current_hash=$(get_hash "$requirements_file")
    local stored_hash=$(read_state "$state_file")

    # We check the hash *before* activating the venv for efficiency
    if [ "$current_hash" != "$stored_hash" ]; then
        echo "$requirements_file hash changed ('$current_hash' vs '$stored_hash')."
        needs_install=true
    fi

    # Optional: Add a check to see if the venv python version matches the system's intended python
    # This is more complex, involves storing python version in state file too.

    if $needs_install; then
        echo "Installing/updating $target_desc dependencies from $requirements_file..."
        # Activate venv in a subshell to install packages
        (
            # shellcheck source=/dev/null
            source "$venv_dir/bin/activate"
            # Use --upgrade for potentially faster updates if needed, but default -r is usually fine
            if pip install -r "$requirements_file"; then
                echo "$target_desc dependencies installed/updated successfully."
                # Write state *outside* the subshell after success
            else
                echo "ERROR: pip install failed for $target_desc" >&2
                # Exit the subshell with error code
                exit 1
            fi
        )
        # Check subshell exit status
        local pip_status=$?
        if [ $pip_status -eq 0 ]; then
             write_state "$state_file" "$current_hash"
        else
            # Optional: remove state file on failure?
            # rm -f "$state_file"
            exit 1 # Propagate the error
        fi
    else
        echo "$target_desc dependencies are up to date (hash: $current_hash)."
    fi
}


# --- Main Setup ---

echo "Starting development environment setup..."

# 1. Setup Frontend
check_and_install_npm "$FRONTEND_DIR" "open-webui"
check_and_build_npm "$FRONTEND_DIR" "open-webui"

# 2. Setup Backend Environment (Install requirements)
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

SESSION_NAME="openwebui_dev_session"

# Backend start command
PORT="${PORT:-8080}"
BACKEND_START_COMMAND="uvicorn open_webui.main:app --port $PORT --host 127.0.0.1 --forwarded-allow-ips '*' --reload"

# Construct the full command for the backend window
BACKEND_COMMAND="cd '$BACKEND_DIR' && \
                   echo 'Activating backend venv...' && \
                   . '$BACKEND_VENV_DIR/bin/activate' && \
                   echo '--- Starting Backend ---' && \
                   exec $BACKEND_START_COMMAND"

# Construct the full command for the function updater window
UPDATER_COMMAND="cd '$SCRIPT_DIR' && \
                   echo 'Activating updater venv...' && \
                   . '$UPDATER_VENV_DIR/bin/activate' && \
                   echo '--- Starting Function Updater ---' && \
                   exec python function_updater.py"

# Create a new tmux session with two windows, executing the commands directly
tmux new-session -d -s "$SESSION_NAME" -n "backend" "$BACKEND_COMMAND" \; \
     new-window -n "function-updater" "$UPDATER_COMMAND"

# Select the first window (backend) to be active when attaching
tmux select-window -t "$SESSION_NAME:backend"

# Enable mouse mode for convenience
tmux set-option -g mouse on

# Attach to the created tmux session
echo "Attaching to tmux session '$SESSION_NAME'..."
tmux attach-session -t "$SESSION_NAME"

echo "Tmux session '$SESSION_NAME' ended."