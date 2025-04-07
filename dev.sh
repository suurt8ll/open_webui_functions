#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# Prevent errors in a pipeline from being masked.
set -euo pipefail

# --- Configuration ---

# Project structure definition
# Get the directory containing this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT="$SCRIPT_DIR" # Assuming this script is at the project root
FRONTEND_DIR="$PROJECT_ROOT/submodules/open-webui"
BACKEND_DIR="$FRONTEND_DIR/backend" # OpenWebUI's backend is inside its own dir

# Virtual environment paths (adjust if different)
BACKEND_VENV_DIR="$BACKEND_DIR/.venv"
UPDATER_VENV_DIR="$PROJECT_ROOT/.venv" # Assumes a separate venv for the updater

# --- Helper Functions ---

# Function to check and install npm dependencies
check_and_install_npm() {
    local target_dir="$1"
    local needs_install=false

    echo "Checking frontend dependencies in '$target_dir'..."
    cd "$target_dir"

    if [ ! -d "node_modules" ]; then
        echo "Node modules not found."
        needs_install=true
    # Check package-lock.json first if it exists
    elif [ -f "package-lock.json" ] && [ "package-lock.json" -nt "node_modules" ]; then
        echo "package-lock.json is newer than node_modules."
        needs_install=true
    # Otherwise check package.json
    elif [ -f "package.json" ] && [ "package.json" -nt "node_modules" ]; then
        echo "package.json is newer than node_modules."
        needs_install=true
    elif [ ! -f "package.json" ] && [ ! -f "package-lock.json" ]; then
         echo "Warning: Neither package.json nor package-lock.json found in $target_dir."
         # Decide if you want to proceed or exit here
         # exit 1
    fi

    if $needs_install; then
        echo "Installing frontend dependencies..."
        npm install || { echo "ERROR: npm install failed in $target_dir"; exit 1; }
        echo "Frontend dependencies installed."
    else
        echo "Frontend dependencies are up to date."
    fi

    cd "$PROJECT_ROOT" # Return to original root dir
}

# Function to check and build the frontend
check_and_build_npm() {
    local target_dir="$1"
    local build_dir="$target_dir/build" # Standard build output dir
    local src_dir="$target_dir/src"
    local needs_rebuild=false

    echo "Checking frontend build status in '$target_dir'..."
    cd "$target_dir"

    if [ ! -d "$build_dir" ]; then
        echo "Build directory not found."
        needs_rebuild=true
    # Check if any file in src is newer than the build directory timestamp
    # Note: This isn't perfect, but a common heuristic. A more robust check
    # might involve comparing against a specific file in the build dir.
    elif find "$src_dir" -type f -newer "$build_dir" -print -quit | grep -q .; then
        echo "Source files are newer than the build directory."
        needs_rebuild=true
    fi

    if $needs_rebuild; then
        echo "Building OpenWebUI frontend..."
        npm run build || { echo "ERROR: npm run build failed in $target_dir"; exit 1; }
        echo "Frontend build complete."
    else
        echo "OpenWebUI frontend is up to date. No build needed."
    fi

    cd "$PROJECT_ROOT" # Return to original root dir
}

# Function to set up backend python environment
setup_backend_venv() {
    local backend_dir="$1"
    local venv_dir="$2"
    local requirements_file="$backend_dir/requirements.txt"

    echo "Checking backend environment in '$backend_dir'..."
    if [ ! -f "$venv_dir/bin/activate" ]; then
        echo "ERROR: Backend virtual environment not found at '$venv_dir'."
        echo "Please create it first (e.g., python -m venv $venv_dir)"
        exit 1
    fi

    if [ ! -f "$requirements_file" ]; then
        echo "Warning: requirements.txt not found in $backend_dir."
        return
    fi

    echo "Activating backend venv and installing/updating dependencies..."
    # Activate venv in a subshell to install packages
    (
        # shellcheck source=/dev/null
        source "$venv_dir/bin/activate"
        pip install -r "$requirements_file" || { echo "ERROR: pip install failed for backend"; exit 1; }
        echo "Backend dependencies installed/updated."
    )
}

# Function to check updater python environment
check_updater_venv() {
    local venv_dir="$1"
    echo "Checking function_updater environment..."
    if [ ! -f "$venv_dir/bin/activate" ]; then
        echo "ERROR: Updater virtual environment not found at '$venv_dir'."
        echo "Please create it first (e.g., python -m venv $venv_dir)"
        exit 1
    fi
     # You could add a pip install here too if function_updater has requirements
    echo "Updater venv found."
}


# --- Main Setup ---

echo "Starting development environment setup..."

# 1. Setup Frontend
check_and_install_npm "$FRONTEND_DIR"
check_and_build_npm "$FRONTEND_DIR"

# 2. Setup Backend Environment (Install requirements)
setup_backend_venv "$BACKEND_DIR" "$BACKEND_VENV_DIR"

# 3. Check Updater Environment
check_updater_venv "$UPDATER_VENV_DIR"

echo "Setup complete. Launching tmux session..."

# --- tmux Launch ---

# Check if already inside tmux
set +u  # Disable "nounset" temporarily
if [ -n "$TMUX" ]; then
    echo "Already inside a tmux session. Skipping new session creation."
    exit 0
fi
set -u  # Re-enable "nounset"

SESSION_NAME="openwebui_dev_session" # Use a more descriptive name

# Start a new tmux session, detached (-d), with a specific name (-s)
# The first window is created automatically.
tmux new-session -d -s "$SESSION_NAME" -n "backend" # Name the first window 'backend'

# Send commands to the first window (index 0: backend)
# Activate venv and use 'exec' to run the main backend command.
# When the backend command exits, the shell running it exits, closing the pane/window.
PORT="${PORT:-8080}"
BACKEND_START_COMMAND="uvicorn open_webui.main:app --port $PORT --host 127.0.0.1 --forwarded-allow-ips '*' --reload"
tmux send-keys -t "$SESSION_NAME:backend" "cd '$BACKEND_DIR'" Enter
tmux send-keys -t "$SESSION_NAME:backend" ". '$BACKEND_VENV_DIR/bin/activate'" Enter
tmux send-keys -t "$SESSION_NAME:backend" "echo '--- Starting Backend ---'" Enter
tmux send-keys -t "$SESSION_NAME:backend" "exec $BACKEND_START_COMMAND" Enter

# Create the second window for function_updater.py
# Pass the command directly to new-window. Use 'exec' for auto-closing.
UPDATER_COMMAND="python function_updater.py"
tmux new-window -t "$SESSION_NAME" -n "function-updater" \
    "cd '$PROJECT_ROOT' && \
     . '$UPDATER_VENV_DIR/bin/activate' && \
     echo '--- Starting Function Updater ---' && \
     exec $UPDATER_COMMAND"

# Select the first window (backend) to be active when attaching
tmux select-window -t "$SESSION_NAME:backend"

# Enable mouse mode for convenience
tmux set-option -g mouse on

# Attach to the created tmux session
echo "Attaching to tmux session '$SESSION_NAME'..."
tmux attach-session -t "$SESSION_NAME"

echo "Tmux session '$SESSION_NAME' ended."