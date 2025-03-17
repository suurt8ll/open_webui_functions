#!/bin/bash

# Get the directory of the script using realpath
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Construct the target directory
TARGET_DIR="$SCRIPT_DIR/submodules/open-webui/"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Target directory '$TARGET_DIR' does not exist."
  exit 1
fi

# Change to the target directory
cd "$TARGET_DIR" || { echo "Failed to cd into '$TARGET_DIR'"; exit 1; }

# Source the nvm initialization script
source /usr/share/nvm/init-nvm.sh || { echo "Failed to source nvm init script"; exit 1; }

# Check if npm install is needed
needs_install=false
if [ ! -d "node_modules" ]; then
  needs_install=true
elif [ -f "package-lock.json" ]; then
  if [[ "package-lock.json" -nt "node_modules" ]]; then
    needs_install=true
  fi
elif [ -f "package.json" ]; then
    if [[ "package.json" -nt "node_modules" ]]; then
        needs_install = true;
    fi
else
  echo "Warning: Neither package.json nor package-lock.json found."
fi

# Run npm install if needed
if $needs_install; then
  echo "Installing dependencies..."
  npm install || { echo "npm install failed"; exit 1; }
  echo "Dependencies installed."
else
  echo "Dependencies are up to date."
fi

# Check if a rebuild is needed
needs_rebuild=false
if [ ! -d "build" ]; then
  needs_rebuild=true
else
  if find src/ -newer build -print -quit | grep -q .; then
    needs_rebuild=true
  fi
fi

# Run npm run build if needed
if $needs_rebuild; then
  echo "Building OpenWebUI..."
  npm run build || { echo "npm run build failed"; exit 1; }
  echo "Build complete."
else
  echo "OpenWebUI is up to date. No build needed."
fi

# --- tmux logic ---

# Check if tmux is already running
if [ -n "$TMUX" ]; then
    echo "Already inside a tmux session.  Skipping tmux launch."
    exit 0
fi

# Start a new tmux session, detached (-d), with a session name (-s)
tmux new-session -d -s openwebui_session

# Rename the first window
tmux rename-window -t openwebui_session:0 "open-webui"

# Send commands to the first window
tmux send-keys -t openwebui_session:0 "cd ./backend" Enter
tmux send-keys -t openwebui_session:0 ". .venv/bin/activate" Enter
tmux send-keys -t openwebui_session:0 "pip install -r requirements.txt" Enter
tmux send-keys -t openwebui_session:0 "sh ./dev.sh" Enter

# Create a new window for function_updater.py
tmux new-window -t openwebui_session -n "function-updater"

# Send commands to the second window
tmux send-keys -t openwebui_session:1 "cd $SCRIPT_DIR" Enter
tmux send-keys -t openwebui_session:1 ". $SCRIPT_DIR/submodules/open-webui/backend/.venv/bin/activate" Enter
tmux send-keys -t openwebui_session:1 "python function_updater.py" Enter

# Switch back to the first window (index 0)
tmux select-window -t openwebui_session:0

# Attach to the tmux session
tmux attach-session -t openwebui_session