#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# The plugin name is passed as an environment variable from the YAML step
PLUGIN_NAME="$PLUGIN_NAME" # From workflow env
PLUGIN_FILE_PATH=""

echo "Looking for plugin file for: ${PLUGIN_NAME}"

# --- Check potential locations for the plugin file ---
PLUGIN_FILTER_FILE="plugins/filters/${PLUGIN_NAME}.py"
PLUGIN_PIPE_FILE="plugins/pipes/${PLUGIN_NAME}.py"

# Check specific file locations
if [[ -f "$PLUGIN_FILTER_FILE" ]]; then
  PLUGIN_FILE_PATH="$PLUGIN_FILTER_FILE"
elif [[ -f "$PLUGIN_PIPE_FILE" ]]; then
  PLUGIN_FILE_PATH="$PLUGIN_PIPE_FILE"
# Add more checks here if plugins can exist elsewhere
fi

# --- Output ---
if [[ -n "$PLUGIN_FILE_PATH" ]]; then
  echo "Found plugin file at: $PLUGIN_FILE_PATH"
  # Output the direct path to the plugin file
  echo "plugin_path=${PLUGIN_FILE_PATH}" >> "$GITHUB_OUTPUT"
else
  echo "::warning::Could not find a primary .py file for plugin '${PLUGIN_NAME}' in expected locations (plugins/filters/ or plugins/pipes/)."
  # Output an empty path if not found
  echo "plugin_path=" >> "$GITHUB_OUTPUT"
fi