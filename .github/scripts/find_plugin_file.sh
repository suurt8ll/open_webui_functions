#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# The plugin name is passed as an environment variable from the YAML step
PLUGIN_NAME="$PLUGIN_NAME" # This variable name must match the 'env' key in the YAML step

PLUGIN_FILE_PATH=""

echo "Looking for plugin file for: ${PLUGIN_NAME}"

# Check potential locations for the plugin file
if [[ -f "plugins/filters/${PLUGIN_NAME}.py" ]]; then
  PLUGIN_FILE_PATH="plugins/filters/${PLUGIN_NAME}.py"
elif [[ -f "plugins/pipes/${PLUGIN_NAME}.py" ]]; then
  PLUGIN_FILE_PATH="plugins/pipes/${PLUGIN_NAME}.py"
# Add more checks here if plugins can exist elsewhere
fi

if [[ -n "$PLUGIN_FILE_PATH" ]]; then
  echo "Found plugin file at: $PLUGIN_FILE_PATH"
  # Output the direct path to the plugin file for the release step
  echo "plugin_path=${PLUGIN_FILE_PATH}" >> "$GITHUB_OUTPUT"
else
  echo "::warning::Could not find a primary .py file for plugin '${PLUGIN_NAME}' in expected locations (plugins/filters/ or plugins/pipes/)."
  # Output an empty path if not found
  echo "plugin_path=" >> "$GITHUB_OUTPUT"
fi