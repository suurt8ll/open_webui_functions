#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# The tag name is available in the GITHUB_REF_NAME environment variable
TAG="$GITHUB_REF_NAME"

# Extract plugin name (part before the last '/')
# rev reverses the string, cut splits by '/', -f2- takes the second field onwards, rev reverses back
PLUGIN_NAME=$(echo "$TAG" | rev | cut -d'/' -f2- | rev)

# Extract version (part after the last '/')
# rev reverses, cut splits by '/', -f1 takes the first field, rev reverses back
VERSION=$(echo "$TAG" | rev | cut -d'/' -f1 | rev)

echo "Parsed Tag: Plugin Name = $PLUGIN_NAME, Version = $VERSION"

# Output variables for subsequent steps in the GitHub Actions workflow
# $GITHUB_OUTPUT is an environment variable set by the runner pointing to a file
echo "plugin_name=${PLUGIN_NAME}" >> "$GITHUB_OUTPUT"
echo "version=${VERSION}" >> "$GITHUB_OUTPUT"