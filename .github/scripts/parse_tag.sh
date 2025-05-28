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
VERSION=$(echo "$TAG" | rev | cut -d'/' -f1 | rev) # e.g., v1.1.0rc1

echo "Parsed Tag: Plugin Name = $PLUGIN_NAME, Version = $VERSION"

# Determine if it's a pre-release
# Remove 'v' prefix for pre-release check
VERSION_NO_V="${VERSION#v}" # e.g., 1.1.0rc1
IS_PRERELEASE="false"
# Check for common PEP 440 pre-release identifiers (rc, a, b)
if [[ "$VERSION_NO_V" == *rc* || "$VERSION_NO_V" == *a* || "$VERSION_NO_V" == *b* ]]; then
  IS_PRERELEASE="true"
fi
echo "Is Pre-release: $IS_PRERELEASE"

# Output variables for subsequent steps in the GitHub Actions workflow
# $GITHUB_OUTPUT is an environment variable set by the runner pointing to a file
echo "plugin_name=${PLUGIN_NAME}" >> "$GITHUB_OUTPUT"
echo "version=${VERSION}" >> "$GITHUB_OUTPUT"
echo "is_prerelease=${IS_PRERELEASE}" >> "$GITHUB_OUTPUT"