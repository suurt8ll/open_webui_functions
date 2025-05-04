#!/bin/bash

set -e

# Inputs from environment variables set in the workflow
PLUGIN_NAME="$PLUGIN_NAME"
CURRENT_TAG="$CURRENT_TAG"
PLUGIN_PATH="$PLUGIN_PATH" # Note: singular path now

echo "Generating changelog for Plugin: $PLUGIN_NAME"
echo "Current Tag: $CURRENT_TAG"
echo "Plugin Path: $PLUGIN_PATH"

# Default body if path is missing (shouldn't happen if find_plugin_file succeeded)
GENERATED_BODY="Could not determine plugin file path."

if [[ -z "$PLUGIN_PATH" ]]; then
  echo "::error:: Plugin path is empty. Cannot generate specific changelog."
  echo "changelog_body=${GENERATED_BODY}" >> "$GITHUB_OUTPUT"
  exit 1 # Exit with error if path is missing
fi

# Find the previous tag for *this specific plugin*
# List tags matching the pattern, sort by version, get the one before the current tag
PREVIOUS_TAG=$(git tag -l "${PLUGIN_NAME}/v*" | sort -V | grep -B 1 "^${CURRENT_TAG}$" | head -n 1 || echo "")

# Handle finding the previous tag (or not)
COMMIT_RANGE=""
if [[ -z "$PREVIOUS_TAG" ]]; then
  echo "No previous tag found for ${PLUGIN_NAME}. Generating changelog from the beginning of history for the plugin path."
  # Log commits from the beginning up to the current tag, filtered by path
  CURRENT_COMMIT=$(git rev-list -n 1 "$CURRENT_TAG")
  COMMIT_RANGE="$CURRENT_COMMIT" # Log commits reachable from CURRENT_COMMIT
else
  echo "Found previous tag: $PREVIOUS_TAG"
  echo "Generating changelog from ${PREVIOUS_TAG} to ${CURRENT_TAG}"
  COMMIT_RANGE="${PREVIOUS_TAG}..${CURRENT_TAG}"
fi

# Generate the changelog body using git log
# Format: "* Subject (sha)"
# Filter by the specific plugin path
CHANGELOG_HEADING="## What's Changed"
# Use --no-merges to potentially exclude merge commits unless they changed the file
COMMIT_LOG=$(git log --no-merges --pretty="format:* %s (%h)" "$COMMIT_RANGE" -- "$PLUGIN_PATH")

# Check if any relevant commits were found
if [[ -z "$COMMIT_LOG" ]]; then
  echo "No relevant non-merge commits found in range $COMMIT_RANGE for path $PLUGIN_PATH."
  # Check if *any* commits exist in the range for the file, including merges
  ANY_COMMIT_LOG=$(git log --pretty="format:* %s (%h)" "$COMMIT_RANGE" -- "$PLUGIN_PATH")
   if [[ -z "$ANY_COMMIT_LOG" ]]; then
      GENERATED_BODY="No code changes found for this plugin (${PLUGIN_PATH}) since the last release (${PREVIOUS_TAG:-first release})."
   else
      # If only merge commits were found, mention that or show them? Let's show them for now.
      echo "Only merge commits found affecting the path. Including them."
      GENERATED_BODY=$(cat <<EOF
${CHANGELOG_HEADING}

${ANY_COMMIT_LOG}

*Note: Only merge commits were found affecting this file in this release range.*
EOF
)
   fi
else
  echo "Generated Commit Log:"
  echo "$COMMIT_LOG"
  GENERATED_BODY=$(cat <<EOF
${CHANGELOG_HEADING}

${COMMIT_LOG}
EOF
)
fi

# Use EOF delimiter for multiline output
echo "changelog_body<<EOF" >> "$GITHUB_OUTPUT"
echo "$GENERATED_BODY" >> "$GITHUB_OUTPUT"
echo "EOF" >> "$GITHUB_OUTPUT"

echo "Changelog generation complete."