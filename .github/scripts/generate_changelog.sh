#!/bin/bash

set -e

# Inputs from environment variables set in the workflow
PLUGIN_NAME="$PLUGIN_NAME"
CURRENT_TAG="$CURRENT_TAG"
PLUGIN_PATH="$PLUGIN_PATH"

echo "Generating changelog for Plugin: $PLUGIN_NAME"
echo "Current Tag: $CURRENT_TAG"
echo "Plugin Path: $PLUGIN_PATH"

# Default body if path is missing
GENERATED_BODY="Could not determine plugin file path."

if [[ -z "$PLUGIN_PATH" ]]; then
  echo "::error:: Plugin path is empty. Cannot generate specific changelog."
  echo "changelog_body=${GENERATED_BODY}" >> "$GITHUB_OUTPUT"
  exit 1
fi

# --- Determine if the CURRENT_TAG is a pre-release ---
# Extract version part from current tag (e.g., v1.1.0rc1 from plugin/v1.1.0rc1)
CURRENT_VERSION_PART=$(echo "$CURRENT_TAG" | rev | cut -d'/' -f1 | rev)
# Remove 'v' prefix for pre-release check
CURRENT_VERSION_NO_V="${CURRENT_VERSION_PART#v}" # e.g., 1.1.0rc1 or 1.1.0

IS_CURRENT_TAG_PRERELEASE="false"
# Check for common PEP 440 pre-release identifiers (rc, a, b)
if [[ "$CURRENT_VERSION_NO_V" == *rc* || "$CURRENT_VERSION_NO_V" == *a* || "$CURRENT_VERSION_NO_V" == *b* ]]; then
  IS_CURRENT_TAG_PRERELEASE="true"
fi
echo "Current tag (${CURRENT_TAG}) is pre-release: $IS_CURRENT_TAG_PRERELEASE"

# --- Find the previous tag for *this specific plugin* ---
PREVIOUS_TAG=""
ALL_PLUGIN_TAGS_SORTED=$(git tag -l "${PLUGIN_NAME}/v*" | sort -V)

if [[ "$IS_CURRENT_TAG_PRERELEASE" == "true" ]]; then
  # For pre-releases, find the immediately preceding tag (can be stable or another pre-release)
  PREVIOUS_TAG=$(echo "$ALL_PLUGIN_TAGS_SORTED" | grep -B 1 "^${CURRENT_TAG}$" | head -n 1 || echo "")
  if [[ "$PREVIOUS_TAG" == "$CURRENT_TAG" ]]; then # Handles case where current tag is the very first tag
    PREVIOUS_TAG=""
  fi
else
  # For stable releases, find the latest STABLE tag strictly before this one.
  # Filter out pre-release tags from the list of all plugin tags
  # Regex: /v<digits>.<digits>...<rc|a|b><digits>
  # We want to KEEP tags that DO NOT match this pre-release pattern.
  STABLE_PLUGIN_TAGS_SORTED=$(echo "$ALL_PLUGIN_TAGS_SORTED" | grep -Ev "/v[0-9.]+(a|b|rc|alpha|beta|c|pre|post|dev)[0-9]*$")

  if [[ -n "$STABLE_PLUGIN_TAGS_SORTED" ]]; then
    PREVIOUS_TAG=$(echo "$STABLE_PLUGIN_TAGS_SORTED" | grep -B 1 "^${CURRENT_TAG}$" | head -n 1 || echo "")
    if [[ "$PREVIOUS_TAG" == "$CURRENT_TAG" ]]; then # Handles case where current tag is the first stable tag
      PREVIOUS_TAG=""
    fi
  else
    # This case should ideally not be hit if CURRENT_TAG is stable and present in ALL_PLUGIN_TAGS_SORTED
    # but as a fallback, if no stable tags are found (e.g. only RCs exist), treat as first release.
    PREVIOUS_TAG=""
  fi
fi


# Handle finding the previous tag (or not)
COMMIT_RANGE=""
if [[ -z "$PREVIOUS_TAG" ]]; then
  echo "No suitable previous tag found for ${PLUGIN_NAME} based on release type. Generating changelog from the beginning of history for the plugin path."
  CURRENT_COMMIT=$(git rev-list -n 1 "$CURRENT_TAG")
  COMMIT_RANGE="$CURRENT_COMMIT"
else
  echo "Found previous tag for changelog: $PREVIOUS_TAG"
  echo "Generating changelog from ${PREVIOUS_TAG} to ${CURRENT_TAG}"
  COMMIT_RANGE="${PREVIOUS_TAG}..${CURRENT_TAG}"
fi

# Generate the changelog body using git log
CHANGELOG_HEADING="## What's Changed"
COMMIT_LOG=$(git log --no-merges --pretty="format:* %s (%h)" "$COMMIT_RANGE" -- "$PLUGIN_PATH")

if [[ -z "$COMMIT_LOG" ]]; then
  echo "No relevant non-merge commits found in range $COMMIT_RANGE for path $PLUGIN_PATH."
  ANY_COMMIT_LOG=$(git log --pretty="format:* %s (%h)" "$COMMIT_RANGE" -- "$PLUGIN_PATH")
   if [[ -z "$ANY_COMMIT_LOG" ]]; then
      GENERATED_BODY="No code changes found for this plugin (${PLUGIN_PATH}) since ${PREVIOUS_TAG:-the first release}."
   else
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

echo "changelog_body<<EOF" >> "$GITHUB_OUTPUT"
echo "$GENERATED_BODY" >> "$GITHUB_OUTPUT"
echo "EOF" >> "$GITHUB_OUTPUT"

echo "Changelog generation complete."