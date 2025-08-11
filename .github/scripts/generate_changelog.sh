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
GENERATED_CHANGELOG_BODY="Could not determine plugin file path."

if [[ -z "$PLUGIN_PATH" ]]; then
  echo "::error:: Plugin path is empty. Cannot generate specific changelog."
  echo "changelog_body=${GENERATED_CHANGELOG_BODY}" >> "$GITHUB_OUTPUT"
  exit 1
fi

# --- Determine if the CURRENT_TAG is a pre-release ---
CURRENT_VERSION_PART=$(echo "$CURRENT_TAG" | rev | cut -d'/' -f1 | rev)
CURRENT_VERSION_NO_V="${CURRENT_VERSION_PART#v}"
IS_CURRENT_TAG_PRERELEASE="false"
if [[ "$CURRENT_VERSION_NO_V" == *rc* || "$CURRENT_VERSION_NO_V" == *a* || "$CURRENT_VERSION_NO_V" == *b* ]]; then
  IS_CURRENT_TAG_PRERELEASE="true"
fi
echo "Current tag (${CURRENT_TAG}) is pre-release: $IS_CURRENT_TAG_PRERELEASE"

# --- Find the previous tag for *this specific plugin* ---
PREVIOUS_TAG=""
ALL_PLUGIN_TAGS_SORTED=$(git tag -l "${PLUGIN_NAME}/v*" | sort -V)

if [[ "$IS_CURRENT_TAG_PRERELEASE" == "true" ]]; then
  PREVIOUS_TAG=$(echo "$ALL_PLUGIN_TAGS_SORTED" | grep -B 1 "^${CURRENT_TAG}$" | head -n 1 || echo "")
  if [[ "$PREVIOUS_TAG" == "$CURRENT_TAG" ]]; then
    PREVIOUS_TAG=""
  fi
else
  STABLE_PLUGIN_TAGS_SORTED=$(echo "$ALL_PLUGIN_TAGS_SORTED" | grep -Ev "/v[0-9.]+(a|b|rc|alpha|beta|c|pre|post|dev)[0-9]*$")
  if [[ -n "$STABLE_PLUGIN_TAGS_SORTED" ]]; then
    PREVIOUS_TAG=$(echo "$STABLE_PLUGIN_TAGS_SORTED" | grep -B 1 "^${CURRENT_TAG}$" | head -n 1 || echo "")
    if [[ "$PREVIOUS_TAG" == "$CURRENT_TAG" ]]; then
      PREVIOUS_TAG=""
    fi
  else
    PREVIOUS_TAG=""
  fi
fi

# --- Determine Commit Range ---
COMMIT_RANGE=""
if [[ -z "$PREVIOUS_TAG" ]]; then
  echo "No suitable previous tag found for ${PLUGIN_NAME}. Generating changelog from the beginning of history for the plugin path up to ${CURRENT_TAG}."
  # For the first release, the range is all commits leading up to CURRENT_TAG affecting PLUGIN_PATH
  COMMIT_RANGE="$CURRENT_TAG"
else
  echo "Found previous tag for changelog: $PREVIOUS_TAG"
  echo "Generating changelog from ${PREVIOUS_TAG} to ${CURRENT_TAG}"
  COMMIT_RANGE="${PREVIOUS_TAG}..${CURRENT_TAG}"
fi

# --- Generate the "What's Changed" section ---
CHANGELOG_HEADING="## 👀 What's Changed"
# Use %B for full commit message body, or %s for subject only. %b for body only.
# Using %s for brevity.
COMMIT_LOG=$(git log --no-merges --pretty="format:* %s (%h by %an)" "$COMMIT_RANGE" -- "$PLUGIN_PATH")

if [[ -z "$COMMIT_LOG" ]]; then
  echo "No relevant non-merge commits found in range $COMMIT_RANGE for path $PLUGIN_PATH."
  # Check for any commits, including merges, if no non-merges were found
  ANY_COMMIT_LOG=$(git log --pretty="format:* %s (%h by %an)" "$COMMIT_RANGE" -- "$PLUGIN_PATH")
   if [[ -z "$ANY_COMMIT_LOG" ]]; then
      GENERATED_CHANGELOG_BODY="No code changes found for this plugin (${PLUGIN_PATH}) since ${PREVIOUS_TAG:-the first release}."
   else
      echo "Only merge commits found affecting the path. Including them."
      GENERATED_CHANGELOG_BODY=$(cat <<EOF
${CHANGELOG_HEADING}

${ANY_COMMIT_LOG}

*Note: Only merge commits were found affecting this file in this release range.*
EOF
)
   fi
else
  echo "Generated Commit Log:"
  echo "$COMMIT_LOG"
  GENERATED_CHANGELOG_BODY=$(cat <<EOF
${CHANGELOG_HEADING}

${COMMIT_LOG}
EOF
)
fi

# --- Generate Contributors Section ---
CONTRIBUTORS_SECTION_CONTENT=""
declare -a FIRST_TIME_CONTRIBUTORS_ARRAY=()
declare -a RETURNING_CONTRIBUTORS_ARRAY=()

# Get unique author names (respecting .mailmap) for commits in the range affecting the plugin path
# This list includes authors from both merge and non-merge commits if ANY_COMMIT_LOG was populated.
# If only non-merge commits, then authors from those.
# We should base authors on the same set of commits used for the changelog body.
# If COMMIT_LOG is non-empty, use that as basis. Else if ANY_COMMIT_LOG is non-empty, use that.
AUTHORS_SOURCE_COMMIT_LOG="$COMMIT_LOG"
if [[ -z "$COMMIT_LOG" && -n "$ANY_COMMIT_LOG" ]]; then
    AUTHORS_SOURCE_COMMIT_LOG="$ANY_COMMIT_LOG" # Not the log itself, but indicates we should use broader author search
fi

if [[ -n "$AUTHORS_SOURCE_COMMIT_LOG" ]]; then # If there's any log (merge or non-merge)
    # Get authors from all commits (merge or non-merge) in the range for the plugin path
    AUTHORS_IN_THIS_RELEASE_FOR_PLUGIN_LIST=$(git log --format='%aN' "$COMMIT_RANGE" -- "$PLUGIN_PATH" | sort -u | grep -v "^\s*$") # Get Author Names, sort unique, remove empty lines

    if [[ -n "$AUTHORS_IN_THIS_RELEASE_FOR_PLUGIN_LIST" ]]; then
        echo "Authors in this release for $PLUGIN_NAME ($PLUGIN_PATH):"
        echo "$AUTHORS_IN_THIS_RELEASE_FOR_PLUGIN_LIST"

        while IFS= read -r AUTHOR_NAME; do
            if [[ -z "$AUTHOR_NAME" ]]; then continue; fi

            # Find the SHA of the very first commit by this author to this specific plugin path in the entire repo history
            FIRST_COMMIT_EVER_BY_AUTHOR_TO_PLUGIN_PATH=$(git log --author="$AUTHOR_NAME" --format='%H' --reverse -- "$PLUGIN_PATH" | head -n 1)

            if [[ -z "$FIRST_COMMIT_EVER_BY_AUTHOR_TO_PLUGIN_PATH" ]]; then
                echo "::warning:: Could not find any commit for author '$AUTHOR_NAME' on path '$PLUGIN_PATH'. Classifying as returning contributor by default."
                RETURNING_CONTRIBUTORS_ARRAY+=("$AUTHOR_NAME")
                continue
            fi

            # Check if this earliest commit is part of the current release's commit list for the plugin
            # `git rev-list` lists commit SHAs. We use it to get all SHAs in the range for the path.
            if git rev-list "$COMMIT_RANGE" -- "$PLUGIN_PATH" | grep -Fxq "$FIRST_COMMIT_EVER_BY_AUTHOR_TO_PLUGIN_PATH"; then
                FIRST_TIME_CONTRIBUTORS_ARRAY+=("$AUTHOR_NAME")
            else
                RETURNING_CONTRIBUTORS_ARRAY+=("$AUTHOR_NAME")
            fi
        done <<< "$AUTHORS_IN_THIS_RELEASE_FOR_PLUGIN_LIST"
    fi


    # Construct the shout-out markdown string
    CONTRIBUTOR_SHOUTOUT_LINES=()
    if [[ ${#FIRST_TIME_CONTRIBUTORS_ARRAY[@]} -gt 0 || ${#RETURNING_CONTRIBUTORS_ARRAY[@]} -gt 0 ]]; then
        CONTRIBUTOR_SHOUTOUT_LINES+=("## 💖 Contributors")
        CONTRIBUTOR_SHOUTOUT_LINES+=("")
        CONTRIBUTOR_SHOUTOUT_LINES+=("A big thank you to everyone who contributed to **${PLUGIN_NAME} ${CURRENT_VERSION_PART}**!")
        CONTRIBUTOR_SHOUTOUT_LINES+=("")

        if [[ ${#FIRST_TIME_CONTRIBUTORS_ARRAY[@]} -gt 0 ]]; then
            CONTRIBUTOR_SHOUTOUT_LINES+=("### 🎉 New Contributors")
            CONTRIBUTOR_SHOUTOUT_LINES+=("Welcome and thank you to the first-time contributors to this plugin:")
            mapfile -t SORTED_FIRST_TIME_CONTRIBUTORS < <(printf "%s\n" "${FIRST_TIME_CONTRIBUTORS_ARRAY[@]}" | sort -u)
            for contributor in "${SORTED_FIRST_TIME_CONTRIBUTORS[@]}"; do
                CONTRIBUTOR_SHOUTOUT_LINES+=("* @${contributor}") # Using @ for mention style
            done
            CONTRIBUTOR_SHOUTOUT_LINES+=("")
        fi

        if [[ ${#RETURNING_CONTRIBUTORS_ARRAY[@]} -gt 0 ]]; then
            if [[ ${#FIRST_TIME_CONTRIBUTORS_ARRAY[@]} -gt 0 ]]; then
                CONTRIBUTOR_SHOUTOUT_LINES+=("### 🚀 Returning Contributors")
                CONTRIBUTOR_SHOUTOUT_LINES+=("And a huge thanks to the returning contributors:")
            else # Only returning contributors
                CONTRIBUTOR_SHOUTOUT_LINES+=("### 🚀 Contributors")
            fi
            mapfile -t SORTED_RETURNING_CONTRIBUTORS < <(printf "%s\n" "${RETURNING_CONTRIBUTORS_ARRAY[@]}" | sort -u)
            for contributor in "${SORTED_RETURNING_CONTRIBUTORS[@]}"; do
                CONTRIBUTOR_SHOUTOUT_LINES+=("* @${contributor}")
            done
            CONTRIBUTOR_SHOUTOUT_LINES+=("")
        fi
        CONTRIBUTOR_SHOUTOUT_LINES+=("*I appreciate all your hard work!*")
    fi

    if [[ ${#CONTRIBUTOR_SHOUTOUT_LINES[@]} -gt 0 ]]; then
        printf -v CONTRIBUTORS_SECTION_CONTENT '%s\n' "${CONTRIBUTOR_SHOUTOUT_LINES[@]}"
        # Remove last trailing newline as we'll add specific spacing later
        CONTRIBUTORS_SECTION_CONTENT="${CONTRIBUTORS_SECTION_CONTENT%\\n}"
    fi
fi

# --- Combine Changelog and Contributors Section ---
FINAL_RELEASE_BODY="$GENERATED_CHANGELOG_BODY"

if [[ -n "$CONTRIBUTORS_SECTION_CONTENT" ]]; then
    # Add two newlines (one blank line in Markdown) before appending the contributors section
    FINAL_RELEASE_BODY="${FINAL_RELEASE_BODY}"$'\n\n'"${CONTRIBUTORS_SECTION_CONTENT}"
fi

# --- Output ---
echo "changelog_body<<EOFBODY" >> "$GITHUB_OUTPUT"
echo "$FINAL_RELEASE_BODY" >> "$GITHUB_OUTPUT"
echo "EOFBODY" >> "$GITHUB_OUTPUT"

echo "Changelog and contributor list generation complete."
