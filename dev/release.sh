#!/bin/bash

# A helper script to interactively create and push a release tag for a plugin.
# It finds all plugins, lets the user select one, extracts the version from
# the plugin's docstring, and then creates and pushes the appropriate git tag.

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Main Logic ---

echo -e "${BLUE}🔍 Finding available plugins...${NC}"

# Find all potential plugin files in the standard directories.
# -path '*/__pycache__/*' -prune -o : Excludes pycache directories
# -name '*.py' : Finds python files
# ! -name '__init__.py' : Excludes __init__.py files
mapfile -t PLUGIN_FILES < <(find plugins/filters plugins/pipes -path '*/__pycache__/*' -prune -o -name '*.py' ! -name '__init__.py' -print | sort)

if [ ${#PLUGIN_FILES[@]} -eq 0 ]; then
  echo -e "${RED}Error: No plugin files found in 'plugins/filters/' or 'plugins/pipes/'.${NC}"
  exit 1
fi

# Create a more user-friendly list of names for the menu
declare -a PLUGIN_DISPLAY_NAMES
for file in "${PLUGIN_FILES[@]}"; do
  # Get the filename without the path and .py extension
  PLUGIN_DISPLAY_NAMES+=("$(basename "$file" .py)")
done

# --- Interactive Selection Menu ---
PS3=$'\n'"Please select a plugin to release (enter the number): "
select plugin_name in "${PLUGIN_DISPLAY_NAMES[@]}"; do
  if [[ -n "$plugin_name" ]]; then
    # The index of the selection is in REPLY-1
    SELECTED_PLUGIN_PATH="${PLUGIN_FILES[$REPLY - 1]}"
    echo -e "${BLUE}▶️ You selected plugin:${NC} ${plugin_name}"
    echo -e "${BLUE}   File path:${NC} ${SELECTED_PLUGIN_PATH}"
    break
  else
    echo -e "${RED}Invalid selection. Please try again.${NC}"
  fi
done

# --- Version Extraction ---
echo -e "\n${BLUE}📝 Extracting version from ${SELECTED_PLUGIN_PATH}...${NC}"

# Use grep with Perl-compatible regex (-P) to find the version line.
# \s*       - zero or more whitespace characters
# version:  - literal text
# \s*       - zero or more whitespace characters
# \K        - discards the text matched so far
# \S+       - matches one or more non-whitespace characters (the version string)
VERSION=$(grep -oP '^\s*version:\s*\K\S+' "$SELECTED_PLUGIN_PATH")

if [[ -z "$VERSION" ]]; then
  echo -e "${RED}Error: Could not find a 'version: ...' key in the docstring of ${SELECTED_PLUGIN_PATH}.${NC}"
  echo "Please ensure the file has a line like 'version: 1.2.3' at the top."
  exit 1
fi

echo -e "${GREEN}✅ Found version:${NC} ${VERSION}"

# --- Tag Creation and Confirmation ---
TAG_NAME="${plugin_name}/v${VERSION}"

echo -e "\n${BLUE}Ready to create and push the following Git tag:${NC}"
echo -e "  ${YELLOW}${TAG_NAME}${NC}"
echo "This will trigger the 'Plugin Release' GitHub Action."

# Prompt for confirmation
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo # Move to a new line

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

# --- Git Operations ---
echo -e "\n${BLUE}🚀 Creating and pushing tag...${NC}"

# Check if the tag already exists locally
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Tag '${TAG_NAME}' already exists locally. Aborting to prevent issues.${NC}"
    echo "If you need to re-tag, please delete the local and remote tag first."
    exit 1
fi

git tag "$TAG_NAME"
echo "  - Local tag '${TAG_NAME}' created."

git push origin "$TAG_NAME"
echo "  - Tag '${TAG_NAME}' pushed to origin."

echo -e "\n${GREEN}✨ Success! The release workflow has been triggered.${NC}"
echo "Check the 'Actions' tab in your GitHub repository to monitor its progress."