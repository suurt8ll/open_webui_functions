#!/bin/bash
#
# uv-shared-env.sh — keep uv-managed Python + uv cache inside the repo so
# venvs are portable between the host and the pi-fluid dev container.
#
# Why: uv venvs hardlink packages from the uv cache and symlink bin/python
# into the uv-managed Python install, using absolute paths. The container
# only bind-mounts this repo (same path on both sides), so when uv state
# lives in $HOME (~/.cache/uv, ~/.local/share/uv), each side creates venvs
# pointing at its own home — and any `uv` run on one side breaks the venv
# for the other. Pointing both env vars at this repo fixes that: same path,
# same inodes, venvs work everywhere.
#
# Usage: source this file before running uv, e.g.
#   source dev/uv-shared-env.sh && uv sync
# or add that source line to your shell rc / direnv / .env handling.
# Already-set variables are respected (override here wins otherwise).

_uv_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$_uv_env_dir/../.uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$_uv_env_dir/../.uv-cache}"
unset _uv_env_dir
