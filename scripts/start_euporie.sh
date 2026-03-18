#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export HOME="$SCRIPT_DIR/.home"
export XDG_CONFIG_HOME="$SCRIPT_DIR/.config"
export XDG_CACHE_HOME="$SCRIPT_DIR/.cache"
export XDG_DATA_HOME="$SCRIPT_DIR/.local/share"
export JUPYTER_DATA_DIR="$SCRIPT_DIR/.jupyter_data"
export JUPYTER_RUNTIME_DIR="$SCRIPT_DIR/.jupyter_runtime"
export IPYTHONDIR="$SCRIPT_DIR/.ipython"

mkdir -p \
  "$HOME/Library/Application Support" \
  "$XDG_CONFIG_HOME" \
  "$XDG_CACHE_HOME" \
  "$XDG_DATA_HOME" \
  "$JUPYTER_DATA_DIR" \
  "$JUPYTER_RUNTIME_DIR" \
  "$IPYTHONDIR"

NOTEBOOK_PATH="${1:-$SCRIPT_DIR/demo_euporie.ipynb}"
shift $(( $# > 0 ? 1 : 0 ))

exec "$ROOT_DIR/.venv/bin/euporie-notebook" --kernel-name python3 "$NOTEBOOK_PATH" "$@"
