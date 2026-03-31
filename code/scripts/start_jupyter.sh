#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export JUPYTER_CONFIG_DIR="$PROJECT_ROOT/.jupyter/config"
export JUPYTER_DATA_DIR="$PROJECT_ROOT/.jupyter/data"
export JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.jupyter/runtime"
export IPYTHONDIR="$PROJECT_ROOT/.ipython"

mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"

cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/jupyter" notebook "$@"
