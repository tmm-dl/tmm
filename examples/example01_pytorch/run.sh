#!/usr/bin/env bash
# Run TTM training for example 01 (DistilBERT text classification).
#
# Usage:
#   ./run.sh                          # train with defaults from ttm.yaml
#   ./run.sh --set trainer.epochs=1   # quick smoke-test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Prefer a build directory next to the repo root; fall back to in-tree build.
TTM_BIN="${TTM_BIN:-$REPO_ROOT/build/src/ttm}"

if [[ ! -x "$TTM_BIN" ]]; then
  echo "error: ttm binary not found at $TTM_BIN" >&2
  echo "  Build with:  cmake --build $REPO_ROOT/build" >&2
  echo "  Or set TTM_BIN=/path/to/ttm before running this script." >&2
  exit 1
fi

# Tell TTM where to find the plugin shared libraries.
export TTM_PLUGIN_PATH="${TTM_PLUGIN_PATH:-$REPO_ROOT/build/extensions/python:$REPO_ROOT/build/extensions/core}"

exec "$TTM_BIN" fit "$SCRIPT_DIR/ttm.yaml" "$@"
