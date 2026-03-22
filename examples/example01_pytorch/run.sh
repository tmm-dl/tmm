#!/usr/bin/env bash
# Run TMM training for example 01 (DistilBERT text classification).
#
# Usage:
#   ./run.sh                          # train with defaults from tmm.yaml
#   ./run.sh --set trainer.epochs=1   # quick smoke-test
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Prefer a build directory next to the repo root; fall back to in-tree build.
TMM_BIN="${TMM_BIN:-$REPO_ROOT/build/src/tmm}"

if [[ ! -x "$TMM_BIN" ]]; then
  echo "error: tmm binary not found at $TMM_BIN" >&2
  echo "  Build with:  cmake --build $REPO_ROOT/build" >&2
  echo "  Or set TMM_BIN=/path/to/tmm before running this script." >&2
  exit 1
fi

# Tell TMM where to find the plugin shared libraries.
export TTM_PLUGIN_PATH="${TTM_PLUGIN_PATH:-$REPO_ROOT/build/extensions/python:$REPO_ROOT/build/extensions/core:$REPO_ROOT/build/extensions/console-ui}"

exec "$TMM_BIN" fit "$SCRIPT_DIR/tmm.yaml" "$@"
