#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_BIN="${PYTHON_BIN:-/teamspace/jz/conda/assemlm/bin/python}"
QUERY_ENTRY="$PROJECT_ROOT/main/query_assemlm.py"

cd "$PROJECT_ROOT"
# exec "$PYTHON_BIN" "$QUERY_ENTRY" \
exec python "$QUERY_ENTRY" \
	--hdf5-path "$PROJECT_ROOT/datasets/demo.hdf5" \
	--output-dir "$PROJECT_ROOT/datasets_tmp" \
	--split test \
    --num-assets-batch 1 \
	--api-url "http://127.0.0.1:25557/query" \
	"$@"