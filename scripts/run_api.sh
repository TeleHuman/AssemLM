#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_ENTRY="$PROJECT_ROOT/API/AssemLM.py"

cd "$PROJECT_ROOT"
exec python "$API_ENTRY" \
    --model-path "$PROJECT_ROOT/models/AssemLM-V1" \
    "$@"