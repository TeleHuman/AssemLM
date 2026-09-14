#!/usr/bin/env bash
set -euo pipefail

# Install the release in the currently selected Python environment.  The old
# script hard-coded a CUDA/Python/Torch-specific FlashAttention wheel, which
# could silently fail on a different machine.  FlashAttention is optional:
# the release uses eager attention by default and can be enabled explicitly.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${ASSEMLM_PYTHON:-python}

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -e "${SCRIPT_DIR}"

if [[ "${INSTALL_FLASH_ATTENTION:-false}" == "true" ]]; then
    "${PYTHON_BIN}" -m pip install flash-attn --no-build-isolation
fi
