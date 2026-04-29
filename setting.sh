#!/usr/bin/env bash
set -e

# This is required to enable PEP 660 (editable install) support
python -m pip install --upgrade pip setuptools wheel

python -m pip install -e .

# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl