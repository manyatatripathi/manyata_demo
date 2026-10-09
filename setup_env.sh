#!/usr/bin/env bash
# Create .venv, install PyTorch plus the fine-tuning libraries, then check the GPU.
# Usage (Linux / WSL / cloud GPU VM):  bash setup_env.sh
#        CPU-only machine (~200 MB torch instead of ~3 GB):  bash setup_env.sh --cpu
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
CPU_ONLY=0
[[ "${1:-}" == "--cpu" ]] && CPU_ONLY=1

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
elif [[ $CPU_ONLY == 0 ]]; then
    echo "WARNING: nvidia-smi not found. On a machine without an NVIDIA GPU, use: bash setup_env.sh --cpu"
fi

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ $CPU_ONLY == 1 ]]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
    # On Linux x86_64 the PyPI torch wheel bundles the CUDA runtime,
    # so only the NVIDIA driver is needed on the machine (no CUDA toolkit).
    python -m pip install torch
fi
python -m pip install -r requirements.txt

if [[ $CPU_ONLY == 1 ]]; then
    python scripts/verify_cuda.py || echo "CPU-only setup: CUDA not expected here."
else
    python scripts/verify_cuda.py
fi
