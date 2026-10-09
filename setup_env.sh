#!/usr/bin/env bash
# Create .venv, install PyTorch with CUDA plus the training libraries, then verify the GPU.
# Usage (Linux / WSL):  bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "WARNING: nvidia-smi not found. Install the NVIDIA driver before training on the GPU."
fi

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

# On Linux x86_64 the PyPI torch wheel already bundles the CUDA runtime,
# so only the NVIDIA driver is needed on the machine (no CUDA toolkit).
python -m pip install torch
python -m pip install -r requirements.txt

python scripts/verify_cuda.py
