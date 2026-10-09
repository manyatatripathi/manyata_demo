# Create .venv, install PyTorch with CUDA plus the training libraries, then verify the GPU.
# Usage (Windows PowerShell):  powershell -ExecutionPolicy Bypass -File setup_env.ps1
# To use a different CUDA build, e.g. cu126:  ... -File setup_env.ps1 -CudaTag cu126
param(
    [string]$CudaTag = "cu128"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} else {
    Write-Warning "nvidia-smi not found. Install the NVIDIA driver before training on the GPU."
}

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# On Windows, PyPI only has CPU builds of torch, so install from PyTorch's CUDA index.
# The CUDA runtime is bundled; only the NVIDIA driver is needed on the machine.
python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag"
python -m pip install -r requirements.txt

python scripts\verify_cuda.py
