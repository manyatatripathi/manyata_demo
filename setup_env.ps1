# Create .venv, install PyTorch plus the fine-tuning libraries, then check the GPU.
# Usage (Windows PowerShell):  powershell -ExecutionPolicy Bypass -File setup_env.ps1
#   CPU-only laptop:            powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Cpu
#   Different CUDA build:       powershell -ExecutionPolicy Bypass -File setup_env.ps1 -CudaTag cu128
param(
    [string]$CudaTag = "cu130",
    [switch]$Cpu
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} elseif (-not $Cpu) {
    Write-Warning "nvidia-smi not found. On a machine without an NVIDIA GPU, add -Cpu."
}

python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

if ($Cpu) {
    # PyPI's Windows torch wheel is the CPU-only build.
    python -m pip install torch
} else {
    # PyPI only has CPU builds of torch for Windows, so install from PyTorch's CUDA index.
    # The CUDA runtime is bundled; only the NVIDIA driver is needed on the machine.
    python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag"
}
python -m pip install -r requirements.txt

python scripts\verify_cuda.py
if ($Cpu -and $LASTEXITCODE -ne 0) { Write-Host "CPU-only setup: CUDA not expected here." }
