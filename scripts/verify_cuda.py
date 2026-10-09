"""Check that PyTorch sees the NVIDIA GPU and that every library imports.

Run from the project root:  python scripts/verify_cuda.py
Exits with code 1 if CUDA is not available.
"""

import importlib
import platform
import sys

import torch

from device import get_amp_dtype, get_device


def main() -> int:
    print(f"Python         : {sys.version.split()[0]} ({platform.system()})")

    for name in ["torch", "transformers", "datasets", "tokenizers", "numpy", "matplotlib", "tqdm"]:
        module = importlib.import_module(name)
        print(f"{name:<15}: {getattr(module, '__version__', 'unknown')}")

    print(f"PyTorch CUDA   : {torch.version.cuda or 'none (CPU-only build)'}")
    print(f"cuDNN          : {torch.backends.cudnn.version()}")

    if not torch.cuda.is_available():
        print("\nCUDA available : NO")
        if torch.version.cuda is None:
            print("-> This is a CPU-only PyTorch build. Re-run the setup script to install the CUDA build.")
        else:
            print("-> PyTorch has CUDA support but no GPU was found. Check the NVIDIA driver with `nvidia-smi`.")
        return 1

    print(f"\nCUDA available : yes ({torch.cuda.device_count()} GPU(s))")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB, "
              f"compute capability {props.major}.{props.minor}")

    device = get_device(require_cuda=True)
    amp_dtype = get_amp_dtype()
    print(f"Mixed precision: {amp_dtype}")

    # Run a real computation on the GPU and compare it with the CPU result.
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    gpu_result = (a.to(device) @ b.to(device)).cpu()
    max_diff = (gpu_result - a @ b).abs().max().item()
    print(f"GPU matmul test: OK (max diff vs CPU {max_diff:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
