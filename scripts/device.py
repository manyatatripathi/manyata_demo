"""Shared device setup: import this in training scripts so everything runs on the NVIDIA GPU."""

import torch


def get_device(require_cuda: bool = False) -> torch.device:
    """Return the CUDA device if one is available, otherwise the CPU.

    Pass require_cuda=True to fail loudly instead of silently training on CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Faster float32 matmuls on Ampere (RTX 30xx) and newer GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return device
    if require_cuda:
        raise RuntimeError(
            "CUDA is not available. Check the NVIDIA driver (`nvidia-smi`) and that "
            "a CUDA build of PyTorch is installed (run scripts/verify_cuda.py)."
        )
    return torch.device("cpu")


def get_amp_dtype() -> torch.dtype:
    """Mixed-precision dtype: bfloat16 where the GPU supports it, else float16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
