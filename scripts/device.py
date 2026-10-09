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
    """Mixed-precision dtype: bfloat16 on Ampere (compute capability 8.0) or newer, else float16.

    Checks the architecture rather than torch.cuda.is_bf16_supported(), which also reports
    True for emulated bf16 on older GPUs such as the T4.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16
