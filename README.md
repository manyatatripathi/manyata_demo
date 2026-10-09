# manyata_demo
my first git repository
<br>
Author- Manyata 

## Language model training environment

A Python setup for training a language model from scratch on an NVIDIA GPU, using
PyTorch (CUDA), Hugging Face `transformers`, `datasets` and `tokenizers`, plus `numpy`,
`matplotlib` and `tqdm`.

### Project structure

```
data/
  raw/            original text / downloaded datasets (not committed)
  processed/      tokenized, ready-to-train data (not committed)
models/
  checkpoints/    saved model weights during training (not committed)
  tokenizer/      trained tokenizer files (not committed)
scripts/
  device.py       get_device() / get_amp_dtype(): puts training on the GPU
  verify_cuda.py  checks the GPU, CUDA and every library
requirements.txt  libraries other than PyTorch
setup_env.sh      one-step setup for Linux / WSL
setup_env.ps1     one-step setup for Windows
```

### Requirements

- An NVIDIA GPU with an up-to-date driver. Check with `nvidia-smi`.
- Python 3.10 or newer.
- You don't need to install the CUDA toolkit. The PyTorch wheels bundle the CUDA runtime.

### Setup

Linux / WSL:

```bash
bash setup_env.sh
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
.venv\Scripts\Activate.ps1
```

Each script creates `.venv`, installs PyTorch with CUDA and the other libraries, then runs
`scripts/verify_cuda.py`. A working setup ends with something like:

```
CUDA available : yes (1 GPU(s))
  GPU 0: NVIDIA GeForce RTX ..., 12.0 GB, compute capability 8.6
Mixed precision: torch.bfloat16
GPU matmul test: OK
```

You can re-run the check at any time with `python scripts/verify_cuda.py`.

### Using the GPU in training scripts

```python
from device import get_device, get_amp_dtype

device = get_device(require_cuda=True)   # raises an error instead of silently using the CPU
model = model.to(device)

with torch.autocast(device_type="cuda", dtype=get_amp_dtype()):
    loss = model(**batch).loss
```
