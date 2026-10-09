# manyata_demo
my first git repository
<br>
Author- Manyata 

## Fine-tuning an LLM for CPU process scheduling

This project fine-tunes a **pretrained** instruction-tuned language model with **LoRA**, so it
can answer CPU process-scheduling questions (FCFS, SJF, SRTF, Priority, Round Robin, ...).
It does not train a model from scratch. Libraries: PyTorch, Hugging Face `transformers`,
`datasets`, `tokenizers`, `peft` and `accelerate`.

### Project structure

```
data/
  raw/              put your dataset here: train.jsonl, val.jsonl, test.jsonl (not committed)
  processed/        free for derived data (not committed)
  samples/          small example dataset showing the format (committed)
models/
  checkpoints/      LoRA adapters and training checkpoints (not committed)
scripts/
  data.py           loads and validates the JSONL dataset, applies the chat template
  finetune_lora.py  LoRA fine-tuning of a pretrained model, saves the adapter
  evaluate.py       base model vs fine-tuned model on the unseen test split
  device.py         GPU selection and mixed-precision settings
  verify_cuda.py    reports library versions and whether a CUDA GPU is visible
  make_tiny_model.py  tiny random model for CPU smoke tests (no download needed)
requirements.txt    libraries other than PyTorch
setup_env.sh        setup for Linux / WSL / cloud GPU VM
setup_env.ps1       setup for Windows
```

### Hardware requirements

Fine-tuning a real pretrained model needs an NVIDIA GPU. A CPU-only laptop can prepare and
validate data and run the tiny smoke test below, but you should run real training on a cloud
GPU (Google Colab, Kaggle, Lambda, RunPod, AWS/GCP/Azure, ...).

| Base model | Method | GPU memory needed | Example cloud GPUs |
|---|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | LoRA | ~6–8 GB | T4 16 GB (free Colab / Kaggle) |
| **`Qwen/Qwen2.5-1.5B-Instruct`** (default) | LoRA | ~10–16 GB | T4 16 GB with `--gradient-checkpointing`; L4 / A10G 24 GB |
| `Qwen/Qwen2.5-3B-Instruct` | LoRA | ~18–24 GB | L4 / A10G 24 GB |
| `Qwen/Qwen2.5-7B-Instruct` | QLoRA (`--load-in-4bit`) | ~12–18 GB | L4 / A10G 24 GB, A100 |

These figures assume `--batch-size 4 --max-length 1024`. If you run out of memory, halve
`--batch-size`, double `--grad-accum` and/or add `--gradient-checkpointing`.

Why the default is `Qwen2.5-1.5B-Instruct`: it is Apache-2.0 licensed, already
instruction-tuned with a chat template, good at step-by-step arithmetic for its size, and LoRA
training fits on a single 16–24 GB GPU. A run over a few thousand examples takes minutes to
about an hour.

Other notes:
- The PyTorch build installed here uses CUDA 13.0, which needs an NVIDIA driver version 580
  or newer. Check the driver with `nvidia-smi`.
- You don't need to install the CUDA toolkit. The PyTorch wheel bundles the CUDA runtime.
- You need about 5 GB of disk for the environment plus the model download (about 3 GB for
  the 1.5B model).

### Dataset format

Put three JSONL files (one JSON object per line) in `data/raw/`:
`train.jsonl`, `val.jsonl` and `test.jsonl`. The test questions must not appear in train.
Each line can use any one of these layouts:

```json
{"instruction": "Schedule these processes using Round Robin (q=2)...", "input": "P1: arrival 0, burst 5\n...", "output": "Gantt chart: ...\nAverage waiting time = 4.33, average turnaround time = 8.00"}
{"prompt": "...", "completion": "..."}
{"question": "...", "answer": "..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The model learns only the answer text, because prompt tokens are masked out of the loss.
Put the final result (for example the average waiting time) on the **last line** of each
answer. `evaluate.py` scores answers by checking the numbers on that line. See
`data/samples/` for complete examples.

Validate your dataset. This prints split sizes, flags questions repeated across splits and
shows one formatted example:

```bash
python scripts/data.py --data-dir data/raw
```

### Setup

On your **CPU-only laptop** (data preparation and smoke tests):

```bash
bash setup_env.sh --cpu                                   # Linux / macOS / WSL
powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Cpu   # Windows
```

On the **cloud GPU machine**:

```bash
git clone https://github.com/manyatatripathi/manyata_demo.git
cd manyata_demo
bash setup_env.sh            # installs PyTorch with CUDA and checks the GPU
source .venv/bin/activate
python scripts/verify_cuda.py  # should print "CUDA available : yes"
```

Then copy your dataset to `data/raw/` on that machine (for example with `scp`, or by uploading
it in Colab).

On Google Colab, PyTorch with CUDA is already installed, so skip `setup_env.sh` and run:

```bash
!git clone https://github.com/manyatatripathi/manyata_demo.git
%cd manyata_demo
!pip install -r requirements.txt
!python scripts/verify_cuda.py
```

### 1. Smoke test on CPU (optional, no download)

This builds a tiny random model and runs the whole pipeline on the sample data in about a
minute. It only checks that the code runs; the scores are meaningless.

```bash
python scripts/make_tiny_model.py --data-dir data/samples --out models/tiny-test-model
python scripts/finetune_lora.py --model models/tiny-test-model --data-dir data/samples \
    --output-dir models/checkpoints/smoke-test --allow-cpu --epochs 1 --batch-size 2 --grad-accum 1
python scripts/evaluate.py --adapter models/checkpoints/smoke-test/adapter \
    --data-dir data/samples --allow-cpu --max-new-tokens 32
```

### 2. Fine-tune on the cloud GPU

```bash
python scripts/finetune_lora.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data-dir data/raw \
    --output-dir models/checkpoints/qwen2.5-1.5b-sched-lora \
    --epochs 3 --lr 2e-4 --batch-size 4 --grad-accum 4 --max-length 1024
```

On a T4 16 GB, add `--gradient-checkpointing`. For the 7B model with QLoRA, first run
`pip install bitsandbytes`, then use
`--model Qwen/Qwen2.5-7B-Instruct --load-in-4bit --gradient-checkpointing`.

The run evaluates on `val.jsonl` after every epoch and keeps the best epoch. It then saves the
LoRA adapter (tens of MB) to `models/checkpoints/qwen2.5-1.5b-sched-lora/adapter/`.
Examples longer than `--max-length` tokens are skipped with a warning.

Main options: `--lora-r` (default 16), `--lora-alpha` (32), `--lora-dropout` (0.05),
`--target-modules`. Run `python scripts/finetune_lora.py --help` for the full list.

### 3. Compare the base and fine-tuned models

```bash
python scripts/evaluate.py \
    --adapter models/checkpoints/qwen2.5-1.5b-sched-lora/adapter \
    --data-dir data/raw --split test
```

Both models answer every test question with greedy decoding. The script prints:

```
metric              base  fine-tuned
final_answer       0.xxx       0.xxx   # numbers on the reference's last line all match
token_f1           0.xxx       0.xxx   # word overlap with the reference answer
exact_match        0.xxx       0.xxx   # identical text (strict)
```

Every question's answers and scores are saved in `.../adapter/eval_test.jsonl`, and the
summary is saved in `.../adapter/eval_test_summary.json`. Use `--limit 50` for a quick check.

### 4. Keep the adapter

Instances such as Colab are temporary, so download the `adapter/` folder before the machine
shuts down. To use it later:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "models/checkpoints/qwen2.5-1.5b-sched-lora/adapter")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoints/qwen2.5-1.5b-sched-lora/adapter")
```
