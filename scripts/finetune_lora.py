"""LoRA fine-tuning of a pretrained chat model on the process-scheduling dataset.

Example (cloud GPU):
    python scripts/finetune_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data-dir data/raw \
        --output-dir models/checkpoints/qwen2.5-1.5b-sched-lora

Only the small LoRA adapter (a few MB to ~100 MB) is saved, not the full model.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from data import load_splits, tokenize_example
from device import get_amp_dtype, get_device

# Attention and MLP projections used by Llama / Qwen / Mistral style models.
DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model id or local path")
    p.add_argument("--data-dir", default="data/raw", help="folder with train.jsonl / val.jsonl / test.jsonl")
    p.add_argument("--output-dir", default="models/checkpoints/sched-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size x grad-accum")
    p.add_argument("--max-length", type=int, default=1024, help="max tokens per example")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA: 4-bit base model (needs bitsandbytes)")
    p.add_argument("--gradient-checkpointing", action="store_true", help="less GPU memory, ~30%% slower")
    p.add_argument("--allow-cpu", action="store_true", help="permit training without a GPU (smoke tests only)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(args, device: torch.device):
    use_cuda = device.type == "cuda"
    dtype = get_amp_dtype() if use_cuda else torch.float32
    kwargs = {"dtype": dtype}

    if args.load_in_4bit:
        if not use_cuda:
            raise SystemExit("--load-in-4bit needs a CUDA GPU")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    else:
        model.to(device)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, dtype


def main() -> None:
    args = parse_args()
    device = get_device(require_cuda=not args.allow_cpu)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise SystemExit(f"{args.model} has no chat template; use an instruct/chat model")

    splits = load_splits(args.data_dir, splits=("train", "val"))
    datasets = {}
    for name, examples in splits.items():
        tokenized = [tokenize_example(m, tokenizer, args.max_length) for m in examples]
        kept = [t for t in tokenized if t is not None]
        if len(kept) < len(tokenized):
            print(f"WARNING: skipped {len(tokenized) - len(kept)} {name} examples longer than --max-length {args.max_length}")
        if not kept:
            raise SystemExit(f"no {name} examples fit in --max-length {args.max_length}")
        datasets[name] = Dataset.from_list(kept)
    print(f"Train: {len(datasets['train'])} examples, val: {len(datasets['val'])} examples")

    model, dtype = load_model(args, device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # values < 1 are a fraction of total steps
        weight_decay=0.0,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=device.type == "cuda" and dtype == torch.bfloat16,
        fp16=device.type == "cuda" and dtype == torch.float16,
        use_cpu=device.type == "cpu",
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    # Save only the best LoRA adapter plus the tokenizer.
    adapter_dir = Path(args.output_dir) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics = trainer.evaluate()
    (adapter_dir / "training_info.json").write_text(
        json.dumps({"base_model": args.model, "args": vars(args), "final_eval": metrics}, indent=2)
    )
    print(f"Validation loss: {metrics['eval_loss']:.4f}")
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
