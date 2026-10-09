"""Compare the base model and the LoRA fine-tuned model on unseen test questions.

Example:
    python scripts/evaluate.py --adapter models/checkpoints/qwen2.5-1.5b-sched-lora/adapter \
        --data-dir data/raw --split test

The base model id is read from the adapter folder, so --model is normally not needed.
Both models answer every question with greedy decoding; outputs and scores are written to
<adapter>/eval_<split>.jsonl, and a summary is printed.

Metrics (per question, averaged):
    final_answer  all numbers on the last line of the reference answer (e.g. average waiting
                  time, Gantt-chart order as process numbers) appear on the model's last line
    token_f1      word overlap with the reference answer (0-1)
    exact_match   normalized full-text equality (strict; usually low for long explanations)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_splits, split_prompt_answer
from device import get_amp_dtype, get_device

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w.\s]", " ", text.lower()).split())


def last_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def numbers(text: str) -> list[float]:
    return [round(float(n), 2) for n in NUMBER_RE.findall(text)]


def score(prediction: str, reference: str) -> dict:
    ref_nums = Counter(numbers(last_line(reference)))
    pred_nums = Counter(numbers(last_line(prediction)))
    final_answer = bool(ref_nums) and not (ref_nums - pred_nums)
    if not ref_nums:  # no numbers to check: fall back to the normalized last line
        final_answer = normalize(last_line(reference)) in normalize(prediction)

    pred_tokens, ref_tokens = normalize(prediction).split(), normalize(reference).split()
    common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    f1 = 0.0 if common == 0 else 2 * common / (len(pred_tokens) + len(ref_tokens))
    return {
        "final_answer": float(final_answer),
        "token_f1": f1,
        "exact_match": float(normalize(prediction) == normalize(reference)),
    }


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], device, batch_size: int, max_new_tokens: int) -> list[str]:
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), leave=False):
        batch = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = batch.to(device)
        generated = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generated[:, batch["input_ids"].shape[1] :]
        outputs += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [o.strip() for o in outputs]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="folder saved by finetune_lora.py (…/adapter)")
    p.add_argument("--model", help="base model id/path (default: read from the adapter)")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--limit", type=int, help="only evaluate the first N questions")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--allow-cpu", action="store_true", help="permit running without a GPU (slow)")
    args = p.parse_args()

    device = get_device(require_cuda=not args.allow_cpu)
    base_id = args.model or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    print(f"Device: {device}\nBase model: {base_id}\nAdapter: {args.adapter}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.padding_side = "left"  # required for batched generation with decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_amp_dtype() if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    examples = load_splits(args.data_dir, splits=(args.split,))[args.split][: args.limit]
    prompts, references = [], []
    for messages in examples:
        prompt_messages, answer = split_prompt_answer(messages)
        prompts.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
        references.append(answer)

    print(f"Generating {len(prompts)} answers with the base model...")
    with model.disable_adapter():
        base_outputs = generate(model, tokenizer, prompts, device, args.batch_size, args.max_new_tokens)
    print(f"Generating {len(prompts)} answers with the fine-tuned model...")
    tuned_outputs = generate(model, tokenizer, prompts, device, args.batch_size, args.max_new_tokens)

    totals = {"base": Counter(), "finetuned": Counter()}
    out_path = Path(args.adapter) / f"eval_{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for messages, ref, base_out, tuned_out in zip(examples, references, base_outputs, tuned_outputs):
            row = {
                "question": split_prompt_answer(messages)[0][-1]["content"],
                "reference": ref,
                "base": {"answer": base_out, **score(base_out, ref)},
                "finetuned": {"answer": tuned_out, **score(tuned_out, ref)},
            }
            for name in totals:
                totals[name].update({k: v for k, v in row[name].items() if k != "answer"})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(examples)
    print(f"\nResults on {n} {args.split} questions:")
    print(f"{'metric':<14}{'base':>10}{'fine-tuned':>12}")
    for metric in ("final_answer", "token_f1", "exact_match"):
        print(f"{metric:<14}{totals['base'][metric] / n:>10.3f}{totals['finetuned'][metric] / n:>12.3f}")
    summary = {name: {k: v / n for k, v in t.items()} for name, t in totals.items()}
    (Path(args.adapter) / f"eval_{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPer-question outputs: {out_path}")


if __name__ == "__main__":
    main()
