"""Load the process-scheduling JSONL dataset and turn it into chat-formatted training examples.

Expected files (one JSON object per line):
    <data_dir>/train.jsonl
    <data_dir>/val.jsonl      (or validation.jsonl)
    <data_dir>/test.jsonl

Each line can use any one of these layouts:
    {"instruction": "...", "input": "...optional...", "output": "..."}
    {"prompt": "...", "completion": "..."}
    {"question": "...", "answer": "..."}
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}
Only the final assistant reply is trained on (prompt tokens are masked out of the loss).

Run this file directly to validate a dataset:
    python scripts/data.py --data-dir data/raw
"""

import argparse
import json
from pathlib import Path

SPLITS = ("train", "val", "test")
# Accepted file names for each split, in order of preference.
SPLIT_FILES = {"train": ["train.jsonl"], "val": ["val.jsonl", "validation.jsonl"], "test": ["test.jsonl"]}

SYSTEM_PROMPT = (
    "You are an expert in operating-system CPU process scheduling "
    "(FCFS, SJF, SRTF, Priority, Round Robin, multilevel queues). "
    "Show the schedule step by step and state the final answer clearly."
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
    return records


def to_messages(record: dict, system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Normalize one record into chat messages ending with the assistant answer."""
    if "messages" in record:
        messages = [dict(m) for m in record["messages"]]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("'messages' must end with an assistant message")
        if messages[0].get("role") != "system" and system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    if "instruction" in record and "output" in record:
        user = record["instruction"]
        if record.get("input"):
            user = f"{user}\n\n{record['input']}"
        answer = record["output"]
    elif "prompt" in record and "completion" in record:
        user, answer = record["prompt"], record["completion"]
    elif "question" in record and "answer" in record:
        user, answer = record["question"], record["answer"]
    else:
        raise ValueError(
            f"unrecognized record keys {sorted(record)}; expected instruction/output, "
            "prompt/completion, question/answer or messages"
        )

    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages += [{"role": "user", "content": str(user)}, {"role": "assistant", "content": str(answer)}]
    return messages


def split_prompt_answer(messages: list[dict]) -> tuple[list[dict], str]:
    """Return (conversation up to the final answer, final answer text)."""
    return messages[:-1], messages[-1]["content"]


def load_splits(data_dir: str | Path, splits=SPLITS) -> dict[str, list[list[dict]]]:
    data_dir = Path(data_dir)
    out = {}
    for split in splits:
        candidates = [data_dir / name for name in SPLIT_FILES[split]]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(f"missing {' or '.join(map(str, candidates))}")
        out[split] = []
        for i, record in enumerate(load_jsonl(path), start=1):
            try:
                out[split].append(to_messages(record))
            except ValueError as e:
                raise ValueError(f"{path}: record {i}: {e}") from e
    return out


def tokenize_example(messages: list[dict], tokenizer, max_length: int) -> dict | None:
    """Tokenize a conversation; labels are -100 everywhere except the final answer.

    Returns None if the example is longer than max_length (it is skipped rather than
    truncated, so the model never learns an answer without its question).
    """
    prompt_messages, answer = split_prompt_answer(messages)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the scheduling JSONL dataset.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model", help="also report token lengths using this model's tokenizer")
    parser.add_argument("--max-length", type=int, default=1024, help="length limit used with --model")
    args = parser.parse_args()

    splits = load_splits(args.data_dir)
    for split, examples in splits.items():
        print(f"{split:<5}: {len(examples)} examples")

    prompts = {split: {json.dumps(split_prompt_answer(m)[0]) for m in ex} for split, ex in splits.items()}
    for other in ("val", "test"):
        overlap = len(prompts["train"] & prompts[other])
        if overlap:
            print(f"WARNING: {overlap} {other} question(s) also appear in train")

    print("\nFirst training example:")
    for m in splits["train"][0]:
        print(f"[{m['role']}] {m['content']}\n")

    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print(f"Token lengths with {args.model} (prompt + answer):")
        for split, examples in splits.items():
            texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples]
            lengths = sorted(len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"])
            too_long = sum(n > args.max_length for n in lengths)
            print(f"  {split:<5}: median {lengths[len(lengths) // 2]}, max {lengths[-1]}, "
                  f"{too_long} over --max-length {args.max_length}")


if __name__ == "__main__":
    main()
