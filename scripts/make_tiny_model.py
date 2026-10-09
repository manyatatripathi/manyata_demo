"""Build a tiny, randomly initialized chat model for CPU smoke tests (no download needed).

It checks that the data -> fine-tune -> evaluate pipeline runs end to end on a laptop.
Its answers are meaningless; use a real pretrained model for actual training.

    python scripts/make_tiny_model.py --data-dir data/samples --out models/tiny-test-model
"""

import argparse

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from data import load_splits

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/samples")
    p.add_argument("--out", default="models/tiny-test-model")
    args = p.parse_args()

    texts = [m["content"] for split in load_splits(args.data_dir).values() for ex in split for m in ex]
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=1000, special_tokens=SPECIAL_TOKENS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=CHAT_TEMPLATE
    )

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Tiny test model ({sum(p.numel() for p in model.parameters()):,} parameters) saved to {args.out}")


if __name__ == "__main__":
    main()
