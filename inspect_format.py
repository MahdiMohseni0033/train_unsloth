#!/usr/bin/env python3
"""Show exactly what the model sees during training.

For a few rows of the prepared SFT dataset, this prints:

  1. The raw Alpaca row (instruction / input / output).
  2. The prompt + completion strings the trainer concatenates.
  3. The full token sequence with each token's contribution to the loss
     marked: tokens from the prompt are *masked* (no gradient), tokens from
     the completion are *targets* (the model is graded on predicting them).

This is the file to read first if you have never done supervised fine-tuning
before. The whole point of SFT with `completion_only_loss=True` is that the
prompt is "given" to the model and only the response is being learned.

Runs on CPU. No GPU needed.
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_MODEL = "unsloth/gemma-4-26b-a4b-it"
DEFAULT_DATASET = Path("datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl")
DEFAULT_RAW = Path("datasets/alpaca-cleaned/alpaca_cleaned.jsonl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW,
                   help="Raw alpaca rows (for showing the original instruction/input/output).")
    p.add_argument("--num-examples", type=int, default=2)
    p.add_argument("--show-token-table", action="store_true",
                   help="Print every token with its label (lots of output).")
    return p.parse_args()


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def show_example(idx: int, raw: dict, sft: dict, tokenizer, show_table: bool) -> None:
    banner(f"Example {idx + 1}")

    print("\n[1] RAW ALPACA ROW (yahma/alpaca-cleaned)")
    print(f"  instruction: {raw.get('instruction', '').strip()!r}")
    print(f"  input      : {raw.get('input', '').strip()!r}")
    print(f"  output     : {raw.get('output', '').strip()!r}")

    print("\n[2] SFT PAIR PASSED TO TRL (after prepare_dataset.py)")
    print("    -- prompt (will be loss-masked) --")
    print(repr(sft["prompt"]))
    print("    -- completion (model learns to produce this) --")
    print(repr(sft["completion"]))

    # Tokenize the way SFTTrainer does with completion_only_loss=True:
    # concatenate prompt + completion, then mark tokens belonging to the
    # prompt with label=-100 so they do not contribute to the loss.
    prompt_ids = tokenizer(sft["prompt"], add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(sft["prompt"] + sft["completion"], add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    n_full = len(full_ids)
    n_completion = n_full - n_prompt
    labels = [-100] * n_prompt + full_ids[n_prompt:]

    print("\n[3] TOKEN COUNTS")
    print(f"  prompt tokens     : {n_prompt:>5}  (loss-masked, label = -100)")
    print(f"  completion tokens : {n_completion:>5}  (loss target)")
    print(f"  total tokens      : {n_full:>5}")

    print("\n[4] DECODED FULL INPUT (what the model actually receives)")
    print(repr(tokenizer.decode(full_ids, skip_special_tokens=False)))

    print("\n[5] LOSS TARGET ONLY (what the model is graded on producing)")
    target_ids = [t for t in labels if t != -100]
    print(repr(tokenizer.decode(target_ids, skip_special_tokens=False)))

    if show_table:
        print("\n[6] PER-TOKEN TABLE  (* = contributes to loss)")
        print(f"  {'idx':>4}  {'token_id':>8}  {'loss':>4}  token")
        for i, (tok_id, lab) in enumerate(zip(full_ids, labels)):
            mark = "*" if lab != -100 else " "
            piece = tokenizer.decode([tok_id], skip_special_tokens=False)
            print(f"  {i:>4}  {tok_id:>8}  {mark:>4}  {piece!r}")


def main() -> None:
    args = parse_args()

    if not args.dataset_jsonl.exists():
        raise SystemExit(
            f"SFT dataset not found: {args.dataset_jsonl}\n"
            "Run `python prepare_dataset.py` first."
        )
    if not args.raw_jsonl.exists():
        raise SystemExit(
            f"Raw dataset not found: {args.raw_jsonl}\n"
            "Run `python prepare_dataset.py` first."
        )

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    raw_rows = read_jsonl(args.raw_jsonl, args.num_examples)
    sft_rows = read_jsonl(args.dataset_jsonl, args.num_examples)

    for i, (raw, sft) in enumerate(zip(raw_rows, sft_rows)):
        show_example(i, raw, sft, tokenizer, show_table=args.show_token_table)

    banner("Summary")
    print("Each training step computes cross-entropy loss only over the")
    print("completion tokens. The prompt tokens are part of the *context*")
    print("but the model is never penalised for what it would have predicted")
    print("there. This is what `completion_only_loss=True` does in SFTConfig.")


if __name__ == "__main__":
    main()
