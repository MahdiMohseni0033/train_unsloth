#!/usr/bin/env python3
"""Download Alpaca Cleaned and convert it to a Gemma-4 SFT JSONL.

Reads from the Hugging Face hub (`yahma/alpaca-cleaned`), wraps each row in
the Gemma-4 chat template, and writes:

    datasets/alpaca-cleaned/alpaca_cleaned.jsonl       raw rows for inspection
    datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl    prompt/completion pairs
    datasets/alpaca-cleaned/preview.md                 human-readable preview

The training script (`finetune.py`) consumes the SFT JSONL.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

DEFAULT_DATASET = "yahma/alpaca-cleaned"
DEFAULT_OUT_DIR = Path("datasets/alpaca-cleaned")


def build_user_message(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def to_gemma4_pair(row: dict) -> dict:
    # The Gemma-4 chat template uses <|turn>role\n...<turn|>\n boundaries.
    # Splitting into prompt / completion lets TRL mask the prompt tokens so
    # the loss is only computed on the model's response.
    user_message = build_user_message(row.get("instruction", ""), row.get("input", ""))
    output = (row.get("output") or "").strip()
    return {
        "prompt": f"<|turn>user\n{user_message}<turn|>\n<|turn>model\n",
        "completion": f"{output}<turn|>\n",
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict], limit: int) -> None:
    lines = [
        "# Alpaca Cleaned Dataset Preview",
        "",
        f"Showing {min(limit, len(rows))} examples from `yahma/alpaca-cleaned`.",
        "",
    ]
    for index, row in enumerate(rows[:limit], start=1):
        lines.extend([
            f"## Example {index}",
            "",
            f"**Instruction:** {row.get('instruction', '').strip()}",
            "",
            f"**Input:** {row.get('input', '').strip() or '(empty)'}",
            "",
            f"**Output:** {row.get('output', '').strip()}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-records", type=int, default=0,
        help="Cap the number of rows. 0 means keep the whole split.",
    )
    parser.add_argument("--preview-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    if args.max_records > 0:
        dataset = dataset.select(range(min(args.max_records, len(dataset))))

    raw_rows = [dict(row) for row in dataset]
    sft_rows = [to_gemma4_pair(row) for row in raw_rows]

    raw_path = args.out_dir / "alpaca_cleaned.jsonl"
    sft_path = args.out_dir / "alpaca_gemma4_sft.jsonl"
    preview_path = args.out_dir / "preview.md"

    write_jsonl(raw_path, raw_rows)
    write_jsonl(sft_path, sft_rows)
    write_preview(preview_path, raw_rows, args.preview_examples)

    print(f"Dataset: {args.dataset}")
    print(f"Rows: {len(raw_rows)}")
    print(f"Raw JSONL:         {raw_path}")
    print(f"Gemma-4 SFT JSONL: {sft_path}")
    print(f"Preview:           {preview_path}")


if __name__ == "__main__":
    main()
