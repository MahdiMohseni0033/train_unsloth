#!/usr/bin/env python3
"""Compare the fine-tuned model's outputs against the dataset ground truth.

Holds out the LAST `--num-samples` rows of the SFT dataset (so they were
*not* used for training under the default `--train-samples` cap), generates
a response for each prompt, and writes a side-by-side report:

    outputs/evaluation_report.md   markdown table of prompt | ground truth | generated
    outputs/evaluation_metrics.json  exact-match + char/token overlap numbers

Optionally also runs the *base* model (no LoRA) on the same prompts so you
can see how much fine-tuning shifted the outputs.

Examples:
    python evaluate.py --num-samples 5
    python evaluate.py --num-samples 10 --include-base
    python evaluate.py --num-samples 5 --no-fine-tuned --include-base
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import gc
import json
from pathlib import Path

import unsloth  # noqa: F401
from unsloth import FastModel

import torch

if getattr(torch.version, "hip", None):
    import transformers.integrations.moe as _trf_moe

    def _can_use_grouped_mm_rocm(input, weight, offs):  # noqa: ARG001
        return False

    _trf_moe._can_use_grouped_mm = _can_use_grouped_mm_rocm


DEFAULT_MODEL = "unsloth/gemma-4-26b-a4b-it"
DEFAULT_DATASET = Path("datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl")
DEFAULT_RAW = Path("datasets/alpaca-cleaned/alpaca_cleaned.jsonl")
DEFAULT_ADAPTER = Path("outputs/gemma4-26b-a4b-it-alpaca-lora/final_adapter")
DEFAULT_OUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--raw-jsonl", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--num-samples", type=int, default=5,
                   help="Hold out this many rows from the END of the dataset.")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--no-fine-tuned", action="store_true",
                   help="Skip the fine-tuned model run.")
    p.add_argument("--include-base", action="store_true",
                   help="Also generate with the un-fine-tuned base model.")
    p.add_argument("--greedy", action="store_true",
                   help="Disable sampling (deterministic for comparison).")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_holdout(args) -> list[dict]:
    sft = read_jsonl(args.dataset_jsonl)
    raw = read_jsonl(args.raw_jsonl)
    if len(sft) != len(raw):
        # The two files came from the same prepare_dataset.py call, so
        # this should never happen unless the user re-prepared one.
        raise SystemExit("SFT JSONL and raw JSONL have different lengths. "
                         "Re-run `python prepare_dataset.py`.")
    n = min(args.num_samples, len(sft))
    return [
        {
            "instruction": raw[-n + i].get("instruction", "").strip(),
            "input":       raw[-n + i].get("input", "").strip(),
            "ground_truth": raw[-n + i].get("output", "").strip(),
            "prompt":       sft[-n + i]["prompt"],
            "completion":   sft[-n + i]["completion"],
        }
        for i in range(n)
    ]


def generate_for_rows(model, tokenizer, rows: list[dict], *,
                      max_new_tokens: int, do_sample: bool) -> list[str]:
    if hasattr(model, "for_inference"):
        model.for_inference()
    else:
        model.eval()
    outputs: list[str] = []
    for row in rows:
        inputs = tokenizer(text=row["prompt"], return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=do_sample,
            )
        new_tokens = generated[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = text.split("<turn|>")[0].strip()
        outputs.append(text)
    return outputs


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Cheap, dependency-free metrics. Real eval should use ROUGE/BERTScore."""
    if not predictions:
        return {}
    n = len(predictions)
    exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    char_overlap = []
    word_overlap = []
    for p, r in zip(predictions, references):
        p_set, r_set = set(p.lower()), set(r.lower())
        if r_set:
            char_overlap.append(len(p_set & r_set) / len(r_set))
        p_words, r_words = set(p.lower().split()), set(r.lower().split())
        if r_words:
            word_overlap.append(len(p_words & r_words) / len(r_words))
    return {
        "n": n,
        "exact_match": exact / n,
        "avg_char_recall": sum(char_overlap) / len(char_overlap) if char_overlap else 0.0,
        "avg_word_recall": sum(word_overlap) / len(word_overlap) if word_overlap else 0.0,
    }


def write_report(out_path: Path, rows: list[dict], ft_outputs, base_outputs) -> None:
    lines = [
        "# Evaluation report",
        "",
        f"Held-out rows: **{len(rows)}** (taken from the END of the SFT dataset).",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        lines += [
            f"## Sample {i}",
            "",
            f"**Instruction:** {row['instruction']}",
            "",
            f"**Input:** {row['input'] or '(empty)'}",
            "",
            f"**Ground truth:**",
            "",
            f"> {row['ground_truth'].replace(chr(10), ' ')}",
            "",
        ]
        if ft_outputs is not None:
            lines += [
                "**Fine-tuned model:**",
                "",
                f"> {ft_outputs[i-1].replace(chr(10), ' ')}",
                "",
            ]
        if base_outputs is not None:
            lines += [
                "**Base model:**",
                "",
                f"> {base_outputs[i-1].replace(chr(10), ' ')}",
                "",
            ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.dataset_jsonl.exists() or not args.raw_jsonl.exists():
        raise SystemExit("Dataset JSONL files not found. "
                         "Run `python prepare_dataset.py` first.")
    if not args.no_fine_tuned and not args.adapter_dir.exists():
        raise SystemExit(f"Adapter not found: {args.adapter_dir}\n"
                         "Train first with `python finetune.py`, "
                         "or pass --no-fine-tuned to evaluate the base model only.")
    if args.no_fine_tuned and not args.include_base:
        raise SystemExit("Nothing to evaluate. Pass --include-base to evaluate the base model.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("No GPU is visible. Run from a GPU compute node.")
    torch.manual_seed(args.seed)

    rows = load_holdout(args)
    print(f"Held out {len(rows)} rows for evaluation.")

    ft_outputs = None
    base_outputs = None
    metrics: dict = {}

    if not args.no_fine_tuned:
        # Pointing FastModel.from_pretrained at the adapter dir loads the
        # base model + attaches the LoRA in one call. See inference.py for
        # why we don't use model.load_adapter() here.
        print(f"Loading fine-tuned model from: {args.adapter_dir}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(args.adapter_dir),
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        ft_outputs = generate_for_rows(
            model, tokenizer, rows,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
        )
        metrics["fine_tuned"] = compute_metrics(ft_outputs, [r["ground_truth"] for r in rows])
        # Free the model before loading the base.
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if args.include_base:
        print(f"Loading base model (no adapter): {args.model_name}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
        base_outputs = generate_for_rows(
            model, tokenizer, rows,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
        )
        metrics["base"] = compute_metrics(base_outputs, [r["ground_truth"] for r in rows])

    report_path = args.out_dir / "evaluation_report.md"
    metrics_path = args.out_dir / "evaluation_metrics.json"
    write_report(report_path, rows, ft_outputs, base_outputs)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    print(json.dumps(metrics, indent=2))
    print()
    print(f"Wrote report:  {report_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
