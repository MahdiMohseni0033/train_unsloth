#!/usr/bin/env python3
"""Run inference with the fine-tuned LoRA adapter.

Loads `unsloth/gemma-4-26b-a4b-it` in 4-bit, attaches the LoRA adapter
saved by `finetune.py`, formats one or more prompts using the same
Gemma-4 chat template the trainer used, and prints / saves the model's
response.

Examples:
    # Use one of the built-in demo prompts
    python inference.py --demo

    # Single prompt from CLI
    python inference.py --instruction "Explain backpropagation in one sentence."

    # Single prompt with input
    python inference.py \\
        --instruction "Translate the input to French." \\
        --input "Good morning, how are you?"

    # Batch of prompts from a JSON file ([{"instruction":..,"input":..}, ...])
    python inference.py --prompts-json my_prompts.json
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
from pathlib import Path

import unsloth  # noqa: F401  -- must come before transformers/peft
from unsloth import FastModel

import torch

if getattr(torch.version, "hip", None):
    import transformers.integrations.moe as _trf_moe

    def _can_use_grouped_mm_rocm(input, weight, offs):  # noqa: ARG001
        return False

    _trf_moe._can_use_grouped_mm = _can_use_grouped_mm_rocm


DEFAULT_MODEL = "unsloth/gemma-4-26b-a4b-it"
DEFAULT_ADAPTER = Path("outputs/gemma4-26b-a4b-it-alpaca-lora/final_adapter")

DEMO_PROMPTS = [
    {"instruction": "Give three tips for staying healthy.", "input": ""},
    {"instruction": "Translate the input to French.",
     "input": "Good morning, how are you today?"},
    {"instruction": "Write a short haiku about the ocean.", "input": ""},
]


def build_user_message(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def format_prompt(instruction: str, input_text: str) -> str:
    user_message = build_user_message(instruction, input_text)
    return f"<|turn>user\n{user_message}<turn|>\n<|turn>model\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER,
                   help="Path to the LoRA adapter saved by finetune.py. "
                        "Pass --no-adapter to compare against the base model.")
    p.add_argument("--no-adapter", action="store_true",
                   help="Skip loading the LoRA adapter (run the base model).")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--do-sample", action="store_true", default=True)
    p.add_argument("--greedy", action="store_true", help="Disable sampling.")
    p.add_argument("--seed", type=int, default=3407)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--demo", action="store_true",
                   help="Use the built-in demo prompts.")
    g.add_argument("--prompts-json", type=Path,
                   help="JSON list of {instruction, input} dicts.")
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--input", dest="input_text", type=str, default="")

    p.add_argument("--out-file", type=Path,
                   default=Path("outputs/inference_samples.md"),
                   help="Markdown file to append the prompt/response pairs to.")
    return p.parse_args()


def collect_prompts(args) -> list[dict]:
    if args.demo:
        return DEMO_PROMPTS
    if args.prompts_json is not None:
        with args.prompts_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    if args.instruction:
        return [{"instruction": args.instruction, "input": args.input_text}]
    return DEMO_PROMPTS  # fall back to demo


def main() -> None:
    args = parse_args()

    if not getattr(torch.version, "hip", None) and not torch.cuda.is_available():
        raise SystemExit("No GPU is visible. Run from a GPU compute node.")

    torch.manual_seed(args.seed)

    if args.no_adapter:
        print(f"Loading base model: {args.model_name}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )
    else:
        if not args.adapter_dir.exists():
            raise SystemExit(
                f"Adapter directory not found: {args.adapter_dir}\n"
                "Train first with `python finetune.py`, or pass --no-adapter."
            )
        # Point FastModel.from_pretrained at the adapter dir; Unsloth reads
        # adapter_config.json -> base_model_name_or_path, loads the base in
        # 4-bit, and attaches the LoRA adapter in one call. Going via
        # PeftModel/load_adapter doesn't work because Unsloth wraps the base
        # linear modules (e.g. Gemma4ClippableLinear) which PEFT does not
        # know how to inject into.
        print(f"Loading model + LoRA adapter from: {args.adapter_dir}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=str(args.adapter_dir),
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
        )

    if hasattr(model, "for_inference"):
        model.for_inference()
    else:
        model.eval()

    prompts = collect_prompts(args)
    print(f"Generating for {len(prompts)} prompt(s).")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with args.out_file.open("a", encoding="utf-8") as out:
        out.write("\n# Inference run\n\n")
        out.write(f"- model: `{args.model_name}`\n")
        out.write(f"- adapter: `{None if args.no_adapter else args.adapter_dir}`\n")
        out.write(f"- max_new_tokens: {args.max_new_tokens}\n\n")

        for i, item in enumerate(prompts, start=1):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            prompt_text = format_prompt(instruction, input_text)

            inputs = tokenizer(text=prompt_text, return_tensors="pt").to("cuda")
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=not args.greedy,
                )

            new_tokens = generated[0, prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            response = response.split("<turn|>")[0].strip()

            print()
            print("-" * 78)
            print(f"[{i}] INSTRUCTION: {instruction}")
            if input_text:
                print(f"    INPUT      : {input_text}")
            print(f"    RESPONSE   : {response}")

            out.write(f"## Prompt {i}\n\n")
            out.write(f"**Instruction:** {instruction}\n\n")
            out.write(f"**Input:** {input_text or '(empty)'}\n\n")
            out.write(f"**Response:** {response}\n\n")

    print()
    print(f"Saved transcript to: {args.out_file}")


if __name__ == "__main__":
    main()
