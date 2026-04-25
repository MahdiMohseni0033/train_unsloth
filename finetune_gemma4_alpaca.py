#!/usr/bin/env python3
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# torch 2.9.1+rocm6.3 ships an inductor that reads `binary.metadata.cluster_dims`,
# but the ROCm Triton 3.6.0 KernelMetadata does not expose that field, so any
# torch.compile-generated kernel raises AttributeError at launch time. Until a
# matched torch/triton-rocm pair lands, run the tutorial in eager mode.
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
from pathlib import Path

# Unsloth must be imported before trl/transformers/peft so its patches apply.
import unsloth  # noqa: F401
from unsloth import FastModel

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# torch._grouped_mm exists on ROCm wheels but raises "grouped gemm is not
# supported on ROCM" at runtime, so transformers' MoE path crashes on Gemma-4.
# Force the python-side fallback (transformers::grouped_mm_fallback) instead.
if getattr(torch.version, "hip", None):
    import transformers.integrations.moe as _trf_moe

    def _can_use_grouped_mm_rocm(input, weight, offs):  # noqa: ARG001
        return False

    _trf_moe._can_use_grouped_mm = _can_use_grouped_mm_rocm


DEFAULT_MODEL = "unsloth/gemma-4-26b-a4b-it"
DEFAULT_DATASET = Path("datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl")
DEFAULT_OUTPUT = Path("outputs/gemma4-26b-a4b-it-alpaca-lora")


def require_rocm_gpu() -> None:
    print("torch", torch.__version__)
    print("torch_hip", getattr(torch.version, "hip", None))
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if not getattr(torch.version, "hip", None):
        raise SystemExit("ROCm torch is not installed. Run this on an AMD GPU node.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("No AMD GPU is visible. Do not run this from a login node.")
    print("device_name", torch.cuda.get_device_name(0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tutorial QLoRA fine-tuning for Gemma-4 26B A4B on Alpaca Cleaned."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_rocm_gpu()

    if not args.dataset_jsonl.exists():
        raise SystemExit(
            f"Dataset file not found: {args.dataset_jsonl}\n"
            "Run ./prepare_alpaca_dataset.py first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("json", data_files=str(args.dataset_jsonl), split="train")
    if args.train_samples > 0:
        dataset = dataset.select(range(min(args.train_samples, len(dataset))))

    print(f"Training rows: {len(dataset)}")
    print(f"Output directory: {args.output_dir}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        # Keep tokenization in-process. datasets 4.x spawns a pool whenever
        # num_proc >= 1, and the Unsloth-patched tokenizer captured by TRL's
        # tokenize_fn carries a torch._dynamo ConfigModuleInstance that is not
        # picklable, so any worker fork raises TypeError.
        dataset_num_proc=None,
        seed=args.seed,
        bf16=True,
        fp16=False,
        report_to="none",
        completion_only_loss=True,
        packing=False,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    adapter_dir = args.output_dir / "final_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter and tokenizer to: {adapter_dir}")

    if hasattr(model, "for_inference"):
        model.for_inference()
    else:
        model.eval()
    sample_prompt = dataset[0]["prompt"]
    # Gemma-4 ships a multimodal processor whose first positional argument is
    # `images`, so the prompt has to be passed as `text=` explicitly.
    inputs = tokenizer(text=sample_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=False)
    sample_path = args.output_dir / "sample_generation.txt"
    sample_path.write_text(decoded, encoding="utf-8")
    print(f"Saved sample generation to: {sample_path}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak allocated GPU memory: {peak_gb:.2f} GiB")


if __name__ == "__main__":
    main()
