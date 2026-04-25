# Tutorial: Fine-Tune Gemma-4 26B A4B on Alpaca Cleaned

This tutorial shows a small supervised fine-tuning run using:

- Model: `unsloth/gemma-4-26b-a4b-it`
- Dataset: `yahma/alpaca-cleaned`
- Method: QLoRA / LoRA supervised fine-tuning
- GPU target: Kelvin2 AMD ROCm GPU node, for example MI300X

The default settings are intentionally small for a tutorial: 512 examples and
30 optimizer steps. Increase them after the first successful run.

A 2-step / 16-sample smoke run on the MI300X used 47.4 GiB of GPU memory and
finished training in ~20 s, so the defaults fit comfortably on a single
MI300X (192 GB).

## 1. Files Added

- `prepare_alpaca_dataset.py`
  Downloads `yahma/alpaca-cleaned`, writes the raw dataset locally, and creates
  a Gemma-4 prompt/completion JSONL file.

- `finetune_gemma4_alpaca.py`
  Loads the Gemma-4 model with Unsloth, attaches LoRA adapters, runs TRL
  `SFTTrainer`, saves the adapter, and writes a sample generation.

- `run_finetune_alpaca_amd_kelvin2.sh`
  Shell wrapper that prepares the dataset, attaches to a Slurm GPU job if
  needed, verifies AMD ROCm, and runs fine-tuning.

## 2. Dataset

Alpaca is a well-known instruction-following dataset format from the Stanford
Alpaca project. The cleaned Hugging Face version used here contains rows with:

- `instruction`
- `input`
- `output`

The preparation script writes:

```text
datasets/alpaca-cleaned/alpaca_cleaned.jsonl
datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl
datasets/alpaca-cleaned/preview.md
```

The raw file is for inspection. The `alpaca_gemma4_sft.jsonl` file is what the
trainer uses. Each row has:

```json
{"prompt": "<|turn>user\n...<turn|>\n<|turn>model\n", "completion": "...<turn|>\n"}
```

That format matches Gemma-4's chat template.

## 3. Prepare the Dataset Only

This does not need a GPU, so it can run on `login4`:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
source ./activate.sh
python prepare_alpaca_dataset.py
```

Or use the wrapper:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --prepare-only
```

Inspect the dataset:

```bash
less datasets/alpaca-cleaned/preview.md
head -n 3 datasets/alpaca-cleaned/alpaca_cleaned.jsonl
head -n 3 datasets/alpaca-cleaned/alpaca_gemma4_sft.jsonl
```

## 4. Run Fine-Tuning from login4

You must run training on an AMD GPU compute node, not on `login4`.

If you already have an active Slurm GPU job:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
./run_finetune_alpaca_amd_kelvin2.sh --jobid 8406919
```

Replace `8406919` with the current active GPU job id.

The wrapper first prepares the dataset on the current node, then attaches to
the Slurm job and runs training on the GPU node.

## 5. Run Fine-Tuning from Inside a GPU Allocation

If your shell is already on an AMD GPU node:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
./run_finetune_alpaca_amd_kelvin2.sh
```

## 6. Quick Test Run

For a very short smoke test:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --jobid 8406919 -- --max-steps 5 --train-samples 64
```

This checks that data loading, model loading, LoRA setup, training, and saving
all work.

## 7. Longer Tutorial Run

The default is:

```bash
--max-steps 30 --train-samples 512
```

To run a larger tutorial:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --jobid 8406919 -- --max-steps 100 --train-samples 2048
```

To use all downloaded Alpaca rows, pass `--train-samples 0`:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --jobid 8406919 -- --max-steps 500 --train-samples 0
```

## 8. Outputs

The default output directory is:

```text
outputs/gemma4-26b-a4b-it-alpaca-lora
```

Important files:

```text
outputs/gemma4-26b-a4b-it-alpaca-lora/final_adapter/
outputs/gemma4-26b-a4b-it-alpaca-lora/sample_generation.txt
```

`final_adapter/` contains the LoRA adapter and tokenizer files. It does not
duplicate the full base model weights.

Logs are written to:

```text
logs/prepare-alpaca-YYYYMMDD_HHMMSS.log
logs/finetune-alpaca-YYYYMMDD_HHMMSS.log
```

## 9. Important Settings

The fine-tuning script uses:

```text
load_in_4bit=True
LoRA r=16
LoRA alpha=16
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
batch_size=1
gradient_accumulation_steps=8
learning_rate=2e-4
bf16=True
```

These settings are conservative for a tutorial run on the MI300X node used
earlier. If you hit memory issues, reduce `--max-seq-length`, `--train-samples`,
or `--grad-accum`.

## 10. Common Errors

### `Unsloth cannot find any torch accelerator`

You are on a login node or a CPU-only shell. Use:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --jobid <active_gpu_job_id>
```

### Dataset file not found

Run:

```bash
python prepare_alpaca_dataset.py
```

### Slurm job expired

Get or start a new GPU job, then pass the new job id:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --jobid <new_gpu_job_id>
```

### Kelvin2 ROCm-specific workarounds (already applied in `finetune_gemma4_alpaca.py`)

The fine-tune script bakes in three patches that the current Unsloth +
torch 2.9.1+rocm6.3 + Triton 3.6.0 stack on this node requires. If you copy
the script to another file, keep them.

1. `dataset_num_proc=None` in `SFTConfig`. With `datasets >= 4.0`, any
   `num_proc >= 1` spawns a worker pool; the Unsloth-patched tokenizer
   captured by TRL's `tokenize_fn` carries a `torch._dynamo` `ConfigModuleInstance`
   that is not picklable, so the worker fork fails with
   `TypeError: cannot pickle 'ConfigModuleInstance' object`.
2. `os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"` before importing Unsloth.
   torch 2.9.1's inductor reads `binary.metadata.cluster_dims`, which the
   ROCm Triton 3.6.0 `KernelMetadata` does not expose, so any compiled kernel
   raises `AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`.
3. `transformers.integrations.moe._can_use_grouped_mm` is monkey-patched to
   return `False` on ROCm. `torch._grouped_mm` exists on this wheel but raises
   `RuntimeError: grouped gemm is not supported on ROCM` at runtime, so we
   route Gemma-4's MoE expert call through the python-side
   `transformers::grouped_mm_fallback`.

The post-training sample generation also passes the prompt as `text=prompt`
(not positionally) because Gemma-4's processor is multimodal — its first
positional argument is `images`.

## 11. Manual Commands

The wrapper is recommended, but the manual equivalent is:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
source ./activate.sh
python prepare_alpaca_dataset.py
srun --jobid=<active_gpu_job_id> --overlap bash -lc '
  cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
  source ./activate.sh
  python finetune_gemma4_alpaca.py --max-steps 30 --train-samples 512
'
```

Do not run `python finetune_gemma4_alpaca.py` directly from `login4`.
