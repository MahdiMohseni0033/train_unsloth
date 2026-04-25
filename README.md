# train_unsloth — QLoRA fine-tuning of Gemma-4 26B A4B with Unsloth

A small, self-contained tutorial project that takes you from a fresh ROCm
node to a fine-tuned Gemma-4 model and back to inference. Everything runs
through a handful of Python scripts and one setup shell script.

```text
.
├── setup.sh              # one-shot installer (Unsloth + ROCm PyTorch + deps)
├── prepare_dataset.py    # download yahma/alpaca-cleaned and format for Gemma-4
├── inspect_format.py     # show exactly what the model sees vs the loss target
├── finetune.py           # QLoRA SFT (TRL SFTTrainer + Unsloth FastModel)
├── plot_training.py      # render loss / lr / grad-norm PNGs
├── inference.py          # load the LoRA adapter and generate
├── evaluate.py           # compare generations against held-out ground truth
├── README.md             # this file (quickstart)
└── tutorial.md           # deep walkthrough: SFT, LoRA, prompts, common errors
```

`datasets/` and `outputs/` are git-ignored — they are produced on first run.

---

## 1. Requirements

- A machine with an **AMD ROCm GPU** (the project was developed on Kelvin2
  MI300X, 192 GB) — runs cleanly anywhere ROCm 6.3 + PyTorch 2.9.1 work.
- Python ≥ 3.11 (the installer can pin one with `--python 3.12`).
- Around 50 GB of free scratch for the Hugging Face cache + adapter outputs.
- An NVIDIA + CUDA build of PyTorch will also work; the ROCm-specific
  monkey-patches in `finetune.py` and `inference.py` are no-ops on CUDA.

The defaults assume `/mnt/scratch2/users/mmohseni` as the install root so
that nothing lands in your home quota. Override with `UNSLOTH_INSTALL_ROOT`
before running `./setup.sh`.

---

## 2. Install

Clone, then run the installer **on a GPU compute node** (not the login node):

```bash
git clone git@github.com:MahdiMohseni0033/train_unsloth.git
cd train_unsloth
./setup.sh                 # installs Unsloth Studio + ROCm PyTorch
```

The installer creates a virtual environment at:

```text
$UNSLOTH_INSTALL_ROOT/.unsloth/studio/unsloth_studio
```

For a Kelvin2 Slurm allocation use `srun` to run the installer on the
allocated node, for example:

```bash
srun --jobid <your_gpu_job_id> --overlap --pty bash
cd /path/to/train_unsloth
./setup.sh
```

---

## 3. Activate the environment

Activation is a single command — no helper script needed:

```bash
# Activate the Unsloth virtual environment + scratch HF/torch caches
export UNSLOTH_INSTALL_ROOT=/mnt/scratch2/users/mmohseni
export HF_HOME="$UNSLOTH_INSTALL_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$UNSLOTH_INSTALL_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$UNSLOTH_INSTALL_ROOT/.cache/triton"
source "$UNSLOTH_INSTALL_ROOT/.unsloth/studio/unsloth_studio/bin/activate"
```

If you keep the default install root the **shortest** form is just:

```bash
source /mnt/scratch2/users/mmohseni/.unsloth/studio/unsloth_studio/bin/activate
```

(The HF/torch cache exports are only needed if your `$HOME` is quota-limited.)

---

## 4. End-to-end run

Each script is independent and prints a one-line description of what it
does when called with `--help`.

```bash
# 1) Download Alpaca Cleaned and convert to a Gemma-4 chat-template SFT JSONL.
#    Runs on CPU, no GPU needed.
python prepare_dataset.py

# 2) Look at what the model will actually see during training.
#    Reads the tokenizer only — runs on CPU.
python inspect_format.py --num-examples 2

# 3) QLoRA fine-tune. Needs a GPU. Tutorial defaults are tiny on purpose
#    (~30 steps over 512 samples, ~1-2 min on MI300X).
python finetune.py

# 4) Plot training loss / learning rate / grad norm.
python plot_training.py
#    -> outputs/gemma4-26b-a4b-it-alpaca-lora/plots/{loss,lr,grad_norm}.png

# 5) Run inference with the saved adapter.
python inference.py --demo

# 6) Evaluate against held-out rows of the dataset (last N rows).
#    Add --include-base to compare against the un-fine-tuned model.
python evaluate.py --num-samples 5 --include-base
```

A typical smoke test (so you don't sit through the full default run):

```bash
python finetune.py --max-steps 2 --train-samples 16
python plot_training.py
python inference.py --demo --max-new-tokens 80
python evaluate.py --num-samples 2
```

---

## 5. Outputs

```text
outputs/gemma4-26b-a4b-it-alpaca-lora/
├── checkpoint-XX/               # intermediate Trainer checkpoints
├── final_adapter/               # LoRA weights + tokenizer (small, ~MB)
├── runs/<timestamp>/            # TensorBoard event files (HF Trainer default)
├── plots/{loss,lr,grad_norm}.png  # rendered by plot_training.py
├── trainer_state.json
└── sample_generation.txt
outputs/inference_samples.md     # appended by every inference.py run
outputs/evaluation_report.md     # written by evaluate.py
outputs/evaluation_metrics.json
```

The LoRA adapter is small (tens of MB) — it does **not** include the base
model weights. Combine the two at load time, exactly like `inference.py` does.

---

## 6. Where to read next

- `tutorial.md` — the conceptual walkthrough: what supervised fine-tuning is,
  what the Alpaca dataset looks like, why the chat template matters, why
  only the completion tokens contribute to the loss, what LoRA / QLoRA buy
  you, and how to read the curves `plot_training.py` produces.
- `inspect_format.py` (the script itself) — the most concrete answer to
  "what does the model actually see?". Run it once before training.

---

## 7. Verified versions (Kelvin2 MI300X)

```text
AMD Instinct MI300X
torch 2.9.1+rocm6.3
ROCm Toolkit 6.3.42134-a9a80e791
unsloth 2026.4.8
transformers 5.5.0
trl 0.24.0
datasets 4.3.0
triton 3.6.0
bitsandbytes 0.50.0.dev0
```

A 2-step / 16-sample smoke run uses ~47 GiB of GPU memory and finishes
training in roughly 20 s on a single MI300X.
