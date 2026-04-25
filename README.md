# Unsloth ROCm Install on Kelvin2

This directory contains a Kelvin2-specific Unsloth installer setup for AMD GPUs.
It keeps the virtual environment, Python package caches, Hugging Face model
cache, temp files, Torch cache, and Triton cache under:

```bash
/mnt/scratch2/users/mmohseni
```

Nothing should be installed under the quota-limited home directory.

## Files

- `install.sh` - patched Unsloth installer. It redirects install paths and caches
  to scratch and installs ROCm PyTorch on AMD GPU nodes.
- `activate.sh` - activates the installed Unsloth environment and exports the
  same scratch cache variables.
- `reinstall_unsloth_amd_kelvin2.sh` - one-command reinstall and verification
  wrapper for Kelvin2 AMD GPU jobs.
- `run_main_amd_kelvin2.sh` - runs `main.py` on an AMD GPU node and refuses to
  run if only a login node is visible.
- `prepare_alpaca_dataset.py` - downloads the Alpaca Cleaned dataset locally.
- `finetune_gemma4_alpaca.py` - tutorial QLoRA fine-tuning script.
- `run_finetune_alpaca_amd_kelvin2.sh` - Slurm-aware fine-tuning wrapper.
- `tutorial.md` - step-by-step fine-tuning tutorial.
- `main.py` - smoke test that loads `unsloth/gemma-4-26b-a4b-it` in 4-bit mode
  and applies a PEFT setup.

## Reinstall Everything

You must run the reinstall on an AMD GPU compute node. If you already have an
active Slurm GPU job, pass its job id:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
./reinstall_unsloth_amd_kelvin2.sh --jobid 8406919
```

For a future job, replace `8406919` with the current active Slurm job id. You
can also use an environment variable:

```bash
KELVIN_JOB_ID=8406919 ./reinstall_unsloth_amd_kelvin2.sh
```

If you are already inside an AMD GPU allocation, run:

```bash
./reinstall_unsloth_amd_kelvin2.sh
```

To reinstall packages and run only the ROCm smoke test, skipping the large model
load in `main.py`:

```bash
./reinstall_unsloth_amd_kelvin2.sh --jobid 8406919 --no-main
```

## Activate Later

After installation, activate the environment with:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
source ./activate.sh
```

The virtual environment is:

```bash
/mnt/scratch2/users/mmohseni/.unsloth/studio/unsloth_studio
```

Activating the environment on `login4` is useful for inspecting packages, but it
does not provide a GPU. Do not run `python main.py` directly from a login node.

## Run main.py

Run `main.py` through Slurm so the process lands on the AMD GPU compute node:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
./run_main_amd_kelvin2.sh --jobid 8406919
```

Replace `8406919` with the current active GPU job id.

If you are already inside an AMD GPU allocation, run:

```bash
./run_main_amd_kelvin2.sh
```

## Verification

The reinstall wrapper checks that:

- It is running on an AMD ROCm GPU node.
- No NVIDIA GPU is selected.
- PyTorch is a ROCm build.
- `torch.cuda.is_available()` is true through ROCm.
- The visible GPU can run a small matrix multiplication.
- `bitsandbytes` and `unsloth` import successfully.

Then, unless `--no-main` is used, it runs:

```bash
python main.py
```

The successful run on Kelvin2 used:

```text
AMD Instinct MI300X
torch 2.9.1+rocm6.3
ROCm Toolkit 6.3.42134-a9a80e791
bitsandbytes 0.50.0.dev0
unsloth 2026.4.8
transformers 5.5.0
trl 0.24.0
datasets 4.3.0
triton 3.6.0
```

The same MI300X node also runs a full Gemma-4 26B-A4B QLoRA fine-tune end
to end via `./run_finetune_alpaca_amd_kelvin2.sh` — see `tutorial.md`. A
2-step / 16-sample smoke run uses about 47 GiB of GPU memory and finishes
training in roughly 20 s.

## Logs

Each reinstall creates timestamped logs in this directory:

```text
install-YYYYMMDD_HHMMSS.log
gpu-check-YYYYMMDD_HHMMSS.log
main-YYYYMMDD_HHMMSS.log
```

The earlier successful logs are:

```text
install-rerun.log
main-run.log
```

## Run the Fine-Tune Tutorial

The full tutorial is in `tutorial.md`. To execute it end to end on Kelvin2:

```bash
cd /mnt/scratch2/users/mmohseni/projects/train_unsloth
./run_finetune_alpaca_amd_kelvin2.sh --jobid <active_gpu_job_id>
```

Replace `<active_gpu_job_id>` with the current Slurm AMD GPU job id. The
wrapper prepares the Alpaca dataset on the calling node and then runs
`finetune_gemma4_alpaca.py` on the GPU node. For a quick smoke test:

```bash
./run_finetune_alpaca_amd_kelvin2.sh --jobid <active_gpu_job_id> -- --max-steps 2 --train-samples 16
```

`finetune_gemma4_alpaca.py` ships with three Kelvin2-specific workarounds
for the current Unsloth + ROCm wheel combination (`dataset_num_proc=None`,
`UNSLOTH_COMPILE_DISABLE=1`, and a ROCm-aware patch to bypass
`torch._grouped_mm`). See `tutorial.md` "Common Errors" for details.

## Notes

The wrapper uses the patched installer default that skips Studio's optional
GGUF `llama.cpp` source build. This avoids failing the Python/ROCm training
environment when Unsloth does not publish a compatible Linux prebuilt for the
AMD node.

If you explicitly want to attempt the optional `llama.cpp` source build, run:

```bash
UNSLOTH_SKIP_GGUF_BUILD=0 ./reinstall_unsloth_amd_kelvin2.sh --jobid 8406919
```

That optional Studio component is not required for the `main.py` training smoke
test.


srun --jobid=8408951 --overlap --pty /bin/bash
# train_unsloth
