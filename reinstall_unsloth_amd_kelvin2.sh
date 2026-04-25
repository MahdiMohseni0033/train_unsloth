#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNSLOTH_INSTALL_ROOT="${UNSLOTH_INSTALL_ROOT:-/mnt/scratch2/users/mmohseni}"
JOB_ID="${KELVIN_JOB_ID:-}"
RUN_MAIN=1
INSIDE_SLURM=0
PYTHON_VERSION=""

usage() {
    cat <<'USAGE'
Usage:
  ./reinstall_unsloth_amd_kelvin2.sh --jobid <SLURM_JOB_ID>
  ./reinstall_unsloth_amd_kelvin2.sh --no-main --jobid <SLURM_JOB_ID>

Options:
  --jobid ID       Attach to an existing Kelvin2 Slurm GPU job.
  --no-main        Install and run the ROCm smoke test, but skip main.py.
  --python VER     Pass a Python version to install.sh, for example 3.13.
  --inside-slurm   Internal flag used after srun attaches to the job.
  -h, --help       Show this help.

Environment:
  KELVIN_JOB_ID              Alternative to --jobid.
  UNSLOTH_INSTALL_ROOT       Defaults to /mnt/scratch2/users/mmohseni.
  UNSLOTH_SKIP_GGUF_BUILD    Defaults to 1 in install.sh. Set to 0 to attempt
                             Studio's optional llama.cpp source build.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --jobid)
            [ "$#" -ge 2 ] || { echo "ERROR: --jobid requires a value" >&2; exit 2; }
            JOB_ID="$2"
            shift 2
            ;;
        --no-main)
            RUN_MAIN=0
            shift
            ;;
        --python)
            [ "$#" -ge 2 ] || { echo "ERROR: --python requires a value" >&2; exit 2; }
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --inside-slurm)
            INSIDE_SLURM=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

export UNSLOTH_INSTALL_ROOT
export HOME="$UNSLOTH_INSTALL_ROOT"
export XDG_CACHE_HOME="$UNSLOTH_INSTALL_ROOT/.cache"
export XDG_CONFIG_HOME="$UNSLOTH_INSTALL_ROOT/.config"
export XDG_DATA_HOME="$UNSLOTH_INSTALL_ROOT/.local/share"
export XDG_RUNTIME_DIR="$UNSLOTH_INSTALL_ROOT/.run"
export TMPDIR="$UNSLOTH_INSTALL_ROOT/.tmp"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export WANDB_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export PATH="$UNSLOTH_INSTALL_ROOT/.local/bin:$PATH"

mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_RUNTIME_DIR" "$TMPDIR" \
    "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$WANDB_DIR"
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

has_amd_rocm_gpu() {
    if command -v rocminfo >/dev/null 2>&1 && \
       rocminfo 2>/dev/null | awk '/Name:[[:space:]]*gfx[0-9]/ && !/Name:[[:space:]]*gfx000/{found=1} END{exit !found}'; then
        return 0
    fi
    if command -v amd-smi >/dev/null 2>&1 && \
       amd-smi list 2>/dev/null | awk '/^GPU[[:space:]]*[:\[][[:space:]]*[0-9]/{found=1} END{exit !found}'; then
        return 0
    fi
    return 1
}

if [ "$INSIDE_SLURM" -ne 1 ] && ! has_amd_rocm_gpu; then
    if [ -n "$JOB_ID" ]; then
        forwarded=(--inside-slurm)
        [ "$RUN_MAIN" -eq 0 ] && forwarded+=(--no-main)
        [ -n "$PYTHON_VERSION" ] && forwarded+=(--python "$PYTHON_VERSION")
        echo "Attaching to Slurm job $JOB_ID and running the installer on the GPU node..."
        exec srun --jobid="$JOB_ID" --overlap bash "$SCRIPT_PATH" "${forwarded[@]}"
    fi

    echo "ERROR: no AMD ROCm GPU detected in this shell." >&2
    echo "Run this script on a Kelvin2 AMD GPU node, or pass --jobid <active_job_id>." >&2
    exit 1
fi

if ! has_amd_rocm_gpu; then
    echo "ERROR: attached shell still does not expose an AMD ROCm GPU." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU detected; this script is intended for AMD ROCm only." >&2
    exit 1
fi

cd "$PROJECT_DIR"

INSTALL_ARGS=(--verbose)
[ -n "$PYTHON_VERSION" ] && INSTALL_ARGS+=(--python "$PYTHON_VERSION")

timestamp="$(date +%Y%m%d_%H%M%S)"
INSTALL_LOG="$PROJECT_DIR/install-${timestamp}.log"
GPU_LOG="$PROJECT_DIR/gpu-check-${timestamp}.log"
MAIN_LOG="$PROJECT_DIR/main-${timestamp}.log"

echo "Host: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Install root: $UNSLOTH_INSTALL_ROOT"
echo "Install log: $INSTALL_LOG"

set -o pipefail
"$PROJECT_DIR/install.sh" "${INSTALL_ARGS[@]}" 2>&1 | tee "$INSTALL_LOG"

source "$PROJECT_DIR/activate.sh"

python - <<'PY' 2>&1 | tee "$GPU_LOG"
import os
import torch

print("VIRTUAL_ENV", os.environ.get("VIRTUAL_ENV"))
print("HOME", os.environ.get("HOME"))
print("HF_HOME", os.environ.get("HF_HOME"))
print("torch", torch.__version__)
print("torch_hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if not getattr(torch.version, "hip", None):
    raise SystemExit("ROCm torch is not installed")
if not torch.cuda.is_available():
    raise SystemExit("ROCm/CUDA API is not available")
if torch.cuda.device_count() < 1:
    raise SystemExit("No AMD GPU visible to torch")

print("device_name", torch.cuda.get_device_name(0))
x = torch.ones((64, 64), device="cuda")
print("matmul_sum", float((x @ x).sum().item()))

import bitsandbytes as bnb
import unsloth

print("bitsandbytes", getattr(bnb, "__version__", "unknown"))
print("unsloth", getattr(unsloth, "__version__", "unknown"))
PY

if [ "$RUN_MAIN" -eq 1 ]; then
    echo "Running main.py; log: $MAIN_LOG"
    python "$PROJECT_DIR/main.py" 2>&1 | tee "$MAIN_LOG"
else
    echo "Skipping main.py because --no-main was provided."
fi

echo "Done."
echo "Install log: $INSTALL_LOG"
echo "GPU check log: $GPU_LOG"
[ "$RUN_MAIN" -eq 1 ] && echo "main.py log: $MAIN_LOG"
