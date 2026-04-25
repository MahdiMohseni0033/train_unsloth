#!/usr/bin/env bash

UNSLOTH_INSTALL_ROOT="${UNSLOTH_INSTALL_ROOT:-/mnt/scratch2/users/mmohseni}"
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

mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_RUNTIME_DIR" "$TMPDIR" \
    "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" \
    "$HF_XET_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$WANDB_DIR"
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

export PATH="$UNSLOTH_INSTALL_ROOT/.local/bin:$UNSLOTH_INSTALL_ROOT/.unsloth/studio/unsloth_studio/bin:$PATH"

_UNSLOTH_VENV="$UNSLOTH_INSTALL_ROOT/.unsloth/studio/unsloth_studio"
if [ -f "$_UNSLOTH_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$_UNSLOTH_VENV/bin/activate"
else
    echo "Unsloth virtual environment not found at $_UNSLOTH_VENV" >&2
    return 1 2>/dev/null || exit 1
fi

_unsloth_has_amd_gpu() {
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

if ! _unsloth_has_amd_gpu; then
    echo "Unsloth environment activated, but no AMD ROCm GPU is visible in this shell." >&2
    echo "Run GPU scripts from a compute node, for example:" >&2
    echo "  ./run_main_amd_kelvin2.sh --jobid <active_gpu_job_id>" >&2
fi
unset -f _unsloth_has_amd_gpu
