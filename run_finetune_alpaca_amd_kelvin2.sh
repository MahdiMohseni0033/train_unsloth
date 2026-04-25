#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ID="${KELVIN_JOB_ID:-}"
INSIDE_SLURM=0
PREPARE_ONLY=0
SKIP_PREPARE=0
TRAIN_ARGS=()

usage() {
    cat <<'USAGE'
Usage:
  ./run_finetune_alpaca_amd_kelvin2.sh --jobid <SLURM_JOB_ID>
  ./run_finetune_alpaca_amd_kelvin2.sh --jobid <SLURM_JOB_ID> -- --max-steps 60 --train-samples 1024
  ./run_finetune_alpaca_amd_kelvin2.sh --prepare-only

Options:
  --jobid ID       Attach to an existing Kelvin2 Slurm AMD GPU job.
  --prepare-only   Download/prepare the dataset only. No GPU required.
  --inside-slurm   Internal flag used after srun attaches to the job.
  --skip-prepare   Internal flag used after dataset prep has already run.
  -h, --help       Show this help.

Everything after "--" is passed to finetune_gemma4_alpaca.py.

Useful training overrides:
  -- --max-steps 10
  -- --train-samples 128
  -- --output-dir outputs/my-alpaca-run

Environment:
  KELVIN_JOB_ID    Alternative to --jobid.
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --jobid)
            [ "$#" -ge 2 ] || { echo "ERROR: --jobid requires a value" >&2; exit 2; }
            JOB_ID="$2"
            shift 2
            ;;
        --prepare-only)
            PREPARE_ONLY=1
            shift
            ;;
        --inside-slurm)
            INSIDE_SLURM=1
            shift
            ;;
        --skip-prepare)
            SKIP_PREPARE=1
            shift
            ;;
        --)
            shift
            TRAIN_ARGS=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option before --: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

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

cd "$PROJECT_DIR"
source "$PROJECT_DIR/activate.sh"

mkdir -p "$PROJECT_DIR/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
DATA_LOG="$PROJECT_DIR/logs/prepare-alpaca-${timestamp}.log"
TRAIN_LOG="$PROJECT_DIR/logs/finetune-alpaca-${timestamp}.log"

if [ "$SKIP_PREPARE" -eq 0 ]; then
    echo "Preparing Alpaca dataset; log: $DATA_LOG"
    python "$PROJECT_DIR/prepare_alpaca_dataset.py" 2>&1 | tee "$DATA_LOG"
else
    echo "Skipping dataset preparation because it already ran before srun."
fi

if [ "$PREPARE_ONLY" -eq 1 ]; then
    echo "Dataset prepared. Skipping training because --prepare-only was provided."
    exit 0
fi

if [ "$INSIDE_SLURM" -ne 1 ] && ! has_amd_rocm_gpu; then
    if [ -n "$JOB_ID" ]; then
        forwarded=(--inside-slurm --skip-prepare --)
        forwarded+=("${TRAIN_ARGS[@]}")
        echo "Attaching to Slurm job $JOB_ID and running fine-tuning on the GPU node..."
        exec srun --jobid="$JOB_ID" --overlap bash "$SCRIPT_PATH" "${forwarded[@]}"
    fi

    echo "ERROR: no AMD ROCm GPU detected in this shell." >&2
    echo "Use --jobid <active_gpu_job_id>, or run from inside a GPU allocation." >&2
    exit 1
fi

if ! has_amd_rocm_gpu; then
    echo "ERROR: attached shell still does not expose an AMD ROCm GPU." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU detected; this tutorial is intended for AMD ROCm only." >&2
    exit 1
fi

echo "Running fine-tuning; log: $TRAIN_LOG"
set -o pipefail
python "$PROJECT_DIR/finetune_gemma4_alpaca.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"
