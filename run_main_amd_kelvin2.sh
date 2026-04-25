#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ID="${KELVIN_JOB_ID:-}"
INSIDE_SLURM=0

usage() {
    cat <<'USAGE'
Usage:
  ./run_main_amd_kelvin2.sh --jobid <SLURM_JOB_ID>

Options:
  --jobid ID       Attach to an existing Kelvin2 Slurm GPU job.
  --inside-slurm   Internal flag used after srun attaches to the job.
  -h, --help       Show this help.

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
        echo "Attaching to Slurm job $JOB_ID and running main.py on the GPU node..."
        exec srun --jobid="$JOB_ID" --overlap bash "$SCRIPT_PATH" --inside-slurm
    fi

    echo "ERROR: no AMD ROCm GPU detected in this shell." >&2
    echo "You appear to be on a login node. Use --jobid <active_gpu_job_id>." >&2
    exit 1
fi

if ! has_amd_rocm_gpu; then
    echo "ERROR: attached shell still does not expose an AMD ROCm GPU." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU detected; this runner is intended for AMD ROCm only." >&2
    exit 1
fi

cd "$PROJECT_DIR"
source "$PROJECT_DIR/activate.sh"

python - <<'PY'
import torch

print("torch", torch.__version__)
print("torch_hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if not getattr(torch.version, "hip", None):
    raise SystemExit("ROCm torch is not installed")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("No AMD GPU visible to torch")

print("device_name", torch.cuda.get_device_name(0))
PY

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$PROJECT_DIR/main-${timestamp}.log"
echo "Running main.py; log: $LOG_FILE"
set -o pipefail
python "$PROJECT_DIR/main.py" 2>&1 | tee "$LOG_FILE"
