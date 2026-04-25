#!/usr/bin/env python3
"""Plot training curves from a finished/in-progress fine-tune run.

Reads `trainer_state.json` (always written by the HF Trainer) and renders
loss / learning_rate / grad_norm vs steps as PNGs. If TensorBoard event
files are present (`tb/`) it also reads them so the plots match what
`tensorboard --logdir=outputs/.../tb` would show.

Examples:
    python plot_training.py
    python plot_training.py --run-dir outputs/gemma4-26b-a4b-it-alpaca-lora
    python plot_training.py --plots-dir outputs/.../plots
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

DEFAULT_RUN_DIR = Path("outputs/gemma4-26b-a4b-it-alpaca-lora")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help="Output directory of finetune.py.")
    p.add_argument("--plots-dir", type=Path, default=None,
                   help="Where to write PNGs (default: <run-dir>/plots).")
    return p.parse_args()


def find_trainer_state(run_dir: Path) -> Path | None:
    # Trainer writes one trainer_state.json per checkpoint and one at the
    # final adapter dir. Pick the latest checkpoint with the highest step.
    candidates = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if (run_dir / "trainer_state.json").exists():
        candidates.append(run_dir / "trainer_state.json")
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_log_history(state_path: Path) -> list[dict]:
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    return state.get("log_history", [])


def load_tb_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Read scalars from any TensorBoard event files under run_dir.

    Transformers writes events under `<run_dir>/runs/<timestamp>/`, while
    older configs may put them in `<run_dir>/tb/`. We just scan for the
    newest file and read it.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    candidates = list(run_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        return {}
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    acc = EventAccumulator(str(latest.parent), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception:
        return {}
    scalars: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        scalars[tag] = [(ev.step, ev.value) for ev in acc.Scalars(tag)]
    return scalars


def series_from_log_history(log_history: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for entry in log_history:
        if key in entry and "step" in entry:
            xs.append(int(entry["step"]))
            ys.append(float(entry[key]))
    return xs, ys


def plot_one(xs: list[int], ys: list[float], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plots_dir = args.plots_dir or (args.run_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    state_path = find_trainer_state(args.run_dir)
    if state_path is None:
        print(f"No trainer_state.json under {args.run_dir}.", file=sys.stderr)
        print("Did you run finetune.py yet?", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {state_path}")
    log_history = load_log_history(state_path)

    tb_scalars = load_tb_scalars(args.run_dir)
    if tb_scalars:
        print(f"TensorBoard scalars found: {sorted(tb_scalars)}")

    metrics = [
        ("loss", "loss", "Training loss", "loss"),
        ("learning_rate", "learning_rate", "Learning-rate schedule", "lr"),
        ("grad_norm", "grad_norm", "Gradient norm", "grad_norm"),
    ]
    written: list[Path] = []
    for tb_tag, hist_key, title, fname in metrics:
        # Prefer TensorBoard data (it includes everything the Trainer logged
        # at every logging_steps tick); fall back to log_history.
        xs, ys = [], []
        if tb_tag in tb_scalars:
            xs = [step for step, _ in tb_scalars[tb_tag]]
            ys = [val for _, val in tb_scalars[tb_tag]]
        elif f"train/{tb_tag}" in tb_scalars:
            xs = [step for step, _ in tb_scalars[f"train/{tb_tag}"]]
            ys = [val for _, val in tb_scalars[f"train/{tb_tag}"]]
        else:
            xs, ys = series_from_log_history(log_history, hist_key)

        if not xs:
            print(f"  skip {fname}: no data")
            continue
        out_path = plots_dir / f"{fname}.png"
        plot_one(xs, ys, title, hist_key, out_path)
        written.append(out_path)
        print(f"  wrote {out_path}  ({len(xs)} points)")

    if not written:
        print("Nothing was plotted.", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone. {len(written)} plot(s) in {plots_dir}/")


if __name__ == "__main__":
    main()
