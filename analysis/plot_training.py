"""Draw the gradient-norm and accuracy/loss charts for one training run.

Run from the repository root:
    .venv/bin/python -m analysis.plot_training results/fp32
Reads gradient_norms.csv and training_history.csv from that folder and writes
gradient_norms.png and training_curves.png next to them.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen before pyplot loads)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"
LAYER_COLORS = {"conv1": "#2a78d6", "conv2": "#eb6834", "dense1": "#1baf7a", "dense2": "#eda100"}
TRAINING_COLOR, VALIDATION_COLOR = "#2a78d6", "#eb6834"
LINE_WIDTH = 1.4
BEST_EPOCH_NOTE = "Dashed line = best validation epoch (the saved weights come from it)."


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/fp32")
    best_epoch = json.loads((run_dir / "run_summary.json").read_text())["best_epoch"]
    plot_gradient_norms(read_csv(run_dir / "gradient_norms.csv"), best_epoch, run_dir / "gradient_norms.png")
    plot_training_curves(
        read_csv(run_dir / "training_history.csv"), best_epoch, run_dir / "training_curves.png"
    )


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open() as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def plot_gradient_norms(rows, best_epoch: int, output_path: Path) -> None:
    figure, axes = plt.subplots(len(LAYER_COLORS), 1, sharex=True, sharey=True, figsize=(9, 8.5))
    for axis, (layer, color) in zip(axes, LAYER_COLORS.items()):
        epochs, low, median, high = _per_epoch_spread(rows, layer)
        axis.fill_between(epochs, low, high, color=color, alpha=0.22, linewidth=0)
        axis.plot(epochs, median, color=color, linewidth=LINE_WIDTH)
        axis.set_yscale("log")
        dead_batches = sum(row[layer] == 0 for row in rows)
        _style_axis(axis, f"{layer}   ({dead_batches} of {len(rows)} batches had a gradient of exactly 0)")
        _mark_best_epoch(axis, best_epoch)
    axes[-1].set_xlabel("epoch", color=TEXT_SECONDARY)
    figure.supylabel("gradient norm (log scale)", color=TEXT_SECONDARY)
    _finish(
        figure,
        "Gradient size per layer during training",
        "Line = median batch of each epoch; band = middle 80% of that epoch's batches; "
        "a gradient of exactly 0 means the layer got no learning signal",
        output_path,
    )


def _per_epoch_spread(rows, layer: str):
    """Median and 10th–90th percentile of each epoch's non-zero gradient norms (log scale cannot show 0)."""
    epochs = sorted({row["epoch"] for row in rows})
    spreads = [
        np.percentile([row[layer] for row in rows if row["epoch"] == epoch and row[layer] > 0], (10, 50, 90))
        for epoch in epochs
    ]
    low, median, high = (list(column) for column in zip(*spreads))
    return epochs, low, median, high


def plot_training_curves(rows, best_epoch: int, output_path: Path) -> None:
    epochs = [row["epoch"] for row in rows]
    figure, (accuracy_axis, loss_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_pair(accuracy_axis, epochs, rows, "accuracy", "accuracy")
    _plot_pair(loss_axis, epochs, rows, "loss", "loss (cross-entropy)")
    for axis in (accuracy_axis, loss_axis):
        _mark_best_epoch(axis, best_epoch)
        axis.set_xlabel("epoch", color=TEXT_SECONDARY)
    accuracy_axis.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    _finish(
        figure,
        "Training versus validation, per epoch",
        "Validation = one unseen person (pengxl) plus two no-gesture recordings",
        output_path,
    )


def _plot_pair(axis, epochs, rows, metric: str, title: str) -> None:
    missing = []
    for prefix, label, color in (("", "training", TRAINING_COLOR), ("val_", "validation", VALIDATION_COLOR)):
        values = [row[prefix + metric] for row in rows]
        if not any(values):  # a column of zeros means the metric was not recorded in that run
            missing.append(label)
            continue
        axis.plot(epochs, values, color=color, linewidth=LINE_WIDTH, label=label)
        axis.annotate(
            label,
            (epochs[-1], values[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=8,
            va="center",
        )
    _style_axis(axis, title + "".join(f"   ({label} {metric} not recorded in this run)" for label in missing))


def _style_axis(axis, title: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=9.5, color=TEXT_PRIMARY)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _mark_best_epoch(axis, best_epoch: int) -> None:
    axis.axvline(best_epoch, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")


def _finish(figure, title: str, subtitle: str, output_path: Path) -> None:
    figure.patch.set_facecolor(SURFACE)
    figure.text(
        0.01, _height_from_top(figure, 0.3), title, fontsize=12, color=TEXT_PRIMARY, fontweight="bold"
    )
    for line_number, line in enumerate((f"{subtitle}.", BEST_EPOCH_NOTE)):
        figure.text(
            0.01, _height_from_top(figure, 0.55 + 0.2 * line_number), line, fontsize=8.5, color=TEXT_SECONDARY
        )
    figure.tight_layout(rect=(0, 0, 1, _height_from_top(figure, 0.9)))
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {output_path}")


def _height_from_top(figure, inches: float) -> float:
    """Figure-fraction position a fixed number of inches below the top, so headers never overlap."""
    return 1 - inches / figure.get_figheight()


if __name__ == "__main__":
    main()
