#!/usr/bin/env python3
"""Plot training curves from a Spacebyte metrics.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unfreeze-epoch", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in args.metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError(f"No epoch records found in {args.metrics}")

    epochs = [record["epoch"] for record in records]
    output = args.output or args.metrics.with_name("training_curves.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(
        epochs, [record["train_loss"] for record in records], "o-", label="train"
    )
    axes[0].plot(
        epochs,
        [record["validation_loss"] for record in records],
        "o-",
        label="validation",
    )
    axes[0].set(title="Masked BCE loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    metric_names = {
        "balanced_accuracy": "Macro balanced accuracy",
        "f1": "Macro F1",
        "auroc": "Macro AUROC",
    }
    for key, label in metric_names.items():
        axes[1].plot(
            epochs,
            [record["metrics"]["macro"][key] for record in records],
            "o-",
            label=label,
        )
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].set_ylim(0.45, 0.70)
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.axvline(
            args.unfreeze_epoch,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="backbone unfrozen",
        )
        axis.set_xticks(epochs)

    figure.suptitle(f"SpatialSense ByteFormer — {len(records)} epochs")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
