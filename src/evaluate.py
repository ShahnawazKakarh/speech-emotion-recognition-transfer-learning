"""Standalone evaluation of a trained checkpoint.

Usage:
    python -m src.evaluate --checkpoint outputs/multimodal_ravdess/last.ckpt \
                           --config configs/multimodal_ravdess.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.datamodule import SERDataModule
from src.models.lightning_module import SERLightningModule
from src.train import get_label_names, load_config
from src.utils.metrics import format_metrics_table


def plot_confusion_matrix(cm: np.ndarray, label_names: list[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_names = get_label_names(cfg["dataset"])

    dm = SERDataModule(cfg)
    model = SERLightningModule.load_from_checkpoint(args.checkpoint, cfg=cfg, label_names=label_names)

    trainer = L.Trainer(accelerator="auto", devices=1, logger=False)
    trainer.test(model, datamodule=dm)

    metrics = model.test_metrics
    print(format_metrics_table(metrics, dataset_name=cfg["dataset"]))

    out = Path(args.output_dir or Path(args.checkpoint).parent / "eval")
    out.mkdir(parents=True, exist_ok=True)

    # Save JSON + confusion matrix plot
    cm = metrics.pop("confusion_matrix")
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(cm, label_names, out / "confusion_matrix.png")
    np.save(out / "confusion_matrix.npy", cm)

    print(f"\nSaved evaluation artifacts to {out}")


if __name__ == "__main__":
    main()
