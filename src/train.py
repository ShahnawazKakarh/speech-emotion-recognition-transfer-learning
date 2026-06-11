"""Training entry point.

Usage:
    python -m src.train --config configs/audio_only_ravdess.yaml
    python -m src.train --config configs/multimodal_meld.yaml --max-epochs 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from src.data.datamodule import SERDataModule
from src.data.meld import MELD_EMOTIONS
from src.data.punjabi_rasa import PUNJABI_RASA_EMOTIONS
from src.data.ravdess import RAVDESS_EMOTIONS
from src.models.lightning_module import SERLightningModule
from src.utils import seed_everything


def load_config(path: str | Path) -> dict:
    """Load a YAML config, applying any `defaults: [base]` inheritance."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for default in cfg.pop("defaults", []):
        default_path = path.parent / f"{default}.yaml"
        with open(default_path) as f:
            base_cfg = yaml.safe_load(f)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def get_label_names(dataset: str) -> list[str]:
    if dataset == "ravdess":
        return list(RAVDESS_EMOTIONS.values())
    if dataset == "meld":
        return MELD_EMOTIONS
    if dataset == "punjabi_rasa":
        return list(PUNJABI_RASA_EMOTIONS)
    if dataset == "iemocap":
        from src.data.iemocap import IEMOCAP_EMOTIONS

        return IEMOCAP_EMOTIONS
    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--fast-dev-run", action="store_true", help="Smoke test with 1 batch")
    parser.add_argument("--wandb", action="store_true", help="Force-enable W&B logging")
    parser.add_argument(
        "--skip-test", action="store_true", help="Skip the final trainer.test() pass"
    )
    parser.add_argument(
        "--patience", type=int, default=8, help="EarlyStopping patience (epochs without improvement)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["trainer"]["max_epochs"] = args.max_epochs
    if args.wandb:
        cfg["logging"]["use_wandb"] = True

    seed_everything(cfg.get("seed", 42))

    exp_name = cfg["experiment_name"]
    output_dir = Path(cfg["logging"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = SERDataModule(cfg)
    label_names = get_label_names(cfg["dataset"])
    model = SERLightningModule(cfg, label_names=label_names)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-{val/wf1:.4f}",
            monitor="val/wf1",
            mode="max",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor="val/wf1", mode="max", patience=args.patience),
        LearningRateMonitor(logging_interval="step"),
    ]

    loggers = [TensorBoardLogger(output_dir, name="tb")]
    if cfg["logging"].get("use_wandb", False):
        loggers.append(
            WandbLogger(project=cfg["logging"].get("wandb_project", "ser"), name=exp_name)
        )

    trainer = L.Trainer(
        max_epochs=cfg["trainer"]["max_epochs"],
        accelerator=cfg["trainer"].get("accelerator", "auto"),
        devices=cfg["trainer"].get("devices", "auto"),
        precision=cfg["trainer"].get("precision", "32-true"),
        gradient_clip_val=cfg["trainer"].get("gradient_clip_val", 1.0),
        accumulate_grad_batches=cfg["trainer"].get("accumulate_grad_batches", 1),
        log_every_n_steps=cfg["trainer"].get("log_every_n_steps", 20),
        callbacks=callbacks,
        logger=loggers,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)

    if args.skip_test:
        return

    if args.fast_dev_run:
        trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
