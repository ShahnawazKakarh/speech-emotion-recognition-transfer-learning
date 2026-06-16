"""Generalized SER checkpoint evaluator (within- or cross-language).

Given a trained XLS-R audio-only checkpoint, evaluate it on the *test*
split of any supported target corpus. Used by the multi-seed cross-lingual
runner to produce one JSON per (model, target_dataset, seed) cell.

Unlike `cross_corpus_punjabi_urdu_latif.py`, this script does NOT enforce
that source != target. Within-language evaluation (source == target) is
the within-language ceiling for the same trained model and is used to
build conditions 1 and 2 of the multi-seed cross-lingual study.

Usage:
    python -m scripts.eval_checkpoint \\
        --ckpt outputs/multiseed_pun_s0/best.ckpt \\
        --target-dataset punjabi_rasa \\
        --output results/multiseed/pun_s0_on_pun.json

    # Cross-lingual eval against a specific Urdu held-out show
    python -m scripts.eval_checkpoint \\
        --ckpt outputs/multiseed_pun_s0/best.ckpt \\
        --target-dataset urdu_latif \\
        --urdu-held-out-show SM2 \\
        --output results/multiseed/pun_s0_on_urdu_sm2.json

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.data.base import collate_ser_batch
from src.data.punjabi_rasa import PUNJABI_RASA_EMOTIONS, PunjabiRASADataset
from src.data.urdu_latif import URDU_LATIF_EMOTIONS, URDULatifDataset
from src.models.lightning_module import SERLightningModule
from src.utils import seed_everything

CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

assert PUNJABI_RASA_EMOTIONS == CLASS_NAMES
assert URDU_LATIF_EMOTIONS == CLASS_NAMES


def build_target_loader(
    target: str,
    *,
    batch_size: int = 4,
    max_sec: float = 6.0,
    urdu_held_out_show: str = "SM4",
    urdu_split_mode: str = "show_independent",
    urdu_split_seed: int = 42,
):
    if target == "urdu_latif":
        ds = URDULatifDataset(
            data_dir="data/urdu_latif",
            split="test",
            split_mode=urdu_split_mode,
            held_out_show=urdu_held_out_show,
            sample_rate=16000,
            max_audio_seconds=max_sec,
            seed=urdu_split_seed,
        )
    elif target == "punjabi_rasa":
        ds = PunjabiRASADataset(
            data_dir="data/punjabi_rasa",
            split="test",
            sample_rate=16000,
            max_audio_seconds=max_sec,
        )
    else:
        raise ValueError(f"Unknown target dataset: {target!r}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_ser_batch,
        pin_memory=True,
    )
    return loader, ds


def load_config_for_checkpoint(ckpt_path: Path) -> dict:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in sd:
        raise RuntimeError(f"{ckpt_path}: no hyper_parameters block.")
    hp = dict(sd["hyper_parameters"])
    if set(hp.keys()) == {"cfg"}:
        return hp["cfg"]
    required = {"model", "data", "dataset"}
    missing = required - hp.keys()
    if missing:
        raise RuntimeError(f"{ckpt_path}: hyper_parameters missing {missing}.")
    return hp


def evaluate(model: SERLightningModule, loader: DataLoader, device: str):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            logits = model.model(batch)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the trained .ckpt.")
    parser.add_argument(
        "--target-dataset",
        required=True,
        choices=["urdu_latif", "punjabi_rasa"],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--urdu-held-out-show",
        default="SM4",
        help="When target is urdu_latif: which show is the test set.",
    )
    parser.add_argument("--urdu-split-mode", default="show_independent")
    parser.add_argument("--urdu-split-seed", type=int, default=42)
    parser.add_argument(
        "--seed-tag",
        default=None,
        help="Free-form seed identifier written into the JSON (e.g. 's0').",
    )
    args = parser.parse_args()

    seed_everything(42)  # Eval determinism (no training here).

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    cfg = load_config_for_checkpoint(ckpt_path)
    source_dataset = cfg["dataset"]
    is_cross_lingual = source_dataset != args.target_dataset

    loader, ds = build_target_loader(
        args.target_dataset,
        batch_size=args.batch_size,
        max_sec=cfg["data"].get("max_audio_seconds", 6.0),
        urdu_held_out_show=args.urdu_held_out_show,
        urdu_split_mode=args.urdu_split_mode,
        urdu_split_seed=args.urdu_split_seed,
    )

    model = SERLightningModule.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        label_names=CLASS_NAMES,
        map_location=args.device,
        strict=False,
    )

    y_true, y_pred = evaluate(model, loader, args.device)

    acc = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()

    summary = {
        "source_dataset": source_dataset,
        "target_dataset": args.target_dataset,
        "is_cross_lingual": is_cross_lingual,
        "source_ckpt": str(ckpt_path),
        "seed_tag": args.seed_tag,
        "urdu_held_out_show": args.urdu_held_out_show if args.target_dataset == "urdu_latif" else None,
        "n_target_test": int(len(y_true)),
        "metrics": {
            "accuracy": float(acc),
            "weighted_f1": float(wf1),
            "unweighted_f1": float(uf1),
            "uar_macro_recall": float(uar),
        },
        "per_class": report,
        "confusion_matrix": cm,
        "class_names": CLASS_NAMES,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    direction = "cross-lingual" if is_cross_lingual else "within-language"
    print(
        f"[{direction}] {source_dataset} -> {args.target_dataset}"
        f"{f' (SM={args.urdu_held_out_show})' if args.target_dataset == 'urdu_latif' else ''}  "
        f"UAR={uar:.4f}  WF1={wf1:.4f}  Acc={acc:.4f}  n={len(y_true)}"
    )
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
