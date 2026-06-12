"""Cross-corpus transformer transfer between Punjabi RASA and URDU-Latif.

Evaluates an XLS-R model trained on one corpus directly on the *test*
split of the other corpus, with no fine-tuning on the target language
(zero-shot transfer). Both corpora share the same 4 emotion labels in
the same order (angry, happy, neutral, sad), so class indices align
directly.

This is the natural follow-up to the classical cross-corpus experiment
on Urdu-Sindhi (which failed catastrophically at ~30 pp UAR drop): does
the multilingual pre-trained XLS-R encoder bridge the cross-lingual gap
that hand-crafted acoustic features could not?

Usage:
    python -m scripts.cross_corpus_punjabi_urdu_latif \\
        --source-ckpt outputs/audio_only_xlsr_punjabi_rasa/best-07-0.9897.ckpt \\
        --target-dataset urdu_latif

    # Or symmetric:
    python -m scripts.cross_corpus_punjabi_urdu_latif \\
        --source-ckpt outputs/audio_only_xlsr_urdu_latif/<best>.ckpt \\
        --target-dataset punjabi_rasa

Outputs:
    JSON line on stdout + a `results/cross_corpus_punjabi_urdu_latif/<src>_to_<tgt>.json`
    file with per-class precision/recall/F1 and the confusion matrix.

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
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

assert PUNJABI_RASA_EMOTIONS == CLASS_NAMES, (
    "PUNJABI_RASA_EMOTIONS ordering changed — fix label alignment."
)
assert URDU_LATIF_EMOTIONS == CLASS_NAMES, (
    "URDU_LATIF_EMOTIONS ordering changed — fix label alignment."
)


def build_target_loader(target: str, batch_size: int = 4, max_sec: float = 6.0):
    """Return a DataLoader over the TEST split of the target corpus."""
    if target == "urdu_latif":
        ds = URDULatifDataset(
            data_dir="data/urdu_latif",
            split="test",
            split_mode="random",
            sample_rate=16000,
            max_audio_seconds=max_sec,
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
    """Find the config used to train a checkpoint.

    Lightning stores hparams inside the .ckpt under "hyper_parameters" -> "cfg".
    We load that rather than rummaging through the configs/ directory.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "hyper_parameters" not in sd:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has no hyper_parameters block; cannot reconstruct model."
        )
    hp = sd["hyper_parameters"]
    if "cfg" not in hp:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} hyper_parameters has no 'cfg' key."
        )
    return hp["cfg"]


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
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-ckpt",
        required=True,
        help="Path to the trained source-language .ckpt file.",
    )
    parser.add_argument(
        "--target-dataset",
        required=True,
        choices=["urdu_latif", "punjabi_rasa"],
        help="Target corpus to evaluate on (the OTHER language).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument(
        "--results-dir",
        default="results/cross_corpus_punjabi_urdu_latif",
        help="Where to write per-run JSON outputs.",
    )
    args = parser.parse_args()

    seed_everything(42)

    ckpt_path = Path(args.source_ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    print(f"[load] reading config from {ckpt_path}")
    cfg = load_config_for_checkpoint(ckpt_path)
    source_dataset = cfg["dataset"]

    if source_dataset == args.target_dataset:
        raise SystemExit(
            f"Source ({source_dataset}) and target ({args.target_dataset}) "
            f"are the same -- this is a within-language eval, not cross-corpus."
        )

    print(f"[transfer] source={source_dataset}  target={args.target_dataset}")

    # Build target test loader.
    loader, ds = build_target_loader(
        args.target_dataset,
        batch_size=args.batch_size,
        max_sec=cfg["data"].get("max_audio_seconds", 6.0),
    )
    print(f"[target] {len(ds)} test samples")

    # Reconstruct the trained model from the source-language ckpt.
    print("[load] reconstructing source-language model...")
    model = SERLightningModule.load_from_checkpoint(
        ckpt_path,
        cfg=cfg,
        label_names=CLASS_NAMES,
        map_location=args.device,
    )

    # Zero-shot evaluation.
    print(f"[eval] running zero-shot transfer on {args.device}...")
    y_true, y_pred = evaluate(model, loader, args.device)

    acc = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    uf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()

    summary = {
        "source_dataset": source_dataset,
        "target_dataset": args.target_dataset,
        "source_ckpt": str(ckpt_path),
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

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_dataset}_to_{args.target_dataset}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 70)
    print(f"  Source: {source_dataset}   ->   Target: {args.target_dataset}")
    print("=" * 70)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Weighted F1    : {wf1:.4f}")
    print(f"  Unweighted F1  : {uf1:.4f}")
    print(f"  UAR (macro R)  : {uar:.4f}")
    print()
    print("  Per-class F1:")
    for name in CLASS_NAMES:
        cls = report[name]
        print(
            f"    {name:<8} precision={cls['precision']:.3f}  "
            f"recall={cls['recall']:.3f}  f1={cls['f1-score']:.3f}  "
            f"support={int(cls['support'])}"
        )
    print()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
