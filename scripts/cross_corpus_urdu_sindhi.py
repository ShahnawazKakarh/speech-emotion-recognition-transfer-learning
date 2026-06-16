"""
Cross-corpus Urdu <-> Sindhi transfer learning experiment on the
Urdu-Sindhi Speech Emotion Corpus (Syed et al. 2020).

Research question: How well do classifiers trained on one Indo-Aryan
language generalize to a related-but-different Indo-Aryan language,
when both use the same acoustic feature extraction?

To our knowledge no one has published this specific cross-corpus
transfer result for Urdu and Sindhi. The paper by Syed et al. only
reports within-language baselines; we add the cross-language transfer
dimension.

Setup
-----
For each feature set (eGeMAPS, ComParE, IS09, IS10, Prosody) and
classifier (SVM-RBF, RandomForest, MLP):
  1. Train on FULL Urdu set (734 samples), test on FULL Sindhi set (701)
  2. Train on FULL Sindhi set (701 samples), test on FULL Urdu set (734)

Both languages share the same 7-emotion taxonomy (Anger, Disgust,
Happiness, Neutral, Sadness, Sarcasm, Surprise), so labels are
directly comparable.

We compare against the within-language 5-fold CV numbers from
train_urdu_sindhi_classical.py as the upper bound.

Usage
-----
    python scripts/cross_corpus_urdu_sindhi.py

Outputs
-------
    results/urdu_sindhi/cross_corpus_summary.csv
    results/urdu_sindhi/cross_corpus_<features>_<model>.json (per run)

Author: Muhammad Shahnawaz Khan
ORCID:  0009-0007-4055-6563
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Reuse the dataset loader from the within-language trainer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_urdu_sindhi_classical import (  # noqa: E402
    FEATURE_SETS,
    LANGUAGE_DIRS,
    RESULTS_ROOT,
    load_language_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Within-language UAR upper bounds (from the within-language run):
WITHIN_LANGUAGE_UAR = {
    ("urdu", "eGeMAPS"): 0.3931,
    ("urdu", "ComParE"): 0.5248,
    ("urdu", "IS09"): 0.4961,
    ("urdu", "IS10"): 0.5526,
    ("urdu", "Prosody"): 0.3852,
    ("sindhi", "eGeMAPS"): 0.4915,
    ("sindhi", "ComParE"): 0.5573,
    ("sindhi", "IS09"): 0.5159,
    ("sindhi", "IS10"): 0.5699,
    ("sindhi", "Prosody"): 0.3760,
}


def build_models() -> dict[str, Pipeline]:
    return {
        "SVM-RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=400,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


@dataclass
class CrossCorpusResult:
    train_language: str
    test_language: str
    features: str
    model: str
    n_train: int
    n_test: int
    accuracy: float
    wf1: float
    uar: float
    per_class_f1: dict[str, float]
    within_language_uar_target: float
    transfer_gap: float  # within-language UAR - cross-corpus UAR


def evaluate_transfer(
    X_train: np.ndarray,
    y_train_str: np.ndarray,
    X_test: np.ndarray,
    y_test_str: np.ndarray,
    pipeline: Pipeline,
) -> dict:
    # Fit encoder on the union of labels to keep indices aligned across
    # train and test (both languages share the same 7-emotion taxonomy)
    all_labels = sorted(set(y_train_str) | set(y_test_str))
    le = LabelEncoder().fit(all_labels)
    y_train = le.transform(y_train_str)
    y_test = le.transform(y_test_str)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    wf1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    uar = recall_score(y_test, y_pred, average="macro", zero_division=0)
    per_class = f1_score(
        y_test,
        y_pred,
        average=None,
        labels=range(len(all_labels)),
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "wf1": float(wf1),
        "uar": float(uar),
        "per_class_f1": {c: float(f) for c, f in zip(all_labels, per_class)},
    }


def run_one_direction(
    train_lang: str,
    test_lang: str,
    features: str,
) -> list[CrossCorpusResult]:
    print(f"\n{'='*72}")
    print(
        f"TRAIN: {train_lang.upper():6s} -> TEST: {test_lang.upper():6s}  "
        f"|  features: {features}"
    )
    print(f"{'='*72}")

    X_train, y_train, _ = load_language_dataset(train_lang, features)
    X_test, y_test, _ = load_language_dataset(test_lang, features)

    # Sanity check on dimensionality
    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError(
            f"Feature dim mismatch: {train_lang}={X_train.shape[1]}, "
            f"{test_lang}={X_test.shape[1]}"
        )

    results: list[CrossCorpusResult] = []
    for name, pipe in build_models().items():
        print(f"\n  Model: {name}")
        metrics = evaluate_transfer(X_train, y_train, X_test, y_test, pipe)
        within_target = WITHIN_LANGUAGE_UAR.get((test_lang, features), float("nan"))
        gap = within_target - metrics["uar"]
        r = CrossCorpusResult(
            train_language=train_lang,
            test_language=test_lang,
            features=features,
            model=name,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            accuracy=metrics["accuracy"],
            wf1=metrics["wf1"],
            uar=metrics["uar"],
            per_class_f1=metrics["per_class_f1"],
            within_language_uar_target=within_target,
            transfer_gap=gap,
        )
        results.append(r)

        out_path = (
            RESULTS_ROOT
            / f"cross_corpus_{train_lang}_to_{test_lang}_{features}_{name}.json"
        )
        with out_path.open("w") as f:
            json.dump(asdict(r), f, indent=2)
        print(
            f"    UAR={r.uar:.4f}  WF1={r.wf1:.4f}  acc={r.accuracy:.4f}\n"
            f"    within-language UAR (target) = {within_target:.4f}\n"
            f"    transfer gap = {gap:+.4f}  (positive = within-language is "
            f"better; negative = cross-corpus is better!)"
        )

    return results


def save_summary(all_results: list[CrossCorpusResult]) -> None:
    import csv

    out_path = RESULTS_ROOT / "cross_corpus_summary.csv"
    fields = [
        "train_language",
        "test_language",
        "features",
        "model",
        "n_train",
        "n_test",
        "accuracy",
        "wf1",
        "uar",
        "within_language_uar_target",
        "transfer_gap",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            row = asdict(r)
            w.writerow({k: row[k] for k in fields})
    print(f"\n[OK] {out_path.relative_to(Path(__file__).resolve().parents[1])}")


def print_leaderboard(all_results: list[CrossCorpusResult]) -> None:
    print("\n" + "=" * 110)
    print("CROSS-CORPUS TRANSFER LEADERBOARD  (sorted by UAR within direction)")
    print("=" * 110)
    print(
        f"{'Train->Test':<18} {'Features':<10} {'Model':<14} {'UAR':<10} "
        f"{'WF1':<10} {'Within UAR':<12} {'Gap':<10}"
    )
    print("-" * 110)
    for r in sorted(
        all_results,
        key=lambda r: (r.train_language, r.test_language, -r.uar),
    ):
        direction = f"{r.train_language[:3]}->{r.test_language[:3]}"
        print(
            f"{direction:<18} {r.features:<10} {r.model:<14} "
            f"{r.uar:.4f}    {r.wf1:.4f}    {r.within_language_uar_target:.4f}      "
            f"{r.transfer_gap:+.4f}"
        )
    print("=" * 110)


def main() -> int:
    all_results: list[CrossCorpusResult] = []
    for features in FEATURE_SETS:
        for train_lang, test_lang in [("urdu", "sindhi"), ("sindhi", "urdu")]:
            try:
                all_results.extend(run_one_direction(train_lang, test_lang, features))
            except Exception as e:
                print(f"  ERROR on {train_lang}->{test_lang} ({features}): {e}", file=sys.stderr)
    save_summary(all_results)
    print_leaderboard(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
