"""
Classical-ML baseline on the Urdu-Sindhi Speech Emotion Corpus.

Loads the .mat feature files released by Syed et al. (2020) at
Zenodo DOI 10.5281/zenodo.3685274 and trains several classical
classifiers (SVM, Random Forest, MLP) under a stratified 5-fold
cross-validation regime. Reports UAR (Unweighted Average Recall),
weighted F1, and per-class F1 — directly comparable to the dataset
paper's reported baseline (Urdu UAR 56.96%, Sindhi UAR 55.29%).

Why this matters for our research programme
-------------------------------------------
This establishes our first cross-lingual Indo-Aryan emotion baseline
using a modern classical ML pipeline. When we add the transformer-based
results later (wav2vec2-XLS-R + XLM-R, pending raw-audio access), we
will compare against both the original paper's numbers and these
modernized classical numbers.

Usage
-----
    # Train on Urdu, eGeMAPS features
    python scripts/train_urdu_sindhi_classical.py --language urdu --features eGeMAPS

    # Train on Sindhi, ComParE features
    python scripts/train_urdu_sindhi_classical.py --language sindhi --features ComParE

    # Run everything (both languages, all 5 feature sets) -- ~10-15 min
    python scripts/train_urdu_sindhi_classical.py --all

Outputs
-------
    results/urdu_sindhi/<language>_<features>_<model>.json   per-model summary
    results/urdu_sindhi/summary.csv                          all runs combined

Author: Muhammad Shahnawaz Khan
ORCID:  0009-0007-4055-6563
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants and paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "urdu_sindhi"
RESULTS_ROOT = ROOT / "results" / "urdu_sindhi"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Folder structure inside the Zenodo release:
#   data/urdu_sindhi/Feats - <Language>/<FeatureSet> funcs/Matfiles/<Emotion>_<NNNN>.mat
LANGUAGE_DIRS = {
    "urdu": "Feats - Urdu",
    "sindhi": "Feats - Sindhi",
}

FEATURE_SETS = {
    "eGeMAPS": "eGeMAPS funcs",
    "ComParE": "ComParE funcs",
    "IS09": "IS09Emotion funcs",
    "IS10": "IS10Paraling funcs",
    "Prosody": "Prosody funcs",
}

EMOTIONS = ["Anger", "Disgust", "Happiness", "Neutral", "Sadness", "Sarcasm", "Surprise"]

# Reported numbers from Syed et al. 2020 (test partition):
PAPER_BASELINE = {
    "urdu": {"uar": 0.5696, "note": "Best model in Syed et al. 2020 (test)"},
    "sindhi": {"uar": 0.5529, "note": "Best model in Syed et al. 2020 (test)"},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_mat_feature_vector(mat_path: Path) -> Optional[np.ndarray]:
    """Load one .mat file and return its feature vector (1-D float array).

    The Zenodo release stores one feature vector per file. The variable name
    varies by feature set, so we pick the largest non-private numerical array.
    """
    try:
        data = scipy.io.loadmat(str(mat_path))
    except Exception as e:
        print(f"  WARNING: failed to load {mat_path.name}: {e}", file=sys.stderr)
        return None

    candidates = []
    for k, v in data.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "shape") and v.dtype.kind in ("f", "i", "u"):
            candidates.append((k, np.asarray(v).flatten()))

    if not candidates:
        return None

    # Pick the largest numerical array
    candidates.sort(key=lambda kv: kv[1].size, reverse=True)
    return candidates[0][1].astype(np.float32)


def load_language_dataset(language: str, features: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all .mat files for one language + feature-set combination.

    Returns
    -------
    X : (N, D) float32 feature matrix
    y : (N,)   string label array (emotion names)
    filenames : list of source filenames (for traceability)
    """
    lang_dir = DATA_ROOT / LANGUAGE_DIRS[language] / FEATURE_SETS[features] / "Matfiles"
    if not lang_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {lang_dir}")

    mat_files = sorted(lang_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in: {lang_dir}")

    print(f"  Loading {len(mat_files)} .mat files from {lang_dir.name}/")

    rows = []
    labels = []
    filenames = []
    expected_dim = None

    for i, mp in enumerate(mat_files):
        vec = load_mat_feature_vector(mp)
        if vec is None:
            continue
        if expected_dim is None:
            expected_dim = vec.size
        if vec.size != expected_dim:
            print(
                f"  WARNING: {mp.name} has {vec.size} features, expected "
                f"{expected_dim}. Skipping.",
                file=sys.stderr,
            )
            continue
        # Label is the part of the filename before the first underscore.
        emotion = mp.stem.split("_")[0]
        rows.append(vec)
        labels.append(emotion)
        filenames.append(mp.name)
        if (i + 1) % 200 == 0:
            print(f"    loaded {i + 1}/{len(mat_files)}")

    X = np.vstack(rows)
    y = np.array(labels)
    print(f"  Final shape: X={X.shape}, y={y.shape}, classes={sorted(set(y))}")
    return X, y, filenames


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models() -> dict[str, Pipeline]:
    """Return a dictionary of named classical-ML pipelines."""
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
                        max_depth=None,
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


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    language: str
    features: str
    model: str
    n_samples: int
    n_features: int
    n_classes: int
    classes: list[str]
    accuracy_mean: float
    accuracy_std: float
    wf1_mean: float
    wf1_std: float
    uar_mean: float
    uar_std: float
    per_class_f1_mean: dict[str, float]
    paper_uar: Optional[float]
    delta_vs_paper: Optional[float]


def cross_validate_model(
    name: str, pipeline: Pipeline, X: np.ndarray, y_str: np.ndarray, n_splits: int = 5
) -> dict:
    """Run stratified k-fold CV and return aggregated metrics."""
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes_in_order = list(le.classes_)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, wf1s, uars = [], [], []
    per_class_f1_history: dict[str, list[float]] = {c: [] for c in classes_in_order}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        pipeline.fit(X[train_idx], y[train_idx])
        y_pred = pipeline.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], y_pred)
        wf1 = f1_score(y[test_idx], y_pred, average="weighted", zero_division=0)
        uar = recall_score(y[test_idx], y_pred, average="macro", zero_division=0)

        # Per-class F1 — keys are emotion names
        class_f1 = f1_score(
            y[test_idx],
            y_pred,
            average=None,
            labels=range(len(classes_in_order)),
            zero_division=0,
        )
        for c, f in zip(classes_in_order, class_f1):
            per_class_f1_history[c].append(float(f))

        accs.append(acc)
        wf1s.append(wf1)
        uars.append(uar)

        print(
            f"    fold {fold}: acc={acc:.4f}  WF1={wf1:.4f}  UAR={uar:.4f}"
        )

    per_class_f1_mean = {c: float(np.mean(v)) for c, v in per_class_f1_history.items()}

    return {
        "classes_in_order": classes_in_order,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "wf1_mean": float(np.mean(wf1s)),
        "wf1_std": float(np.std(wf1s)),
        "uar_mean": float(np.mean(uars)),
        "uar_std": float(np.std(uars)),
        "per_class_f1_mean": per_class_f1_mean,
    }


def run_one(language: str, features: str) -> list[RunResult]:
    """Run all classifiers for one (language, features) combination."""
    print(f"\n{'='*72}")
    print(f"LANGUAGE = {language.upper()}   |   FEATURES = {features}")
    print(f"{'='*72}")

    X, y_str, _ = load_language_dataset(language, features)

    results: list[RunResult] = []
    models = build_models()
    for name, pipe in models.items():
        print(f"\n  Model: {name}")
        cv = cross_validate_model(name, pipe, X, y_str)
        paper_uar = PAPER_BASELINE[language]["uar"]
        result = RunResult(
            language=language,
            features=features,
            model=name,
            n_samples=int(X.shape[0]),
            n_features=int(X.shape[1]),
            n_classes=len(cv["classes_in_order"]),
            classes=cv["classes_in_order"],
            accuracy_mean=cv["accuracy_mean"],
            accuracy_std=cv["accuracy_std"],
            wf1_mean=cv["wf1_mean"],
            wf1_std=cv["wf1_std"],
            uar_mean=cv["uar_mean"],
            uar_std=cv["uar_std"],
            per_class_f1_mean=cv["per_class_f1_mean"],
            paper_uar=paper_uar,
            delta_vs_paper=cv["uar_mean"] - paper_uar,
        )
        results.append(result)

        # Save individual JSON
        out_path = RESULTS_ROOT / f"{language}_{features}_{name}.json"
        with out_path.open("w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"    -> wrote {out_path.relative_to(ROOT)}")
        print(
            f"    SUMMARY  UAR={result.uar_mean:.4f} (±{result.uar_std:.4f})  "
            f"WF1={result.wf1_mean:.4f}  vs paper UAR {paper_uar:.4f}  "
            f"-> delta {result.delta_vs_paper:+.4f}"
        )

    return results


def save_summary_csv(all_results: list[RunResult]) -> None:
    """Append a CSV summary of every run."""
    import csv

    out_path = RESULTS_ROOT / "summary.csv"
    fieldnames = [
        "language",
        "features",
        "model",
        "n_samples",
        "n_features",
        "n_classes",
        "accuracy_mean",
        "accuracy_std",
        "wf1_mean",
        "wf1_std",
        "uar_mean",
        "uar_std",
        "paper_uar",
        "delta_vs_paper",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = asdict(r)
            row = {k: row[k] for k in fieldnames}
            writer.writerow(row)
    print(f"\n[OK] Wrote consolidated summary: {out_path.relative_to(ROOT)}")


def print_final_table(all_results: list[RunResult]) -> None:
    """Pretty-print all runs to stdout as a leaderboard."""
    print("\n" + "=" * 92)
    print("FINAL LEADERBOARD  (5-fold CV mean ± std; delta vs paper UAR)")
    print("=" * 92)
    print(
        f"{'Language':<8} {'Features':<10} {'Model':<14} {'UAR':<16} "
        f"{'WF1':<16} {'vs paper':<10}"
    )
    print("-" * 92)
    for r in sorted(
        all_results, key=lambda r: (r.language, -r.uar_mean)
    ):
        uar_str = f"{r.uar_mean:.4f} ±{r.uar_std:.3f}"
        wf1_str = f"{r.wf1_mean:.4f} ±{r.wf1_std:.3f}"
        delta_str = f"{r.delta_vs_paper:+.4f}"
        print(
            f"{r.language:<8} {r.features:<10} {r.model:<14} {uar_str:<16} "
            f"{wf1_str:<16} {delta_str:<10}"
        )
    print("=" * 92)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=list(LANGUAGE_DIRS.keys()))
    parser.add_argument("--features", choices=list(FEATURE_SETS.keys()), default="eGeMAPS")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all languages x all feature sets (~10-15 min total).",
    )
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        print(f"ERROR: Data directory missing: {DATA_ROOT}", file=sys.stderr)
        print(
            "Download the Urdu-Sindhi corpus from https://zenodo.org/records/3685274",
            file=sys.stderr,
        )
        return 1

    all_results: list[RunResult] = []

    if args.all:
        for lang in LANGUAGE_DIRS:
            for feats in FEATURE_SETS:
                try:
                    all_results.extend(run_one(lang, feats))
                except Exception as e:
                    print(f"  ERROR on ({lang}, {feats}): {e}", file=sys.stderr)
    else:
        if not args.language:
            parser.error("Provide --language (or use --all).")
        all_results = run_one(args.language, args.features)

    save_summary_csv(all_results)
    print_final_table(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
