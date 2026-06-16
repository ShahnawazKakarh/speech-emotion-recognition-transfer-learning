"""Aggregate multi-seed cross-lingual results into mean +/- 95% bootstrap CIs.

Reads per-seed JSON files written by `scripts/eval_checkpoint.py` (via the
multiseed orchestrator) and produces:

  1. A console-friendly summary table (mean +/- bootstrap CI).
  2. A LaTeX-ready table (`results/multiseed_cross_lingual/summary.tex`).
  3. A flat CSV (`results/multiseed_cross_lingual/summary.csv`).
  4. A consolidated machine-readable JSON
     (`results/multiseed_cross_lingual/aggregate.json`).

Metrics aggregated:
    - UAR (macro-averaged recall)
    - Weighted F1
    - Per-class recall (angry, happy, neutral, sad)

CIs are 95% percentile bootstrap intervals over the *seed dimension*
(n=3 by default). With only 3 seeds, bootstrap CIs are wide and indicative
rather than tight -- this is honest given the sample size.

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results/multiseed_cross_lingual"
CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

# (condition_label, file_glob_pattern) -- matches run_multiseed_cross_lingual outputs.
CONDITIONS = [
    ("Pun -> Pun  (within)",        "pun_s*_on_pun.json"),
    ("Urdu -> Urdu (within, show-indep)", "urdu_s*_on_urdu.json"),
    ("Pun -> Urdu (cross-lingual)", "pun_s*_on_urdu.json"),
    ("Urdu -> Pun (cross-lingual)", "urdu_s*_on_pun.json"),
]


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, rng_seed: int = 42
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a 95% percentile bootstrap CI."""
    rng = np.random.default_rng(rng_seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        v = float(values[0])
        return v, v, v
    samples = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return float(values.mean()), lo, hi


def load_condition(pattern: str) -> list[dict]:
    paths = sorted(RESULTS_DIR.glob(pattern))
    out = []
    for p in paths:
        out.append(json.loads(p.read_text()))
    return out


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def bootstrap_ci_pooled(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_boot: int = 10_000, rng_seed: int = 42):
    """Bootstrap CI over the sample dimension (paired y_true, y_pred)."""
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])
    return float(metric_fn(y_true, y_pred)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def compute_pooled_block(runs: list[dict], *, n_boot: int = 10_000) -> dict | None:
    """Concatenate predictions across seeds and bootstrap over sample dimension.

    Reads y_true / y_pred reconstructed from each run's confusion matrix (this
    is sufficient because the predictions arrive in fixed order from the test
    DataLoader).
    """
    # We don't have per-sample y_true/y_pred stored; reconstruct from CM by
    # emitting (true, pred) counts from the matrix. CIs computed on this pool.
    y_true_all = []
    y_pred_all = []
    for r in runs:
        cm = np.array(r["confusion_matrix"])
        for t in range(cm.shape[0]):
            for p in range(cm.shape[1]):
                count = int(cm[t, p])
                if count > 0:
                    y_true_all.extend([t] * count)
                    y_pred_all.extend([p] * count)
    if not y_true_all:
        return None
    y_true_all = np.asarray(y_true_all, dtype=int)
    y_pred_all = np.asarray(y_pred_all, dtype=int)

    uar_m, uar_lo, uar_hi = bootstrap_ci_pooled(
        y_true_all, y_pred_all,
        lambda t, p: recall_score(t, p, average="macro", labels=list(range(4)), zero_division=0),
        n_boot=n_boot,
    )
    wf1_m, wf1_lo, wf1_hi = bootstrap_ci_pooled(
        y_true_all, y_pred_all,
        lambda t, p: f1_score(t, p, average="weighted", labels=list(range(4)), zero_division=0),
        n_boot=n_boot,
    )
    acc_m, acc_lo, acc_hi = bootstrap_ci_pooled(
        y_true_all, y_pred_all, accuracy_score, n_boot=n_boot,
    )

    per_class = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        per_class[c_name] = bootstrap_ci_pooled(
            y_true_all, y_pred_all,
            lambda t, p, k=c_idx: recall_score(t, p, labels=list(range(4)), average=None, zero_division=0)[k],
        )

    return {
        "n_pooled": int(len(y_true_all)),
        "uar": {"mean": uar_m, "ci_lo": uar_lo, "ci_hi": uar_hi},
        "wf1": {"mean": wf1_m, "ci_lo": wf1_lo, "ci_hi": wf1_hi},
        "accuracy": {"mean": acc_m, "ci_lo": acc_lo, "ci_hi": acc_hi},
        "per_class_recall": {
            c: {"mean": m, "ci_lo": lo, "ci_hi": hi}
            for c, (m, lo, hi) in per_class.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"No results directory at {RESULTS_DIR}")

    aggregate: dict[str, dict] = {}
    rows_for_csv: list[dict] = []

    # ----- console + structured aggregation -----
    print()
    print("=" * 92)
    print(f"  {'Condition':<40} {'UAR (mean [95% CI])':<26} {'WF1 (mean [95% CI])':<26}")
    print("=" * 92)

    for label, pattern in CONDITIONS:
        runs = load_condition(pattern)
        if not runs:
            print(f"  {label:<40} NO RUNS FOUND for pattern '{pattern}'")
            continue

        uars = np.array([r["metrics"]["uar_macro_recall"] for r in runs])
        wf1s = np.array([r["metrics"]["weighted_f1"] for r in runs])

        uar_mean, uar_lo, uar_hi = bootstrap_ci(uars, n_boot=args.n_boot)
        wf1_mean, wf1_lo, wf1_hi = bootstrap_ci(wf1s, n_boot=args.n_boot)

        per_class_recall = {}
        for c in CLASS_NAMES:
            recalls = np.array([r["per_class"][c]["recall"] for r in runs])
            per_class_recall[c] = bootstrap_ci(recalls, n_boot=args.n_boot)

        pooled = compute_pooled_block(runs, n_boot=args.n_boot)

        agg_entry = {
            "n_seeds": int(len(runs)),
            "per_seed_uar": uars.tolist(),
            "per_seed_wf1": wf1s.tolist(),
            "uar":   {"mean": uar_mean, "ci_lo": uar_lo, "ci_hi": uar_hi},
            "wf1":   {"mean": wf1_mean, "ci_lo": wf1_lo, "ci_hi": wf1_hi},
            "per_class_recall": {
                c: {"mean": m, "ci_lo": lo, "ci_hi": hi}
                for c, (m, lo, hi) in per_class_recall.items()
            },
            "pooled": pooled,
            "source_files": [str(Path(r["source_ckpt"]).name) for r in runs],
        }
        aggregate[label] = agg_entry

        print(
            f"  {label:<40} {fmt_ci(uar_mean, uar_lo, uar_hi):<26} "
            f"{fmt_ci(wf1_mean, wf1_lo, wf1_hi):<26}"
        )

        rows_for_csv.append({
            "condition": label,
            "n_seeds": len(runs),
            "uar_mean": uar_mean, "uar_ci_lo": uar_lo, "uar_ci_hi": uar_hi,
            "wf1_mean": wf1_mean, "wf1_ci_lo": wf1_lo, "wf1_ci_hi": wf1_hi,
            **{f"recall_{c}_mean": per_class_recall[c][0] for c in CLASS_NAMES},
            **{f"recall_{c}_ci_lo": per_class_recall[c][1] for c in CLASS_NAMES},
            **{f"recall_{c}_ci_hi": per_class_recall[c][2] for c in CLASS_NAMES},
        })

    print("=" * 92)

    # ----- per-class recall breakout -----
    print()
    print("Per-class recall (mean [95% CI]) -- diagnoses arousal-vs-valence transfer asymmetry:")
    print()
    header = f"  {'Condition':<40} " + " ".join(f"{c:<18}" for c in CLASS_NAMES)
    print(header)
    print("-" * len(header))
    for label, _ in CONDITIONS:
        if label not in aggregate:
            continue
        row = aggregate[label]["per_class_recall"]
        cells = "  ".join(fmt_ci(row[c]["mean"], row[c]["ci_lo"], row[c]["ci_hi"]).ljust(18) for c in CLASS_NAMES)
        print(f"  {label:<40} {cells}")

    # ----- pooled (sample-level) metrics -----
    print()
    print("Pooled metrics (concatenate predictions across seeds, bootstrap over SAMPLES, not seeds).")
    print("For Pun->Urdu this is the methodologically cleanest single number because per-seed CIs")
    print("would otherwise span DIFFERENT held-out shows (SM2 / SM3 / SM4) with tiny test sizes.")
    print()
    print(f"  {'Condition':<40} {'n':<6} {'UAR (mean [95% CI])':<26} {'WF1 (mean [95% CI])':<26}")
    print("-" * 100)
    for label, _ in CONDITIONS:
        if label not in aggregate:
            continue
        p = aggregate[label].get("pooled")
        if p is None:
            continue
        print(
            f"  {label:<40} {p['n_pooled']:<6} "
            f"{fmt_ci(p['uar']['mean'], p['uar']['ci_lo'], p['uar']['ci_hi']):<26} "
            f"{fmt_ci(p['wf1']['mean'], p['wf1']['ci_lo'], p['wf1']['ci_hi']):<26}"
        )

    # ----- CSV -----
    csv_path = RESULTS_DIR / "summary.csv"
    if rows_for_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"\n[saved] {csv_path.relative_to(REPO_ROOT)}")

    # ----- aggregate JSON -----
    json_path = RESULTS_DIR / "aggregate.json"
    json_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[saved] {json_path.relative_to(REPO_ROOT)}")

    # ----- LaTeX table -----
    tex_path = RESULTS_DIR / "summary.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-lingual SER on Punjabi RASA and URDU-Latif (XLS-R-300M)."
        r" Mean $\pm$ 95\% percentile bootstrap CI across 3 seeds. Urdu seeds use"
        r" show-independent splits (SM2/SM3/SM4 as held-out test); Punjabi seeds"
        r" use the publisher-fixed test set with varied train/val partition.}",
        r"\label{tab:multiseed_xlingual}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Condition & UAR & WF1 \\",
        r"\midrule",
    ]
    for label, _ in CONDITIONS:
        if label not in aggregate:
            continue
        u = aggregate[label]["uar"]
        w = aggregate[label]["wf1"]
        # Escape arrow for LaTeX
        tex_label = label.replace("->", r"$\rightarrow$")
        lines.append(
            f"{tex_label} & "
            f"{u['mean']:.3f} [{u['ci_lo']:.3f}, {u['ci_hi']:.3f}] & "
            f"{w['mean']:.3f} [{w['ci_lo']:.3f}, {w['ci_hi']:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] {tex_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
