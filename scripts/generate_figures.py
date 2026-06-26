"""Generate paper figures from existing results JSONs.

Produces three PDF figures used in paper_v2.pdf (v2.1.0+):

    fig_per_seed_uar         — forest-plot of per-seed UAR + pooled CI for the 4 conditions
    fig_confusion_matrices   — 4-panel pooled confusion matrices for §3.4 conditions
    fig_per_class_heatmap    — per-class recall heatmap (condition × emotion)

Reads from:
    results/multiseed_cross_lingual/aggregate.json     — bootstrap CIs
    results/multiseed_cross_lingual/{pun,urdu}_s{0,1,2}_on_{pun,urdu}.json — per-seed CMs

Writes to:
    docs/writeups/cross_corpus_paper/figures/*.png  (300 DPI, for WeasyPrint embed)
    docs/writeups/cross_corpus_paper/figures/*.pdf  (vector, for archival / standalone view)

Usage:
    cd ~/speech-emotion-recognition-transfer-learning
    python scripts/generate_figures.py

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "multiseed_cross_lingual"
FIGS_DIR = REPO_ROOT / "docs" / "writeups" / "cross_corpus_paper" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Angry", "Happy", "Neutral", "Sad"]

CONDITIONS = [
    ("Pun → Pun  (within)",                "pun_s*_on_pun.json"),
    ("Urdu → Urdu (within, show-indep)",   "urdu_s*_on_urdu.json"),
    ("Pun → Urdu (cross-lingual)",         "pun_s*_on_urdu.json"),
    ("Urdu → Pun (cross-lingual)",         "urdu_s*_on_pun.json"),
]
SHORT_LABELS = [
    "Pun → Pun\n(within)",
    "Urdu → Urdu\n(show-indep)",
    "Pun → Urdu\n(cross-lingual)",
    "Urdu → Pun\n(cross-lingual)",
]
CHANCE = 0.25  # 4-class macro-recall chance baseline

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def _save_both(fig: plt.Figure, basename: str) -> None:
    """Save the figure as both PNG (for WeasyPrint embed) and PDF (for archival)."""
    fig.savefig(FIGS_DIR / f"{basename}.png")
    fig.savefig(FIGS_DIR / f"{basename}.pdf")
    plt.close(fig)
    print(f"[saved] figures/{basename}.{{png,pdf}}")


def _load_aggregate() -> dict:
    return json.loads((RESULTS_DIR / "aggregate.json").read_text())


def _pooled_confusion_matrix(pattern: str) -> np.ndarray:
    """Sum confusion matrices across per-seed JSONs matching the glob pattern."""
    paths = sorted(RESULTS_DIR.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No JSONs matched: {pattern}")
    cms = []
    for p in paths:
        d = json.loads(p.read_text())
        cms.append(np.array(d["confusion_matrix"], dtype=float))
    return np.sum(cms, axis=0)


def fig_per_seed_uar(agg: dict) -> None:
    """Forest plot: per-seed dots + pooled-CI bar for the 4 conditions."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    y_positions = np.arange(len(CONDITIONS))[::-1]  # top-to-bottom in declared order
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for i, ((label, _), y, color) in enumerate(zip(CONDITIONS, y_positions, colors)):
        if label not in agg:
            continue
        entry = agg[label]
        per_seed = entry["per_seed_uar"]
        pooled = entry.get("pooled")

        # Per-seed dots (jittered horizontally for readability if duplicates)
        x_jitter = np.linspace(-0.04, 0.04, len(per_seed)) if len(per_seed) > 1 else [0.0]
        for x, j in zip(per_seed, x_jitter):
            ax.scatter(x, y + j, s=42, color=color, alpha=0.55, edgecolor="white",
                       linewidth=0.6, zorder=3)

        # Pooled mean diamond + 95% CI bar
        if pooled:
            pm = pooled["uar"]["mean"]; lo = pooled["uar"]["ci_lo"]; hi = pooled["uar"]["ci_hi"]
            ax.errorbar(pm, y, xerr=[[pm - lo], [hi - pm]], fmt="D",
                        color=color, markersize=8, markerfacecolor=color,
                        markeredgecolor="black", markeredgewidth=0.7,
                        ecolor="black", elinewidth=1.0, capsize=4, capthick=1.0, zorder=4)

    ax.axvline(CHANCE, color="#666", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(CHANCE + 0.005, len(CONDITIONS) - 0.4, "chance (0.25)", fontsize=8, color="#666")
    ax.axvline(1.0, color="#ccc", linestyle=":", linewidth=0.6, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9)
    ax.set_xlim(-0.02, 1.05)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("UAR (macro-averaged recall)")
    ax.set_title("Per-seed UAR (dots) and pooled-prediction 95% bootstrap CI (diamond + bar)",
                 pad=10, fontweight="normal")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.spines["left"].set_visible(False)

    _save_both(fig, "fig_per_seed_uar")


def fig_confusion_matrices(agg: dict) -> None:
    """2×2 grid of pooled confusion matrices (one per §3.4 condition)."""
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4))
    axes = axes.flatten()

    titles = [
        "Pun → Pun (within, publisher split)",
        "Urdu → Urdu (within, show-independent)",
        "Pun → Urdu (zero-shot cross-lingual)",
        "Urdu → Pun (zero-shot cross-lingual)",
    ]

    for ax, (label, pattern), title in zip(axes, CONDITIONS, titles):
        cm = _pooled_confusion_matrix(pattern)
        # Row-normalize to per-class recall view
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(title, fontsize=10, pad=6)

        # Annotate cells with the raw count (and recall on diagonal)
        for i in range(4):
            for j in range(4):
                count = int(cm[i, j])
                norm = cm_norm[i, j]
                if count == 0:
                    txt = "0"
                else:
                    txt = f"{count}\n{norm:.2f}"
                color = "white" if norm > 0.55 else "#222"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

        ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
        ax.spines["left"].set_linewidth(0.4); ax.spines["bottom"].set_linewidth(0.4)
        ax.spines["top"].set_linewidth(0.4); ax.spines["right"].set_linewidth(0.4)

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.04, shrink=0.65)
    cbar.set_label("Row-normalised (per-class recall)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Pooled confusion matrices across 3 seeds per condition",
                 fontsize=11.5, y=0.995, fontweight="normal")
    _save_both(fig, "fig_confusion_matrices")


def fig_per_class_heatmap(agg: dict) -> None:
    """Heatmap of per-class recall mean across conditions."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    matrix = np.zeros((len(CONDITIONS), 4))
    for i, (label, _) in enumerate(CONDITIONS):
        if label not in agg:
            continue
        for j, c in enumerate(["angry", "happy", "neutral", "sad"]):
            matrix[i, j] = agg[label]["per_class_recall"][c]["mean"]

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9.5)
    ax.set_yticklabels([s.replace("\n", " ") for s in SHORT_LABELS], fontsize=9)

    for i in range(len(CONDITIONS)):
        for j in range(4):
            v = matrix[i, j]
            color = "#111" if 0.35 <= v <= 0.75 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    color=color, fontweight="bold")

    ax.set_title("Per-class recall (mean across 3 seeds)", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.85)
    cbar.set_label("Recall", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    _save_both(fig, "fig_per_class_heatmap")


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Results not found at {RESULTS_DIR}")
    agg = _load_aggregate()
    print(f"[load] {len(agg)} condition(s) in aggregate.json")

    fig_per_seed_uar(agg)
    fig_confusion_matrices(agg)
    fig_per_class_heatmap(agg)

    print(f"\n[done] 3 figures written to {FIGS_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
