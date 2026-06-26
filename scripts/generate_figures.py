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


def fig_methodology(agg: dict) -> None:
    """Schematic: 4 conditions × 3 seeds = 12 evaluations."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.8); ax.axis("off")

    # Two training boxes
    train_boxes = [
        (1.2, 6.3, "#dbeafe", "#1d4ed8", "Punjabi RASA\n(train 8.7k / val 962)\nfine-tune XLS-R-300M"),
        (1.2, 3.0, "#fee2e2", "#b91c1c", "URDU-Latif\n(3 shows train + 1 show test)\nfine-tune XLS-R-300M"),
    ]
    for x, y, fc, ec, label in train_boxes:
        ax.add_patch(plt.Rectangle((x, y), 3.6, 2.0, facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + 1.8, y + 1.0, label, ha="center", va="center", fontsize=9, color="#111")

    # Seed sub-boxes inside each
    for x, y, _, ec, _ in train_boxes:
        for s, sy in enumerate([1.6, 1.05, 0.5]):
            ax.add_patch(plt.Rectangle((x + 0.15, y + sy - 0.1), 0.5, 0.32,
                                       facecolor="white", edgecolor=ec, linewidth=0.6))
            ax.text(x + 0.4, y + sy + 0.06, f"s{s}", fontsize=7.5, ha="center", va="center", color=ec)

    # Evaluation boxes (4 conditions)
    eval_boxes = [
        (7.6, 7.0, "#dbeafe", "#1d4ed8", "Pun → Pun\n(within, n=962)"),
        (7.6, 5.3, "#fef3c7", "#b45309", "Pun → Urdu\n(cross, n=88 pooled)"),
        (7.6, 3.5, "#dcfce7", "#15803d", "Urdu → Pun\n(cross, n=2,886 pooled)"),
        (7.6, 1.8, "#fee2e2", "#b91c1c", "Urdu → Urdu\n(within, n=88 pooled)"),
    ]
    for x, y, fc, ec, label in eval_boxes:
        ax.add_patch(plt.Rectangle((x, y), 3.2, 1.3, facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(x + 1.6, y + 0.65, label, ha="center", va="center", fontsize=8.5)

    # Arrows
    arrow_kw = dict(arrowstyle="->", linewidth=1.0, color="#475569")
    ax.annotate("", xy=(7.55, 7.6), xytext=(4.85, 7.3), arrowprops=arrow_kw)   # Pun -> Pun
    ax.annotate("", xy=(7.55, 5.9), xytext=(4.85, 7.1), arrowprops=arrow_kw)   # Pun -> Urdu
    ax.annotate("", xy=(7.55, 4.1), xytext=(4.85, 4.0), arrowprops=arrow_kw)   # Urdu -> Pun
    ax.annotate("", xy=(7.55, 2.4), xytext=(4.85, 3.8), arrowprops=arrow_kw)   # Urdu -> Urdu

    ax.text(6.0, 8.55, "4 conditions × 3 seeds = 12 evaluations", ha="center",
            fontsize=10.5, fontweight="bold", color="#111")
    ax.text(3.0, 0.2, "Training (3 independent seeds per language)",
            ha="center", fontsize=8.5, color="#666", style="italic")
    ax.text(9.2, 0.2, "Evaluation (each model × both test sets)",
            ha="center", fontsize=8.5, color="#666", style="italic")

    _save_both(fig, "fig_methodology")


def fig_c2_feature_inversion() -> None:
    """C2 grouped bar chart: within-language vs cross-corpus UAR per feature set (best classifier)."""
    import csv
    csv_path = REPO_ROOT / "results" / "urdu_sindhi" / "cross_corpus_summary.csv"
    if not csv_path.is_file():
        print(f"[skip] {csv_path} not found, skipping C2 inversion chart")
        return

    # Pull the BEST cross-corpus UAR per (feature set, direction) and the corresponding
    # within-language target from the same row.
    feature_order = ["eGeMAPS", "ComParE", "IS09", "IS10", "Prosody"]
    data = {f: {"u2s_within": None, "u2s_cross": 0.0, "s2u_within": None, "s2u_cross": 0.0}
            for f in feature_order}

    with csv_path.open() as f:
        for row in csv.DictReader(f):
            feat = row["features"]
            if feat not in data:
                continue
            cross = float(row["uar"])
            within = float(row["within_language_uar_target"])
            direction = (row["train_language"], row["test_language"])
            if direction == ("urdu", "sindhi"):
                if cross > data[feat]["u2s_cross"]:
                    data[feat]["u2s_cross"] = cross
                    data[feat]["u2s_within"] = within
            elif direction == ("sindhi", "urdu"):
                if cross > data[feat]["s2u_cross"]:
                    data[feat]["s2u_cross"] = cross
                    data[feat]["s2u_within"] = within

    x = np.arange(len(feature_order))
    width = 0.2

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    u2s_within = [data[f]["u2s_within"] for f in feature_order]
    u2s_cross  = [data[f]["u2s_cross"]  for f in feature_order]
    s2u_within = [data[f]["s2u_within"] for f in feature_order]
    s2u_cross  = [data[f]["s2u_cross"]  for f in feature_order]

    ax.bar(x - 1.5*width, u2s_within, width, label="Urdu within-language (target)",
           color="#93c5fd", edgecolor="#1d4ed8", linewidth=0.6)
    ax.bar(x - 0.5*width, u2s_cross,  width, label="Urdu → Sindhi cross-corpus",
           color="#1d4ed8", edgecolor="#1d4ed8", linewidth=0.6)
    ax.bar(x + 0.5*width, s2u_within, width, label="Sindhi within-language (target)",
           color="#fca5a5", edgecolor="#b91c1c", linewidth=0.6)
    ax.bar(x + 1.5*width, s2u_cross,  width, label="Sindhi → Urdu cross-corpus",
           color="#b91c1c", edgecolor="#b91c1c", linewidth=0.6)

    ax.axhline(0.143, color="#666", linestyle="--", linewidth=0.8)
    ax.text(len(feature_order) - 0.5, 0.155, "chance (1/7 = 0.143)", fontsize=8, color="#666")

    ax.set_xticks(x); ax.set_xticklabels(feature_order, fontsize=9.5)
    ax.set_ylabel("UAR")
    ax.set_ylim(0, 0.7)
    ax.set_title("C2: Feature-set ranking inversion under cross-corpus transfer", pad=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.set_axisbelow(True); ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    _save_both(fig, "fig_c2_feature_inversion")


def fig_condition_overview(agg: dict) -> None:
    """Grouped bar chart: UAR + WF1 for all 4 C4 conditions side-by-side."""
    labels = [s.replace("\n", " ") for s in SHORT_LABELS]
    uar_means, uar_los, uar_his = [], [], []
    wf1_means, wf1_los, wf1_his = [], [], []
    for label, _ in CONDITIONS:
        entry = agg.get(label, {})
        pooled = entry.get("pooled") or {}
        u = pooled.get("uar", {}); w = pooled.get("wf1", {})
        uar_means.append(u.get("mean", 0)); uar_los.append(u.get("ci_lo", 0)); uar_his.append(u.get("ci_hi", 0))
        wf1_means.append(w.get("mean", 0)); wf1_los.append(w.get("ci_lo", 0)); wf1_his.append(w.get("ci_hi", 0))

    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    uar_err = [[m - lo for m, lo in zip(uar_means, uar_los)],
               [hi - m for m, hi in zip(uar_means, uar_his)]]
    wf1_err = [[m - lo for m, lo in zip(wf1_means, wf1_los)],
               [hi - m for m, hi in zip(wf1_means, wf1_his)]]

    ax.bar(x - width/2, uar_means, width, yerr=uar_err, label="UAR",
           color="#3b82f6", edgecolor="#1e3a8a", linewidth=0.5, capsize=3.5,
           error_kw=dict(elinewidth=0.8, ecolor="#1e3a8a"))
    ax.bar(x + width/2, wf1_means, width, yerr=wf1_err, label="Weighted F1",
           color="#f59e0b", edgecolor="#92400e", linewidth=0.5, capsize=3.5,
           error_kw=dict(elinewidth=0.8, ecolor="#92400e"))
    ax.axhline(CHANCE, color="#666", linestyle="--", linewidth=0.8)
    ax.text(len(labels) - 0.5, CHANCE + 0.015, "chance (0.25)", fontsize=8, color="#666")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("C4: pooled-prediction UAR and Weighted F1 across the 4 conditions (95% CI)", pad=8)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.set_axisbelow(True); ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    _save_both(fig, "fig_condition_overview")


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise SystemExit(f"Results not found at {RESULTS_DIR}")
    agg = _load_aggregate()
    print(f"[load] {len(agg)} condition(s) in aggregate.json")

    fig_methodology(agg)
    fig_c2_feature_inversion()
    fig_condition_overview(agg)
    fig_per_seed_uar(agg)
    fig_confusion_matrices(agg)
    fig_per_class_heatmap(agg)

    print(f"\n[done] 6 figures written to {FIGS_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
