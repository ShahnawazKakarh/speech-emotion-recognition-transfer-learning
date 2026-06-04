"""Classification metrics: Weighted F1, Unweighted F1 (UAR), per-class F1, confusion matrix.

SER literature reports both WF1 (weighted by class frequency) and UF1 / UAR
(macro-average across classes). Both are essential because MELD and IEMOCAP
have severe class imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    label_names: list[str] | None = None,
) -> dict:
    """Compute the full metric suite reported in SER papers.

    Returns:
        dict with keys:
            - accuracy
            - weighted_f1 (WF1)
            - unweighted_f1 (UF1 / macro F1 / UAR proxy)
            - per_class_f1 (dict: name -> f1)
            - per_class_precision (dict)
            - per_class_recall (dict)
            - per_class_support (dict)
            - confusion_matrix (np.ndarray)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = int(max(y_true.max(), y_pred.max()) + 1) if len(y_true) else 0
    if label_names is None:
        label_names = [f"class_{i}" for i in range(n_classes)]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(label_names))), zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "unweighted_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {name: float(s) for name, s in zip(label_names, f1)},
        "per_class_precision": {name: float(s) for name, s in zip(label_names, precision)},
        "per_class_recall": {name: float(s) for name, s in zip(label_names, recall)},
        "per_class_support": {name: int(s) for name, s in zip(label_names, support)},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(label_names)))
        ),
    }


def format_metrics_table(metrics: dict, dataset_name: str = "") -> str:
    """Pretty-print a metrics dict for logging / README."""
    lines = []
    header = f"=== Metrics{' — ' + dataset_name if dataset_name else ''} ==="
    lines.append(header)
    lines.append(f"Accuracy:       {metrics['accuracy']:.4f}")
    lines.append(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    lines.append(f"Unweighted F1:  {metrics['unweighted_f1']:.4f}")
    lines.append("Per-class F1:")
    for name, f1 in metrics["per_class_f1"].items():
        support = metrics["per_class_support"][name]
        lines.append(f"  {name:<12} F1={f1:.4f}  (n={support})")
    return "\n".join(lines)
