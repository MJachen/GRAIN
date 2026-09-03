"""Single official binary-classification metric implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    acc: float
    f1: float
    rec: float
    auc: float
    pre: float
    spec: float
    npv: float
    threshold: float
    positive_class: int = 1

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_metrics(
    y_true: np.ndarray,
    positive_probability: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute every official metric; AUC always receives probabilities."""

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probability = np.asarray(positive_probability, dtype=np.float64).reshape(-1)
    if y_true.shape != probability.shape or y_true.size == 0:
        raise ValueError("Labels and probabilities must be non-empty aligned vectors")
    if not np.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("Positive probabilities must be finite and in [0,1]")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Both binary classes are required for ROC AUC")
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    return BinaryMetrics(
        acc=float(accuracy_score(y_true, prediction)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        rec=float(recall_score(y_true, prediction, pos_label=1, zero_division=0)),
        auc=float(roc_auc_score(y_true, probability)),
        pre=float(precision_score(y_true, prediction, pos_label=1, zero_division=0)),
        spec=float(specificity),
        npv=float(npv),
        threshold=float(threshold),
    )
