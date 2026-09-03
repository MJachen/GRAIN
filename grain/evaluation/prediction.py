"""Patient/sample-aligned prediction records and artifact writing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import BinaryMetrics, compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    split: str
    fold: int
    sample_ids: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray
    loss: float
    metrics: BinaryMetrics

    @classmethod
    def create(
        cls,
        *,
        split: str,
        fold: int,
        sample_ids: tuple[str, ...],
        labels: np.ndarray,
        probabilities: np.ndarray,
        loss: float,
        threshold: float,
    ) -> "EvaluationResult":
        return cls(
            split=split,
            fold=int(fold),
            sample_ids=sample_ids,
            labels=np.asarray(labels, dtype=np.int64),
            probabilities=np.asarray(probabilities, dtype=np.float64),
            loss=float(loss),
            metrics=compute_metrics(labels, probabilities, threshold),
        )

    def write_csv(self, path: str | Path, threshold: float = 0.5) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "patient_id_or_stable_sample_id",
                    "fold",
                    "split",
                    "label",
                    "probability",
                    "prediction",
                ],
            )
            writer.writeheader()
            for sample_id, label, probability in zip(
                self.sample_ids, self.labels, self.probabilities
            ):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "patient_id_or_stable_sample_id": sample_id,
                        "fold": self.fold,
                        "split": self.split,
                        "label": int(label),
                        "probability": float(probability),
                        "prediction": int(probability >= threshold),
                    }
                )
