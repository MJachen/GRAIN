"""Deterministic patient-level outer CV and inner validation manifests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .schema import CohortManifest


class SplitValidationError(ValueError):
    """Raised when a split overlaps, omits patients or is not reproducible."""


@dataclass(frozen=True)
class SplitManifest:
    schema_version: int
    dataset_name: str
    outer_fold: int
    seed: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    k_candidates: tuple[int, ...] = tuple(range(2, 11))
    selected_k: int | None = None

    def validate(self, expected_patient_ids: Iterable[str]) -> None:
        train, validation, test = set(self.train), set(self.validation), set(self.test)
        if train & validation or train & test or validation & test:
            raise SplitValidationError(
                f"Fold {self.outer_fold}: train/validation/test patient overlap"
            )
        expected = set(expected_patient_ids)
        assigned = train | validation | test
        if assigned != expected:
            missing = expected.difference(assigned)
            unknown = assigned.difference(expected)
            raise SplitValidationError(
                f"Fold {self.outer_fold}: missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not train or not validation or not test:
            raise SplitValidationError(f"Fold {self.outer_fold}: empty partition")
        if tuple(self.k_candidates) != tuple(range(2, 11)):
            raise SplitValidationError("K candidates must be exactly 2 through 10")
        if self.selected_k is not None and self.selected_k not in self.k_candidates:
            raise SplitValidationError("selected_k is outside the candidate set")

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["train"] = list(self.train)
        payload["validation"] = list(self.validation)
        payload["test"] = list(self.test)
        payload["k_candidates"] = list(self.k_candidates)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SplitManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("train", "validation", "test", "k_candidates"):
            payload[key] = tuple(payload[key])
        return cls(**payload)


def generate_nested_manifests(
    cohort: CohortManifest,
    dataset_name: str,
    n_outer_folds: int = 10,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> list[SplitManifest]:
    """Generate outer-test and inner-validation splits using patient IDs.

    The validation split is drawn only from the current outer-development set.
    Outer-test IDs are never passed to the inner split function.
    """

    patient_ids = np.asarray(cohort.patient_ids, dtype=object)
    labels = np.asarray(cohort.labels, dtype=np.int64)
    if len(np.unique(patient_ids)) != len(patient_ids):
        raise SplitValidationError("patient_id values must be unique before splitting")
    class_counts = np.bincount(labels, minlength=2)
    if class_counts.min() < n_outer_folds:
        raise SplitValidationError(
            f"Each class needs at least {n_outer_folds} patients; got {class_counts.tolist()}"
        )

    outer = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    manifests: list[SplitManifest] = []
    for fold_index, (development_index, test_index) in enumerate(
        outer.split(patient_ids, labels), start=1
    ):
        development_ids = patient_ids[development_index]
        development_labels = labels[development_index]
        train_ids, validation_ids = train_test_split(
            development_ids,
            test_size=validation_fraction,
            stratify=development_labels,
            random_state=seed + fold_index,
        )
        manifest = SplitManifest(
            schema_version=1,
            dataset_name=dataset_name,
            outer_fold=fold_index,
            seed=seed,
            train=tuple(sorted(str(value) for value in train_ids)),
            validation=tuple(sorted(str(value) for value in validation_ids)),
            test=tuple(sorted(str(value) for value in patient_ids[test_index])),
        )
        manifest.validate(cohort.patient_ids)
        manifests.append(manifest)
    return manifests


def write_manifests(manifests: Iterable[SplitManifest], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    for manifest in manifests:
        filename = f"{manifest.dataset_name}_fold_{manifest.outer_fold:02d}.json"
        manifest.to_json(output_dir / filename)

