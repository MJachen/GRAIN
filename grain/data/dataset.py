"""Feature dataset whose explicit mask is authoritative."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .schema import CohortManifest, PatientRecord


class FeatureValidationError(ValueError):
    """Raised when a feature vector violates the configured data contract."""


FeatureLoader = Callable[[Path], np.ndarray]


def _default_loader(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


class FeatureDataset:
    """Return one explicit patient sample without inferring missingness from values.

    Shapes:
        plain: [D_plain]
        ce: [D_ce]
        mask: [2] in the fixed order [plain, ce]
        label: scalar integer
    """

    def __init__(
        self,
        manifest: CohortManifest,
        feature_root: str | Path,
        plain_dimension: int,
        ce_dimension: int,
        loader: FeatureLoader = _default_loader,
    ) -> None:
        self.manifest = manifest
        self.feature_root = Path(feature_root)
        self.dimensions = (int(plain_dimension), int(ce_dimension))
        self.loader = loader

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_modality(
        self,
        record: PatientRecord,
        available: bool,
        relative_path: str | None,
        expected_dimension: int,
        name: str,
    ) -> np.ndarray:
        if not available:
            return np.zeros(expected_dimension, dtype=np.float32)
        if relative_path is None:
            raise FeatureValidationError(
                f"Patient {record.patient_id}: available {name} modality has no path"
            )
        path = self.feature_root / relative_path
        value = np.asarray(self.loader(path), dtype=np.float32).reshape(-1)
        if value.shape != (expected_dimension,):
            raise FeatureValidationError(
                f"Patient {record.patient_id}: {name} shape {value.shape}, "
                f"expected {(expected_dimension,)}"
            )
        if not np.isfinite(value).all():
            raise FeatureValidationError(
                f"Patient {record.patient_id}: observed {name} feature contains NaN/Inf"
            )
        return value

    def __getitem__(self, index: int) -> dict[str, object]:
        record = tuple(self.manifest)[index]
        plain = self._load_modality(
            record,
            record.plain_available,
            record.plain_feature_path,
            self.dimensions[0],
            "plain",
        )
        ce = self._load_modality(
            record,
            record.ce_available,
            record.ce_feature_path,
            self.dimensions[1],
            "ce",
        )
        return {
            "patient_id": record.patient_id,
            "plain": plain,
            "ce": ce,
            "mask": np.asarray(record.modality_mask, dtype=np.bool_),
            "label": np.int64(record.label),
        }


def collate_samples(samples: Sequence[dict[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    return {
        "patient_id": [str(sample["patient_id"]) for sample in samples],
        "plain": np.stack([np.asarray(sample["plain"]) for sample in samples]),
        "ce": np.stack([np.asarray(sample["ce"]) for sample in samples]),
        "mask": np.stack([np.asarray(sample["mask"]) for sample in samples]),
        "label": np.asarray([sample["label"] for sample in samples], dtype=np.int64),
    }

