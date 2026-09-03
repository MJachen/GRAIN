"""Cohort schema with explicit patient identity and modality availability."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


class CohortValidationError(ValueError):
    """Raised when cohort metadata are ambiguous or leakage-prone."""


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise CohortValidationError(f"Invalid boolean value: {value!r}")


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    center: str
    label: int
    plain_available: bool
    ce_available: bool
    plain_feature_path: str | None
    ce_feature_path: str | None

    @property
    def modality_mask(self) -> tuple[bool, bool]:
        """Mask order is always (plain, contrast-enhanced)."""

        return self.plain_available, self.ce_available

    @property
    def is_complete(self) -> bool:
        return self.plain_available and self.ce_available

    def validate(self) -> None:
        if not self.patient_id.strip():
            raise CohortValidationError("patient_id must be non-empty")
        if self.label not in (0, 1):
            raise CohortValidationError(
                f"Patient {self.patient_id}: label must be 0 or 1, got {self.label}"
            )
        if not (self.plain_available or self.ce_available):
            raise CohortValidationError(
                f"Patient {self.patient_id}: at least one CT modality is required"
            )
        if self.plain_available and not self.plain_feature_path:
            raise CohortValidationError(
                f"Patient {self.patient_id}: plain modality is available but path is blank"
            )
        if self.ce_available and not self.ce_feature_path:
            raise CohortValidationError(
                f"Patient {self.patient_id}: CE modality is available but path is blank"
            )


class CohortManifest:
    """Validated one-row-per-patient cohort metadata."""

    REQUIRED_COLUMNS = {
        "patient_id",
        "center",
        "label",
        "plain_available",
        "ce_available",
        "plain_feature_path",
        "ce_feature_path",
    }

    def __init__(self, records: Iterable[PatientRecord]):
        self._records = tuple(records)
        if not self._records:
            raise CohortValidationError("Cohort manifest is empty")
        seen: set[str] = set()
        for record in self._records:
            record.validate()
            if record.patient_id in seen:
                raise CohortValidationError(
                    f"Duplicate patient_id would invalidate patient-level splitting: "
                    f"{record.patient_id}"
                )
            seen.add(record.patient_id)

    @classmethod
    def from_csv(cls, path: str | Path) -> "CohortManifest":
        path = Path(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = cls.REQUIRED_COLUMNS.difference(fields)
            if missing:
                raise CohortValidationError(
                    f"Manifest is missing explicit columns: {sorted(missing)}"
                )
            records = []
            for row in reader:
                records.append(
                    PatientRecord(
                        patient_id=row["patient_id"].strip(),
                        center=row["center"].strip(),
                        label=int(row["label"]),
                        plain_available=parse_bool(row["plain_available"]),
                        ce_available=parse_bool(row["ce_available"]),
                        plain_feature_path=row["plain_feature_path"].strip() or None,
                        ce_feature_path=row["ce_feature_path"].strip() or None,
                    )
                )
        return cls(records)

    def __iter__(self) -> Iterator[PatientRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def patient_ids(self) -> tuple[str, ...]:
        return tuple(record.patient_id for record in self._records)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(record.label for record in self._records)

    def by_id(self) -> dict[str, PatientRecord]:
        return {record.patient_id: record for record in self._records}

