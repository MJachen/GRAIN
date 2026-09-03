"""Training-only anchor identity boundary for future graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import PatientRecord


class AnchorPolicyError(ValueError):
    """Raised when validation/test identity enters a training anchor bank."""


@dataclass(frozen=True)
class AnchorBank:
    """Immutable IDs of complete-modality patients from one training split only."""

    training_patient_ids: frozenset[str]
    anchor_patient_ids: tuple[str, ...]

    @classmethod
    def from_training_records(
        cls,
        records: Iterable[PatientRecord],
        training_patient_ids: Iterable[str],
    ) -> "AnchorBank":
        training_ids = frozenset(training_patient_ids)
        record_map = {record.patient_id: record for record in records}
        unknown = training_ids.difference(record_map)
        if unknown:
            raise AnchorPolicyError(f"Unknown training patient IDs: {sorted(unknown)}")
        anchors = tuple(
            sorted(patient_id for patient_id in training_ids if record_map[patient_id].is_complete)
        )
        if not anchors:
            raise AnchorPolicyError("Training split has no complete-modality anchors")
        return cls(training_patient_ids=training_ids, anchor_patient_ids=anchors)

    def eligible_ids_for(self, query_patient_id: str) -> tuple[str, ...]:
        """Return every eligible training anchor, excluding the query itself."""

        return tuple(
            patient_id
            for patient_id in self.anchor_patient_ids
            if patient_id != query_patient_id
        )

    def effective_k(self, query_patient_id: str, requested_k: int) -> int:
        """Cap K without choosing neighbors before Phase 3 similarity ranking."""

        if requested_k < 1:
            raise AnchorPolicyError("requested_k must be positive")
        return min(requested_k, len(self.eligible_ids_for(query_patient_id)))

    def assert_no_forbidden_ids(self, forbidden_patient_ids: Iterable[str]) -> None:
        overlap = set(forbidden_patient_ids).intersection(self.anchor_patient_ids)
        if overlap:
            raise AnchorPolicyError(
                f"Validation/test patients entered the anchor bank: {sorted(overlap)}"
            )
