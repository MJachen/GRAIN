"""Cross-artifact validation for cohort and split metadata."""

from __future__ import annotations

from collections import Counter

from .schema import CohortManifest
from .splits import SplitManifest


def cohort_summary(cohort: CohortManifest) -> dict[str, object]:
    centers = Counter(record.center for record in cohort)
    labels = Counter(record.label for record in cohort)
    patterns = Counter(
        "plain+ce"
        if record.is_complete
        else "plain_only"
        if record.plain_available
        else "ce_only"
        for record in cohort
    )
    return {
        "patients": len(cohort),
        "centers": dict(sorted(centers.items())),
        "labels": {str(key): value for key, value in sorted(labels.items())},
        "modality_patterns": dict(sorted(patterns.items())),
    }


def validate_anchor_boundary(
    split: SplitManifest,
    anchor_patient_ids: set[str],
) -> None:
    forbidden = set(split.validation) | set(split.test)
    overlap = forbidden.intersection(anchor_patient_ids)
    if overlap:
        raise ValueError(f"Non-training patients found in anchor bank: {sorted(overlap)}")

