from __future__ import annotations

from grain.data.schema import CohortManifest, PatientRecord


def synthetic_cohort(size: int = 40) -> CohortManifest:
    records = []
    for index in range(size):
        records.append(
            PatientRecord(
                patient_id=f"P{index:04d}",
                center="A",
                label=index % 2,
                plain_available=True,
                ce_available=index % 3 != 0,
                plain_feature_path=f"plain/P{index:04d}.npy",
                ce_feature_path=(f"ce/P{index:04d}.npy" if index % 3 != 0 else None),
            )
        )
    return CohortManifest(records)

