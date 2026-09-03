"""Read-only inspection of legacy concatenated feature tables."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_legacy_table(
    path: str | Path,
    *,
    skip_leading_columns: int,
    plain_dimension: int,
    ce_dimension: int,
    label_column: str,
    confidence: str,
    paper_expected: dict[str, Any],
    code_evidence: list[str],
) -> dict[str, Any]:
    """Fingerprint one legacy CSV without modifying or rewriting it."""

    source = Path(path).resolve()
    frame = pd.read_csv(source)
    expected_columns = skip_leading_columns + plain_dimension + ce_dimension + 1
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"{source}: expected {expected_columns} columns, found {frame.shape[1]}"
        )
    if frame.columns[-1] != label_column:
        raise ValueError(
            f"{source}: expected label {label_column!r}, found {frame.columns[-1]!r}"
        )

    feature_start = skip_leading_columns
    plain = frame.iloc[:, feature_start : feature_start + plain_dimension]
    ce = frame.iloc[
        :, feature_start + plain_dimension : feature_start + plain_dimension + ce_dimension
    ]
    plain_available = ~plain.isna().all(axis=1)
    ce_available = ~ce.isna().all(axis=1)
    labels = pd.to_numeric(frame[label_column], errors="raise").astype(int)
    distribution = labels.value_counts().sort_index()
    first_column = str(frame.columns[0])
    identity_status = "blocked"
    identity_reason = "No configured stable patient ID column"
    if skip_leading_columns and not first_column.lower().startswith("unnamed"):
        identity_status = "candidate"
        identity_reason = f"Leading column {first_column!r} requires source verification"

    return {
        "absolute_path": source.as_posix(),
        "filename": source.name,
        "file_size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "label_column": label_column,
        "label_distribution": {
            "negative_0": int(distribution.get(0, 0)),
            "positive_1": int(distribution.get(1, 0)),
        },
        "feature_dimension": plain_dimension + ce_dimension,
        "modality_layout": {
            "skip_leading_columns": skip_leading_columns,
            "plain_columns_zero_based": [feature_start, feature_start + plain_dimension - 1],
            "ce_columns_zero_based": [
                feature_start + plain_dimension,
                feature_start + plain_dimension + ce_dimension - 1,
            ],
            "order": ["plain", "ce"],
            "availability_rule_for_inventory_only": "block is not entirely NaN",
        },
        "modality_availability": {
            "plain_only": int((plain_available & ~ce_available).sum()),
            "ce_only": int((~plain_available & ce_available).sum()),
            "both": int((plain_available & ce_available).sum()),
            "neither": int((~plain_available & ~ce_available).sum()),
        },
        "patient_identity": {
            "status": identity_status,
            "reason": identity_reason,
            "first_column": first_column,
        },
        "confidence": confidence.upper(),
        "paper_expected": paper_expected,
        "legacy_code_evidence": code_evidence,
    }

