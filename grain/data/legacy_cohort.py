"""Read-only adapter for bound legacy clinical feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True)
class LegacyFeatureCohort:
    """In-memory view of one legacy CSV without modifying its source.

    ``sample_ids`` are experiment-local stable row identifiers. They are not
    recovered clinical patient identities.
    """

    sample_ids: tuple[str, ...]
    features: Tensor  # [N, M=2, D]
    modality_mask: Tensor  # boolean [N, M=2]
    labels: Tensor  # [N]
    source_path: str
    source_sha256: str
    id_kind: str = "experiment_local_stable_sample_id"

    def __post_init__(self) -> None:
        count = len(self.sample_ids)
        if len(set(self.sample_ids)) != count:
            raise ValueError("Experiment-local sample IDs must be unique")
        if self.features.ndim != 3 or self.features.shape[:2] != (count, 2):
            raise ValueError("features must have shape [N, 2, D]")
        if self.modality_mask.shape != (count, 2) or self.modality_mask.dtype != torch.bool:
            raise ValueError("modality_mask must be boolean [N, 2]")
        if self.labels.shape != (count,) or not bool(((self.labels == 0) | (self.labels == 1)).all()):
            raise ValueError("labels must be binary with shape [N]")
        if bool((self.modality_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every sample must have at least one modality")

    @property
    def feature_dimension(self) -> int:
        return int(self.features.shape[2])

    def indices(self, sample_ids: Iterable[str]) -> Tensor:
        lookup = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        requested = tuple(sample_ids)
        unknown = set(requested).difference(lookup)
        if unknown:
            raise KeyError(f"Unknown sample IDs: {sorted(unknown)}")
        return torch.tensor([lookup[value] for value in requested], dtype=torch.long)

    def select(self, sample_ids: Iterable[str]) -> tuple[Tensor, Tensor, Tensor, tuple[str, ...]]:
        requested = tuple(sample_ids)
        index = self.indices(requested)
        return (
            self.features[index],
            self.modality_mask[index],
            self.labels[index],
            requested,
        )


def load_legacy_feature_table(
    path: str | Path,
    *,
    sample_id_prefix: str,
    skip_leading_columns: int,
    plain_dimension: int,
    ce_dimension: int,
    label_column: str,
    expected_sha256: str,
) -> LegacyFeatureCohort:
    """Load the bound artifact after checking its immutable fingerprint."""

    from .legacy import sha256_file

    source = Path(path).resolve()
    observed_sha256 = sha256_file(source)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Legacy source fingerprint changed: {observed_sha256}; expected {expected_sha256}"
        )
    frame = pd.read_csv(source)
    if frame.columns[-1] != label_column:
        raise ValueError("Configured label column is not the final legacy column")
    start = int(skip_leading_columns)
    plain = frame.iloc[:, start : start + plain_dimension]
    ce = frame.iloc[:, start + plain_dimension : start + plain_dimension + ce_dimension]
    if plain_dimension != ce_dimension:
        raise ValueError("Current GRAIN core requires equal within-cohort modality dimensions")
    masks = np.stack(
        [~plain.isna().all(axis=1).to_numpy(), ~ce.isna().all(axis=1).to_numpy()], axis=1
    )
    for name, block, available in (("plain", plain, masks[:, 0]), ("ce", ce, masks[:, 1])):
        partial = block.isna().any(axis=1).to_numpy() & available
        if bool(partial.any()):
            raise ValueError(f"{name} contains partially missing feature rows")
    features = np.stack(
        [plain.fillna(0.0).to_numpy(np.float32), ce.fillna(0.0).to_numpy(np.float32)],
        axis=1,
    )
    labels = frame[label_column].to_numpy(np.int64)
    sample_ids = tuple(
        f"{sample_id_prefix}_row_{index + 1:06d}" for index in range(len(frame))
    )
    return LegacyFeatureCohort(
        sample_ids=sample_ids,
        features=torch.from_numpy(features),
        modality_mask=torch.from_numpy(masks),
        labels=torch.from_numpy(labels),
        source_path=source.as_posix(),
        source_sha256=observed_sha256,
    )
