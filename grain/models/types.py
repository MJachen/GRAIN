"""Typed tensor containers for label-free GRAIN inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AnchorFeatureBank:
    """Complete-modality features from exactly one outer-training partition.

    Shapes:
        features: [A, M=2, D]
        modality_mask: [A, M=2], required to be all True
    """

    patient_ids: tuple[str, ...]
    training_patient_ids: frozenset[str]
    features: Tensor
    modality_mask: Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 3 or self.features.shape[1] != 2:
            raise ValueError("Anchor features must have shape [A, 2, D]")
        if self.modality_mask.shape != self.features.shape[:2]:
            raise ValueError("Anchor modality_mask must have shape [A, 2]")
        if self.modality_mask.dtype != torch.bool:
            raise TypeError("Anchor modality_mask must be boolean")
        if not bool(self.modality_mask.all()):
            raise ValueError("Every graph anchor must have both modalities")
        if len(self.patient_ids) != self.features.shape[0]:
            raise ValueError("Anchor patient ID count does not match tensor rows")
        if len(set(self.patient_ids)) != len(self.patient_ids):
            raise ValueError("Anchor patient IDs must be unique")
        if not set(self.patient_ids).issubset(self.training_patient_ids):
            raise ValueError("Anchor bank contains a non-training patient")
        if len(self.patient_ids) < 2:
            raise ValueError("Paper protocol requires at least two complete anchors")

    def assert_excludes(self, patient_ids: tuple[str, ...]) -> None:
        overlap = set(patient_ids).intersection(self.patient_ids)
        if overlap:
            raise ValueError(f"Forbidden patients are present as anchors: {sorted(overlap)}")

    def to(self, device: torch.device | str) -> "AnchorFeatureBank":
        return AnchorFeatureBank(
            patient_ids=self.patient_ids,
            training_patient_ids=self.training_patient_ids,
            features=self.features.to(device),
            modality_mask=self.modality_mask.to(device),
        )


@dataclass
class GraphSelection:
    """Target-to-anchor sparse graph.

    Shapes:
        dense_scores: [B, A]
        adjacency: [B, A]
        selected_mask: [B, A]
        effective_k: [B]
    """

    dense_scores: Tensor
    adjacency: Tensor
    selected_mask: Tensor
    effective_k: Tensor


@dataclass
class GRAINOutput:
    """All label-free model outputs needed by losses and diagnostics."""

    logits: Tensor
    probabilities: Tensor
    modality_logits: Tensor
    modality_probabilities: Tensor
    modality_entropy: Tensor
    modality_weights: Tensor
    patient_representation: Tensor
    reconstructed_features: Tensor
    updated_features: Tensor
    graph: GraphSelection

