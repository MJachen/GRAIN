"""Paper Eq. (3)–(4): explicit-mask patient representation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MaskAwarePatientRepresentation(nn.Module):
    """Project each modality and average only observed modalities.

    Input shapes:
        features: [B, M=2, D]
        modality_mask: [B, M=2]
    Output:
        patient representation Q: [B, d]
    """

    def __init__(self, feature_dimension: int, patient_dimension: int, epsilon: float = 1e-8):
        super().__init__()
        self.feature_dimension = int(feature_dimension)
        self.patient_dimension = int(patient_dimension)
        self.epsilon = float(epsilon)
        self.modality_projections = nn.ModuleList(
            nn.Linear(self.feature_dimension, self.patient_dimension) for _ in range(2)
        )

    def forward(self, features: Tensor, modality_mask: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1:] != (2, self.feature_dimension):
            raise ValueError(
                f"features must be [B, 2, {self.feature_dimension}], got {tuple(features.shape)}"
            )
        if modality_mask.shape != features.shape[:2] or modality_mask.dtype != torch.bool:
            raise ValueError("modality_mask must be boolean with shape [B, 2]")
        observed_count = modality_mask.sum(dim=1, keepdim=True)
        if bool((observed_count == 0).any()):
            raise ValueError("Every target patient must have at least one observed modality")
        projected = torch.stack(
            [
                projection(features[:, modality_index, :])
                for modality_index, projection in enumerate(self.modality_projections)
            ],
            dim=1,
        )
        mask = modality_mask.to(dtype=features.dtype).unsqueeze(-1)
        return (projected * mask).sum(dim=1) / (observed_count.to(features.dtype) + self.epsilon)

