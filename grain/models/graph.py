"""Paper Eq. (5)–(15): adaptive target-to-training-anchor graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import GraphSelection


def _activation(name: str) -> nn.Module:
    choices = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
    if name not in choices:
        raise ValueError(f"Unsupported relation activation {name!r}")
    return choices[name]


class AdaptivePatientGraph(nn.Module):
    """Compute asymmetric bilinear plus cosine relations and sparse Top-K.

    B is the number of target patients and A is the number of immutable
    outer-training complete-modality anchors. No target-target edges exist.
    """

    def __init__(
        self,
        patient_dimension: int,
        relation_dimension: int,
        activation: str,
        alpha: float,
        gamma: float,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.embedding_target = nn.Sequential(
            nn.Linear(patient_dimension, relation_dimension), _activation(activation)
        )
        self.embedding_source = nn.Sequential(
            nn.Linear(patient_dimension, relation_dimension), _activation(activation)
        )
        self.enhance_target = nn.Linear(relation_dimension, relation_dimension, bias=False)
        self.enhance_source = nn.Linear(relation_dimension, relation_dimension, bias=False)

    def forward(
        self,
        target_representation: Tensor,
        anchor_representation: Tensor,
        target_patient_ids: tuple[str, ...],
        anchor_patient_ids: tuple[str, ...],
        requested_k: int,
    ) -> GraphSelection:
        if requested_k < 1:
            raise ValueError("requested_k must be positive")
        batch_size = target_representation.shape[0]
        anchor_count = anchor_representation.shape[0]
        if len(target_patient_ids) != batch_size or len(anchor_patient_ids) != anchor_count:
            raise ValueError("Patient IDs must align with target and anchor tensor rows")

        target_r1 = torch.tanh(
            self.alpha * self.enhance_target(self.embedding_target(target_representation))
        )
        anchor_r2 = torch.tanh(
            self.alpha * self.enhance_source(self.embedding_source(anchor_representation))
        )
        bilinear = self.gamma * torch.matmul(target_r1, anchor_r2.transpose(0, 1))
        cosine = F.cosine_similarity(
            target_r1.unsqueeze(1), anchor_r2.unsqueeze(0), dim=-1, eps=self.epsilon
        )
        scores = bilinear + cosine

        adjacency = torch.zeros_like(scores)
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        effective_k = torch.zeros(batch_size, dtype=torch.long, device=scores.device)
        for target_index, target_id in enumerate(target_patient_ids):
            eligible = torch.tensor(
                [anchor_id != target_id for anchor_id in anchor_patient_ids],
                dtype=torch.bool,
                device=scores.device,
            )
            available = int(eligible.sum().item())
            if available == 0:
                raise ValueError(f"No eligible anchor for target {target_id}")
            k_effective = min(int(requested_k), available)
            effective_k[target_index] = k_effective

            normalized = F.softmax(
                F.relu(scores[target_index]).masked_fill(~eligible, float("-inf")), dim=0
            )
            _, selected = torch.topk(normalized, k=k_effective)
            selected_weights = normalized[selected]
            selected_weights = selected_weights / selected_weights.sum().clamp_min(self.epsilon)
            adjacency[target_index, selected] = selected_weights
            selected_mask[target_index, selected] = True

        return GraphSelection(
            dense_scores=scores,
            adjacency=adjacency,
            selected_mask=selected_mask,
            effective_k=effective_k,
        )

