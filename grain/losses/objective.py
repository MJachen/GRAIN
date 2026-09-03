"""Differentiable implementation of paper Eq. (17), (18), (25), (35), (38), (39)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from grain.models.types import GRAINOutput


@dataclass
class GRAINLoss:
    total: Tensor
    classification: Tensor
    auxiliary: Tensor
    contrastive: Tensor
    balance: Tensor


def neighborhood_contrastive_loss(
    original_features: Tensor,
    reconstructed_features: Tensor,
    complete_mask: Tensor,
    temperature: float,
) -> Tensor:
    """Eq. (17)–(18), using complete patients as paired positives.

    Inputs are [B,M=2,D], while ``complete_mask`` is boolean [B]. At least two
    complete patients are needed to define a positive against negatives.
    """

    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    if complete_mask.dtype != torch.bool or complete_mask.shape != original_features.shape[:1]:
        raise ValueError("complete_mask must be boolean with shape [B]")
    original = original_features[complete_mask]
    reconstructed = reconstructed_features[complete_mask]
    if original.shape[0] < 2:
        raise ValueError("Contrastive loss requires at least two complete patients")
    losses = []
    labels = torch.arange(original.shape[0], device=original.device)
    for modality_index in range(original.shape[1]):
        similarity = torch.matmul(
            F.normalize(original[:, modality_index, :], dim=-1),
            F.normalize(reconstructed[:, modality_index, :], dim=-1).transpose(0, 1),
        )
        losses.append(F.cross_entropy(similarity / temperature, labels))
    return torch.stack(losses).mean()


def modality_balance_loss(modality_weights: Tensor, margin: float) -> Tensor:
    """Eq. (35): hinge penalty only beyond the allowed weight difference."""

    if not 0 <= margin <= 1:
        raise ValueError("balance margin must be in [0, 1]")
    if modality_weights.ndim != 2 or modality_weights.shape[1] != 2:
        raise ValueError("modality_weights must have shape [B, 2]")
    difference = (modality_weights[:, 0] - modality_weights[:, 1]).abs()
    return F.relu(difference - margin).mean()


def compute_grain_loss(
    output: GRAINOutput,
    labels: Tensor,
    original_features: Tensor,
    modality_mask: Tensor,
    *,
    contrastive_temperature: float,
    balance_margin: float,
    contrastive_weight: float,
    balance_weight: float,
    auxiliary_weight: float = 1.0,
) -> GRAINLoss:
    """Eq. (39), without ``detach`` or ``.data`` gradient breaks."""

    labels = labels.to(dtype=output.logits.dtype)
    if labels.shape != output.logits.shape:
        raise ValueError("labels and final logits must have identical shape [B]")
    classification = F.binary_cross_entropy_with_logits(output.logits, labels)
    auxiliary_targets = labels.unsqueeze(1).expand_as(output.modality_logits)
    auxiliary = F.binary_cross_entropy_with_logits(
        output.modality_logits, auxiliary_targets
    )
    contrastive = neighborhood_contrastive_loss(
        original_features,
        output.reconstructed_features,
        modality_mask.all(dim=1),
        contrastive_temperature,
    )
    balance = modality_balance_loss(output.modality_weights, balance_margin)
    total = (
        classification
        + auxiliary_weight * auxiliary
        + contrastive_weight * contrastive
        + balance_weight * balance
    )
    return GRAINLoss(total, classification, auxiliary, contrastive, balance)
