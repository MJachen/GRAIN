"""Explicit train, validation and one-shot test APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from grain.data.legacy_cohort import LegacyFeatureCohort
from grain.evaluation import EvaluationResult
from grain.losses import compute_grain_loss
from grain.models import AnchorFeatureBank, GRAIN


@dataclass(frozen=True)
class EpochResult:
    loss: float
    classification: float
    auxiliary: float
    contrastive: float
    balance: float
    gradient_norm: float
    batches: int


def build_anchor_feature_bank(
    cohort: LegacyFeatureCohort,
    training_ids: Iterable[str],
    device: torch.device,
) -> AnchorFeatureBank:
    training_ids = tuple(training_ids)
    features, masks, _, ids = cohort.select(training_ids)
    complete = masks.all(dim=1)
    anchor_ids = tuple(value for value, keep in zip(ids, complete.tolist()) if keep)
    return AnchorFeatureBank(
        patient_ids=anchor_ids,
        training_patient_ids=frozenset(training_ids),
        features=features[complete].to(device),
        modality_mask=masks[complete].to(device),
    )


def _batch_indices(count: int, batch_size: int, seed: int) -> list[Tensor]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 because the model uses BatchNorm")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    batches = list(permutation.split(batch_size))
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat([batches[-2], batches[-1]])
        batches.pop()
    return batches


def train_one_epoch(
    *,
    model: GRAIN,
    optimizer: torch.optim.Optimizer,
    cohort: LegacyFeatureCohort,
    training_ids: Iterable[str],
    anchor_bank: AnchorFeatureBank,
    batch_size: int,
    device: torch.device,
    epoch_seed: int,
    contrastive_temperature: float,
    balance_margin: float,
    contrastive_weight: float,
    balance_weight: float,
    auxiliary_weight: float = 1.0,
) -> EpochResult:
    """Train only on the supplied outer-training IDs and training anchors."""

    features, masks, labels, ids = cohort.select(training_ids)
    totals = np.zeros(5, dtype=np.float64)
    maximum_gradient_norm = 0.0
    batches = _batch_indices(len(ids), batch_size, epoch_seed)
    model.train()
    for indices in batches:
        batch_features = features[indices].to(device)
        batch_masks = masks[indices].to(device)
        batch_labels = labels[indices].to(device)
        batch_ids = tuple(ids[index] for index in indices.tolist())
        optimizer.zero_grad(set_to_none=True)
        output = model(batch_features, batch_masks, anchor_bank, batch_ids)
        anchor_output = model(
            anchor_bank.features,
            anchor_bank.modality_mask,
            anchor_bank,
            anchor_bank.patient_ids,
        )
        loss = compute_grain_loss(
            output,
            batch_labels,
            batch_features,
            batch_masks,
            contrastive_original_features=anchor_bank.features,
            contrastive_reconstructed_features=anchor_output.reconstructed_features,
            contrastive_complete_mask=torch.ones(
                len(anchor_bank.patient_ids), dtype=torch.bool, device=device
            ),
            contrastive_temperature=contrastive_temperature,
            balance_margin=balance_margin,
            contrastive_weight=contrastive_weight,
            balance_weight=balance_weight,
            auxiliary_weight=auxiliary_weight,
        )
        if not bool(torch.isfinite(loss.total)):
            raise FloatingPointError("Non-finite training loss")
        loss.total.backward()
        squared = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError("Non-finite model gradient")
                squared += float(parameter.grad.detach().pow(2).sum())
        gradient_norm = squared ** 0.5
        if gradient_norm == 0.0:
            raise FloatingPointError("Zero total gradient")
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        totals += np.asarray(
            [
                float(loss.total.detach()),
                float(loss.classification.detach()),
                float(loss.auxiliary.detach()),
                float(loss.contrastive.detach()),
                float(loss.balance.detach()),
            ],
            dtype=np.float64,
        )
    means = totals / len(batches)
    return EpochResult(*map(float, means), maximum_gradient_norm, len(batches))


def _evaluate(
    *,
    split: str,
    model: GRAIN,
    cohort: LegacyFeatureCohort,
    sample_ids: Iterable[str],
    anchor_bank: AnchorFeatureBank,
    fold: int,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> EvaluationResult:
    features, masks, labels, ids = cohort.select(sample_ids)
    probabilities = []
    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            stop = min(start + batch_size, len(ids))
            batch_features = features[start:stop].to(device)
            batch_masks = masks[start:stop].to(device)
            batch_ids = ids[start:stop]
            output = model(batch_features, batch_masks, anchor_bank, batch_ids)
            targets = labels[start:stop].to(device, dtype=output.logits.dtype)
            primary = F.binary_cross_entropy_with_logits(output.logits, targets)
            auxiliary = F.binary_cross_entropy_with_logits(
                output.modality_logits, targets.unsqueeze(1).expand_as(output.modality_logits)
            )
            losses.append(float(primary + auxiliary))
            probabilities.append(output.probabilities.cpu())
    return EvaluationResult.create(
        split=split,
        fold=fold,
        sample_ids=ids,
        labels=labels.numpy(),
        probabilities=torch.cat(probabilities).numpy(),
        loss=float(np.mean(losses)),
        threshold=threshold,
    )


def validate(**kwargs) -> EvaluationResult:
    return _evaluate(split="validation", **kwargs)


def test(**kwargs) -> EvaluationResult:
    return _evaluate(split="test", **kwargs)
