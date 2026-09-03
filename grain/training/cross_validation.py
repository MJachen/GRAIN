"""Fold-isolated runtime construction and validation-only K selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from grain.data import LegacyFeatureCohort, SplitManifest
from grain.evaluation import EvaluationResult
from grain.models import AnchorFeatureBank, GRAIN
from grain.utils.reproducibility import set_global_seed

from .trainer import build_anchor_feature_bank, train_one_epoch, validate


@dataclass
class FoldRuntime:
    fold: int
    seed: int
    selected_k: int
    model: GRAIN
    optimizer: torch.optim.Optimizer
    scheduler: Any
    anchor_bank: AnchorFeatureBank


@dataclass(frozen=True)
class KCandidateResult:
    k: int
    validation_auc: float
    validation_loss: float


def build_model(config: dict[str, Any], selected_k: int) -> GRAIN:
    model = config["model"]
    required = (
        "feature_dimension",
        "patient_dimension",
        "relation_dimension",
        "gat_hidden_dimension",
        "transformer_dimension",
        "fusion_dimension",
        "classifier_hidden_dimension",
        "attention_tokens",
        "transformer_ffn_dimension",
        "relation_activation",
        "relation_alpha",
        "relation_gamma",
        "gat_dropout",
        "transformer_dropout",
    )
    unresolved = [key for key in required if model.get(key) is None]
    if unresolved:
        raise ValueError(f"Model configuration remains unresolved: {unresolved}")
    return GRAIN(
        feature_dimension=int(model["feature_dimension"]),
        patient_dimension=int(model["patient_dimension"]),
        relation_dimension=int(model["relation_dimension"]),
        gat_hidden_dimension=int(model["gat_hidden_dimension"]),
        transformer_dimension=int(model["transformer_dimension"]),
        fusion_dimension=int(model["fusion_dimension"]),
        classifier_hidden_dimension=int(model["classifier_hidden_dimension"]),
        attention_tokens=int(model["attention_tokens"]),
        attention_heads=int(model["fusion_attention_heads"]),
        transformer_ffn_dimension=int(model["transformer_ffn_dimension"]),
        relation_activation=str(model["relation_activation"]),
        relation_alpha=float(model["relation_alpha"]),
        relation_gamma=float(model["relation_gamma"]),
        requested_k=int(selected_k),
        gat_dropout=float(model["gat_dropout"]),
        transformer_dropout=float(model["transformer_dropout"]),
        uncertainty_temperature=float(config["training"]["uncertainty_temperature"]),
    )


def build_fold_runtime(
    *,
    config: dict[str, Any],
    cohort: LegacyFeatureCohort,
    split: SplitManifest,
    selected_k: int,
    device: torch.device,
    seed: int,
) -> FoldRuntime:
    """Create all mutable fold state after setting that fold's seed."""

    set_global_seed(seed)
    model = build_model(config, selected_k).to(device)
    training = config["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler_config = training.get("scheduler")
    scheduler = None
    if scheduler_config:
        if scheduler_config.get("name") != "step":
            raise ValueError("Only explicitly configured step scheduler is supported")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config["step_size"]),
            gamma=float(scheduler_config["gamma"]),
        )
    anchor_bank = build_anchor_feature_bank(cohort, split.train, device)
    anchor_bank.assert_excludes(tuple(split.validation) + tuple(split.test))
    return FoldRuntime(
        fold=split.outer_fold,
        seed=seed,
        selected_k=selected_k,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        anchor_bank=anchor_bank,
    )


def select_k_on_validation(
    *,
    config: dict[str, Any],
    cohort: LegacyFeatureCohort,
    split: SplitManifest,
    device: torch.device,
    seed: int,
    epochs: int,
    observer: Callable[[int, EvaluationResult], None] | None = None,
) -> tuple[int, tuple[KCandidateResult, ...]]:
    """Train candidates with train/validation only; no test argument exists."""

    mode = config["training"]["k_selection"]["mode"]
    if mode == "fixed":
        fixed = int(config["training"]["k_selection"]["fixed_k"])
        if fixed not in split.k_candidates:
            raise ValueError("fixed_k must be in the declared candidate set")
        return fixed, tuple()
    if mode != "validation":
        raise ValueError("K selection mode must be fixed or validation")
    results = []
    for candidate in split.k_candidates:
        runtime = build_fold_runtime(
            config=config,
            cohort=cohort,
            split=split,
            selected_k=candidate,
            device=device,
            seed=seed,
        )
        for epoch in range(epochs):
            train_one_epoch(
                model=runtime.model,
                optimizer=runtime.optimizer,
                cohort=cohort,
                training_ids=split.train,
                anchor_bank=runtime.anchor_bank,
                batch_size=int(config["training"]["batch_size"]),
                device=device,
                epoch_seed=seed + epoch,
                contrastive_temperature=float(config["training"]["contrastive_temperature"]),
                balance_margin=float(config["training"]["balance_margin"]),
                contrastive_weight=float(config["training"]["loss_weights"]["contrastive"]),
                balance_weight=float(config["training"]["loss_weights"]["balance"]),
                auxiliary_weight=float(config["training"]["loss_weights"]["auxiliary"]),
            )
            if runtime.scheduler is not None:
                runtime.scheduler.step()
        validation = validate(
            model=runtime.model,
            cohort=cohort,
            sample_ids=split.validation,
            anchor_bank=runtime.anchor_bank,
            fold=split.outer_fold,
            batch_size=int(config["training"]["batch_size"]),
            device=device,
            threshold=float(config["evaluation"]["threshold"]),
        )
        if observer is not None:
            observer(candidate, validation)
        results.append(KCandidateResult(candidate, validation.metrics.auc, validation.loss))
    selected = max(results, key=lambda item: (item.validation_auc, -item.k)).k
    return selected, tuple(results)
