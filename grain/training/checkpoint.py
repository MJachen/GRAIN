"""Validation-only checkpoint selection and complete provenance payloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from grain.evaluation import EvaluationResult


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_auc: float,
    fold: int,
    seed: int,
    selected_k: int,
    resolved_config: dict[str, Any],
    data_fingerprint: dict[str, Any],
    git_commit: str,
    scheduler: Any = None,
) -> dict[str, Any]:
    return {
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": None if scheduler is None else deepcopy(scheduler.state_dict()),
        "epoch": int(epoch),
        "best_val_auc": float(best_val_auc),
        "fold": int(fold),
        "seed": int(seed),
        "selected_k": int(selected_k),
        "resolved_config": deepcopy(resolved_config),
        "data_fingerprint": deepcopy(data_fingerprint),
        "git_commit": str(git_commit),
    }


class ValidationCheckpointManager:
    """Accept only validation results; test results are rejected by construction."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.best_auc = float("-inf")
        self.best_epoch: int | None = None

    def consider(
        self,
        result: EvaluationResult,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        fold: int,
        seed: int,
        selected_k: int,
        resolved_config: dict[str, Any],
        data_fingerprint: dict[str, Any],
        git_commit: str,
        scheduler: Any = None,
    ) -> bool:
        if result.split != "validation":
            raise ValueError("Only validation AUC may select a checkpoint")
        if result.metrics.auc <= self.best_auc:
            return False
        self.best_auc = result.metrics.auc
        self.best_epoch = int(epoch)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_auc=result.metrics.auc,
                fold=fold,
                seed=seed,
                selected_k=selected_k,
                resolved_config=resolved_config,
                data_fingerprint=data_fingerprint,
                git_commit=git_commit,
            ),
            self.path,
        )
        return True


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload
