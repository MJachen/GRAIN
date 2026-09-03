"""Phase 4 training package placeholder."""
"""Leakage-safe training infrastructure."""

from .checkpoint import ValidationCheckpointManager, load_checkpoint
from .cross_validation import FoldRuntime, build_fold_runtime, select_k_on_validation
from .trainer import EpochResult, build_anchor_feature_bank, test, train_one_epoch, validate
from .preflight import run_preflight

__all__ = [
    "EpochResult",
    "FoldRuntime",
    "ValidationCheckpointManager",
    "build_anchor_feature_bank",
    "build_fold_runtime",
    "load_checkpoint",
    "run_preflight",
    "select_k_on_validation",
    "test",
    "train_one_epoch",
    "validate",
]
