"""Phase 5 evaluation package placeholder."""
"""Shared official prediction and metric API."""

from .metrics import BinaryMetrics, compute_metrics
from .prediction import EvaluationResult

__all__ = ["BinaryMetrics", "EvaluationResult", "compute_metrics"]
