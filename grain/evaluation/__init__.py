"""Phase 5 evaluation package placeholder."""
"""Shared official prediction and metric API."""

from .metrics import BinaryMetrics, compute_metrics
from .prediction import EvaluationResult
from .audit import audit_formal_run

__all__ = ["BinaryMetrics", "EvaluationResult", "audit_formal_run", "compute_metrics"]
