"""Explicit patient, modality-mask, anchor and split interfaces."""

from .anchors import AnchorBank
from .dataset import FeatureDataset, collate_samples
from .schema import CohortManifest, PatientRecord
from .splits import SplitManifest, generate_nested_manifests

__all__ = [
    "AnchorBank",
    "CohortManifest",
    "FeatureDataset",
    "PatientRecord",
    "SplitManifest",
    "collate_samples",
    "generate_nested_manifests",
]

