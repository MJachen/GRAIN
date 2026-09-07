"""Explicit patient, modality-mask, anchor and split interfaces."""

from .anchors import AnchorBank
from .dataset import FeatureDataset, collate_samples
from .legacy_cohort import (
    LegacyFeatureCohort,
    load_legacy_feature_table,
    resolve_data_fingerprint,
)
from .schema import CohortManifest, PatientRecord
from .splits import (
    SplitManifest,
    generate_nested_manifests,
    generate_nested_manifests_from_arrays,
)

__all__ = [
    "AnchorBank",
    "CohortManifest",
    "FeatureDataset",
    "LegacyFeatureCohort",
    "PatientRecord",
    "SplitManifest",
    "collate_samples",
    "generate_nested_manifests",
    "generate_nested_manifests_from_arrays",
    "load_legacy_feature_table",
    "resolve_data_fingerprint",
]
