# GRAIN Official Implementation

This repository is a clean, independent implementation of **Graph-based
Imputation Network (GRAIN)** for predicting immune-related adverse events with
missing plain and contrast-enhanced CT modalities.

The latest paper is the specification. Legacy code and historical results are
reference material only: they are not copied into this repository and are not
treated as ground truth when they conflict with the paper or leakage-safe
experimental practice.

## Current status

- Phase 1 — repository bootstrap: implemented.
- Phase 2 — explicit data contract and patient-level split manifests:
  implemented and covered by unit tests.
- Phase 3 onward — GRAIN model, trainer, evaluation, full reproduction,
  baselines and ablations: intentionally not implemented yet.
- No paper result is claimed reproduced by this bootstrap.

## Non-negotiable protocol rules

1. Model inference accepts features and modality masks, never labels.
2. Validation and test patients may query only complete-modality anchors from
   the corresponding outer-training partition.
3. Every outer fold owns a new model, optimizer, scheduler, anchor bank and
   normalization state.
4. Checkpoints and K are selected using validation AUC only. Outer test data are
   evaluated once after selection.
5. ROC AUC is computed from continuous positive-class probabilities.
6. Patient IDs, splits, configs, predictions and metrics are saved as provenance.
7. Missingness is represented by an explicit two-element mask in the order
   `[plain, contrast_enhanced]`; all-zero features are never used as a mask.

## Expected cohort manifest

One row represents one de-identified patient and must contain these columns:

| Column | Meaning |
|---|---|
| `patient_id` | Stable de-identified patient identifier |
| `center` | `A`, `B`, or `C` |
| `label` | Binary outcome; positive class is `1` |
| `plain_available` | Explicit boolean availability |
| `ce_available` | Explicit boolean availability |
| `plain_feature_path` | Relative path to a `.npy` feature vector, blank if unavailable |
| `ce_feature_path` | Relative path to a `.npy` feature vector, blank if unavailable |

Observed feature vectors must be finite and match the configured modality
dimension. A false availability flag is authoritative even if a path is
accidentally present.

Feature dimension, hidden dimension and dropout remain `null` until the
approved feature extractor artifact and the corresponding paper specification
are reconciled. Legacy defaults are deliberately not promoted to official
configuration values.

## Bootstrap usage

Create reproducible outer/validation manifests after an approved cohort
manifest with real de-identified patient IDs is available:

```bash
python scripts/validate_data.py --config configs/center_a.yaml
python scripts/create_splits.py --config configs/center_a.yaml
```

Run Phase 1–2 tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The future official entry point is reserved as:

```bash
python scripts/run_cv.py --config configs/center_a.yaml
```

At the current phase it validates the configuration and exits without
training, preventing accidental use of an incomplete pipeline.

## Reproducibility layout

Future runs will write only under `outputs/<experiment_name>/fold_XX/` and will
contain resolved config, split, checkpoint, logs, validation/test predictions
and metrics. Raw clinical data, feature arrays, generated splits, checkpoints
and run outputs are ignored by Git.

See [docs/paper_code_mapping.md](docs/paper_code_mapping.md) for equation-level
implementation status and [docs/legacy_migration.md](docs/legacy_migration.md)
for the legacy migration boundary.
