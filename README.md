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
- Phase 2.5 — read-only legacy-data and exact-interpreter binding: implemented.
- Phase 3 — label-free core GRAIN model and paper losses: implemented and
  covered by synthetic shape, isolation, uncertainty and gradient tests.
- Phase 4A — train/validation/test, validation-only checkpoint/K selection and
  probability metric infrastructure: implemented.
- Phase 4B — one real-data fold is available only through the explicitly
  labelled development smoke entry point.
- Phase 4C — formal architecture/training parameters and per-dataset readiness:
  frozen. Center A and Center B are eligible for internal full CV. Mixed A+B
  is blocked by 14 partial-modality rows; Mixed-to-C external inference is also
  blocked by feature-space mismatch.
- Phase 5A — formal Center A/B runner and independent integrity audit are
  implemented; formal execution is permitted only after regression tests,
  preflight and a clean commit.
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

Formal architecture values are declared in
`configs/architecture_grain_official.json`. Four paper-underspecified choices
remain visibly marked `ASSUMED`; no value was selected from outer-test
performance. See `docs/architecture_provenance.md`.

## Bound environment and development usage

Every command must use the interpreter that the historical attention IDE
configuration identifies. Do not install `requirements.txt` into it; that file
documents the observed environment.

```bash
E:/cjj/anaconda3/envs/irae/python.exe scripts/check_environment.py
```

The old data root is configured once as `data.source_root`. Code contains no
cohort-specific absolute data path and the repository path policy rejects
writes outside this repository's output areas.

## Cohort and sample identity

Bound legacy tables are the operative experimental datasets. Deterministic IDs
such as `centerA_row_000001` are experiment-local stable sample identifiers,
not recovered clinical patient identities. Paper cohort-count agreement and
historical identity recovery are recorded provenance limitations, not training
gates.

## Optional explicit-manifest bootstrap

Create reproducible outer/validation manifests after an approved cohort
manifest with real de-identified patient IDs is available:

```bash
E:/cjj/anaconda3/envs/irae/python.exe scripts/validate_data.py --config configs/center_a.json
E:/cjj/anaconda3/envs/irae/python.exe scripts/create_splits.py --config configs/center_a.json
```

Run all Phase 1–3 tests:

```bash
E:/cjj/anaconda3/envs/irae/python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The Phase 5A formal entry points are:

```bash
E:/cjj/anaconda3/envs/irae/python.exe scripts/run_cv.py --config configs/formal_center_a.json
E:/cjj/anaconda3/envs/irae/python.exe scripts/audit_formal_run.py --output outputs/formal_center_a
```

`run_cv.py` accepts only the frozen Center A/B formal configs. It refuses a
dirty worktree or existing formal output directory, evaluates each outer test
fold exactly once after checkpoint reload, and automatically runs the audit.

The only enabled real-data experiment is the single-fold smoke:

```bash
E:/cjj/anaconda3/envs/irae/python.exe scripts/run_smoke_fold.py --config configs/smoke_center_a.json --fold 0
```

Its output is always marked `DEVELOPMENT SMOKE TEST - NOT PAPER RESULT`.

## Reproducibility layout

Future runs will write only under `outputs/<experiment_name>/fold_XX/` and will
contain resolved config, split, checkpoint, logs, validation/test predictions
and metrics. Raw clinical data, feature arrays, generated splits, checkpoints
and run outputs are ignored by Git.

See [docs/paper_code_mapping.md](docs/paper_code_mapping.md) for equation-level
implementation status, [docs/architecture_provenance.md](docs/architecture_provenance.md)
for the frozen parameter evidence, [docs/feature_alignment.md](docs/feature_alignment.md)
for the cross-center decision, and [docs/legacy_migration.md](docs/legacy_migration.md)
for the legacy migration boundary. Phase 2.5 evidence is documented in
[docs/data_environment_binding.md](docs/data_environment_binding.md).
