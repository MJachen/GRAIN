# Legacy migration boundary

The historical repository remains read-only and is not a package dependency.

## Safe to migrate after verification

- Generic seed-setting intent.
- High-level two-modality feature dimensionality where confirmed by the
  approved feature manifest.
- Paper-stated optimizer, epoch and loss-weight values as configuration, not as
  proof of reproduced results.

## Migrate only after rewrite

- Adaptive graph construction and Top-K neighbor logic.
- GAT imputation.
- Intra-modal and cross-modal attention.
- Entropy-based adaptive fusion.
- Auxiliary, contrastive and modality-balance losses.
- Cross-validation, K selection, checkpointing and evaluation.

These areas contain label dependence, full-cohort graph construction,
cross-fold state reuse, detached losses or hard-label AUC in the legacy code.

## Legacy only

- BraTS experiments and artificial dropped-feature data.
- Old attention variants, debugging scripts and placeholder entry points.
- Hard-coded plotting/result scripts and historical spreadsheets.
- Existing baseline prototypes and serialized legacy model objects.
- Duplicate `new`, `new1`, `new2`, backup or copy artifacts.

Baselines will be ported only after the official GRAIN pipeline passes the
shared data, split, validation and metric contracts.

