# Cross-Center Feature Alignment

This investigation used the legacy repository read-only. It establishes schema
and observed table relationships; it does not claim recovery of the original
feature extractor or clinical patient identity.

| Dataset | Bound file | N | Per-modality width | Availability plain/CE/both | Feature evidence | Status |
|---|---|---:|---:|---:|---|---|
| Center A | `E:/EXPS/attention/data/qilu_spleen_new.csv` | 221 | 120 | 185 / 23 / 13 | two 120-column modality blocks; current A and C feature names match | READY for internal CV |
| Center B | `E:/EXPS/attention/data/sd_spleen_label2.csv` | 185 | 60 | 14 / 0 / 171 | two 60-column modality blocks; no verified mapping to 120-d artifacts | READY for internal CV |
| Mixed A+B | `E:/EXPS/attention/data/merged_file.csv` | 344 | 60 | UNVERIFIED: 14 partial-CE rows | active K-search input; 171 rows exactly match complete Center B rows; generation of remaining rows is unavailable | BLOCKED for strict explicit-mask loading and external use |
| Center C | `E:/EXPS/attention/data/qfsALL_ln_new.csv` | 83 | 120 | 46 / 0 / 37 | two 120-column blocks; column names match Center A, but extractor provenance is absent | target artifact valid; Mixed-model inference blocked |

## `qilu&sd.csv` versus `merged_file.csv`

- `qilu&sd.csv` has 406 rows and 240 feature columns. Its first 221 rows are
  exactly the current Center A table. For the appended 185 Center B rows, the
  entire 120-column Center B feature vector occupies the first 120 columns and
  the last 120 columns are NaN. It is therefore heterogeneous row
  concatenation, not a verified common 120-d-per-modality representation.
- `merged_file.csv` has 344 rows and 120 feature columns (60 plus 60) and is the
  active input of `search4k.py`. It contains the 171 complete Center B rows
  exactly; no checked-in script uniquely explains how the other 173 rows were
  derived from current Center A.
- In rows 5 through 18 (one-based), the Mixed CE block contains 8 finite values
  followed by 52 NaNs. These are the same positions as the 14 plain-only rows
  in Center B; however, four labels in the first 185 Mixed rows also differ
  from the Center B table. The coarse inventory rule previously counted these
  rows as `both` because the CE block was not entirely NaN. That rule is not a
  valid explicit mask, and the strict official loader correctly rejects the
  artifact rather than silently applying `nan_to_num`.
- `zhexiantu.py` reads `qilu&sd.csv` for result/distribution visualization. That
  use does not establish it as the final mixed training artifact.
- `evaluate.py` applies D=120 Qilu-family checkpoints to Center C paths. This is
  evidence for an A-like 120-d external path, not evidence that the D=60
  `merged_file.csv` mixed model is compatible with Center C.

## Compatibility decision

`merged_file.csv` is additionally blocked for internal CV until its 14 partial
modality rows have an evidence-backed mask definition. Independently,
`merged_file.csv` (60 dimensions/modality) and Center C (120
dimensions/modality) are **not compatible** for direct external inference.
Internal projection layers do not establish cross-space correspondence. No
verified adapter, PCA, column selection, padding rule, or common re-extraction
artifact was found, so the official repository performs none of those
operations.
