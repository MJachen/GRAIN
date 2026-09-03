# Legacy Data and Environment Binding

This document records the Phase 2.5 evidence boundary. The historical
`E:/EXPS/attention` tree is read-only. Official code reads paths from JSON
configuration and may write only below this repository's `outputs`, `splits`,
or `artifacts` directories.

## Bound interpreter

- Executable: `E:/cjj/anaconda3/envs/irae/python.exe`
- Evidence: `.idea/attention.iml` and `.idea/misc.xml` name `Python 3.10 (irae)`;
  the executable was then queried directly.
- Observed: Python 3.10.19, PyTorch 1.13.1+cpu, CUDA unavailable, NumPy 1.26.4,
  pandas 2.2.3, scikit-learn 1.5.2, SciPy 1.15.3.
- Conflict: the paper reports PyTorch 2.3.1 and an NVIDIA RTX 3090. The bound
  local environment supports unit tests but is not evidence of that reported
  training environment.

No package was installed, upgraded, or downgraded. Exact inventories are in
`artifacts/environment.json`, `artifacts/pip_freeze.txt`, and
`artifacts/conda_list.json`.

## Data identity boundary

Metadata fingerprints are stored in `artifacts/data_inventory.json`; no
features or patient rows are copied. The inspected clinical CSV files do not
contain a trustworthy stable patient identifier. A numeric first feature
column and the `Unnamed: 0` row index in the Center C candidate must not be
promoted to official patient identities.

By explicit project decision after Phase 2.5, historical identity recovery and
paper cohort-count agreement are no longer formal-training gates. Each bound
table now receives a deterministic experiment-local identifier of the form
`<cohort>_row_XXXXXX`. This identifier is used only for split reproducibility,
prediction alignment and paired analysis; it is never described as a recovered
clinical patient identifier. The source fingerprint remains the data identity.
