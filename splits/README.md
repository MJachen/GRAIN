# Split manifests

Generated official split JSON files belong here and should be committed after
the cohort manifest is approved. They must contain de-identified IDs; do not
create row-number-derived patient IDs.

Every fold JSON must pass `SplitManifest.validate()` and the schema in
`schema.json`. `selected_k` remains null during initial split generation; a
future Phase 4 K-selection record will store every candidate validation AUC and
then write the selected value as provenance.
