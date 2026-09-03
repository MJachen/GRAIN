# Phase 4C Architecture Provenance

This is a parameter-recovery record, not a test-performance search. No outer
test label, prediction, or metric was used to choose any value. Dataset-specific
`D` denotes the feature width of one modality.

| Parameter | Official value | Paper evidence | Legacy-code evidence | Checkpoint evidence | Confidence |
|---|---:|---|---|---|---|
| `input_dim_plain`, `input_dim_ce` | A/C: 120; B/Mixed: 60 | MedicalNet features, exact width not stated | Active CSV slicing in final scripts | First linear/GAT widths agree with D-specific models | VERIFIED for bound artifact schema |
| `projection_dim` | D | Eq. (20) requires a fully connected projection | `attention1.py` uses D to D | projection weights are D by D | LIKELY |
| `patient_dim` | D | Eq. (3)-(4), width not stated | FAMI construction preserves D | Q is not independently identifiable in saved state | ASSUMED |
| `relation_dim` | D | Eq. (5)-(9), width not stated | `FAMI.py` constructs D-width relation layers | relation matrices preserve D | LIKELY |
| `gat_hidden_dim` | D | two-layer/two-head GAT stated | legacy graph path preserves D | GAT weights are D by D | VERIFIED |
| `gat_heads` | 2 | Section III-D explicitly states two heads | legacy implementation does not faithfully implement multi-head GAT | not uniquely recoverable | VERIFIED from paper |
| `gat_layers` | 2 | Section III-D explicitly states two layers | two legacy graph stages | consistent | VERIFIED from paper |
| `transformer_dim` | D | attention width not numerically stated | final attention construction uses `embed_dim=D` | MHSA projection shapes are 3D by D | VERIFIED |
| `transformer_heads` | 4 | not stated | final `attention1.py` default and loaded model objects use 4 | D is divisible by 4 in saved models | VERIFIED from final legacy object |
| `transformer_ffn_dimension` | 2048 | not stated | PyTorch encoder default | saved FFN weights are 2048 by D and D by 2048 | VERIFIED |
| `attention_tokens` | 1 | no feature-to-token decomposition is defined | legacy code ambiguously treats patients as sequence | cannot be recovered | ASSUMED; minimal within-patient choice |
| `fusion_dim` | D | Eq. (36) maps fused representation to H | fusion path preserves D | fusion weight shape is D by 4D | LIKELY |
| `classifier_hidden_dim` | D | two-layer MLP stated, width not stated | final classifier is 2D to D to 2 | fc1/fc2 shapes verify D | VERIFIED |
| `gat_dropout` | 0.2 | not stated | `FAMI.py` and loaded models use 0.2 | model-object metadata agrees | VERIFIED |
| `transformer_dropout` | 0.1 | not stated | `attention1.py` and loaded models use 0.1 | model-object metadata agrees | VERIFIED |
| `relation_activation` | ReLU | paper denotes unspecified nonlinear sigma | legacy relation path uses a different/ambiguous operation | not identifiable | ASSUMED |
| `alpha` | 3.0 | Eq. (7)-(8) defines alpha, not its value | `FAMI.py` and final objects use 3 | model-object metadata agrees | VERIFIED |
| `gamma` | 1.0 | Eq. (9) defines gamma, not its value | no unique stored setting found | not identifiable | ASSUMED; unit coefficient |
| uncertainty temperature | 0.1 | Experimental Setup | compatible with rewritten label-free entropy path | not needed | VERIFIED |
| contrastive temperature | 0.5 | Experimental Setup | not relied upon | not needed | VERIFIED |

Remaining `ASSUMED` parameters: **4** (`attention_tokens`, `patient_dim`,
`relation_activation`, `relation_gamma`). They are declared in the machine
readable manifest and may only be revisited through paper/historical evidence
or development/validation-only selection.

The legacy checkpoint is evidence for dimensions only. It is not loaded as an
official checkpoint because the corrected graph, uncertainty, fusion, and
attention semantics are intentionally not state-compatible with the unsafe
legacy implementation.

## Frozen training configuration

| Item | Official setting | Evidence | Confidence |
|---|---|---|---|
| Optimizer | Adam | paper and final scripts | VERIFIED |
| Learning rate | 1e-4 | paper and final scripts | VERIFIED |
| Weight decay | 5e-4 | final `trainerv1.py` and `search4k.py` | LIKELY |
| Batch size | 32 | paper and final scripts | VERIFIED |
| Epochs | 100 | paper and final scripts | VERIFIED |
| Scheduler | none | no scheduler in paper/final path | LIKELY |
| K | per-fold validation selection over 2 through 10 | latest paper | VERIFIED |
| Loss weights | CE 1, auxiliary 1, contrastive 0.1, balance 0.01 | paper | VERIFIED |
| Seed | base 42 plus zero-based outer-fold index | reproducible fold-isolation policy | LIKELY |
| Checkpoint metric | validation AUC | latest paper | VERIFIED |
| Classification threshold | 0.5 | binary default; no historical calibrated threshold found | ASSUMED |

