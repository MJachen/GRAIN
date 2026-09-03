# Paper–Code Mapping

Specification source: latest 10-page GRAIN manuscript supplied with the
historical repository. Shapes use `B` target patients, `A` training anchors,
`M=2` modalities, `D` feature dimension, `P` patient dimension, `R` relation
dimension, `T` within-patient tokens, and `H` fusion dimension.

| Paper item | Paper description | Official implementation | Input → output | Verification |
|---|---|---|---|---|
| Data contract | Plain CT, CE CT, explicit availability and stable patient ID | `grain/data/schema.py`; `grain/data/dataset.py` | manifest row → named sample and bool mask `[M]` | Unit tested; real identity BLOCKED |
| Patient/sample-level nested split | Outer stratified 10-fold plus inner validation | `grain/data/splits.py` | stable IDs → disjoint train/validation/test IDs | Unit tested; legacy rows use experiment-local stable IDs |
| Eq. (2) | MedicalNet 3D-ResNet-50 feature extraction | External legacy feature artifact only | CT → `h_n^m [D]` | UNVERIFIED; extractor provenance unavailable |
| Eq. (3)–(4) | Availability-mask weighted patient representation | `MaskAwarePatientRepresentation.forward` | `[B,M,D]`, `[B,M]` → `Q [B,P]` | Shape/mask/gradient tested |
| Eq. (5)–(8) | Separate patient embeddings and relation enhancement | `AdaptivePatientGraph.forward` | target `[B,P]`, anchors `[A,P]` → relation embeddings | Gradient tested; activation/scales UNVERIFIED |
| Eq. (9)–(11) | Bilinear-plus-cosine score, ReLU and row softmax | `AdaptivePatientGraph.forward` | relation embeddings → dense scores `[B,A]` | Shape/normalization tested |
| Eq. (12) | Complete patients from the current outer-training fold only | `AnchorBank`; `AnchorFeatureBank` | training IDs/features → immutable anchor bank | Leakage rejection tested |
| Eq. (13)–(15) | Self exclusion, `K_eff`, Top-K and sparse renormalization | `AdaptivePatientGraph.forward` | scores `[B,A]` → adjacency `[B,A]` | Self exclusion/K cap/row sum tested |
| Eq. (16) | Two-layer, two-head modality-specific GAT reconstruction | `AnchorGraphImputer`; `TwoLayerTwoHeadGAT` | target and anchor features → `h_hat [B,M,D]` | Shape/gradient tested |
| Eq. (17)–(18) | Complete-patient neighborhood contrastive objective | `neighborhood_contrastive_loss` | original/reconstructed complete features → scalar | Differentiable gradient tested |
| Eq. (19) | Missing: reconstruction; observed: original plus reconstruction | `AnchorGraphImputer.forward` | `h`, `h_hat`, mask → `h_tilde [B,M,D]` | Three mask patterns tested |
| Eq. (20) | Fully connected projection before fusion | `FeatureTokenizer` | `[B,D]` → `[B,T,H]` | Shape tested; tokenization UNVERIFIED |
| Eq. (21)–(23) | Modality-specific self-attention, residual, LN and FFN | `TransformerBlock` | `[B,T,H]` → `[B,T,H]` | Patient isolation/gradient tested |
| Eq. (24)–(25) | Modality-specific logits and auxiliary BCE | `AdaptiveIntraInterFusion`; `compute_grain_loss` | modality vectors → logits `[B,M]` and scalar loss | Gradient tested |
| Eq. (26) | Binary predictive entropy from unimodal probability | `binary_predictive_entropy` | probabilities `[B,M]` → entropy `[B,M]` | Compared with manual calculation |
| Eq. (27)–(28) | Softmax negative entropy and weighted concatenation | `entropy_modality_weights`; fusion forward | entropy/intra vectors → weights `[B,M]` | Direction and row sum tested |
| Eq. (29)–(34) | Bidirectional cross-attention and concatenation | `CrossAttentionBlock`; fusion forward | two `[B,T,H]` streams → cross feature `[B,2H]` | Patient isolation/gradient tested |
| Eq. (35) | Hinge penalty beyond modality-weight tolerance | `modality_balance_loss` | weights `[B,M]` → scalar | Differentiable gradient tested |
| Eq. (36) | Concatenation, linear mapping and batch normalization | `AdaptiveIntraInterFusion.forward` | intra/cross `[B,4H]` → `[B,H]` | Shape/gradient tested |
| Eq. (37)–(38) | Two-layer MLP, sigmoid, primary BCE | fusion classifier; `compute_grain_loss` | `[B,H]` → logit/probability `[B]` and scalar | Forward/gradient tested |
| Eq. (39) | `L_ce + L_aux + lambda L_cl + mu L_bc` | `compute_grain_loss` | component losses → total scalar | All components stay in autograd graph |
| Label-free inference | Uncertainty is prediction entropy, not label-derived CE | `GRAIN.forward` | features/mask/anchor bank/IDs only | Signature invariant tested |
| Target isolation | No target-target transductive interaction | complete model | adding unrelated batch target leaves prediction unchanged | Unit tested in eval mode |
| Validation-AUC checkpoint rule | No outer-test model selection | `ValidationCheckpointManager` | validation result → best checkpoint | Test result rejection and reload tested |
| K in 2–10 | Select only inside outer development data | `select_k_on_validation` | train/validation → candidate trace and selected K | Test-blindness tested |
| Table II/VI/VII | Center A/B, mixed A+B, external Center C | `configs/*.json` | approved data/splits → future experiment | UNVERIFIED; cohort provenance conflicts |
| Table III–V | Graph/fusion/loss ablations | future Phase 8 | shared official protocol | Not implemented |
| Probability metrics | ACC/F1/REC/AUC/PRE/SPEC/NPV; positive class 1 | `grain/evaluation/metrics.py` | labels/probabilities → metrics | Probability-ranking AUC tested |
| Paired DeLong and Holm | Patient-aligned statistics | future statistics module | prediction CSV → paired tests | Deferred beyond Phase 4 |

## Unresolved architecture provenance

The paper does not uniquely specify `D`, `P`, `R`, `H`, dropout, the mapping
from one feature vector to multiple within-patient attention tokens, the
nonlinearity in Eq. (5)–(6), or the relation scales in Eq. (7)–(9). These stay
explicit and nullable in official configs. Synthetic tests supply values only
to verify mechanics; those values are not official experimental settings.

No row is marked verified merely because a similarly named legacy class
exists. `FAMI.py`, `GAT.py`, `attention1.py`, `trainerv1.py`, and `function.py`
were inspected for intent but their unsafe implementations were not copied.
