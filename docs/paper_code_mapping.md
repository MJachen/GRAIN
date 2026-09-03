# Paper–Code Mapping

Specification source: latest 10-page GRAIN manuscript supplied with the
historical repository. Status reflects the official repository only.

| Paper item | Official location | Status after Phase 2 |
|---|---|---|
| Centers A, B, C and mixed A+B | `configs/*.yaml`; `grain/data/schema.py` | Interface complete; official cohort manifests unverified |
| Plain and contrast-enhanced CT | `PatientRecord`; `FeatureDataset` | Implemented with explicit names and fixed mask order |
| Patient-level splitting | `grain/data/splits.py` | Implemented and tested |
| Training-only complete-modality anchors | `grain/data/anchors.py` | Identity/policy boundary implemented; feature bank deferred |
| Eq. (2), MedicalNet feature extraction | planned `grain/data/features.py` | Not implemented |
| Eq. (3)–(4), patient representation | planned `grain/models/representation.py` | Not implemented |
| Eq. (5)–(15), adaptive graph and Top-K | planned `grain/models/graph.py` | Not implemented |
| Eq. (16), two-layer/two-head GAT | planned `grain/models/imputation.py` | Not implemented |
| Eq. (17)–(19), contrastive learning/update | planned model/loss modules | Not implemented |
| Eq. (20)–(23), intra-modal modeling | planned `grain/models/fusion.py` | Not implemented |
| Eq. (24)–(28), auxiliary prediction, entropy and weighting | planned fusion/loss modules | Not implemented |
| Eq. (29)–(34), bidirectional cross-attention | planned `grain/models/fusion.py` | Not implemented |
| Eq. (35), modality-balance loss | planned `grain/losses/balance.py` | Not implemented |
| Eq. (36)–(38), final classifier | planned `grain/models/classifier.py` | Not implemented |
| Eq. (39), combined loss | planned `grain/losses/objective.py` | Not implemented |
| Validation-AUC checkpoint rule | enforced by config invariant | Trainer not implemented |
| K in 2–10 selected inside outer development data | config + split contract | Selection routine deferred to Phase 4 |
| Table II, Center A/B | `configs/center_a.yaml`, `center_b.yaml` | Experiment not run |
| Table III–V, graph/fusion/loss ablations | future Phase 8 configs | Not implemented |
| Table VI, mixed A+B | `configs/mixed_ab.yaml` | Experiment not run |
| Table VII, external Center C | `configs/external_center_c.yaml` | Experiment not run |
| Probability metrics, paired DeLong and Holm | planned `grain/evaluation/` | Not implemented |

No row in this document may be marked implemented solely because a similarly
named legacy class exists. Equation-level tests are required in Phase 3.

