"""Complete label-free GRAIN forward path defined by paper Eq. (3)–(37)."""

from __future__ import annotations

from torch import Tensor, nn

from .fusion import AdaptiveIntraInterFusion
from .graph import AdaptivePatientGraph
from .imputation import AnchorGraphImputer
from .representation import MaskAwarePatientRepresentation
from .types import AnchorFeatureBank, GRAINOutput


class GRAIN(nn.Module):
    """Official core model with an immutable training-only anchor interface.

    Shape symbols:
        B: target patients; A: training anchors; M: modalities (always 2)
        D: input feature dimension; P: patient representation dimension
        R: relation dimension; T: within-patient attention tokens
        H: fusion hidden dimension; C: binary classes represented by one logit

    Ground-truth labels are deliberately absent from ``forward``. Labels may
    enter only the external loss and metric functions.
    """

    def __init__(
        self,
        *,
        feature_dimension: int,
        patient_dimension: int,
        relation_dimension: int,
        gat_hidden_dimension: int,
        transformer_dimension: int,
        fusion_dimension: int,
        classifier_hidden_dimension: int,
        attention_tokens: int,
        attention_heads: int,
        transformer_ffn_dimension: int,
        relation_activation: str,
        relation_alpha: float,
        relation_gamma: float,
        requested_k: int,
        gat_dropout: float,
        transformer_dropout: float,
        uncertainty_temperature: float,
    ) -> None:
        super().__init__()
        if feature_dimension % 2 != 0 or gat_hidden_dimension % 2 != 0:
            raise ValueError("GAT input/output dimensions must support two heads")
        if transformer_dimension % attention_heads != 0:
            raise ValueError("transformer_dimension must be divisible by attention_heads")
        self.feature_dimension = int(feature_dimension)
        self.requested_k = int(requested_k)
        self.patient_representation = MaskAwarePatientRepresentation(
            feature_dimension, patient_dimension
        )
        self.patient_graph = AdaptivePatientGraph(
            patient_dimension=patient_dimension,
            relation_dimension=relation_dimension,
            activation=relation_activation,
            alpha=relation_alpha,
            gamma=relation_gamma,
        )
        self.imputer = AnchorGraphImputer(
            feature_dimension, gat_hidden_dimension, gat_dropout
        )
        self.fusion = AdaptiveIntraInterFusion(
            feature_dimension=feature_dimension,
            transformer_dimension=transformer_dimension,
            fusion_dimension=fusion_dimension,
            classifier_hidden_dimension=classifier_hidden_dimension,
            attention_tokens=attention_tokens,
            attention_heads=attention_heads,
            ffn_dimension=transformer_ffn_dimension,
            dropout=transformer_dropout,
            uncertainty_temperature=uncertainty_temperature,
        )

    def forward(
        self,
        features: Tensor,
        modality_mask: Tensor,
        anchor_bank: AnchorFeatureBank,
        patient_ids: tuple[str, ...],
    ) -> GRAINOutput:
        """Predict B targets without labels or target-target interaction.

        Inputs:
            features: [B, M=2, D]
            modality_mask: boolean [B, M=2]
            anchor_bank.features: [A, M=2, D], complete training patients only
            patient_ids: B stable IDs, used solely for anchor self-exclusion
        """

        if len(patient_ids) != features.shape[0]:
            raise ValueError("patient_ids must align with target rows")
        if features.device != anchor_bank.features.device:
            raise ValueError("Target and anchor tensors must be on the same device")
        if anchor_bank.features.shape[2] != self.feature_dimension:
            raise ValueError("Anchor feature dimension does not match the model")

        target_representation = self.patient_representation(features, modality_mask)
        anchor_representation = self.patient_representation(
            anchor_bank.features, anchor_bank.modality_mask
        )
        graph = self.patient_graph(
            target_representation=target_representation,
            anchor_representation=anchor_representation,
            target_patient_ids=patient_ids,
            anchor_patient_ids=anchor_bank.patient_ids,
            requested_k=self.requested_k,
        )
        reconstructed, updated = self.imputer(
            target_features=features,
            modality_mask=modality_mask,
            anchor_features=anchor_bank.features,
            target_to_anchor_adjacency=graph.adjacency,
        )
        (
            logits,
            probabilities,
            modality_logits,
            modality_probabilities,
            modality_entropy,
            modality_weights,
        ) = self.fusion(updated)
        return GRAINOutput(
            logits=logits,
            probabilities=probabilities,
            modality_logits=modality_logits,
            modality_probabilities=modality_probabilities,
            modality_entropy=modality_entropy,
            modality_weights=modality_weights,
            patient_representation=target_representation,
            reconstructed_features=reconstructed,
            updated_features=updated,
            graph=graph,
        )
