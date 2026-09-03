"""Paper Eq. (16) and (19): anchor-only graph completion and update."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .gat import TwoLayerTwoHeadGAT


class AnchorGraphImputer(nn.Module):
    """Reconstruct each modality from a target-to-training-anchor star graph.

    B = target count, A = anchor count, M = 2, D = feature dimension.
    Target nodes never send messages to another target patient.
    """

    def __init__(
        self, feature_dimension: int, hidden_dimension: int, dropout: float
    ) -> None:
        super().__init__()
        self.feature_dimension = int(feature_dimension)
        self.modality_gat = nn.ModuleList(
            TwoLayerTwoHeadGAT(
                self.feature_dimension, hidden_dimension, dropout=dropout
            )
            for _ in range(2)
        )

    def forward(
        self,
        target_features: Tensor,
        modality_mask: Tensor,
        anchor_features: Tensor,
        target_to_anchor_adjacency: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if target_features.ndim != 3 or target_features.shape[1:] != (
            2,
            self.feature_dimension,
        ):
            raise ValueError("target_features must be [B, 2, D]")
        batch_size = target_features.shape[0]
        anchor_count = anchor_features.shape[0]
        if anchor_features.shape[1:] != (2, self.feature_dimension):
            raise ValueError("anchor_features must be [A, 2, D]")
        if target_to_anchor_adjacency.shape != (batch_size, anchor_count):
            raise ValueError("target-to-anchor adjacency must be [B, A]")

        node_count = anchor_count + 1
        graph_adjacency = target_features.new_zeros(batch_size, node_count, node_count)
        graph_adjacency[:, 0, 1:] = target_to_anchor_adjacency
        anchor_indices = torch.arange(1, node_count, device=target_features.device)
        graph_adjacency[:, anchor_indices, anchor_indices] = 1.0

        reconstructed_modalities = []
        for modality_index, gat in enumerate(self.modality_gat):
            target_node = target_features[:, modality_index, :].unsqueeze(1)
            anchors = anchor_features[:, modality_index, :].unsqueeze(0).expand(
                batch_size, -1, -1
            )
            nodes = torch.cat([target_node, anchors], dim=1)
            reconstructed_modalities.append(gat(nodes, graph_adjacency)[:, 0, :])
        reconstructed = torch.stack(reconstructed_modalities, dim=1)
        observed = modality_mask.to(dtype=target_features.dtype).unsqueeze(-1)
        updated = reconstructed + observed * target_features
        return reconstructed, updated
