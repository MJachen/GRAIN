"""Two-layer, two-head weighted graph attention for Eq. (16)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class WeightedMultiHeadGraphAttention(nn.Module):
    """Graph attention where rows receive messages from columns.

    Shapes:
        node_features: [B, N, D_in]
        adjacency: [B, N, N], non-negative edge priors
        output: [B, N, D_out]
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        heads: int,
        dropout: float,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if output_dimension % heads != 0:
            raise ValueError("GAT output dimension must be divisible by head count")
        self.heads = int(heads)
        self.head_dimension = output_dimension // heads
        self.weight = nn.Parameter(
            torch.empty(self.heads, input_dimension, self.head_dimension)
        )
        self.attention_target = nn.Parameter(torch.empty(self.heads, self.head_dimension))
        self.attention_source = nn.Parameter(torch.empty(self.heads, self.head_dimension))
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = float(negative_slope)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attention_target.unsqueeze(-1))
        nn.init.xavier_uniform_(self.attention_source.unsqueeze(-1))

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        if node_features.ndim != 3 or adjacency.ndim != 3:
            raise ValueError("GAT expects node_features [B,N,D] and adjacency [B,N,N]")
        if adjacency.shape[:2] != node_features.shape[:2] or adjacency.shape[2] != node_features.shape[1]:
            raise ValueError("GAT adjacency does not align with node count")
        transformed = torch.einsum("bnd,hdo->bhno", node_features, self.weight)
        target_term = torch.einsum("bhnd,hd->bhn", transformed, self.attention_target)
        source_term = torch.einsum("bhnd,hd->bhn", transformed, self.attention_source)
        logits = F.leaky_relu(
            target_term.unsqueeze(-1) + source_term.unsqueeze(-2),
            negative_slope=self.negative_slope,
        )
        edge_mask = adjacency.unsqueeze(1) > 0
        logits = logits.masked_fill(~edge_mask, float("-inf"))
        learned = F.softmax(logits, dim=-1)
        weighted = learned * adjacency.unsqueeze(1)
        weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weighted = self.dropout(weighted)
        aggregated = torch.einsum("bhij,bhjd->bhid", weighted, transformed)
        return aggregated.transpose(1, 2).reshape(
            node_features.shape[0], node_features.shape[1], -1
        )


class TwoLayerTwoHeadGAT(nn.Module):
    """Paper-specified two GAT layers and two attention heads."""

    def __init__(self, feature_dimension: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = WeightedMultiHeadGraphAttention(
            feature_dimension, feature_dimension, heads=2, dropout=dropout
        )
        self.layer2 = WeightedMultiHeadGraphAttention(
            feature_dimension, feature_dimension, heads=2, dropout=dropout
        )

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        hidden = F.elu(self.layer1(node_features, adjacency))
        return self.layer2(hidden, adjacency)

