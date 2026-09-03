"""Paper Eq. (20)–(37): patient-isolated uncertainty-aware fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def binary_predictive_entropy(probabilities: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Eq. (26), elementwise binary entropy for probabilities in [0, 1]."""

    probabilities = probabilities.clamp(epsilon, 1.0 - epsilon)
    return -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )


def entropy_modality_weights(entropy: Tensor, temperature: float) -> Tensor:
    """Eq. (27): higher uncertainty receives lower sample-wise weight."""

    if temperature <= 0:
        raise ValueError("uncertainty temperature must be positive")
    if entropy.ndim != 2 or entropy.shape[1] != 2:
        raise ValueError("entropy must have shape [B, 2]")
    return F.softmax(-entropy / temperature, dim=1)


class FeatureTokenizer(nn.Module):
    """Projection head g from [B,D] to within-patient tokens [B,T,H].

    The paper does not specify tokenization. T is therefore a required explicit
    experimental parameter rather than a hidden default.
    """

    def __init__(self, input_dimension: int, hidden_dimension: int, tokens: int) -> None:
        super().__init__()
        if tokens < 1:
            raise ValueError("attention token count must be positive")
        self.hidden_dimension = int(hidden_dimension)
        self.tokens = int(tokens)
        self.projection = nn.Linear(input_dimension, hidden_dimension * tokens)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features).reshape(
            features.shape[0], self.tokens, self.hidden_dimension
        )


class TransformerBlock(nn.Module):
    """Eq. (21)–(23), operating only over tokens within each patient."""

    def __init__(self, hidden_dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dimension)
        self.norm2 = nn.LayerNorm(hidden_dimension)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension * 4),
            nn.ReLU(),
            nn.Linear(hidden_dimension * 4, hidden_dimension),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        hidden = self.norm1(tokens + self.dropout(attended))
        return self.norm2(hidden + self.dropout(self.feed_forward(hidden)))


class CrossAttentionBlock(nn.Module):
    """One direction of Eq. (29)–(33), isolated within each patient."""

    def __init__(self, hidden_dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dimension)
        self.norm2 = nn.LayerNorm(hidden_dimension)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension * 4),
            nn.ReLU(),
            nn.Linear(hidden_dimension * 4, hidden_dimension),
        )

    def forward(self, query_tokens: Tensor, source_tokens: Tensor) -> Tensor:
        attended, _ = self.attention(
            query_tokens, source_tokens, source_tokens, need_weights=False
        )
        hidden = self.norm1(query_tokens + self.dropout(attended))
        return self.norm2(hidden + self.dropout(self.feed_forward(hidden)))


class AdaptiveIntraInterFusion(nn.Module):
    """AIIFM with auxiliary predictions, entropy weights and cross-attention.

    Input updated_features is [B, M=2, D]. All attention tensors are [B,T,H],
    so B is never interpreted as a Transformer sequence dimension.
    """

    def __init__(
        self,
        feature_dimension: int,
        hidden_dimension: int,
        attention_tokens: int,
        attention_heads: int,
        dropout: float,
        uncertainty_temperature: float,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            feature_dimension, hidden_dimension, attention_tokens
        )
        self.intra_blocks = nn.ModuleList(
            TransformerBlock(hidden_dimension, attention_heads, dropout) for _ in range(2)
        )
        self.auxiliary_classifiers = nn.ModuleList(
            nn.Linear(hidden_dimension, 1) for _ in range(2)
        )
        self.cross_a_from_b = CrossAttentionBlock(
            hidden_dimension, attention_heads, dropout
        )
        self.cross_b_from_a = CrossAttentionBlock(
            hidden_dimension, attention_heads, dropout
        )
        self.final_projection = nn.Linear(hidden_dimension * 4, hidden_dimension)
        self.batch_norm = nn.BatchNorm1d(hidden_dimension)
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.uncertainty_temperature = float(uncertainty_temperature)

    def forward(
        self, updated_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if updated_features.ndim != 3 or updated_features.shape[1] != 2:
            raise ValueError("updated_features must have shape [B, 2, D]")

        projected = [
            self.tokenizer(updated_features[:, modality_index, :])
            for modality_index in range(2)
        ]
        intra_tokens = [
            block(tokens) for block, tokens in zip(self.intra_blocks, projected)
        ]
        intra_vectors = [tokens.mean(dim=1) for tokens in intra_tokens]
        modality_logits = torch.cat(
            [
                classifier(vector)
                for classifier, vector in zip(
                    self.auxiliary_classifiers, intra_vectors
                )
            ],
            dim=1,
        )
        modality_probabilities = torch.sigmoid(modality_logits)
        modality_entropy = binary_predictive_entropy(modality_probabilities)
        modality_weights = entropy_modality_weights(
            modality_entropy, self.uncertainty_temperature
        )
        weighted_intra = torch.cat(
            [
                intra_vectors[index] * modality_weights[:, index : index + 1]
                for index in range(2)
            ],
            dim=1,
        )

        cross_a = self.cross_a_from_b(projected[0], projected[1]).mean(dim=1)
        cross_b = self.cross_b_from_a(projected[1], projected[0]).mean(dim=1)
        cross = torch.cat([cross_a, cross_b], dim=1)
        final_features = self.batch_norm(
            self.final_projection(torch.cat([weighted_intra, cross], dim=1))
        )
        logits = self.final_classifier(final_features).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        return (
            logits,
            probabilities,
            modality_logits,
            modality_probabilities,
            modality_entropy,
            modality_weights,
        )
