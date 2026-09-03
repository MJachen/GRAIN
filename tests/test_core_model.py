from __future__ import annotations

import inspect
import math
import unittest

import torch

from grain.losses import compute_grain_loss
from grain.models import (
    AnchorFeatureBank,
    GRAIN,
    binary_predictive_entropy,
    entropy_modality_weights,
)


def make_model(dropout: float = 0.1, requested_k: int = 2) -> GRAIN:
    torch.manual_seed(7)
    return GRAIN(
        feature_dimension=6,
        patient_dimension=8,
        relation_dimension=8,
        gat_hidden_dimension=8,
        transformer_dimension=8,
        fusion_dimension=8,
        classifier_hidden_dimension=8,
        attention_tokens=3,
        attention_heads=2,
        transformer_ffn_dimension=32,
        relation_activation="relu",
        relation_alpha=1.0,
        relation_gamma=1.0,
        requested_k=requested_k,
        gat_dropout=dropout,
        transformer_dropout=dropout,
        uncertainty_temperature=0.1,
    )


def make_bank() -> AnchorFeatureBank:
    generator = torch.Generator().manual_seed(17)
    return AnchorFeatureBank(
        patient_ids=("A0", "A1", "A2", "A3"),
        training_patient_ids=frozenset({"A0", "A1", "A2", "A3", "T0"}),
        features=torch.randn(4, 2, 6, generator=generator),
        modality_mask=torch.ones(4, 2, dtype=torch.bool),
    )


class CoreGRAINTests(unittest.TestCase):
    def test_forward_signature_is_label_free(self) -> None:
        parameters = inspect.signature(GRAIN.forward).parameters
        self.assertNotIn("label", parameters)
        self.assertNotIn("labels", parameters)
        self.assertEqual(
            tuple(parameters),
            ("self", "features", "modality_mask", "anchor_bank", "patient_ids"),
        )

    def test_masks_self_exclusion_and_k_eff(self) -> None:
        model = make_model(requested_k=99).eval()
        features = torch.randn(3, 2, 6)
        masks = torch.tensor([[True, False], [False, True], [True, True]])
        with torch.no_grad():
            output = model(features, masks, make_bank(), ("T0", "T1", "A0"))
        self.assertEqual(tuple(output.probabilities.shape), (3,))
        self.assertEqual(tuple(output.updated_features.shape), (3, 2, 6))
        self.assertEqual(output.graph.effective_k.tolist(), [4, 4, 3])
        self.assertFalse(bool(output.graph.selected_mask[2, 0]))
        self.assertEqual(output.graph.adjacency[2, 0].item(), 0.0)
        torch.testing.assert_close(output.graph.adjacency.sum(dim=1), torch.ones(3))
        # Eq. (19): missing uses reconstruction; observed adds original feature.
        torch.testing.assert_close(
            output.updated_features[0, 1], output.reconstructed_features[0, 1]
        )
        torch.testing.assert_close(
            output.updated_features[1, 0], output.reconstructed_features[1, 0]
        )
        torch.testing.assert_close(
            output.updated_features[2], output.reconstructed_features[2] + features[2]
        )

    def test_entropy_matches_manual_binary_entropy(self) -> None:
        probabilities = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
        observed = binary_predictive_entropy(probabilities)
        expected_first = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
        expected = torch.tensor([[expected_first, math.log(2)]], dtype=torch.float64)
        torch.testing.assert_close(observed, expected)

    def test_higher_uncertainty_receives_lower_weight(self) -> None:
        entropy = torch.tensor([[0.1, 0.6]])
        weights = entropy_modality_weights(entropy, temperature=0.2)
        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))

    def test_all_declared_module_groups_receive_gradient(self) -> None:
        model = make_model(dropout=0.0).train()
        bank = make_bank()
        features = torch.randn(4, 2, 6)
        masks = torch.tensor(
            [[True, True], [True, True], [True, False], [False, True]]
        )
        output = model(features, masks, bank, ("A0", "A1", "T0", "T1"))
        losses = compute_grain_loss(
            output,
            labels=torch.tensor([0.0, 1.0, 0.0, 1.0]),
            original_features=features,
            modality_mask=masks,
            contrastive_temperature=0.5,
            balance_margin=0.3,
            contrastive_weight=0.1,
            balance_weight=0.01,
        )
        losses.total.backward()
        for name, module in (
            ("patient_representation", model.patient_representation),
            ("patient_graph", model.patient_graph),
            ("imputer", model.imputer),
            ("fusion", model.fusion),
        ):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            self.assertTrue(gradients, f"{name} received no gradients")
            self.assertTrue(
                any(bool(torch.isfinite(value).all()) and value.abs().sum() > 0 for value in gradients),
                f"{name} received no finite non-zero gradient",
            )

    def test_patient_isolation_and_deterministic_eval(self) -> None:
        model = make_model().eval()
        bank = make_bank()
        first = torch.randn(1, 2, 6)
        mask_first = torch.tensor([[True, False]])
        unrelated = torch.randn(1, 2, 6)
        with torch.no_grad():
            alone = model(first, mask_first, bank, ("T0",))
            repeated = model(first, mask_first, bank, ("T0",))
            together = model(
                torch.cat([first, unrelated]),
                torch.cat([mask_first, torch.tensor([[False, True]])]),
                bank,
                ("T0", "T9"),
            )
        torch.testing.assert_close(alone.probabilities, repeated.probabilities)
        torch.testing.assert_close(alone.probabilities[0], together.probabilities[0])
        torch.testing.assert_close(alone.graph.adjacency[0], together.graph.adjacency[0])

    def test_anchor_bank_rejects_non_training_patient(self) -> None:
        with self.assertRaises(ValueError):
            AnchorFeatureBank(
                patient_ids=("A0", "TEST"),
                training_patient_ids=frozenset({"A0"}),
                features=torch.randn(2, 2, 6),
                modality_mask=torch.ones(2, 2, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
