from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from grain.data import LegacyFeatureCohort, generate_nested_manifests_from_arrays
from grain.evaluation import EvaluationResult, compute_metrics
from grain.training import (
    ValidationCheckpointManager,
    build_fold_runtime,
    load_checkpoint,
    select_k_on_validation,
    test,
    train_one_epoch,
    validate,
)


def synthetic_legacy_cohort(seed: int = 3) -> LegacyFeatureCohort:
    generator = torch.Generator().manual_seed(seed)
    count = 40
    features = torch.randn(count, 2, 6, generator=generator)
    masks = torch.ones(count, 2, dtype=torch.bool)
    masks[::4] = torch.tensor([True, False])
    masks[1::4] = torch.tensor([False, True])
    features[~masks] = 0.0
    return LegacyFeatureCohort(
        sample_ids=tuple(f"sample_{index:03d}" for index in range(count)),
        features=features,
        modality_mask=masks,
        labels=torch.tensor([index % 2 for index in range(count)]),
        source_path="synthetic-only",
        source_sha256="synthetic-only",
    )


def synthetic_config() -> dict:
    return {
        "model": {
            "feature_dimension": 6,
            "patient_dimension": 8,
            "relation_dimension": 8,
            "gat_hidden_dimension": 8,
            "transformer_dimension": 8,
            "fusion_dimension": 8,
            "classifier_hidden_dimension": 8,
            "attention_tokens": 2,
            "fusion_attention_heads": 2,
            "transformer_ffn_dimension": 32,
            "relation_activation": "relu",
            "relation_alpha": 1.0,
            "relation_gamma": 1.0,
            "gat_dropout": 0.0,
            "transformer_dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 32,
            "scheduler": {"name": "step", "step_size": 2, "gamma": 0.5},
            "k_selection": {"mode": "validation", "fixed_k": None},
            "contrastive_temperature": 0.5,
            "uncertainty_temperature": 0.1,
            "balance_margin": 0.3,
            "loss_weights": {"auxiliary": 1.0, "contrastive": 0.1, "balance": 0.01},
        },
        "evaluation": {"threshold": 0.5},
    }


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = synthetic_legacy_cohort()
        self.split = generate_nested_manifests_from_arrays(
            self.cohort.sample_ids, self.cohort.labels.tolist(), "synthetic", 4, 0.25, 42
        )[0]
        self.device = torch.device("cpu")

    def runtime(self, seed: int = 42, k: int = 2):
        return build_fold_runtime(
            config=synthetic_config(), cohort=self.cohort, split=self.split,
            selected_k=k, device=self.device, seed=seed
        )

    def test_fold_model_optimizer_scheduler_and_anchor_independence(self) -> None:
        first, second = self.runtime(42), self.runtime(42)
        self.assertIsNot(first.model, second.model)
        self.assertIsNot(first.optimizer, second.optimizer)
        self.assertIsNot(first.scheduler, second.scheduler)
        self.assertIsNot(first.anchor_bank, second.anchor_bank)
        self.assertNotEqual(
            next(first.model.parameters()).data_ptr(), next(second.model.parameters()).data_ptr()
        )

    def test_anchor_isolation(self) -> None:
        runtime = self.runtime()
        self.assertTrue(set(runtime.anchor_bank.patient_ids).issubset(self.split.train))
        self.assertFalse(
            set(runtime.anchor_bank.patient_ids) & (set(self.split.validation) | set(self.split.test))
        )

    def test_checkpoint_rejects_test_and_round_trip_preserves_prediction(self) -> None:
        runtime = self.runtime()
        validation = validate(
            model=runtime.model, cohort=self.cohort, sample_ids=self.split.validation,
            anchor_bank=runtime.anchor_bank, fold=0, batch_size=32,
            device=self.device, threshold=0.5
        )
        held_out = test(
            model=runtime.model, cohort=self.cohort, sample_ids=self.split.test,
            anchor_bank=runtime.anchor_bank, fold=0, batch_size=32,
            device=self.device, threshold=0.5
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            manager = ValidationCheckpointManager(path)
            common = dict(
                model=runtime.model, optimizer=runtime.optimizer, epoch=0, fold=0,
                seed=42, selected_k=2, resolved_config=synthetic_config(),
                data_fingerprint={"sha256": "synthetic"}, git_commit="test",
                scheduler=runtime.scheduler,
            )
            with self.assertRaises(ValueError):
                manager.consider(held_out, **common)
            self.assertTrue(manager.consider(validation, **common))
            before = held_out.probabilities.copy()
            with torch.no_grad():
                next(runtime.model.parameters()).add_(1.0)
            load_checkpoint(path, runtime.model, runtime.optimizer, runtime.scheduler)
            after = test(
                model=runtime.model, cohort=self.cohort, sample_ids=self.split.test,
                anchor_bank=runtime.anchor_bank, fold=0, batch_size=32,
                device=self.device, threshold=0.5
            ).probabilities
            np.testing.assert_allclose(before, after, rtol=0, atol=0)

    def test_auc_uses_probability_ranking(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        probability = np.asarray([0.10, 0.40, 0.35, 0.30])
        self.assertAlmostEqual(compute_metrics(labels, probability).auc, 0.75)

    def test_same_seed_reproducible_and_different_seed_independent(self) -> None:
        same_a, same_b, different = self.runtime(9), self.runtime(9), self.runtime(10)
        state_a = same_a.model.state_dict()
        state_b = same_b.model.state_dict()
        state_different = different.model.state_dict()
        self.assertTrue(all(torch.equal(state_a[key], state_b[key]) for key in state_a))
        self.assertTrue(any(not torch.equal(state_a[key], state_different[key]) for key in state_a))

    def test_k_search_never_requests_outer_test(self) -> None:
        class TrackingCohort:
            def __init__(self, base):
                self.base = base
                self.requested = []
            def select(self, ids):
                values = tuple(ids)
                self.requested.append(values)
                return self.base.select(values)

        tracked = TrackingCohort(self.cohort)
        short_split = replace(self.split, k_candidates=(2, 3))
        observed_splits = []
        selected, results = select_k_on_validation(
            config=synthetic_config(), cohort=tracked, split=short_split,
            device=self.device, seed=42, epochs=1,
            observer=lambda _, value: observed_splits.append(value.split),
        )
        self.assertIn(selected, (2, 3))
        self.assertEqual(len(results), 2)
        self.assertEqual(observed_splits, ["validation", "validation"])
        test_ids = set(self.split.test)
        self.assertTrue(all(not (set(request) & test_ids) for request in tracked.requested))
        self.assertNotIn("test", inspect.signature(select_k_on_validation).parameters)

    def test_one_epoch_handles_all_modality_patterns_without_labels_in_forward(self) -> None:
        runtime = self.runtime()
        result = train_one_epoch(
            model=runtime.model, optimizer=runtime.optimizer, cohort=self.cohort,
            training_ids=self.split.train, anchor_bank=runtime.anchor_bank,
            batch_size=32, device=self.device, epoch_seed=42,
            contrastive_temperature=0.5, balance_margin=0.3,
            contrastive_weight=0.1, balance_weight=0.01,
        )
        training_masks = self.cohort.select(self.split.train)[1]
        patterns = {tuple(value) for value in training_masks.tolist()}
        self.assertEqual(patterns, {(True, False), (False, True), (True, True)})
        self.assertTrue(np.isfinite(result.loss))
        self.assertGreater(result.gradient_norm, 0)
        self.assertNotIn("labels", inspect.signature(runtime.model.forward).parameters)


if __name__ == "__main__":
    unittest.main()
