from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from grain.config import load_config
from grain.data import load_legacy_feature_table
from grain.models import AnchorFeatureBank
from grain.training.cross_validation import build_model


ROOT = Path(__file__).parents[1]
FORMAL_CONFIGS = {
    "center_a": "formal_center_a.json",
    "center_b": "formal_center_b.json",
    "mixed_ab": "formal_mixed_ab.json",
    "center_c": "formal_external_center_c.json",
}


def load_bound_cohort(config: dict):
    data = config["data"]
    layout = data["legacy_layout"]
    return load_legacy_feature_table(
        Path(data["source_root"]) / data["feature_source"],
        sample_id_prefix=data["sample_id_prefix"],
        skip_leading_columns=int(layout["skip_leading_columns"]),
        plain_dimension=int(layout["plain_dimension"]),
        ce_dimension=int(layout["ce_dimension"]),
        label_column=data["label_column"],
        expected_sha256=data["expected_sha256"],
    )


class ArchitectureFreezeTests(unittest.TestCase):
    def test_manifest_declares_only_four_assumptions(self) -> None:
        manifest = json.loads(
            (ROOT / "configs" / "architecture_grain_official.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            manifest["remaining_assumed_parameters"],
            ["attention_tokens", "patient_dim", "relation_activation", "relation_gamma"],
        )

    def test_all_formal_profiles_construct_with_resolved_dimensions(self) -> None:
        expected = {"center_a": 120, "center_b": 60, "mixed_ab": 60, "center_c": 120}
        for name, filename in FORMAL_CONFIGS.items():
            with self.subTest(name=name):
                config = load_config(ROOT / "configs" / filename)
                self.assertEqual(config["model"]["feature_dimension"], expected[name])
                self.assertEqual(config["model"]["parameter_status"], "FROZEN_PHASE_4C")
                model = build_model(config, selected_k=2)
                self.assertEqual(model.feature_dimension, expected[name])

    def test_bound_modality_availability_is_unchanged(self) -> None:
        expected = {
            "center_a": (185, 23, 13),
            "center_b": (14, 0, 171),
            "center_c": (46, 0, 37),
        }
        for name, observed_expected in expected.items():
            with self.subTest(name=name):
                filename = FORMAL_CONFIGS[name]
                cohort = load_bound_cohort(load_config(ROOT / "configs" / filename))
                mask = cohort.modality_mask
                observed = (
                    int((mask[:, 0] & ~mask[:, 1]).sum()),
                    int((~mask[:, 0] & mask[:, 1]).sum()),
                    int(mask.all(dim=1).sum()),
                )
                self.assertEqual(observed, observed_expected)

    def test_mixed_partial_modality_rows_are_rejected(self) -> None:
        config = load_config(ROOT / "configs" / FORMAL_CONFIGS["mixed_ab"])
        with self.assertRaisesRegex(ValueError, "ce contains partially missing feature rows"):
            load_bound_cohort(config)

    def test_mixed_to_center_c_is_explicitly_blocked(self) -> None:
        mixed = load_config(ROOT / "configs" / FORMAL_CONFIGS["mixed_ab"])
        center_c = load_config(ROOT / "configs" / FORMAL_CONFIGS["center_c"])
        self.assertNotEqual(
            mixed["model"]["feature_dimension"], center_c["model"]["feature_dimension"]
        )
        self.assertEqual(mixed["external_alignment"]["status"], "BLOCKED")
        self.assertEqual(center_c["external_alignment"]["status"], "BLOCKED")

    def test_formal_center_a_real_batch_forward_is_finite(self) -> None:
        config = load_config(ROOT / "configs" / FORMAL_CONFIGS["center_a"])
        cohort = load_bound_cohort(config)
        complete = cohort.modality_mask.all(dim=1).nonzero().flatten()[:4]
        targets = cohort.modality_mask.logical_not().any(dim=1).nonzero().flatten()[:2]
        anchor_ids = tuple(cohort.sample_ids[index] for index in complete.tolist())
        target_ids = tuple(cohort.sample_ids[index] for index in targets.tolist())
        anchor_bank = AnchorFeatureBank(
            patient_ids=anchor_ids,
            training_patient_ids=frozenset(anchor_ids),
            features=cohort.features[complete],
            modality_mask=cohort.modality_mask[complete],
        )
        model = build_model(config, selected_k=5).eval()
        with torch.no_grad():
            output = model(
                cohort.features[targets], cohort.modality_mask[targets], anchor_bank, target_ids
            )
        self.assertEqual(output.probabilities.shape, (2,))
        self.assertTrue(bool(torch.isfinite(output.probabilities).all()))


if __name__ == "__main__":
    unittest.main()
