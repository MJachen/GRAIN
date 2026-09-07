from copy import deepcopy
from pathlib import Path
import unittest

from grain.config import ConfigurationError, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_center_config_inherits_protocol_invariants(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "center_a.json"
        config = load_config(path)
        self.assertEqual(config["split"]["k_candidates"], list(range(2, 11)))
        self.assertEqual(config["training"]["checkpoint_metric"], "validation_auc")
        self.assertEqual(tuple(config["data"]["modalities"]), ("plain", "ce"))
        self.assertFalse(config["data"]["enforce_fingerprint"])
        self.assertFalse(config["data"]["require_inventory_registration"])

    def test_expected_sha256_is_optional_when_enforcement_is_disabled(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "center_a.json"
        config = deepcopy(load_config(path))
        config["data"].pop("expected_sha256")
        validate_config(config)

        config["data"]["enforce_fingerprint"] = True
        with self.assertRaisesRegex(ConfigurationError, "expected_sha256"):
            validate_config(config)

    def test_test_metric_cannot_select_checkpoint(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "center_a.json"
        config = load_config(path)
        config["training"]["checkpoint_metric"] = "test_accuracy"
        with self.assertRaises(ConfigurationError):
            validate_config(config)

    def test_phase5a_formal_configs_are_frozen(self) -> None:
        for filename, output_subdir, feature_dimension in (
            ("formal_center_a.json", "formal_center_a", 120),
            ("formal_center_b.json", "formal_center_b", 60),
        ):
            with self.subTest(filename=filename):
                config = load_config(Path(__file__).parents[1] / "configs" / filename)
                self.assertEqual(config["experiment"]["output_subdir"], output_subdir)
                self.assertEqual(config["model"]["feature_dimension"], feature_dimension)
                self.assertEqual(config["training"]["epochs"], 100)
                self.assertEqual(config["training"]["batch_size"], 32)
                self.assertEqual(config["training"]["optimizer"], "adam")
                self.assertEqual(config["training"]["learning_rate"], 1e-4)
                self.assertEqual(config["training"]["weight_decay"], 5e-4)
                self.assertIsNone(config["training"]["scheduler"])
                self.assertEqual(config["training"]["k_selection"]["mode"], "validation")
                self.assertEqual(config["training"]["k_selection"]["search_epochs"], 100)
                self.assertEqual(config["evaluation"]["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
