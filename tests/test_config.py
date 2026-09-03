from pathlib import Path
import unittest

from grain.config import ConfigurationError, load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_center_config_inherits_protocol_invariants(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "center_a.yaml"
        config = load_config(path)
        self.assertEqual(config["split"]["k_candidates"], list(range(2, 11)))
        self.assertEqual(config["training"]["checkpoint_metric"], "validation_auc")
        self.assertEqual(tuple(config["data"]["modalities"]), ("plain", "ce"))

    def test_test_metric_cannot_select_checkpoint(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "center_a.yaml"
        config = load_config(path)
        config["training"]["checkpoint_metric"] = "test_accuracy"
        with self.assertRaises(ConfigurationError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

