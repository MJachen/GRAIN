from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from grain.data import load_legacy_feature_table, resolve_data_fingerprint


class ProvenancePolicyTests(unittest.TestCase):
    def _write_source(self, root: Path) -> tuple[Path, str]:
        source = root / "features.csv"
        source.write_text(
            "plain_0,plain_1,ce_0,ce_1,label\n"
            "1.0,2.0,,,0\n"
            ",,3.0,4.0,1\n",
            encoding="utf-8",
        )
        return source, hashlib.sha256(source.read_bytes()).hexdigest()

    def _load(self, source: Path, **kwargs):
        return load_legacy_feature_table(
            source,
            sample_id_prefix="test",
            skip_leading_columns=0,
            plain_dimension=2,
            ce_dimension=2,
            label_column="label",
            **kwargs,
        )

    def test_missing_expected_sha_uses_actual_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, actual_sha = self._write_source(Path(directory))
            cohort = self._load(source)
            self.assertEqual(cohort.source_sha256, actual_sha)

    def test_stale_expected_sha_is_ignored_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, actual_sha = self._write_source(Path(directory))
            cohort = self._load(source, expected_sha256="0" * 64)
            self.assertEqual(cohort.source_sha256, actual_sha)

    def test_strict_fingerprint_enforcement_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, _ = self._write_source(Path(directory))
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                self._load(
                    source,
                    expected_sha256="0" * 64,
                    enforce_fingerprint=True,
                )

    def test_unregistered_dataset_records_actual_sha_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, actual_sha = self._write_source(root)
            inventory = root / "data_inventory.json"
            inventory.write_text(json.dumps({"datasets": []}), encoding="utf-8")
            fingerprint = resolve_data_fingerprint(
                source_path=source,
                source_sha256=actual_sha,
                inventory_path=inventory,
            )
            self.assertEqual(fingerprint["sha256"], actual_sha)
            self.assertFalse(fingerprint["inventory_registered"])
            self.assertEqual(
                fingerprint["registration_status"], "unregistered_dataset"
            )

    def test_strict_inventory_registration_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, actual_sha = self._write_source(root)
            inventory = root / "data_inventory.json"
            inventory.write_text(json.dumps({"datasets": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not registered"):
                resolve_data_fingerprint(
                    source_path=source,
                    source_sha256=actual_sha,
                    inventory_path=inventory,
                    require_inventory_registration=True,
                )

    def test_registered_inventory_metadata_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, actual_sha = self._write_source(root)
            inventory = root / "data_inventory.json"
            inventory.write_text(
                json.dumps({"datasets": [{"sha256": actual_sha, "dataset": "test"}]}),
                encoding="utf-8",
            )
            fingerprint = resolve_data_fingerprint(
                source_path=source,
                source_sha256=actual_sha,
                inventory_path=inventory,
            )
            self.assertEqual(fingerprint["dataset"], "test")
            self.assertTrue(fingerprint["inventory_registered"])
            self.assertEqual(fingerprint["registration_status"], "registered_dataset")


if __name__ == "__main__":
    unittest.main()
