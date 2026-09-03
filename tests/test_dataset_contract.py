from __future__ import annotations

import unittest

import numpy as np

from grain.data.dataset import FeatureDataset
from grain.data.schema import CohortManifest, CohortValidationError, PatientRecord


class DatasetContractTests(unittest.TestCase):
    def test_mask_is_explicit_and_authoritative(self) -> None:
        record = PatientRecord(
            patient_id="P001",
            center="A",
            label=1,
            plain_available=True,
            ce_available=False,
            plain_feature_path="plain.npy",
            ce_feature_path="should_not_be_loaded.npy",
        )
        requested_paths: list[str] = []

        def loader(path):
            requested_paths.append(path.name)
            return np.asarray([1.0, 2.0], dtype=np.float32)

        dataset = FeatureDataset(CohortManifest([record]), ".", 2, 2, loader=loader)
        sample = dataset[0]
        np.testing.assert_array_equal(sample["mask"], [True, False])
        np.testing.assert_array_equal(sample["ce"], np.zeros(2, dtype=np.float32))
        self.assertEqual(requested_paths, ["plain.npy"])

    def test_duplicate_patient_ids_are_rejected(self) -> None:
        record = PatientRecord("P001", "A", 0, True, False, "p.npy", None)
        with self.assertRaises(CohortValidationError):
            CohortManifest([record, record])

    def test_patient_must_have_at_least_one_modality(self) -> None:
        record = PatientRecord("P001", "A", 0, False, False, None, None)
        with self.assertRaises(CohortValidationError):
            CohortManifest([record])


if __name__ == "__main__":
    unittest.main()

