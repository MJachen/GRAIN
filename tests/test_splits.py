from __future__ import annotations

import unittest

from grain.data.splits import generate_nested_manifests
from tests.helpers import synthetic_cohort


class SplitTests(unittest.TestCase):
    def test_outer_and_validation_partitions_are_disjoint_and_complete(self) -> None:
        cohort = synthetic_cohort(40)
        manifests = generate_nested_manifests(
            cohort, "synthetic", n_outer_folds=4, validation_fraction=0.25, seed=42
        )
        self.assertEqual(len(manifests), 4)
        seen_as_test: list[str] = []
        for manifest in manifests:
            manifest.validate(cohort.patient_ids)
            self.assertFalse(set(manifest.test) & set(manifest.train))
            self.assertFalse(set(manifest.test) & set(manifest.validation))
            seen_as_test.extend(manifest.test)
        self.assertEqual(sorted(seen_as_test), sorted(cohort.patient_ids))

    def test_generation_is_reproducible(self) -> None:
        cohort = synthetic_cohort(40)
        first = generate_nested_manifests(cohort, "synthetic", 4, 0.25, 11)
        second = generate_nested_manifests(cohort, "synthetic", 4, 0.25, 11)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

