from __future__ import annotations

import unittest

from grain.data.anchors import AnchorBank, AnchorPolicyError
from grain.data.splits import generate_nested_manifests
from tests.helpers import synthetic_cohort


class AnchorPolicyTests(unittest.TestCase):
    def test_only_complete_outer_training_patients_become_anchors(self) -> None:
        cohort = synthetic_cohort(40)
        split = generate_nested_manifests(cohort, "synthetic", 4, 0.25, 42)[0]
        bank = AnchorBank.from_training_records(cohort, split.train)
        bank.assert_no_forbidden_ids(set(split.validation) | set(split.test))
        records = cohort.by_id()
        self.assertTrue(all(records[value].is_complete for value in bank.anchor_patient_ids))
        self.assertTrue(set(bank.anchor_patient_ids).issubset(set(split.train)))

    def test_forbidden_identity_is_detected(self) -> None:
        cohort = synthetic_cohort(40)
        split = generate_nested_manifests(cohort, "synthetic", 4, 0.25, 42)[0]
        bank = AnchorBank.from_training_records(cohort, split.train)
        with self.assertRaises(AnchorPolicyError):
            bank.assert_no_forbidden_ids([bank.anchor_patient_ids[0]])

    def test_k_is_capped_and_self_is_excluded(self) -> None:
        cohort = synthetic_cohort(40)
        split = generate_nested_manifests(cohort, "synthetic", 4, 0.25, 42)[0]
        bank = AnchorBank.from_training_records(cohort, split.train)
        query = bank.anchor_patient_ids[0]
        candidates = bank.eligible_ids_for(query)
        self.assertNotIn(query, candidates)
        self.assertEqual(len(candidates), len(bank.anchor_patient_ids) - 1)
        self.assertEqual(bank.effective_k(query, 10_000), len(candidates))


if __name__ == "__main__":
    unittest.main()
