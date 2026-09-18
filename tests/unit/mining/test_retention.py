from __future__ import annotations

import unittest

from recommendation.app.server import Lab, MINING_CONFIG_KEYS
from recommendation.mining.retention import retain_complete_units


class CompleteUnitRetentionTest(unittest.TestCase):
    def test_keeps_newest_units_without_splitting_interleaved_unit(self):
        selected = retain_complete_units(
            [
                ("impression-1", "a"),
                ("impression-2", "b"),
                ("impression-1", "c"),
                ("impression-3", "d"),
            ],
            max_units=2,
            max_cases=10,
        )

        # Completion order is impression-2, impression-1, impression-3.
        # Both records from impression-1 survive even though they interleave.
        self.assertEqual(selected.unit_ids, ("impression-1", "impression-3"))
        self.assertEqual(selected.records, ("a", "c", "d"))
        self.assertEqual(selected.audit.source_units, 3)
        self.assertEqual(selected.audit.retained_units, 2)
        self.assertEqual(selected.audit.expired_units, 1)
        self.assertEqual(selected.audit.expired_cases, 1)
        self.assertTrue(selected.audit.unit_limit_applied)
        self.assertFalse(selected.audit.split_units)
        self.assertTrue(selected.audit.exact)

    def test_case_limit_keeps_a_contiguous_whole_unit_suffix(self):
        selected = retain_complete_units(
            [
                ("old", 1), ("old", 2),
                ("middle", 3), ("middle", 4),
                ("new", 5),
            ],
            max_units=10,
            max_cases=3,
        )

        self.assertEqual(selected.unit_ids, ("middle", "new"))
        self.assertEqual(selected.records, (3, 4, 5))
        self.assertTrue(selected.audit.case_limit_applied)
        self.assertFalse(selected.audit.unit_limit_applied)
        self.assertEqual(selected.audit.retained_cases, 3)
        self.assertEqual(selected.audit.expired_cases, 2)
        self.assertEqual(
            selected.audit.expiration_update, "exact_full_rebuild"
        )

    def test_rejects_newest_oversized_unit_instead_of_splitting_it(self):
        with self.assertRaisesRegex(MemoryError, "refusing to split"):
            retain_complete_units(
                [("old", 1), ("new", 2), ("new", 3), ("new", 4)],
                max_units=2,
                max_cases=2,
            )

    def test_empty_snapshot_is_well_formed_and_deterministic(self):
        first = retain_complete_units([], max_units=3, max_cases=4)
        second = retain_complete_units([], max_units=3, max_cases=4)
        self.assertEqual(first.records, ())
        self.assertEqual(first.unit_ids, ())
        self.assertEqual(
            first.audit.retained_membership_sha256,
            second.audit.retained_membership_sha256,
        )

    def test_lab_uses_impressions_as_units_and_keeps_source_case_ids(self):
        lab = object.__new__(Lab)
        lab.data = {"events": [
            {"impression": "old", "article": "a", "action": "click"},
            {"impression": "old", "article": "b", "action": "skip"},
            {"impression": "middle", "article": "c", "action": "click"},
            {"impression": "new", "article": "d", "action": "skip"},
        ]}
        lab.config = {
            "mining_retention_max_units": 2,
            "mining_retention_max_cases": 10,
            "negative_ratio": 0,
        }

        retained = lab._retained_mining_population()
        self.assertEqual(
            tuple(index for index, _event in retained.records), (2, 3)
        )
        self.assertEqual(
            [case_id for case_id, _event
             in lab._mining_event_cases(retained.records)],
            ["point_case_2", "point_case_3"],
        )
        self.assertEqual(retained.audit.expired_units, 1)
        self.assertEqual(retained.audit.expired_cases, 2)
        self.assertIn("mining_retention_max_units", MINING_CONFIG_KEYS)
        self.assertIn("mining_retention_max_cases", MINING_CONFIG_KEYS)


if __name__ == "__main__":
    unittest.main()
