import json
import math
import unittest

from recommendation.core.symbolic import QuantileNumericEvidence


class QuantileNumericEvidenceTest(unittest.TestCase):
    def test_fit_is_deterministic_and_transform_does_not_refit(self):
        values = [7, 1, 6, 0, 5, 2, 4, 3, None, float("nan")]
        pairs = [(0, 1), (0, 2), (0, 5), (0, 10), (None, 3)]
        encoder = QuantileNumericEvidence.fit(
            "quality", values, pairs, bins=4
        )
        reversed_encoder = QuantileNumericEvidence.fit(
            "quality", reversed(values), reversed(pairs), bins=4
        )

        self.assertEqual(encoder.to_json(), reversed_encoder.to_json())
        self.assertEqual(encoder.scalar_thresholds, (2.0, 4.0, 6.0))
        self.assertEqual(encoder.delta_thresholds, (2.0, 5.0, 10.0))
        self.assertEqual(
            [encoder.encode_scalar(value) for value in (0, 2, 4, 7)],
            ["q1", "q2", "q3", "q4"],
        )

        # An evaluation outlier is transformed against frozen train metadata.
        before = encoder.to_json()
        self.assertEqual(encoder.encode_scalar(1e100), "q4")
        self.assertEqual(encoder.to_json(), before)

    def test_ties_collapse_boundaries_without_splitting_equal_values(self):
        encoder = QuantileNumericEvidence.fit(
            "count", [0, 0, 0, 1, 1, 2], [(0, 0), (0, 1), (0, 1), (0, 2)],
            bins=4,
            relative_tolerance=0,
            scale_tolerance=0,
        )

        self.assertEqual(encoder.scalar_thresholds, (1.0, 2.0))
        self.assertEqual(encoder.scalar_bin_count, 3)
        self.assertEqual(encoder.encode_scalar(0), "q1")
        self.assertEqual(encoder.encode_scalar(1), "q2")
        self.assertEqual(encoder.encode_scalar(2), "q3")
        self.assertEqual(encoder.encode_pair(2, 2), "equal")
        self.assertEqual(encoder.delta_sample_count, 3)

    def test_near_identical_delta_boundaries_collapse(self):
        encoder = QuantileNumericEvidence.fit(
            "rounded_llm_score",
            [0.0, 1.0],
            [(0.0, 0.04), (0.1, 0.14),
             (0.0, 0.4), (0.0, 0.8)],
            bins=4,
        )

        thresholds = encoder.delta_thresholds
        self.assertFalse(any(
            math.isclose(left, right, rel_tol=encoder.relative_tolerance,
                         abs_tol=encoder.absolute_tolerance)
            for left, right in zip(thresholds, thresholds[1:])
        ))
        self.assertEqual(
            encoder.encode_pair(0.0, 0.04),
            encoder.encode_pair(0.1, 0.14),
        )

    def test_weighted_pair_quantiles_can_equalize_source_groups(self):
        pairs = [(0, 1), (0, 2), (0, 3), (0, 100)]
        uniform = QuantileNumericEvidence.fit(
            "delta", [], pairs, bins=2
        )
        group_balanced = QuantileNumericEvidence.fit(
            "delta", [], pairs, training_pair_weights=[1 / 3, 1 / 3, 1 / 3, 1],
            bins=2,
        )

        self.assertEqual(uniform.delta_thresholds, (3.0,))
        self.assertEqual(group_balanced.delta_thresholds, (100.0,))

    def test_pair_weight_validation_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "length"):
            QuantileNumericEvidence.fit(
                "delta", [], [(0, 1)], training_pair_weights=[]
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            QuantileNumericEvidence.fit(
                "delta", [], [(0, 1)], training_pair_weights=[-1]
            )

    def test_pair_labels_are_directionally_antisymmetric(self):
        encoder = QuantileNumericEvidence.fit(
            "similarity",
            [0, 1, 2, 5, 10],
            [(0, 1), (0, 2), (0, 5), (0, 10)],
            bins=4,
        )
        for left, right in ((1, 0), (5, 0), (-3, 7), (100, -100)):
            forward = encoder.encode_pair(left, right)
            reverse = encoder.encode_pair(right, left)
            expected_reverse = (
                "right_" + forward.removeprefix("left_")
                if forward.startswith("left_")
                else "left_" + forward.removeprefix("right_")
            )
            self.assertEqual(expected_reverse, reverse)

        self.assertEqual(encoder.encode_pair(5, 0), "left_q3")
        self.assertEqual(encoder.encode_pair(0, 5), "right_q3")

    def test_missing_and_near_equal_values_have_explicit_states(self):
        encoder = QuantileNumericEvidence.fit(
            "score", [0, 100], [(0, 10), (0, 50)], bins=3
        )
        missing = (None, "bad", float("nan"), float("inf"), True)
        for value in missing:
            self.assertEqual(encoder.encode_scalar(value), "unknown")
            self.assertEqual(encoder.encode_pair(value, value), "unknown")
            self.assertEqual(encoder.encode_pair(1, value), "left_known")
            self.assertEqual(encoder.encode_pair(value, 1), "right_known")

        self.assertEqual(encoder.encode_pair(50, 50 + 1e-11), "equal")

    def test_positive_rescaling_preserves_symbols(self):
        values = [-4, -1, 0, 2, 9, 15]
        pairs = [(-4, -1), (-4, 2), (-4, 9), (-4, 15)]
        base = QuantileNumericEvidence.fit("signal", values, pairs, bins=4)
        factor = 1000
        scaled = QuantileNumericEvidence.fit(
            "signal",
            [value * factor for value in values],
            [(left * factor, right * factor) for left, right in pairs],
            bins=4,
        )

        for value in (-10, -4, 0, 7, 20):
            self.assertEqual(
                base.encode_scalar(value), scaled.encode_scalar(value * factor)
            )
        for left, right in ((-4, -1), (9, -4), (100, 0), (2, 2)):
            self.assertEqual(
                base.encode_pair(left, right),
                scaled.encode_pair(left * factor, right * factor),
            )

    def test_metadata_round_trip_is_exact_and_validated(self):
        encoder = QuantileNumericEvidence.fit(
            "recency", [1, 2, 3, 4], [(1, 2), (1, 4)], bins=3
        )
        restored = QuantileNumericEvidence.from_json(encoder.to_json())
        self.assertEqual(restored, encoder)
        self.assertEqual(restored.to_metadata(), encoder.to_metadata())
        self.assertEqual(json.loads(restored.to_json())["version"], 1)

        bad = restored.to_metadata()
        bad["scalar"]["thresholds"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            QuantileNumericEvidence.from_metadata(bad)

        non_finite = restored.to_metadata()
        non_finite["pair_delta"]["thresholds"] = [math.inf]
        with self.assertRaisesRegex(ValueError, "finite"):
            QuantileNumericEvidence.from_metadata(non_finite)


if __name__ == "__main__":
    unittest.main()
