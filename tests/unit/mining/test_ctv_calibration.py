import math
import unittest

from recommendation.core.ctv_calibration import (
    CTVObservation,
    calibrate_ctv,
    reencode_ctv_confidence,
)


class ImpressionWeightedCTVTests(unittest.TestCase):
    def test_point_strength_weights_each_impression_equally(self):
        observations = [
            CTVObservation("small", matched=True, target=True),
            CTVObservation("small", matched=False, target=False),
        ]
        observations.extend(
            CTVObservation("large", matched=index < 4, target=index == 4)
            for index in range(8)
        )

        result = calibrate_ctv(observations, evidence_k=2, rule_kind="point")

        # Raw matched precision is 1/5. Impression-normalized matched precision
        # is (0.5 from small + 0 from large) / (0.5 + 0.5) = 0.5.
        self.assertEqual(result.raw.matched_target, 1)
        self.assertEqual(result.raw.matched_non_target, 4)
        self.assertAlmostEqual(result.positive.strength, 0.5)
        self.assertAlmostEqual(result.positive.weighted_support, 1.0)
        self.assertAlmostEqual(result.positive.effective_impressions, 2.0)
        self.assertAlmostEqual(result.positive.confidence, 0.5)
        self.assertAlmostEqual(result.negative.strength, 0.125)
        self.assertEqual(result.impressions, 2)
        self.assertEqual(result.observations, 10)

    def test_pair_applicability_is_not_multiplied_into_confidence(self):
        rows = [
            {"imp": "a", "match": True, "win": True, "known": True},
            {"imp": "a", "match": False, "win": False, "known": True},
            {"imp": "b", "match": True, "win": False, "known": True},
            {"imp": "b", "match": False, "win": True, "known": True},
            {"imp": "b", "match": False, "win": True, "known": False},
            {"imp": "b", "match": False, "win": False, "known": False},
        ]
        result = calibrate_ctv(
            [
                CTVObservation(
                    row["imp"], row["match"], row["win"], row["known"]
                )
                for row in rows
            ],
            evidence_k=2,
            rule_kind="pair",
        )

        self.assertEqual(result.rule_kind, "pair")
        self.assertAlmostEqual(result.applicability.weighted_fraction, 0.75)
        self.assertAlmostEqual(result.activation.weighted_fraction, 0.375)
        self.assertAlmostEqual(result.activation_given_applicable, 0.5)
        self.assertAlmostEqual(result.positive.strength, 2 / 3)
        self.assertAlmostEqual(result.negative.strength, 1 / 3)
        self.assertAlmostEqual(result.positive.effective_impressions, 1.8)
        # Confidence uses n_eff only; it is not 0.75 times this value.
        self.assertAlmostEqual(result.positive.confidence, 1.8 / 3.8)
        self.assertEqual(result.raw.inapplicable_target, 1)
        self.assertEqual(result.raw.inapplicable_non_target, 1)

    def test_custom_weights_are_normalized_inside_each_impression(self):
        observations = [
            CTVObservation("a", True, True, weight=3),
            CTVObservation("a", True, False, weight=1),
            CTVObservation("b", True, False, weight=1),
            CTVObservation("b", False, True, weight=1),
        ]
        result = calibrate_ctv(observations, evidence_k=1)

        self.assertAlmostEqual(result.positive.weighted_support, 1.5)
        self.assertAlmostEqual(result.positive.weighted_target_support, 0.75)
        self.assertAlmostEqual(result.positive.strength, 0.5)
        expected_neff = 1.5**2 / (1.0**2 + 0.5**2)
        self.assertAlmostEqual(result.positive.effective_impressions, expected_neff)
        self.assertAlmostEqual(
            result.positive.confidence, expected_neff / (expected_neff + 1)
        )
        self.assertAlmostEqual(result.weighted.mass, 2.0)

    def test_repeating_dependent_rows_inside_one_impression_adds_no_evidence(self):
        original = [
            CTVObservation("small", True, True),
            CTVObservation("small", False, False),
            CTVObservation("large", True, False),
            CTVObservation("large", False, True),
        ]
        repeated = original[:2] + original[2:] * 50

        baseline = calibrate_ctv(original, evidence_k=2, rule_kind="pair")
        duplicated = calibrate_ctv(repeated, evidence_k=2, rule_kind="pair")

        # The 100 dependent rows still represent one ``large`` impression.
        # They alter raw audit counts, but not its unit mass, CTV, Kish ESS or
        # confidence.
        self.assertGreater(
            duplicated.positive.raw_support, baseline.positive.raw_support
        )
        self.assertAlmostEqual(
            duplicated.positive.weighted_support,
            baseline.positive.weighted_support,
        )
        self.assertAlmostEqual(
            duplicated.positive.strength, baseline.positive.strength
        )
        self.assertAlmostEqual(
            duplicated.positive.effective_impressions,
            baseline.positive.effective_impressions,
        )
        self.assertAlmostEqual(
            duplicated.positive.confidence, baseline.positive.confidence
        )
        self.assertEqual(duplicated.positive.distinct_impressions, 2)

    def test_weighted_support_gate_lower_bounds_effective_impressions(self):
        """A macro-support gate already implies the same numeric Kish-ESS gate.

        Normalization gives every branch at most unit mass in each source
        impression.  Therefore ``sum(m_i**2) <= sum(m_i)`` and
        ``(sum(m_i)**2 / sum(m_i**2)) >= sum(m_i)``.  This guards the reason
        that mirrored pair rows cannot pass a support threshold while having
        fewer effective independent impressions than that threshold.
        """
        rows = [
            CTVObservation("a", True, True),
            CTVObservation("a", False, False),
            CTVObservation("b", True, True),
            CTVObservation("b", True, False),
            CTVObservation("b", False, True),
            CTVObservation("c", True, False),
            CTVObservation("c", False, True),
            CTVObservation("c", False, False),
            CTVObservation("c", False, False),
        ]

        result = calibrate_ctv(rows, evidence_k=800, rule_kind="pair")

        for branch in (result.positive, result.negative):
            self.assertLessEqual(
                branch.weighted_support, branch.effective_impressions
            )
            self.assertLessEqual(
                branch.effective_impressions, branch.distinct_impressions
            )

    def test_missing_branch_is_explicit_and_neutral(self):
        result = calibrate_ctv([
            CTVObservation("a", True, True),
            CTVObservation("b", True, False),
        ], evidence_k=4)

        self.assertTrue(result.positive.defined)
        self.assertFalse(result.negative.defined)
        self.assertEqual(result.negative.confidence, 0.0)
        self.assertEqual(result.negative.strength, result.applicable_target_base_rate)
        self.assertIn("(CTV (STV", result.metta_ctv())
        self.assertEqual(result.as_dict()["negative"]["defined"], False)

    def test_confidence_can_be_reencoded_without_changing_population_statistics(self):
        selected = calibrate_ctv([
            CTVObservation("a", True, True),
            CTVObservation("a", False, False),
            CTVObservation("b", True, False),
            CTVObservation("b", False, True),
        ], evidence_k=20, rule_kind="pair")

        petta = reencode_ctv_confidence(selected, evidence_k=800)

        self.assertEqual(selected.evidence_k, 20.0)
        self.assertEqual(petta.evidence_k, 800.0)
        self.assertEqual(petta.positive.strength, selected.positive.strength)
        self.assertEqual(
            petta.positive.effective_impressions,
            selected.positive.effective_impressions,
        )
        self.assertEqual(
            petta.positive.weighted_support,
            selected.positive.weighted_support,
        )
        self.assertAlmostEqual(
            petta.positive.confidence,
            petta.positive.effective_impressions
            / (petta.positive.effective_impressions + 800.0),
        )
        self.assertLess(petta.positive.confidence, selected.positive.confidence)
        with self.assertRaisesRegex(ValueError, "calibration"):
            reencode_ctv_confidence(None)
        with self.assertRaisesRegex(ValueError, "evidence_k"):
            reencode_ctv_confidence(selected, evidence_k=0)

    def test_rejects_invalid_evidence_and_observations(self):
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            calibrate_ctv([])
        with self.assertRaisesRegex(ValueError, "evidence_k"):
            calibrate_ctv([CTVObservation("i", True, True)], evidence_k=0)
        with self.assertRaisesRegex(ValueError, "weight"):
            CTVObservation("i", True, True, weight=math.inf)
        with self.assertRaisesRegex(ValueError, "positive total weight"):
            calibrate_ctv([CTVObservation("i", True, True, weight=0)])
        with self.assertRaisesRegex(ValueError, "matched must be bool"):
            CTVObservation("i", 1, True)


if __name__ == "__main__":
    unittest.main()
