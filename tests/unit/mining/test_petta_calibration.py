from __future__ import annotations

import math
import unittest

from recommendation.mining.petta_calibration import (
    calibrate_point_rules,
    parse_point_calibrations,
    update_point_rules,
)


def calibration_form(rule_id: str = "mined_1") -> str:
    return (
        f"(PointCalibration {rule_id} "
        "(CTV (STV 0.5 0.25) (STV 0.75 0.2)) "
        "(Counts 2 4 10 5) (Metrics 0.5 1.0 1 0.4023594781))"
    )


class PointCalibrationParserTest(unittest.TestCase):
    def test_parses_complete_metta_result(self):
        result = parse_point_calibrations(
            [f"({calibration_form()})"], expected_ids=("mined_1",)
        )["mined_1"]

        self.assertEqual(result.joint_support, 2)
        self.assertEqual(result.antecedent_support, 4)
        self.assertEqual(result.population_size, 10)
        self.assertEqual(result.target_support, 5)
        self.assertEqual(result.strength, 0.5)
        self.assertEqual(result.negative_strength, 0.75)
        self.assertEqual(result.as_rule_update()["quality"], result.quality)

    def test_rejects_missing_or_inconsistent_results(self):
        with self.assertRaisesRegex(ValueError, "does not match requested"):
            parse_point_calibrations([], expected_ids=("mined_1",))
        inconsistent = calibration_form().replace(
            "(Counts 2 4 10 5)", "(Counts 6 4 10 5)"
        )
        with self.assertRaisesRegex(ValueError, "inconsistent contingency"):
            parse_point_calibrations([inconsistent], expected_ids=("mined_1",))


class RecordingPeTTa:
    def __init__(self):
        self.calls: list[str] = []

    def process_metta_string(self, source: str):
        self.calls.append(source)
        return [f"({calibration_form()})"]


class PointCalibrationQueryTest(unittest.TestCase):
    def test_host_only_serializes_rule_and_deserializes_metta_result(self):
        petta = RecordingPeTTa()
        result, query_count = calibrate_point_rules(
            petta,
            space="&complete_population",
            rules=[{
                "id": "mined_1",
                "premises": (("topic", "news"),),
            }],
            evidence_k=800.0,
        )

        self.assertEqual(query_count, 1)
        self.assertEqual(set(result), {"mined_1"})
        self.assertIn("recommendation-calibrate-rule", petta.calls[0])
        self.assertIn('(topic $case "news")', petta.calls[0])
        self.assertNotIn(str(math.log1p(4)), petta.calls[0])

    def test_append_update_sends_prior_counts_to_metta(self):
        petta = RecordingPeTTa()
        prior = parse_point_calibrations(
            [calibration_form()], expected_ids=("mined_1",)
        )["mined_1"]

        result, query_count = update_point_rules(
            petta,
            delta_space="&population_delta",
            rules=[{
                "id": "mined_1",
                "premises": (("topic", "news"),),
            }],
            previous={"mined_1": prior},
            evidence_k=800.0,
        )

        self.assertEqual(query_count, 1)
        self.assertEqual(set(result), {"mined_1"})
        self.assertIn("recommendation-update-calibration", petta.calls[0])
        self.assertIn(" 2 4 10 5)", petta.calls[0])


if __name__ == "__main__":
    unittest.main()
