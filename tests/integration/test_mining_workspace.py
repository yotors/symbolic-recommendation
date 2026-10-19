from __future__ import annotations

import math
import unittest

from recommendation.app.server import Lab, fixture


class MiningWorkspaceIntegrationTest(unittest.TestCase):
    def assert_point_calibration_matches_oracle(self, lab):
        population = [
            (event, lab._bounded_features(lab.event_features(event)))
            for event in lab.data["events"]
        ]
        target_total = sum(
            event["action"] in {"click", "like", "complete"}
            for event, _attrs in population
        )
        base_rate = target_total / len(population)
        for rule in lab.mined_rules:
            matching = [
                event for event, attrs in population
                if all(attrs.get(predicate) == value
                       for predicate, value in rule["premises"])
            ]
            joint = sum(
                event["action"] in {"click", "like", "complete"}
                for event in matching
            )
            strength = joint / len(matching)
            confidence = len(matching) / (
                len(matching) + lab.config["ctv_evidence_k"]
            )
            expected_quality = (
                confidence * abs(strength - base_rate)
                * (1 + 0.25 * (len(rule["premises"]) - 1))
                * math.log1p(len(matching))
            )
            self.assertEqual(rule["joint_support"], joint)
            self.assertEqual(rule["antecedent_support"], len(matching))
            self.assertAlmostEqual(rule["strength"], strength)
            self.assertAlmostEqual(rule["confidence"], confidence)
            outside = len(population) - len(matching)
            self.assertAlmostEqual(
                rule["negative_strength"],
                (target_total - joint) / outside if outside else 0.0,
            )
            self.assertAlmostEqual(rule["quality"], expected_quality)

    def test_stable_case_ids_enable_reuse_and_append_only_sync(self):
        lab = Lab(data=fixture())
        try:
            self.assert_point_calibration_matches_oracle(lab)

            initial_point_ids = [case_id for case_id, _event in lab._mining_event_cases()]
            sampled_pair_ids = {
                case["case_id"] for case in lab._pair_training_cases(negative_ratio=1)
            }
            population_pair_ids = {
                case["case_id"] for case in lab._pair_training_cases(negative_ratio=0)
            }
            self.assertEqual(len(initial_point_ids), len(set(initial_point_ids)))
            self.assertTrue(sampled_pair_ids)
            self.assertTrue(sampled_pair_ids.issubset(population_pair_ids))
            self.assertEqual(
                len(population_pair_ids),
                len(lab._pair_training_cases(negative_ratio=0)),
            )

            unchanged = lab.mine()
            all_sync = [
                *unchanged["workspace_sync"],
                *unchanged["pairwise"]["workspace_sync"],
            ]
            self.assertTrue(all_sync)
            self.assertEqual({item["mode"] for item in all_sync}, {"reused"})
            self.assertTrue(all(item["appended_facts"] == 0 for item in all_sync))

            lab.data["events"].append({
                "user":"u1", "article":"n1", "action":"click",
                "impression":"workspace_append_only_impression",
            })
            appended = lab.mine()
            point_sync = appended["workspace_sync"]
            self.assertTrue(point_sync)
            self.assertEqual({item["mode"] for item in point_sync}, {"appended"})
            self.assertTrue(all(item["appended"] == 1 for item in point_sync))
            self.assertEqual(
                [case_id for case_id, _event in lab._mining_event_cases()][:-1],
                initial_point_ids,
            )
            self.assertEqual(
                appended["workspace_mode"],
                "incremental_fpminer_support_petta_full_population_ctv",
            )
            self.assertEqual(
                appended["full_population_calibration"]["executor"], "petta"
            )
            self.assertEqual(
                appended["full_population_calibration"]["statistics_mode"],
                "delta_updated",
            )
            self.assertEqual(
                appended["full_population_calibration"]["appended_cases"], 1
            )
            self.assert_point_calibration_matches_oracle(lab)
            self.assertFalse(appended["full_structure_research"])
            self.assertEqual(
                {item["mode"] for item in appended["fpminer_incremental"]},
                {"delta_updated"},
            )
            self.assertTrue(all(
                item["researched_cases"] == 1
                for item in appended["fpminer_incremental"]
            ))
            self.assertEqual(
                {item["mode"] for item in appended["pairwise"]["fpminer_incremental"]},
                {"reused"},
            )
            self.assertEqual(
                lab.state()["mining_workspace"]["last_sync"],
                point_sync,
            )
        finally:
            lab.close()

    def test_all_negative_live_impression_delta_mines_its_skip(self):
        lab = Lab(data=fixture(), config={"negative_ratio":1})
        try:
            case_index=len(lab.data["events"])
            case_id=f"point_case_{case_index}"
            lab.data["events"].append({
                "user":"u1", "article":"n1", "action":"skip",
                "impression":"live_incremental_skip_0_1",
            })

            selected=dict(lab._mining_event_cases())
            self.assertIn(case_id,selected)
            self.assertEqual(selected[case_id]["action"],"skip")

            mined=lab.mine()
            self.assertEqual(mined["online_negative_sampling"],{
                "policy":"first_observed_append_stable",
                "cap_per_all_negative_live_impression":1,
                "eligible":1,
                "retained":1,
            })
            self.assertEqual(
                {item["mode"] for item in mined["fpminer_incremental"]},
                {"delta_updated"},
            )
            self.assertTrue(all(
                item["researched_cases"]==1
                for item in mined["fpminer_incremental"]
            ))
        finally:
            lab.close()


if __name__ == "__main__":
    unittest.main()
