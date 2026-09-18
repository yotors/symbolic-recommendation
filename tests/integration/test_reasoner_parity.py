"""Matched fixture cases through direct reconstruction and real PeTTa proofs."""

from __future__ import annotations

import unittest

from recommendation.evaluation.reasoner_parity import evaluate_pair_reasoner_parity
from recommendation.app.server import Lab, fixture


class PairReasonerParityIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = Lab(data=fixture(), config={
            "min_support": 2,
            "pair_min_support": 2,
            "max_rules": 8,
            "pair_max_rules": 8,
            "conjunctions": 2,
            "pair_conjunctions": 2,
            "pair_aggregation": "proof_margin",
            "pair_margin_transform": "linear",
            "pair_margin_power": 1.0,
        })
        cls.addClassCleanup(cls.lab.engine.close)

    def _fixture_pair_specs(self):
        events = self.lab.data["events"]
        specs = {}
        # These are ordinary ordered article comparisons built from the same
        # causal fixture snapshots that feed serving/benchmark pair features.
        for left_index, left_event in enumerate(events):
            for right_event in events[left_index + 1:]:
                left = (
                    self.lab.article(left_event["article"]),
                    self.lab.event_features(left_event),
                )
                right = (
                    self.lab.article(right_event["article"]),
                    self.lab.event_features(right_event),
                )
                for spec in (self.lab._pair_spec(left, right),
                             self.lab._pair_spec(right, left)):
                    specs.setdefault(spec[0], spec)
                if len(specs) >= 16:
                    return list(specs.values())
        return list(specs.values())

    def test_direct_table_matches_real_pettachainer_on_exact_fixture_cases(self):
        specs = self._fixture_pair_specs()
        self.assertTrue(specs)
        live_state = {
            "pid": self.lab.engine.pid,
            "case_attrs": dict(self.lab._pair_case_attrs),
            "proof_cache": dict(self.lab._pair_proof_cache),
            "channel_cache": dict(self.lab._pair_channel_proof_cache),
            "loaded_pairs": set(self.lab._loaded_pairs),
            "loaded_channels": set(self.lab._loaded_pair_channels),
        }

        report = evaluate_pair_reasoner_parity(
            self.lab, specs, max_cases=32, max_rules=32,
            absolute_tolerance=1e-8,
        )

        self.assertGreater(report["summary"]["expected_channel_activations"], 0)
        self.assertTrue(report["sampled_semantic_parity"], report["errors"])
        self.assertIn(report["status"], {"full_model_match", "sampled_match"})
        self.assertGreater(report["summary"]["compiled_channel_coverage"], 0.0)
        self.assertEqual(report["summary"]["case_match_rate"], 1.0)
        self.assertEqual(report["summary"]["expected_activation_coverage"], 1.0)
        self.assertEqual(report["summary"]["root_stv_match_rate"], 1.0)
        self.assertEqual(report["summary"]["dependency_margin_match_rate"], 1.0)
        self.assertLessEqual(
            report["summary"]["max_confidence_absolute_error"], 1e-8
        )
        self.assertTrue(any(
            case["expected_active_channels"] for case in report["cases"]
        ), "fixture cohort must contain a real active antecedent")
        self.assertTrue(any(
            not case["expected_active_channels"] for case in report["cases"]
        ), "fixture cohort must contain a real inactive antecedent")
        self.assertTrue(all(
            "grounded_relevant_facts" in case for case in report["cases"]
        ))
        self.assertTrue(all(
            case["isolated_audit_case"].startswith("parity_audit_")
            and case["isolated_audit_case"] != case["case"]
            for case in report["cases"]
        ))
        self.assertEqual(
            report["summary"]["queried_pair_signal_roots"],
            report["summary"]["cases"]
            * report["summary"]["compiled_channels"],
        )
        self.assertEqual(
            report["summary"]["host_optimized_proof_path_calls"], 0
        )
        self.assertFalse(report["summary"]["live_worker_mutated"])
        self.assertEqual(self.lab.engine.pid, live_state["pid"])
        self.assertEqual(self.lab._pair_case_attrs, live_state["case_attrs"])
        self.assertEqual(self.lab._pair_proof_cache, live_state["proof_cache"])
        self.assertEqual(
            self.lab._pair_channel_proof_cache, live_state["channel_cache"]
        )
        self.assertEqual(self.lab._loaded_pairs, live_state["loaded_pairs"])
        self.assertEqual(
            self.lab._loaded_pair_channels, live_state["loaded_channels"]
        )


if __name__ == "__main__":
    unittest.main()
