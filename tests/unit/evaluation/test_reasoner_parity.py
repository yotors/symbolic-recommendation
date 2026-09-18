"""Unit contracts for the bounded pair-reasoner parity evaluator."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from unittest.mock import patch

import recommendation.evaluation.reasoner_parity as parity
from recommendation.evaluation.reasoner_parity import (
    INTERPRETATION,
    UnsupportedParityTopology,
    evaluate_pair_reasoner_parity,
)


DEPENDENCY = "pair_mined_cluster_1"
CHANNEL = f"{DEPENDENCY}_v1"
PREMISES = (("pair_topic", "left"),)


def _sources():
    variant = (
        f'(: {CHANNEL} (Implication (Pair_Topic $pair "left") '
        f'(MinedPairPreference $pair "{DEPENDENCY}" "{CHANNEL}")) '
        '(CTV (STV 0.8 0.7) (STV 0.2 0.7)))'
    )
    return [
        variant,
        f'(: pair_decision_rule_1 (Implication '
        f'(MinedPairPreference $pair "{DEPENDENCY}" "{CHANNEL}") '
        f'(PairSignal $pair "{DEPENDENCY}" "{CHANNEL}")) '
        '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
        f'(: (no_inverse pair_merge_rule_1) (Implication '
        f'(PairSignal $pair "{DEPENDENCY}" "{CHANNEL}") (PairWin $pair)) '
        '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
    ]


class _FakeLab:
    def __init__(self, *, confidence_offset=0.0, aggregation="proof_margin"):
        self.config = {
            "pair_aggregation": aggregation,
            "pair_margin_transform": "linear",
            "pair_margin_power": 1.0,
        }
        self._pair_rule_sources = _sources()
        source_hash = hashlib.sha256(
            self._pair_rule_sources[0].encode("utf-8")
        ).hexdigest()
        self.pair_rules = [{
            "id": DEPENDENCY,
            "dependency_id": DEPENDENCY,
            "variant_id": CHANNEL,
            "proof_channel_id": CHANNEL,
            "premises": PREMISES,
            "strength": 0.8,
            "confidence": 0.7,
            "negative_strength": 0.2,
            "negative_confidence": 0.7,
            "proof_factorization": {
                "schema": "isolated_extensional_pair_channel_v1",
                "case_variable": "$pair",
                "premise_tv": [1.0, 1.0],
                "single_channel_producer": True,
                "dependency_id": DEPENDENCY,
                "proof_channel_id": CHANNEL,
                "premises": [list(item) for item in PREMISES],
                "rule_source_sha256": source_hash,
            },
        }]
        self._pair_case_attrs = {}
        self.confidence_offset = confidence_offset
        self.mined_rules = [{
            "id": "mined_1",
            "point_variant_id": "point_variant_1",
            "point_decision_id": "point_decision_rule_1",
            "point_proof_channel_id": "point_mined_1",
            "premises": (("topic", "news"),),
            "target": "click",
            "strength": 0.75,
            "confidence": 0.6,
            "negative_strength": 0.25,
            "negative_confidence": 0.6,
        }]
        self._point_rule_sources = [
            '(: mined_1 (Implication (Topic $case "news") '
            '(Engagement $case "click")) '
            '(CTV (STV 0.75 0.6) (STV 0.25 0.6)))'
        ]
        point_variant = (
            '(: point_variant_1 (Implication (Topic $case "news") '
            '(MinedPointPreference $case "mined_1" "point_mined_1")) '
            '(CTV (STV 0.75 0.6) (STV 0.25 0.6)))'
        )
        point_decision = (
            '(: point_decision_rule_1 (Implication '
            '(MinedPointPreference $case "mined_1" "point_mined_1") '
            '(PointSignal $case "mined_1" "point_mined_1")) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
        )
        self._point_channel_sources = [point_variant, point_decision]
        self.mined_rules[0]["point_proof_factorization"] = {
            "schema": "isolated_extensional_point_channel_v1",
            "case_variable": "$case",
            "premise_tv": [1.0, 1.0],
            "single_channel_producer": True,
            "rule_id": "mined_1",
            "variant_id": "point_variant_1",
            "decision_id": "point_decision_rule_1",
            "proof_channel_id": "point_mined_1",
            "premises": [["topic", "news"]],
            "rule_source_sha256": hashlib.sha256(
                point_variant.encode("utf-8")
            ).hexdigest(),
            "decision_source_sha256": hashlib.sha256(
                point_decision.encode("utf-8")
            ).hexdigest(),
        }

    def _ensure_pair_specs(self, specs):
        raise AssertionError("parity must not call the host grounding path")

    def _proofs_for_pair_specs(self, specs):
        raise AssertionError("parity must not call the host proof/cache path")


class _FakeIsolatedEngine:
    """Small process-boundary stand-in; activation comes from grounded facts."""

    def __init__(self, *, confidence_offset=0.0, force_inactive=False):
        self.confidence_offset = confidence_offset
        self.force_inactive = force_inactive
        self.sources = []
        self.facts = {}
        self.closed = False

    def replace(self, sources):
        self.sources = list(sources)

    def add_atoms_no_check(self, atoms, timeout_sec=None):
        for atom in atoms:
            match = re.search(
                r'\(([A-Z][A-Za-z0-9_]*) (parity_audit_[A-Za-z0-9_]+) '
                r'("(?:[^"\\]|\\.)*")\)',
                atom,
            )
            if match:
                predicate, case, encoded = match.groups()
                self.facts.setdefault(case, {})[predicate] = json.loads(encoded)

    def query_many(self, queries, *, steps, timeout_sec):
        results = []
        for query in queries:
            root = re.search(
                r'\(PairSignal (parity_audit_[A-Za-z0-9_]+) '
                r'("(?:[^"\\]|\\.)*") '
                r'("(?:[^"\\]|\\.)*")\)',
                query,
            )
            if root is None:
                results.append([])
                continue
            case, dependency_encoded, channel_encoded = root.groups()
            dependency = json.loads(dependency_encoded)
            channel = json.loads(channel_encoded)
            conclusion = (
                f'(MinedPairPreference $pair {dependency_encoded} '
                f'{channel_encoded})'
            )
            source = next(
                item for item in self.sources
                if item.startswith(f'(: {channel} ') and conclusion in item
            )
            premise_source = source.split(conclusion, 1)[0]
            premises = [
                (predicate, json.loads(encoded))
                for predicate, encoded in re.findall(
                    r'\(([A-Z][A-Za-z0-9_]*) \$pair '
                    r'("(?:[^"\\]|\\.)*")\)',
                    premise_source,
                )
            ]
            active = all(
                self.facts.get(case, {}).get(predicate) == value
                for predicate, value in premises
            )
            if not active and not self.force_inactive:
                results.append([])
                continue
            ctv = parity._CTV_AT_END.search(source)
            assert ctv is not None
            values = tuple(map(float, ctv.groups()))
            antecedent = (1.0, 1.0)
            for _premise in premises[1:]:
                antecedent = parity._and_formula(antecedent, (1.0, 1.0))
            mined = parity._ctv_modus_ponens(
                antecedent, (values[0], values[1]), (values[2], values[3])
            )
            truth_value = parity._ctv_modus_ponens(
                mined, (1.0, 1.0), (0.0, 1.0)
            )
            truth_value = (
                truth_value[0],
                max(0.0, truth_value[1] + self.confidence_offset),
            )
            results.append([
                f'(by {channel} (PairSignal {case} {dependency_encoded} '
                f'{channel_encoded}) (STV {truth_value[0]} '
                f'{truth_value[1]}))'
            ])
        return results

    def close(self):
        self.closed = True


def _evaluate(lab, specs, *, engine=None, **kwargs):
    isolated = engine or _FakeIsolatedEngine(
        confidence_offset=lab.confidence_offset
    )
    with patch.object(
        parity, "_create_isolated_audit_engine", return_value=isolated
    ) as creator:
        try:
            return evaluate_pair_reasoner_parity(lab, specs, **kwargs)
        finally:
            if creator.called:
                assert isolated.closed


class ReasonerParityUnitTest(unittest.TestCase):
    def test_point_reconstructor_validates_sources_and_is_lazy(self):
        lab=_FakeLab()
        reconstructor=parity.DirectPointReconstructor(lab)

        active=reconstructor.reconstruct({"topic":"news"})
        inactive=reconstructor.reconstruct({"topic":"sports"})
        proofs=reconstructor.proof_rows("candidate_1",{"topic":"news"})

        self.assertEqual(len(active),1)
        self.assertEqual(active[0]["rule_id"],"mined_1")
        self.assertEqual(inactive,[])
        self.assertIn("(by point_variant_1)",proofs[0])
        self.assertIn('(PointSignal candidate_1 "mined_1" "point_mined_1")',proofs[0])

        lab._point_channel_sources[0]=lab._point_channel_sources[0].replace(
            "0.75","0.70"
        )
        with self.assertRaisesRegex(
                UnsupportedParityTopology,"metadata disagrees"):
            parity.DirectPointReconstructor(lab)

    def test_compiled_direct_reconstructor_is_reusable_and_lazy(self):
        reconstructor = parity.DirectPairReconstructor(_FakeLab())

        active = reconstructor.reconstruct({"pair_topic": "left"})
        inactive = reconstructor.reconstruct({"pair_topic": "right"})

        self.assertEqual(len(active["active_channels"]), 1)
        self.assertEqual(
            active["active_channels"][0]["proof_channel_id"], CHANNEL
        )
        self.assertEqual(
            active["selected_dependencies"][DEPENDENCY]["proof_channel_id"],
            CHANNEL,
        )
        self.assertEqual(inactive["active_channels"], [])
        self.assertEqual(inactive["selected_dependencies"], {})

    def test_reports_exact_activation_stv_and_margin_parity(self):
        report = _evaluate(
            _FakeLab(),
            [("active", {"pair_topic": "left"}),
             ("inactive", {"pair_topic": "right"})],
        )

        self.assertTrue(report["semantic_parity"])
        self.assertTrue(report["full_model_semantic_parity"])
        self.assertTrue(report["sampled_semantic_parity"])
        self.assertEqual(report["status"], "full_model_match")
        self.assertEqual(report["parity_scope"], "full_model")
        self.assertEqual(report["summary"]["compiled_channel_coverage"], 1.0)
        self.assertEqual(report["interpretation"], INTERPRETATION)
        self.assertIn("not an accuracy causal ablation", report["interpretation"])
        self.assertEqual(report["summary"]["exact_case_matches"], 2)
        self.assertEqual(report["summary"]["expected_activation_coverage"], 1.0)
        self.assertEqual(report["summary"]["root_stv_match_rate"], 1.0)
        self.assertEqual(report["summary"]["dependency_margin_match_rate"], 1.0)
        self.assertEqual(
            report["summary"]["worker_kind"],
            "disposable_isolated_pettachainer",
        )
        self.assertFalse(report["summary"]["live_worker_mutated"])
        self.assertFalse(report["summary"]["live_proof_cache_mutated"])
        self.assertEqual(
            report["summary"]["host_optimized_proof_path_calls"], 0
        )
        self.assertEqual(report["summary"]["queried_pair_signal_roots"], 2)
        self.assertEqual(report["errors"], [])

    def test_deliberately_wrong_host_matcher_is_never_consulted(self):
        # _FakeLab's host grounding/proof methods both raise.  A real match is
        # still obtained because the audit grounds facts and queries all roots
        # in its own worker.
        report = _evaluate(
            _FakeLab(),
            [("active", {"pair_topic": "left"}),
             ("inactive", {"pair_topic": "right"})],
        )

        self.assertTrue(report["sampled_semantic_parity"])
        self.assertEqual(
            report["summary"]["host_optimized_proof_path_calls"], 0
        )

    def test_isolated_activation_mismatch_is_detected(self):
        report = _evaluate(
            _FakeLab(),
            [("active", {"pair_topic": "left"}),
             ("inactive", {"pair_topic": "right"})],
            engine=_FakeIsolatedEngine(force_inactive=True),
        )

        self.assertFalse(report["sampled_semantic_parity"])
        self.assertEqual(report["summary"]["unexpected_channel_activations"], 1)
        self.assertIn(
            "unexpected_proof", {error["kind"] for error in report["errors"]}
        )

    def test_numerical_mismatch_is_reported_without_becoming_an_auc_claim(self):
        report = _evaluate(
            _FakeLab(confidence_offset=-0.1),
            [("active", {"pair_topic": "left"})],
        )

        self.assertFalse(report["semantic_parity"])
        self.assertGreater(report["summary"]["max_confidence_absolute_error"], 0.09)
        self.assertGreater(report["summary"]["max_dependency_margin_absolute_error"], 0.05)
        self.assertEqual(
            {error["kind"] for error in report["errors"]},
            {"root_stv_mismatch", "dependency_margin_mismatch"},
        )
        self.assertNotIn("auc", report["summary"])

    def test_partial_channel_coverage_is_only_a_sampled_match(self):
        lab = _FakeLab()
        dependency = "pair_mined_cluster_2"
        channel = f"{dependency}_v1"
        variant = (
            f'(: {channel} (Implication (Pair_Topic $pair "right") '
            f'(MinedPairPreference $pair "{dependency}" "{channel}")) '
            '(CTV (STV 0.7 0.6) (STV 0.3 0.6)))'
        )
        lab._pair_rule_sources.extend([
            variant,
            f'(: pair_decision_rule_2 (Implication '
            f'(MinedPairPreference $pair "{dependency}" "{channel}") '
            f'(PairSignal $pair "{dependency}" "{channel}")) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
            f'(: (no_inverse pair_merge_rule_2) (Implication '
            f'(PairSignal $pair "{dependency}" "{channel}") (PairWin $pair)) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
        ])
        lab.pair_rules.append({
            "id": dependency,
            "dependency_id": dependency,
            "variant_id": channel,
            "proof_channel_id": channel,
            "premises": (("pair_topic", "right"),),
            "strength": 0.7,
            "confidence": 0.6,
            "negative_strength": 0.3,
            "negative_confidence": 0.6,
            "proof_factorization": {
                "schema": "isolated_extensional_pair_channel_v1",
                "case_variable": "$pair",
                "premise_tv": [1.0, 1.0],
                "single_channel_producer": True,
                "dependency_id": dependency,
                "proof_channel_id": channel,
                "premises": [["pair_topic", "right"]],
                "rule_source_sha256": hashlib.sha256(
                    variant.encode("utf-8")
                ).hexdigest(),
            },
        })

        report = _evaluate(
            lab, [("active", {"pair_topic": "left"})]
        )

        self.assertFalse(report["semantic_parity"])
        self.assertFalse(report["full_model_semantic_parity"])
        self.assertTrue(report["sampled_semantic_parity"])
        self.assertEqual(report["status"], "sampled_match")
        self.assertEqual(report["parity_scope"], "sampled_channels")
        self.assertEqual(report["summary"]["compiled_channels"], 2)
        self.assertEqual(report["summary"]["exercised_channels"], 1)
        self.assertEqual(report["summary"]["unexercised_channels"], [channel])
        self.assertEqual(report["summary"]["compiled_channel_coverage"], 0.5)

    def test_unsupported_posterior_topology_fails_before_query(self):
        lab = _FakeLab(aggregation="posterior")
        with self.assertRaisesRegex(
            UnsupportedParityTopology, "only isolated proof_margin"
        ):
            _evaluate(
                lab, [("active", {"pair_topic": "left"})]
            )

    def test_tampered_compiled_source_fails_closed(self):
        lab = _FakeLab()
        lab._pair_rule_sources[0] = lab._pair_rule_sources[0].replace("0.8", "0.9")
        with self.assertRaisesRegex(UnsupportedParityTopology, "source hash"):
            _evaluate(
                lab, [("active", {"pair_topic": "left"})]
            )

    def test_case_bound_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "case count exceeds bound"):
            _evaluate(
                _FakeLab(),
                [("case_1", {}), ("case_2", {})],
                max_cases=1,
            )

    def test_case_by_channel_query_bound_fails_closed(self):
        with patch.object(parity,"_HARD_MAX_QUERY_ROOTS",1):
            with self.assertRaisesRegex(
                    UnsupportedParityTopology,"case x channel root count"):
                _evaluate(
                    _FakeLab(),
                    [("active", {"pair_topic": "left"}),
                     ("inactive", {"pair_topic": "right"})],
                )

    def test_all_inactive_cohort_cannot_report_vacuous_parity(self):
        with self.assertRaisesRegex(ValueError, "neither the direct reconstructor"):
            _evaluate(
                _FakeLab(), [("inactive", {"pair_topic": "right"})]
            )


if __name__ == "__main__":
    unittest.main()
