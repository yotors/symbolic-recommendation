"""Regression coverage for correlated pair-rule proof channels.

Proof-margin mode may retain several mined variants for one logical evidence
dependency.  The variants need distinct PeTTa conclusions so that PeTTa does
not revise correlated alternatives together; the host scorer must then retain
only the strongest inferred margin for the dependency.
"""

import hashlib
import re
import unittest

from recommendation.app.server import IsolatedPeTTaChainer, Lab, proof_tv
from recommendation.tests.fixtures import conditional_annotation_fixture


DEPENDENCY = "pair_mined_cluster_1"
CHANNEL_A = f"{DEPENDENCY}_v1"
CHANNEL_B = f"{DEPENDENCY}_v2"


def _variant_sources():
    variants = (
        (CHANNEL_A, "Pair_Test_Variant_A", 0.65, 0.80),
        (CHANNEL_B, "Pair_Test_Variant_B", 0.80, 0.90),
    )
    sources = []
    for index, (channel, predicate, strength, confidence) in enumerate(variants, 1):
        sources.extend((
            f'(: {channel} (Implication ({predicate} $pair "left") '
            f'(MinedPairPreference $pair "{DEPENDENCY}" "{channel}")) '
            f'(CTV (STV {strength} {confidence}) '
            f'(STV {1.0 - strength} {confidence})))',
            f'(: pair_decision_rule_{index} (Implication '
            f'(MinedPairPreference $pair "{DEPENDENCY}" "{channel}") '
            f'(PairSignal $pair "{DEPENDENCY}" "{channel}")) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
            f'(: (no_inverse pair_merge_rule_{index}) (Implication '
            f'(PairSignal $pair "{DEPENDENCY}" "{channel}") '
            '(PairWin $pair)) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
        ))
    return sources


def _factorization_contract(rule, source):
    return {
        "schema": "isolated_extensional_pair_channel_v1",
        "case_variable": "$pair",
        "premise_tv": [1.0, 1.0],
        "single_channel_producer": True,
        "dependency_id": rule["dependency_id"],
        "proof_channel_id": rule["proof_channel_id"],
        "premises": [list(item) for item in rule["premises"]],
        "rule_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _channel_root(case, channel):
    return (
        f'(: $proof (PairSignal {case} "{DEPENDENCY}" "{channel}") $tv)'
    )


def _fact(case, predicate):
    return f'(: fact_{case}_{predicate} ({predicate} {case} "left") (STV 1.0 1.0))'


def _truth_values(proofs):
    return sorted(proof_tv(proof) for proof in proofs)


class RealPeTTaVariantIsolationTest(unittest.TestCase):
    """Exercise simultaneous same-dependency variants in the real reasoner."""

    @classmethod
    def setUpClass(cls):
        cls.engine = IsolatedPeTTaChainer()
        cls.addClassCleanup(cls.engine.close)
        cls.engine.replace(_variant_sources())
        cls.engine.add_atoms_no_check([
            _fact("both", "Pair_Test_Variant_A"),
            _fact("both", "Pair_Test_Variant_B"),
            _fact("only_a", "Pair_Test_Variant_A"),
            _fact("only_b", "Pair_Test_Variant_B"),
        ])

    def query(self, roots):
        return self.engine.query_many(roots, steps=400, timeout_sec=0)

    def test_simultaneous_variants_keep_their_own_truth_values_and_lineages(self):
        roots = [
            _channel_root("both", CHANNEL_A),
            _channel_root("both", CHANNEL_B),
            _channel_root("only_a", CHANNEL_A),
            _channel_root("only_b", CHANNEL_B),
        ]
        both_a, both_b, alone_a, alone_b = self.query(roots)

        self.assertTrue(all((both_a, both_b, alone_a, alone_b)))
        self.assertEqual(_truth_values(both_a), _truth_values(alone_a))
        self.assertEqual(_truth_values(both_b), _truth_values(alone_b))
        self.assertNotEqual(_truth_values(both_a), _truth_values(both_b))

        joined_a = " ".join(both_a)
        joined_b = " ".join(both_b)
        self.assertIn(CHANNEL_A, joined_a)
        self.assertNotIn(CHANNEL_B, joined_a)
        self.assertIn(CHANNEL_B, joined_b)
        self.assertNotIn(CHANNEL_A, joined_b)

        # Query order and batching cannot change a channel's inferred STV.
        reversed_results = self.query(list(reversed(roots)))
        for expected, actual in zip(
            (both_a, both_b, alone_a, alone_b), reversed(reversed_results)
        ):
            self.assertEqual(_truth_values(expected), _truth_values(actual))

    def test_absent_sibling_stays_unproved_after_shared_pairwin_query(self):
        missing = _channel_root("only_a", CHANNEL_B)
        self.assertEqual(self.query([missing])[0], [])

        decision = self.query(["(: $proof (PairWin only_a) $tv)"])[0]
        self.assertTrue(decision)
        self.assertTrue(any("no_inverse pair_merge_rule_1" in proof
                            for proof in decision))

        # Searching the common decision root must not create a witness for the
        # absent sibling channel, and the old shared 2-argument root is unused.
        missing_again, legacy_shared = self.query([
            missing,
            f'(: $proof (PairSignal only_a "{DEPENDENCY}") $tv)',
        ])
        self.assertEqual(missing_again, [])
        self.assertEqual(legacy_shared, [])

    def test_alpha_normalized_channel_proofs_equal_direct_grounded_proofs(self):
        """Exhaust the two-rule activation lattice in the real reasoner."""
        lab = object.__new__(Lab)
        lab.config = {
            "query_batch_size": 32,
            "pair_aggregation": "proof_margin",
            "pair_chain_steps": 12,
            "pair_margin_transform": "linear",
            "pair_margin_power": 1.0,
        }
        lab.version = 11
        lab.engine = self.engine
        lab.pair_rules = [
            {
                "id": DEPENDENCY, "dependency_id": DEPENDENCY,
                "variant_id": CHANNEL_A, "proof_channel_id": CHANNEL_A,
                "premises": (("pair_test_variant_a", "left"),),
            },
            {
                "id": DEPENDENCY, "dependency_id": DEPENDENCY,
                "variant_id": CHANNEL_B, "proof_channel_id": CHANNEL_B,
                "premises": (("pair_test_variant_b", "left"),),
            },
        ]
        lab._pair_rule_sources = _variant_sources()
        for rule in lab.pair_rules:
            source = next(item for item in lab._pair_rule_sources
                          if item.startswith(f'(: {rule["variant_id"]} '))
            rule["proof_factorization"] = _factorization_contract(rule, source)
        lab._pair_proof_cache = {}; lab._pair_channel_proof_cache = {}
        lab._loaded_pair_channels = set(); lab._pair_channel_templates = {}
        lab._pair_proof_origins = {}; lab._pair_margin_cache = {}
        lab._pair_query_calls = 0; lab._pair_query_roots = 0
        lab._pair_pruned_query_roots = 0
        lab._pair_channel_activations = 0
        lab._pair_reused_channel_activations = 0
        specs = [
            ("neither", {}),
            ("only_a", {"pair_test_variant_a": "left"}),
            ("only_b", {"pair_test_variant_b": "left"}),
            ("both", {"pair_test_variant_a": "left",
                      "pair_test_variant_b": "left"}),
        ]

        factored, calls = lab._proofs_for_pair_specs(specs)

        direct = {
            "neither": [],
            "only_a": self.query([_channel_root("only_a", CHANNEL_A)])[0],
            "only_b": self.query([_channel_root("only_b", CHANNEL_B)])[0],
            "both": sum(self.query([
                _channel_root("both", CHANNEL_A),
                _channel_root("both", CHANNEL_B),
            ]), []),
        }
        self.assertEqual(calls, 1)
        self.assertEqual(lab._pair_query_roots, 2)
        self.assertEqual(lab._pair_channel_activations, 4)
        self.assertEqual(lab._pair_reused_channel_activations, 2)
        self.assertEqual(len(lab._pair_channel_templates), 2)
        for case in direct:
            self.assertEqual(_truth_values(factored[case]),
                             _truth_values(direct[case]))
        self.assertTrue(all(
            "pair_channel_" in proof
            for case in ("only_a", "only_b", "both")
            for proof in factored[case]
        ))

    def test_conjunction_template_equals_direct_conjunction_proof(self):
        dependency = "pair_mined_cluster_9"
        channel = f"{dependency}_v1"
        variant = (
            f'(: {channel} (Implication (And (Pair_A $pair "left") '
            f'(Pair_Context $pair "same")) (MinedPairPreference $pair '
            f'"{dependency}" "{channel}")) '
            '(CTV (STV 0.77 0.86) (STV 0.23 0.86)))'
        )
        decision = (
            '(: pair_decision_rule_9 (Implication '
            f'(MinedPairPreference $pair "{dependency}" "{channel}") '
            f'(PairSignal $pair "{dependency}" "{channel}")) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
        )
        bridge = (
            '(: (no_inverse pair_merge_rule_9) (Implication '
            f'(PairSignal $pair "{dependency}" "{channel}") '
            '(PairWin $pair)) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
        )
        engine = IsolatedPeTTaChainer()
        self.addCleanup(engine.close)
        engine.replace([variant, decision, bridge])
        engine.add_atoms_no_check([
            '(: direct_a (Pair_A direct "left") (STV 1.0 1.0))',
            '(: direct_context (Pair_Context direct "same") (STV 1.0 1.0))',
        ])
        rule = {
            "id": dependency, "dependency_id": dependency,
            "variant_id": channel, "proof_channel_id": channel,
            "premises": (("pair_a", "left"), ("pair_context", "same")),
        }
        rule["proof_factorization"] = _factorization_contract(rule, variant)
        lab = object.__new__(Lab)
        lab.config = {"query_batch_size": 8, "pair_aggregation": "proof_margin",
                      "pair_chain_steps": 12, "pair_margin_transform": "linear",
                      "pair_margin_power": 1.0}
        lab.version = 12; lab.engine = engine; lab.pair_rules = [rule]
        lab._pair_rule_sources = [variant, decision, bridge]
        lab._pair_proof_cache = {}; lab._pair_channel_proof_cache = {}
        lab._loaded_pair_channels = set(); lab._pair_channel_templates = {}
        lab._pair_proof_origins = {}; lab._pair_margin_cache = {}
        lab._pair_query_calls = 0; lab._pair_query_roots = 0
        lab._pair_pruned_query_roots = 0
        lab._pair_channel_activations = 0
        lab._pair_reused_channel_activations = 0

        factored, _calls = lab._proofs_for_pair_specs([
            ("logical_case", {"pair_a": "left", "pair_context": "same"}),
        ])
        direct = engine.query_many([
            f'(: $proof (PairSignal direct "{dependency}" "{channel}") $tv)'
        ], steps=12, timeout_sec=0)[0]

        self.assertTrue(direct)
        self.assertEqual(_truth_values(factored["logical_case"]),
                         _truth_values(direct))
        audit = next(iter(lab._pair_channel_templates.values()))
        self.assertEqual(audit["facts"], [
            ["pair_a", "left"], ["pair_context", "same"],
        ])


class _RecordingEngine:
    def __init__(self):
        self.calls = []
        self.added = []

    def add_atoms_no_check(self, atoms, timeout_sec=None):
        self.added.extend(atoms)

    def query_many(self, queries, *, steps, timeout_sec):
        self.calls.append((tuple(queries), steps, timeout_sec))
        return [
            [f'(by recorded_source_{index} {query} (STV 0.7 0.8))']
            for index, query in enumerate(queries)
        ]


class _FailingEngine(_RecordingEngine):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on
        self.query_count = 0

    def add_atoms_no_check(self, atoms, timeout_sec=None):
        if self.fail_on == "add":
            raise RuntimeError("injected add failure")
        return super().add_atoms_no_check(atoms,timeout_sec=timeout_sec)

    def query_many(self, queries, *, steps, timeout_sec):
        self.query_count += 1
        if self.fail_on == "query" or (
                self.fail_on == "second_query" and self.query_count == 2):
            raise RuntimeError("injected query failure")
        return super().query_many(
            queries, steps=steps, timeout_sec=timeout_sec
        )


def _bare_lab(aggregation="proof_margin"):
    lab = object.__new__(Lab)
    lab.config = {
        "query_batch_size": 32,
        "pair_aggregation": aggregation,
        "pair_chain_steps": 12,
        "pair_margin_transform": "linear",
        "pair_margin_power": 1.0,
    }
    lab.version = 7
    lab.pair_rules = [
        {
            "id": DEPENDENCY,
            "dependency_id": DEPENDENCY,
            "variant_id": CHANNEL_A,
            "proof_channel_id": (
                CHANNEL_A if aggregation == "proof_margin" else DEPENDENCY
            ),
            "premises": (("pair_test_variant_a", "left"),),
        },
        {
            "id": DEPENDENCY,
            "dependency_id": DEPENDENCY,
            "variant_id": CHANNEL_B,
            "proof_channel_id": (
                CHANNEL_B if aggregation == "proof_margin" else DEPENDENCY
            ),
            "premises": (("pair_test_variant_b", "left"),),
        },
    ]
    lab._pair_rule_sources = _variant_sources()
    if aggregation == "proof_margin":
        for rule in lab.pair_rules:
            source = next(item for item in lab._pair_rule_sources
                          if item.startswith(f'(: {rule["variant_id"]} '))
            rule["proof_factorization"] = _factorization_contract(rule, source)
    lab._pair_proof_cache = {}
    lab._pair_margin_cache = {}
    lab._pair_query_calls = 0
    lab._pair_query_roots = 0
    lab._pair_pruned_query_roots = 0
    lab.engine = _RecordingEngine()
    return lab


class HostVariantAggregationTest(unittest.TestCase):
    def test_proof_margin_queries_each_channel_and_cache_key_tracks_roots(self):
        lab = _bare_lab()
        attrs = {"pair_test_variant_a": "left",
                 "pair_test_variant_b": "left"}
        specs = [("case_1", attrs), ("case_1", attrs)]

        proof_map, calls = lab._proofs_for_pair_specs(specs)

        self.assertEqual(calls, 1)
        self.assertEqual(set(proof_map), {"case_1"})
        queries, steps, timeout = lab.engine.calls[0]
        template_a = Lab.pair_channel_case(
            DEPENDENCY, CHANNEL_A, (("pair_test_variant_a", "left"),)
        )
        template_b = Lab.pair_channel_case(
            DEPENDENCY, CHANNEL_B, (("pair_test_variant_b", "left"),)
        )
        self.assertEqual(queries, (
            _channel_root(template_a, CHANNEL_A),
            _channel_root(template_b, CHANNEL_B),
        ))
        self.assertEqual(steps, 24)
        self.assertEqual(timeout, 30.0)
        self.assertFalse(any("PairWin" in query for query in queries))
        self.assertEqual(lab._pair_query_roots, 2)
        self.assertEqual(lab._pair_pruned_query_roots, 0)

        _proof_map, cached_calls = lab._proofs_for_pair_specs(specs)
        self.assertEqual(cached_calls, 0)
        self.assertEqual(len(lab.engine.calls), 1)

        # Proof roots are part of the cache identity. Adding a new variant in
        # the same version cannot accidentally reuse an older channel set.
        third = f"{DEPENDENCY}_v3"
        lab.pair_rules.append({
            "id": DEPENDENCY,
            "dependency_id": DEPENDENCY,
            "variant_id": third,
            "proof_channel_id": third,
            "premises": (("pair_test_variant_c", "left"),),
        })
        third_source = (
            f'(: {third} (Implication (Pair_Test_Variant_C $pair "left") '
            f'(MinedPairPreference $pair "{DEPENDENCY}" "{third}")) '
            '(CTV (STV 0.7 0.8) (STV 0.3 0.8)))'
        )
        lab._pair_rule_sources.append(third_source)
        lab.pair_rules[-1]["proof_factorization"] = _factorization_contract(
            lab.pair_rules[-1], third_source
        )
        changed_attrs = {**attrs, "pair_test_variant_c": "left"}
        _proof_map, changed_calls = lab._proofs_for_pair_specs([
            ("case_2", changed_attrs),
        ])
        self.assertEqual(changed_calls, 1)
        self.assertEqual(len(lab.engine.calls[-1][0]), 3)
        self.assertIn(third, lab.engine.calls[-1][0][-1])

    def test_known_nonmatching_channels_are_not_queried(self):
        lab = _bare_lab()
        lab.pair_rules[0]["premises"] = (("pair_view_a", "left"),)
        lab.pair_rules[1]["premises"] = (("pair_view_b", "left"),)
        for rule in lab.pair_rules:
            predicate = rule["premises"][0][0].title()
            source = (
                f'(: {rule["variant_id"]} (Implication '
                f'({predicate} $pair "left") (MinedPairPreference $pair '
                f'"{DEPENDENCY}" "{rule["variant_id"]}")) '
                '(CTV (STV 0.7 0.8) (STV 0.3 0.8)))'
            )
            source_index = next(
                index for index, item in enumerate(lab._pair_rule_sources)
                if item.startswith(f'(: {rule["variant_id"]} ')
            )
            lab._pair_rule_sources[source_index] = source
            rule["proof_factorization"] = _factorization_contract(rule, source)
        lab._pair_case_attrs = {
            "only_a": {"pair_view_a": "left", "pair_view_b": "right"},
        }

        proofs, calls = lab._proofs_for_pair_specs([("only_a", {})])

        self.assertEqual(calls, 1)
        self.assertTrue(proofs["only_a"])
        queries, _steps, _timeout = lab.engine.calls[0]
        template = Lab.pair_channel_case(
            DEPENDENCY, CHANNEL_A, (("pair_view_a", "left"),)
        )
        self.assertEqual(queries, (_channel_root(template, CHANNEL_A),))
        self.assertEqual(lab._pair_query_roots, 1)
        self.assertEqual(lab._pair_pruned_query_roots, 1)

    def test_channel_templates_are_reused_across_joint_activation_cases(self):
        lab = _bare_lab()
        specs = [
            ("only_a", {"pair_test_variant_a": "left"}),
            ("only_b", {"pair_test_variant_b": "left"}),
            ("both", {"pair_test_variant_a": "left",
                      "pair_test_variant_b": "left"}),
            ("neither", {}),
        ]

        proofs, calls = lab._proofs_for_pair_specs(specs)

        self.assertEqual(calls, 1)
        self.assertEqual(len(lab.engine.calls[0][0]), 2)
        self.assertEqual(lab._pair_query_roots, 2)
        self.assertEqual(lab._pair_channel_activations, 4)
        self.assertEqual(lab._pair_reused_channel_activations, 2)
        self.assertEqual(len(proofs["only_a"]), 1)
        self.assertEqual(len(proofs["only_b"]), 1)
        self.assertEqual(len(proofs["both"]), 2)
        self.assertEqual(proofs["neither"], [])
        self.assertEqual(len(lab.engine.added), 2)
        self.assertTrue(all("pair_channel_" in fact for fact in lab.engine.added))

    def test_failed_fact_install_does_not_publish_any_cache_state(self):
        lab = _bare_lab()
        lab.engine = _FailingEngine("add")
        attrs = {"pair_test_variant_a": "left"}

        with self.assertRaisesRegex(RuntimeError, "injected add failure"):
            lab._proofs_for_pair_specs([("case", attrs)])

        self.assertEqual(lab._pair_proof_cache, {})
        self.assertEqual(lab._pair_channel_proof_cache, {})
        self.assertEqual(lab._loaded_pair_channels, set())
        self.assertEqual(lab._pair_channel_templates, {})
        self.assertEqual(lab._pair_query_roots, 0)

    def test_failed_later_query_does_not_publish_partial_cache_state(self):
        lab = _bare_lab()
        lab.config["query_batch_size"] = 1
        lab.engine = _FailingEngine("second_query")
        attrs = {"pair_test_variant_a": "left",
                 "pair_test_variant_b": "left"}

        with self.assertRaisesRegex(RuntimeError, "injected query failure"):
            lab._proofs_for_pair_specs([("case", attrs)])

        self.assertEqual(lab._pair_proof_cache, {})
        self.assertEqual(lab._pair_channel_proof_cache, {})
        self.assertEqual(lab._loaded_pair_channels, set())
        self.assertEqual(lab._pair_channel_templates, {})
        self.assertEqual(lab._pair_query_roots, 0)

    def test_untrusted_or_mutated_rule_shape_is_rejected_before_query(self):
        lab = _bare_lab()
        del lab.pair_rules[0]["proof_factorization"]

        with self.assertRaisesRegex(RuntimeError, "not safely factorable"):
            lab._proofs_for_pair_specs([
                ("case", {"pair_test_variant_a": "left"}),
            ])

        self.assertEqual(lab.engine.calls, [])
        self.assertEqual(lab.engine.added, [])

    def test_only_maximum_variant_margin_counts_once_per_dependency(self):
        lab = _bare_lab()
        other_dependency = "pair_mined_cluster_2"
        proofs = [
            f'(by {CHANNEL_A} (: p (PairSignal c "{DEPENDENCY}" '
            f'"{CHANNEL_A}") (STV 0.65 0.8)))',
            f'(by {CHANNEL_B} (: p (PairSignal c "{DEPENDENCY}" '
            f'"{CHANNEL_B}") (STV 0.8 0.9)))',
            # A duplicate view of the weaker variant is correlated evidence.
            f'(by {CHANNEL_A} (: p (PairSignal c "{DEPENDENCY}" '
            f'"{CHANNEL_A}") (STV 0.65 0.8)))',
            f'(by {other_dependency}_v1 (: p (PairSignal c '
            f'"{other_dependency}" "{other_dependency}_v1") (STV 0.7 0.6)))',
        ]

        margins = lab._proof_dependency_margins("c", proofs)

        self.assertEqual(set(margins), {DEPENDENCY, other_dependency})
        self.assertAlmostEqual(margins[DEPENDENCY], 0.54)
        self.assertAlmostEqual(margins[other_dependency], 0.24)
        self.assertAlmostEqual(lab._proof_vote_margin("c", proofs), 0.78)
        self.assertEqual(
            margins,
            lab._proof_dependency_margins("c_reversed", list(reversed(proofs))),
        )

    def test_posterior_mode_retains_one_shared_pairwin_query(self):
        lab = _bare_lab(aggregation="posterior")

        proof_map, calls = lab._proofs_for_pair_specs([
            ("case_2", {}), ("case_2", {}),
        ])

        self.assertEqual(calls, 1)
        self.assertTrue(proof_map["case_2"])
        queries, steps, timeout = lab.engine.calls[0]
        self.assertEqual(queries, ("(: $proof (PairWin case_2) $tv)",))
        self.assertEqual(steps, 12)
        self.assertEqual(timeout, 30.0)
        self.assertFalse(any("PairSignal" in query for query in queries))
        self.assertEqual(
            {rule["proof_channel_id"] for rule in lab.pair_rules},
            {DEPENDENCY},
            "posterior compatibility deliberately retains the shared target",
        )


class GeneratedChannelIdentityTest(unittest.TestCase):
    """IDs and serialized source order must repeat across the same mining run."""

    @classmethod
    def setUpClass(cls):
        cls.lab = Lab(conditional_annotation_fixture(), config={
            "miner_strategy": "conditional_llm",
            "pair_feature_profile": "llm_conditional",
            "pair_conjunctions": 3,
            "min_support": 2,
            "pair_min_support": 2,
            "max_rules": 8,
            "pair_max_rules": 8,
            "pair_negative_ratio": 0,
        })
        cls.addClassCleanup(cls.lab.engine.close)

    @staticmethod
    def _rule_identities(lab):
        return tuple(
            (
                tuple(rule["premises"]),
                rule["dependency_id"],
                rule["variant_id"],
                rule["proof_channel_id"],
            )
            for rule in lab.pair_rules
        )

    def test_correlated_generated_variants_have_deterministic_channel_sources(self):
        grouped = {}
        for rule in self.lab.pair_rules:
            grouped.setdefault(rule["dependency_id"], []).append(rule)
        correlated = next(
            (rules for rules in grouped.values() if len(rules) > 1), None
        )
        self.assertIsNotNone(correlated, "fixture must retain correlated variants")

        channels = {rule["proof_channel_id"] for rule in self.lab.pair_rules}
        self.assertEqual(len(channels), len(self.lab.pair_rules))
        self.assertEqual(self.lab.last_pair_mining["proof_channels"], len(channels))
        self.assertEqual(
            self.lab.last_pair_mining["proof_channel_mode"], "variant_isolated"
        )
        for rule in self.lab.pair_rules:
            self.assertEqual(rule["proof_channel_id"], rule["variant_id"])
            rule_sources = [
                source for source in self.lab._pair_rule_sources
                if source.startswith(f'(: {rule["variant_id"]} ')
            ]
            self.assertEqual(len(rule_sources), 1)
            self.assertIn(
                f'(MinedPairPreference $pair "{rule["dependency_id"]}" '
                f'"{rule["proof_channel_id"]}")',
                rule_sources[0],
            )

        decisions = [source for source in self.lab._pair_rule_sources
                     if re.match(r"^\(: pair_decision_rule_\d+ ", source)]
        bridges = [source for source in self.lab._pair_rule_sources
                   if re.match(r"^\(: \(no_inverse pair_merge_rule_\d+\) ", source)]
        self.assertEqual(len(decisions), len(channels))
        self.assertEqual(len(bridges), len(channels))

        identities = self._rule_identities(self.lab)
        sources = tuple(self.lab._pair_rule_sources)
        self.lab.mine()
        self.assertEqual(self._rule_identities(self.lab), identities)
        self.assertEqual(tuple(self.lab._pair_rule_sources), sources)

    def test_serving_feature_projection_equals_full_feature_construction(self):
        events = self.lab.data["events"][:4]
        for left_event in events:
            for right_event in events:
                if left_event is right_event:
                    continue
                left_article = self.lab.article(left_event["article"])
                right_article = self.lab.article(right_event["article"])
                left_attrs = self.lab.event_features(left_event)
                right_attrs = self.lab.event_features(right_event)
                full = self.lab._bounded_pair_features(self.lab._pair_features(
                    left_attrs, right_attrs, left_article, right_article
                ))
                projected = self.lab._bounded_pair_features(
                    self.lab._pair_features(
                        left_attrs, right_attrs, left_article, right_article,
                        needed=self.lab._pair_feature_vocabulary,
                    )
                )
                self.assertEqual(projected, full)


if __name__ == "__main__":
    unittest.main()
