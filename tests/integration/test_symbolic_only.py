"""Symbolic-only boundary and real miner -> PeTTa lexical integration tests."""

import copy
import io
import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from recommendation.app.server import (
    Handler, Lab, PAIR_FEATURE_PROFILES, PeTTaChainerConfigurationError,
)
from recommendation.features.lexical_workspace import LEXICAL_FEATURES
from recommendation.tests.fixtures import lexical_fixture


class SymbolicOnlyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_data = lexical_fixture()
        with patch("recommendation.app.server.load_text_embedding_sidecar",
                   side_effect=AssertionError("Neural sidecar was loaded")):
            cls.lab = Lab(data=cls.source_data, symbolic_only=True, config={
                "pair_feature_profile": "symbolic_lexical",
                "pair_family_fusion": "symbolic_balanced",
                "min_support": 2, "pair_min_support": 2,
                "max_rules": 8, "pair_max_rules": 8,
                "conjunctions": 2, "pair_conjunctions": 2,
            })

    @classmethod
    def tearDownClass(cls):
        cls.lab.engine.close()

    def test_constructor_removes_vector_maps_and_does_not_load_sidecar(self):
        self.assertTrue(self.lab.symbolic_only)
        self.assertFalse(self.lab._article_entity_vectors)
        self.assertFalse(self.lab._article_text_vectors)
        self.assertIsNone(self.lab._text_embedding_metadata)
        # Input ownership is retained: stripping must not silently change the
        # reusable source dataset from which a separate baseline is built.
        self.assertTrue(self.source_data["article_entity_vectors"])
        self.assertTrue(self.source_data["article_text_vectors"])

    def test_config_rejects_every_legacy_profile_with_neural_or_entity_evidence(self):
        before = copy.deepcopy(self.lab.config)
        for profile, predicates in PAIR_FEATURE_PROFILES.items():
            if any("semantic" in p or "entity" in p for p in predicates):
                with self.subTest(profile=profile):
                    with self.assertRaisesRegex(ValueError, "symbolic-only"):
                        self.lab.configure({"pair_feature_profile": profile})
                    self.assertEqual(self.lab.config, before)
        with self.assertRaisesRegex(ValueError, "symbolic-only"):
            self.lab.configure({"feature_profile": "accuracy_detail"})
        self.assertEqual(self.lab.config, before)

    def test_portable_symbolic_profiles_are_configurable(self):
        previous = self.lab.config["pair_feature_profile"]
        try:
            for suffix in ("baseline", "precision", "lexical", "coverage", "rich"):
                profile = "symbolic_" + suffix
                self.lab.configure({"pair_feature_profile": profile})
                self.assertEqual(self.lab.config["pair_feature_profile"], profile)
        finally:
            self.lab.configure({"pair_feature_profile": previous})

    def test_history_scope_is_shared_and_outcome_free(self):
        left={"history_size_bucket":"light","long_affinity":"high"}
        right={"history_size_bucket":"light","long_affinity":"none"}
        article={"topic":"renamed_topic","subcategory":"renamed_subcategory"}
        forward=self.lab._pair_features(left,right,article,article)
        reverse=self.lab._pair_features(right,left,article,article)
        self.assertEqual(forward["pair_history_scope"],"light")
        self.assertEqual(reverse["pair_history_scope"],"light")
        self.assertEqual(forward["pair_long_affinity"],"left")
        self.assertEqual(reverse["pair_long_affinity"],"right")
        right["history_size_bucket"]="heavy"
        self.assertEqual(self.lab._pair_features(left,right,article,article)["pair_history_scope"],"unknown")

    def test_nl2pln_is_disabled_before_any_external_request(self):
        with patch.object(self.lab, "semantic_client") as client:
            with self.assertRaisesRegex(PeTTaChainerConfigurationError, "disabled"):
                self.lab.preview_semantics(self.lab.article("matched"))
            client.parse_article.assert_not_called()

    def test_historical_missing_evidence_does_not_read_later_profile(self):
        article = self.lab.article("matched")
        self.assertGreater(self.lab.features("reader", article)["lexical_peak_match"], 0)
        snapshot = {"history_size_bucket": "cold", "lexical_peak_match": None}
        with patch.object(self.lab, "features", side_effect=AssertionError("Read later profile")):
            attrs = self.lab.contextual_features("reader", article, snapshot)
        self.assertEqual(attrs["history_size_bucket"], "cold")
        self.assertTrue(set(LEXICAL_FEATURES).isdisjoint(attrs))

    def test_historical_measured_zero_is_preserved(self):
        article = self.lab.article("matched")
        snapshot = {"history_size_bucket": "light", "lexical_peak_match": 0.0}
        with patch.object(self.lab, "features", side_effect=AssertionError("Read later profile")):
            attrs = self.lab.contextual_features("reader", article, snapshot)
        self.assertEqual(attrs["lexical_peak_match"], "0.0")
        self.assertNotIn("lexical_history_coverage", attrs)

    def test_lexical_preference_is_mined_and_proved_by_pettachainer(self):
        lab = self.lab
        rules = [rule for rule in lab.pair_rules
                 if ("pair_lexical_peak_match", "left") in rule["premises"]]
        self.assertTrue(rules, "Actual miner did not discover the lexical preference")
        self.assertTrue(all(rule["source"].endswith("fpMiner.metta") for rule in rules))
        self.assertTrue(all(rule["strength"] > 0.5 for rule in rules))
        first, second = lab.data["events"][:2]
        forward = lab._pair_spec(
            (lab.article(first["article"]), lab.event_features(first)),
            (lab.article(second["article"]), lab.event_features(second)),
        )
        reverse = lab._pair_spec(
            (lab.article(second["article"]), lab.event_features(second)),
            (lab.article(first["article"]), lab.event_features(first)),
        )
        lab._ensure_pair_specs([forward, reverse])
        groups, _ = lab._proofs_for_pair_specs([forward, reverse])
        proof = " ".join(groups[forward[0]])
        self.assertTrue(any(rule["dependency_id"] in proof for rule in rules))
        self.assertIn("(STV", proof)
        self.assertGreater(lab._proof_vote_margin(forward[0], groups[forward[0]]),
                           lab._proof_vote_margin(reverse[0], groups[reverse[0]]))

    def test_remining_identical_symbolic_workspace_is_deterministic(self):
        def signatures():
            return [(rule["id"], tuple(rule["premises"]), rule["strength"],
                     rule["confidence"], rule["dependency_id"])
                    for rule in self.lab.pair_rules]
        before = signatures()
        self.lab.mine()
        self.assertTrue(before)
        self.assertEqual(before, signatures())


def _rank_one_dependency(rules):
    lab = object.__new__(Lab)
    lab.config = {
        "ranking_mode": "pairwise", "pair_aggregation": "proof_margin",
        "pair_family_fusion": "symbolic_balanced", "pairwise_weight": 1.0,
        "pairwise_fusion": "rank", "pair_margin_transform": "linear",
        "pair_margin_power": 1.0, "pairwise_opponents": 0,
    }
    lab.pair_rules = rules
    lab.version = 1
    lab._pair_margin_cache = {}
    rows = [{
        "article": {"id": article}, "score": 0.5,
        "stv": {"strength": 0.5, "confidence": 0.0},
        "tie_break": {"topic_prior": 0, "format_prior": 0, "subcategory_prior": 0},
    } for article in ("a", "b")]
    plan = ([(0, 1, "ab", "ba")], [])
    proofs = {"ab": ["(by pair_mined_cluster_1 (STV 0.8 1.0))"], "ba": []}
    return lab._pairwise_rank(rows, [], plan, proofs)


class SymbolicDatasetTransferTest(unittest.TestCase):
    def test_dataset_reload_preserves_selected_configuration(self):
        active=MagicMock()
        active.symbolic_only=True
        active.config={"pair_feature_profile":"symbolic_coverage",
                       "pair_ctv_mode":"impression_macro",
                       "pair_family_fusion":"symbolic_balanced"}
        active.lock=threading.RLock()
        replacement=MagicMock()
        body=json.dumps({"path":"mindplex.json.gz"}).encode()
        handler=object.__new__(Handler)
        handler.path="/api/dataset/load"
        handler.headers={"Content-Length":str(len(body))}
        handler.rfile=io.BytesIO(body)
        handler.send_json=MagicMock()
        data={"articles":[],"events":[],"users":{},"tests":[]}
        with patch("recommendation.app.server.LAB",active), \
                patch("recommendation.app.server.load_symbolic_snapshot",return_value=data) as loader, \
                patch("recommendation.app.server.Lab",return_value=replacement) as constructor:
            handler.do_POST()
        loader.assert_called_once_with("mindplex.json.gz")
        constructor.assert_called_once_with(data=data,symbolic_only=True,config=active.config)
        self.assertIsNot(constructor.call_args.kwargs["config"],active.config)
        # Dataset replacement retires the complete Lab lifecycle (including
        # its asynchronous miner and workspaces), not only the scorer worker.
        active.close.assert_called_once_with()
        self.assertIn("state",handler.send_json.call_args.args[0])


class SymbolicFamilyOwnershipTest(unittest.TestCase):
    @staticmethod
    def rule(rule_id, *predicates):
        return {"id": rule_id, "dependency_id": "pair_mined_cluster_1",
                "premises": [(predicate, "left") for predicate in predicates],
                "strength": 0.8, "confidence": 1.0}

    def test_mixed_proof_contributes_to_one_family_only(self):
        rows = _rank_one_dependency([self.rule(
            "variant1", "pair_long_affinity", "pair_lexical_peak_match",
            "pair_recent_subcategory_transition")])
        self.assertEqual(rows[0]["pairwise_family_rank_scores"], {"lexical": 1.0})
        self.assertEqual(rows[1]["pairwise_family_rank_scores"], {"lexical": 0.0})
        self.assertAlmostEqual(rows[0]["pairwise_margin_score"], 0.6)

    def test_dependency_owner_does_not_depend_on_variant_order(self):
        rules = [self.rule("variant1", "pair_long_affinity"),
                 self.rule("variant2", "pair_lexical_peak_match")]
        first = _rank_one_dependency(rules)
        second = _rank_one_dependency(list(reversed(rules)))
        self.assertEqual(first, second)
        self.assertEqual(set(first[0]["pairwise_family_rank_scores"]), {"lexical"})

    def test_transition_takes_ownership_over_interest_without_new_family(self):
        rows = _rank_one_dependency([self.rule(
            "variant1", "pair_recent_subcategory_transition", "pair_long_affinity")])
        self.assertEqual(set(rows[0]["pairwise_family_rank_scores"]), {"transition"})


if __name__ == "__main__":
    unittest.main()
