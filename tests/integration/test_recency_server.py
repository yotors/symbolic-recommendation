"""Recency observations require real mined preference rules and PeTTa proofs."""

import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from recommendation.app.server import Handler, Lab, load_symbolic_snapshot
from recommendation.pipelines.recency_data import build_recency_projection
from recommendation.features.recency_workspace import (
    RECENCY_WORKSPACE_FEATURES,
    build_recency_workspace_facts,
)
from recommendation.features.text_embeddings import build_text_embedding_sidecar
from recommendation.tests.fixtures import FakeEncoder, lexical_fixture


class RecencyLabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        source = lexical_fixture()
        source.pop("article_text_vectors")
        source["tests"][0]["id"] = "evaluation_1"
        sidecar = root / "vectors.npz"
        source["metadata"]["text_embedding_sidecar"] = str(sidecar)
        corpus = root / "corpus.json"
        corpus.write_text(json.dumps({"articles": source["articles"]}), encoding="utf-8")
        # Test-only encoder: sorted IDs are matched, past, unrelated. Production
        # experiments reuse the independently fingerprinted frozen sidecar.
        build_text_embedding_sidecar(
            corpus, sidecar, encoder=FakeEncoder([[1, 0], [1, 0], [0, 1]]),
            model_name="unit-fixture", model_revision="v1",
        )
        histories = copy.deepcopy(source)
        for event in histories["events"]:
            event["history"] = ["past"]
        cls.source = source
        cls.data = build_recency_projection(source, histories)
        cls.lab = Lab(cls.data, config={
            "pair_feature_profile": "text_semantic_recency_h8",
            "pair_family_fusion": "balanced_rank",
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
        })
        cls.addClassCleanup(cls.lab.engine.close)

    def test_recency_preference_is_discovered_by_fpminer_and_proved(self):
        predicate = "pair_text_semantic_recency_attention_h8_similarity"
        rules = [rule for rule in self.lab.pair_rules
                 if (predicate, "left") in rule["premises"]]
        self.assertTrue(rules, "Actual miner did not discover the recency preference")
        self.assertTrue(all(rule["source"].endswith("fpMiner.metta") for rule in rules))
        self.assertTrue(all(rule["strength"] > 0.5 for rule in rules))
        result = self.lab.benchmark({"remine": False})
        self.assertEqual(result["auc_proof_only"], 1.0)
        self.assertEqual(result["pairwise_proof_coverage"], 1.0)
        first, second = self.lab.data["events"][:2]
        forward = self.lab._pair_spec(
            (self.lab.article(first["article"]), self.lab.event_features(first)),
            (self.lab.article(second["article"]), self.lab.event_features(second)),
        )
        self.lab._ensure_pair_specs([forward])
        proof_map, _ = self.lab._proofs_for_pair_specs([forward])
        proof = " ".join(proof_map[forward[0]])
        self.assertIn("(STV", proof)
        self.assertTrue(any(rule["dependency_id"] in proof for rule in rules))

    def test_semantic_observations_cannot_rank_without_preference_proofs(self):
        def no_proofs(specs, **_kwargs):
            return {spec[0]: [] for spec in specs}, 0

        with patch.object(self.lab, "_proofs_for_pair_specs", side_effect=no_proofs):
            result = self.lab.benchmark({"remine": False})
        self.assertEqual(result["auc_proof_only"], 0.5)
        self.assertEqual(result["pairwise_proof_coverage"], 0.0)

    def test_live_training_and_replay_observations_use_identical_history_semantics(self):
        expected = build_recency_workspace_facts(
            "matched", ["past"], self.lab._article_text_vectors
        )
        live = self.lab.features("reader", self.lab.article("matched"))
        event = self.lab.data["events"][0]
        historical = self.lab.event_features(event)
        context = self.lab.data["tests"][0]["candidate_context"]["matched"]
        replay = self.lab.contextual_features("reader", self.lab.article("matched"), context)
        for feature, value in expected.items():
            self.assertEqual(live[feature], value, feature)
            self.assertEqual(event[feature], value, feature)
            self.assertEqual(context[feature], value, feature)
            self.assertEqual(float(historical[feature]), value, feature)
            self.assertEqual(float(replay[feature]), value, feature)

    def test_original_entity_evidence_is_retained_in_the_new_workspace(self):
        self.assertEqual(self.lab.data["article_entity_vectors"],
                         self.source["article_entity_vectors"])
        self.assertEqual(self.lab._article_entity_vectors,
                         {key: tuple(value) for key, value
                          in self.source["article_entity_vectors"].items()})
        self.assertTrue(self.lab.state()["engine"]["recency_workspace"])

    def test_strict_symbolic_constructor_strips_recency_and_vector_evidence(self):
        with patch("recommendation.app.server.load_text_embedding_sidecar",
                   side_effect=AssertionError("strict mode opened the vector sidecar")):
            strict = Lab(self.data, symbolic_only=True, config={
                "min_support": 2, "pair_min_support": 2,
                "max_rules": 8, "pair_max_rules": 8,
            })
        try:
            self.assertFalse(strict._recency_workspace)
            self.assertNotIn("recency_workspace", strict.data["metadata"])
            self.assertFalse(strict._article_text_vectors)
            self.assertFalse(strict._article_entity_vectors)
            for event in strict.data["events"]:
                self.assertTrue(set(RECENCY_WORKSPACE_FEATURES).isdisjoint(event))
            for context in strict.data["tests"][0]["candidate_context"].values():
                self.assertTrue(set(RECENCY_WORKSPACE_FEATURES).isdisjoint(context))
            self.assertTrue(strict.mined_rules)
            result = strict.benchmark({"remine": False})
            self.assertEqual(result["cases"], 1)
            self.assertIsNotNone(result["auc_proof_only"])
            # Constructing the strict copy must not mutate the reusable source.
            self.assertTrue(self.data["metadata"]["recency_workspace"])
            self.assertIn(RECENCY_WORKSPACE_FEATURES[0], self.data["events"][0])
        finally:
            strict.engine.close()

    def test_wrong_sidecar_hash_fails_before_any_mining(self):
        bad = copy.deepcopy(self.data)
        bad["metadata"]["recency_workspace"]["embedding_file_sha256"] = "0" * 64
        with patch("recommendation.app.server.PeTTa", side_effect=AssertionError("mining started")):
            with self.assertRaisesRegex(ValueError, "frozen provenance"):
                Lab(bad)

    def test_wrong_projection_or_observation_schema_fails_before_mining(self):
        for field in ("schema", "observation_schema"):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.data)
                bad["metadata"]["recency_workspace"][field] = "different-formula"
                with patch("recommendation.app.server.PeTTa", side_effect=AssertionError("mining started")):
                    with self.assertRaisesRegex(ValueError, "formula/schema"):
                        Lab(bad)

    def test_inline_text_vectors_cannot_override_the_verified_sidecar(self):
        bad = copy.deepcopy(self.data)
        bad["article_text_vectors"] = {"unverified": [1, 0]}
        with patch("recommendation.app.server.PeTTa", side_effect=AssertionError("mining started")):
            with self.assertRaisesRegex(ValueError, "inline text vectors"):
                Lab(bad)

    def test_recency_profiles_require_a_prepared_snapshot_and_preserve_config_on_failure(self):
        # Configuration validation does not require starting another reasoner.
        plain = object.__new__(Lab)
        plain.data = self.source
        plain.symbolic_only = False
        plain._semantic_workspace_model = {}
        plain._recency_workspace = {}
        plain.config = {**self.lab.config, "pair_feature_profile": "stable_multi_interest"}
        before = plain.config.copy()
        for half_life in (8, 16):
            with self.subTest(half_life=half_life):
                with self.assertRaisesRegex(ValueError, "prepared recency workspace"):
                    plain.configure({"pair_feature_profile": f"text_semantic_recency_h{half_life}"})
                self.assertEqual(plain.config, before)

    def test_residual_mode_rejects_shared_encoder_views_without_mutating_config(self):
        before = self.lab.config.copy()
        with self.assertRaisesRegex(ValueError, "clustered dependencies"):
            self.lab.configure({"pair_dependency_mode": "residual_hypergraph"})
        self.assertEqual(self.lab.config, before)

    def test_missing_recency_observation_stays_unknown_in_both_orientations(self):
        predicate = "pair_text_semantic_recency_attention_h8_similarity"
        missing = {RECENCY_WORKSPACE_FEATURES[0]: None}
        known = {RECENCY_WORKSPACE_FEATURES[0]: 0.0}
        self.assertEqual(self.lab._pair_features(missing, known, {}, {})[predicate], "right_known")
        self.assertEqual(self.lab._pair_features(known, missing, {}, {})[predicate], "left_known")

    def test_prepared_missing_observations_never_use_a_later_live_profile_without_old_markers(self):
        article = self.lab.article("matched")
        live = self.lab.features("reader", article)
        self.assertTrue(self.lab.data["users"]["reader"]["history"])
        self.assertTrue(all(live[feature] is not None for feature in RECENCY_WORKSPACE_FEATURES))
        event = copy.deepcopy(self.lab.data["events"][0])
        context = copy.deepcopy(self.lab.data["tests"][0]["candidate_context"]["matched"])
        for snapshot in (event, context):
            # Portable adapters need not use MIND's older snapshot sentinels.
            for key in ("recent_affinity", "long_affinity", "history_size_bucket",
                        "history_topic_count_bucket", "recent_topic_count_bucket"):
                snapshot.pop(key, None)
            for feature in RECENCY_WORKSPACE_FEATURES:
                snapshot[feature] = None
        with patch.object(self.lab, "features", side_effect=AssertionError("read future live profile")):
            historical = self.lab.event_features(event)
            replay = self.lab.contextual_features("reader", article, context)
        for feature in RECENCY_WORKSPACE_FEATURES:
            self.assertNotIn(feature, historical)
            self.assertNotIn(feature, replay)

    def test_generic_replay_reload_keeps_json_route_and_complete_configuration(self):
        path = Path(self.temporary.name) / "generic-replay.json"
        path.write_text(json.dumps(self.source), encoding="utf-8")
        prepared = load_symbolic_snapshot(path)
        self.assertTrue(prepared["metadata"]["replay_snapshot"])
        self.assertNotIn("recency_workspace", prepared["metadata"])
        active = MagicMock()
        active.symbolic_only = False
        active._semantic_workspace_model = {}
        active._recency_workspace = {}
        active._replay_snapshot = True
        active.config = {
            "pair_feature_profile": "text_semantic_attention_t8",
            "pair_family_fusion": "balanced_rank", "pair_min_support": 19,
        }
        active.lock = threading.RLock()
        replacement = MagicMock()
        body = json.dumps({"path": str(path)}).encode()
        handler = object.__new__(Handler)
        handler.path = "/api/dataset/load"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler.send_json = MagicMock()
        with patch("recommendation.app.server.LAB", active), \
                patch("recommendation.app.server.load_symbolic_snapshot", return_value=prepared) as loader, \
                patch("recommendation.app.server.load_mind", side_effect=AssertionError("opened raw dataset adapter")), \
                patch("recommendation.app.server.Lab", return_value=replacement) as constructor:
            handler.do_POST()
        loader.assert_called_once_with(str(path))
        constructor.assert_called_once_with(data=prepared, symbolic_only=False, config=active.config)
        self.assertIsNot(constructor.call_args.kwargs["config"], active.config)
        active.close.assert_called_once_with()
        self.assertIn("state", handler.send_json.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
