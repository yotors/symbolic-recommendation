from __future__ import annotations

import copy
import gzip
import hashlib
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from recommendation.pipelines.recency_data import build_recency_projection, main
from recommendation.features.recency_workspace import RECENCY_WORKSPACE_FEATURES, build_recency_workspace_facts
from recommendation.features.text_embeddings import TextEmbeddingError, build_text_embedding_sidecar, load_text_embedding_sidecar
from recommendation.tests.fixtures import FakeEncoder


class RecencyProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = {
            "articles": [
                {"id": "item_A", "source_id": "A", "title": "Robotics research", "abstract": "New results", "entities": ["robot"]},
                {"id": "item_B", "source_id": "B", "title": "Forest birds", "abstract": "Observations"},
                {"id": "item_H", "source_id": "H", "title": "Earlier science", "abstract": "Prior reading"},
            ],
            "users": {"u": {"history": ["item_A", "item_B"]}},
            "events": [
                {"user": "u", "article": "item_A", "action": "click", "impression": "train", "topic": "science", "entity_overlap": "high", "text_semantic_attention_t8_similarity": 0.123},
                {"user": "u", "article": "item_B", "action": "skip", "impression": "train", "topic": "nature"},
            ],
            "tests": [{"id": "test", "user": "u", "candidates": ["item_A", "item_B"],
                       "history": ["item_H", "missing"], "labels": {"item_A": 1, "item_B": 0},
                       "relevant": ["item_A"], "candidate_context": {
                           "item_A": {"entity_overlap": "high", "text_semantic_attention_t8_similarity": 0.321},
                           "item_B": {"topic": "nature", "prior_ctr": 0.01},
                       }}],
            "article_entity_vectors": {"item_A": [1.0, 2.0]},
            "title_idf_model": {"documents": 23},
            "subcategory_transition_model": {"sample": 0.23},
            "config": {"pair_feature_profile": "text_semantic_attention_t8"},
            "metadata": {"dataset": "portable-fixture", "original": {"recorded": True}},
        }
        self.sidecar = self.root / "vectors.npz"
        corpus = self.root / "corpus.json"
        corpus.write_text(json.dumps({"articles": self.source["articles"]}), encoding="utf-8")
        build_text_embedding_sidecar(
            corpus, self.sidecar, encoder=FakeEncoder([[1, 0], [0, 1], [1, 1]]),
            model_name="test/frozen-content", model_revision="fixture-revision",
        )
        self.source["metadata"]["text_embedding_sidecar"] = str(self.sidecar)
        self.histories = copy.deepcopy(self.source)
        for event in self.histories["events"]:
            event["history"] = ["item_H", "missing"]
        # Companion feature values are deliberately different: only history is copied.
        self.histories["events"][0]["entity_overlap"] = "none"
        self.histories["tests"][0]["candidate_context"]["item_A"]["text_semantic_attention_t8_similarity"] = 0.999

    def tearDown(self):
        self.temporary.cleanup()

    def project(self, source=None, histories=None):
        return build_recency_projection(
            self.source if source is None else source,
            self.histories if histories is None else histories,
        )

    def test_all_original_evidence_survives_without_mutating_either_input(self):
        before_source = copy.deepcopy(self.source)
        before_histories = copy.deepcopy(self.histories)
        result = self.project()
        self.assertEqual(self.source, before_source)
        self.assertEqual(self.histories, before_histories)
        self.assertEqual(result["events"][0]["history"], ["item_H", "missing"])
        restored = copy.deepcopy(result)
        for event in restored["events"]:
            event.pop("history")
            for feature in RECENCY_WORKSPACE_FEATURES:
                event.pop(feature)
        for context in restored["tests"][0]["candidate_context"].values():
            for feature in RECENCY_WORKSPACE_FEATURES:
                context.pop(feature)
        restored["metadata"].pop("recency_workspace")
        self.assertEqual(restored, self.source)
        result["article_entity_vectors"]["item_A"][0] = -1
        self.assertEqual(self.source["article_entity_vectors"]["item_A"][0], 1)

    def test_same_live_projector_produces_training_and_evaluation_observations(self):
        result = self.project()
        vectors = load_text_embedding_sidecar(self.sidecar).as_mapping()
        expected = build_recency_workspace_facts("item_A", ["item_H", "missing"], vectors)
        for name, value in expected.items():
            self.assertEqual(result["events"][0][name], value)
            self.assertEqual(result["tests"][0]["candidate_context"]["item_A"][name], value)
        audit = result["metadata"]["recency_workspace"]
        self.assertEqual(audit["embedding_file_sha256"], hashlib.sha256(self.sidecar.read_bytes()).hexdigest())
        self.assertEqual(audit["covered_article_texts_verified"], 3)
        self.assertTrue(audit["preserved_original_evidence"])
        self.assertFalse(audit["nl2pln_used"])

    def test_future_user_profiles_and_outcomes_do_not_supply_observations(self):
        before = self.project()
        changed = copy.deepcopy(self.source)
        changed["users"]["u"]["history"] = ["item_A"] * 10
        companion = copy.deepcopy(self.histories)
        for data in (changed, companion):
            data["tests"][0]["labels"] = {"item_A": 0, "item_B": 1}
            data["tests"][0]["relevant"] = ["item_B"]
        after = self.project(changed, companion)
        self.assertEqual(before["events"], after["events"])
        self.assertEqual(before["tests"][0]["candidate_context"], after["tests"][0]["candidate_context"])

    def test_missing_explicit_histories_and_mismatched_training_rows_are_rejected(self):
        for key in ("user", "article", "action", "impression"):
            changed = copy.deepcopy(self.histories)
            changed["events"][0][key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "training row identity"):
                self.project(histories=changed)
        changed = copy.deepcopy(self.histories)
        del changed["events"][0]["history"]
        with self.assertRaisesRegex(ValueError, "explicit ordered history"):
            self.project(histories=changed)
        changed = copy.deepcopy(self.histories)
        changed["events"][1]["history"] = ["item_A"]
        with self.assertRaisesRegex(ValueError, "different histories"):
            self.project(histories=changed)

    def test_eval_reordering_and_history_changes_are_rejected(self):
        for field, value in (("id", "other"), ("candidates", ["item_B", "item_A"]),
                             ("history", ["item_A"]), ("labels", {"item_A": 0, "item_B": 1})):
            changed = copy.deepcopy(self.histories)
            changed["tests"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "evaluation slate mismatch"):
                self.project(histories=changed)

    def test_stale_text_in_companion_or_sidecar_is_rejected(self):
        changed = copy.deepcopy(self.histories)
        changed["articles"][0]["abstract"] = "Different meaning"
        with self.assertRaisesRegex(TextEmbeddingError, "companion article content"):
            self.project(histories=changed)
        original = copy.deepcopy(self.source)
        original["articles"][0]["abstract"] = "Different meaning"
        with self.assertRaisesRegex(TextEmbeddingError, "content provenance"):
            self.project(original, changed)

    def test_cold_explicit_history_stays_missing_and_cannot_overwrite_existing_features(self):
        changed = copy.deepcopy(self.histories)
        for event in changed["events"]:
            event["history"] = []
        result = self.project(histories=changed)
        for feature in RECENCY_WORKSPACE_FEATURES:
            self.assertIsNone(result["events"][0][feature])
        modified = copy.deepcopy(self.source)
        modified["events"][0][RECENCY_WORKSPACE_FEATURES[0]] = 0.5
        with self.assertRaisesRegex(ValueError, "already contains recency"):
            self.project(source=modified)

    def test_cli_new_file_atomic_reproducibility_and_input_fingerprints(self):
        source_path = self.root / "source.json.gz"
        history_path = self.root / "histories.json"
        with gzip.open(source_path, "wt", encoding="utf-8") as stream:
            json.dump(self.source, stream)
        history_path.write_text(json.dumps(self.histories), encoding="utf-8")
        arguments = ["--data", str(source_path), "--histories", str(history_path)]
        outputs = [self.root / "first.json.gz", self.root / "second.json.gz"]
        for output in outputs:
            with redirect_stdout(StringIO()):
                self.assertEqual(main([*arguments, "--output", str(output)]), 0)
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        with gzip.open(outputs[0], "rt", encoding="utf-8") as stream:
            audit = json.load(stream)["metadata"]["recency_workspace"]
        self.assertEqual(audit["source_dataset_sha256_kind"], "file-bytes")
        self.assertEqual(audit["source_dataset_sha256"], hashlib.sha256(source_path.read_bytes()).hexdigest())
        self.assertEqual(audit["histories_dataset_sha256"], hashlib.sha256(history_path.read_bytes()).hexdigest())
        for path in (source_path, history_path, self.sidecar, outputs[0]):
            before = path.read_bytes()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main([*arguments, "--output", str(path)])
            self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".first.json.gz.*")))


if __name__ == "__main__":
    unittest.main()
