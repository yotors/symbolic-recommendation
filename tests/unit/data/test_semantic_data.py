from __future__ import annotations

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path

import numpy as np

from recommendation.pipelines.semantic_data import build_semantic_projection, main
from recommendation.features.semantic_workspace import SEMANTIC_WORKSPACE_FEATURES, build_semantic_workspace_facts
from recommendation.features.text_embeddings import TextEmbeddingError, build_text_embedding_sidecar, load_text_embedding_sidecar
from recommendation.tests.fixtures import FakeEncoder


class SemanticProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        articles = [
            {"id": f"item_{key}", "source_id": key, "title": title,
             "abstract": "", "topic": "science", "subcategory": "research", "format": "short"}
            for key, title in (("A", "New robotics research"), ("B", "Forest bird observations"),
                               ("H", "Robotics history"), ("E", "Unseen evaluation science"),
                               ("U", "Article without a vector"))
        ]
        self.data = {
            "articles": articles,
            "users": {"u": {"history": ["item_E"], "topics": ["science"]}},
            "events": [
                {"user": "u", "impression": "train", "article": "item_A", "action": "click", "history": ["item_H"],
                 "text_semantic_top1_similarity": 0.123, "entity_overlap": "high"},
                {"user": "u", "impression": "train", "article": "item_B", "action": "skip", "history": ["item_H"]},
                {"user": "u", "impression": "later", "article": "item_A", "action": "click", "history": ["item_H", "missing"]},
            ],
            "tests": [{"id": "test", "user": "u", "history": ["item_H", "missing"],
                       "candidates": ["item_E", "item_U"], "relevant": ["item_E"],
                       "labels": {"item_E": 1, "item_U": 0}}],
            "metadata": {"dataset": "portable-fixture", "text_embedding_sidecar": "stale.npz"},
            "article_entity_vectors": {"item_A": [1, 2]},
            "article_text_vectors": {"item_A": [3, 4]},
        }
        self.corpus = self.root / "corpus.json"
        self.corpus.write_text(json.dumps({"articles": [article for article in articles if article["source_id"] != "U"]}), encoding="utf-8")
        self.sidecar = self.root / "embeddings.npz"
        # Corpus canonical order is A, B, E, H. E never belongs to training.
        build_text_embedding_sidecar(
            self.corpus, self.sidecar,
            encoder=FakeEncoder([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]),
            model_name="test/frozen-content-encoder", model_revision="revision-1",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def project(self, data=None, corpus=None):
        return build_semantic_projection(
            self.data if data is None else data, self.sidecar,
            provenance_corpus=self.corpus if corpus is None else corpus,
        )

    def test_training_only_unique_background_and_verified_provenance(self):
        result = self.project()
        model = result["semantic_workspace_model"]
        self.assertEqual(model["training_vector_count"], 3)
        expected = (np.array([1, 0, 0]) + np.array([0, 1, 0]) + np.array([1, 1, 0]) / np.sqrt(2)) / 3
        np.testing.assert_allclose(model["mean_vector"], expected)
        audit = result["metadata"]["semantic_workspace"]
        self.assertEqual(audit["fitted_item_ids"], 3)
        self.assertEqual(audit["embedding_file_sha256"], hashlib.sha256(self.sidecar.read_bytes()).hexdigest())
        self.assertEqual(audit["source_checks"]["covered_article_texts_verified"], 4)
        self.assertEqual(result["metadata"]["text_embedding_sidecar"], str(self.sidecar.resolve()))
        self.assertEqual(audit["embedding_model"]["revision"], "revision-1")

    def test_same_projector_serves_explicit_history_and_missingness(self):
        result = self.project()
        vectors = load_text_embedding_sidecar(self.sidecar).as_mapping()
        model = result["semantic_workspace_model"]
        case = result["tests"][0]
        expected = build_semantic_workspace_facts("item_E", ["item_H", "missing"], vectors, model)
        context = case["candidate_context"]["item_E"]
        for feature, value in expected.items():
            self.assertEqual(context[feature], value)
        unknown = case["candidate_context"]["item_U"]
        for feature in SEMANTIC_WORKSPACE_FEATURES:
            self.assertIsNone(unknown[feature])
        self.assertIsNone(unknown["text_semantic_top1_similarity"])
        coverage = result["metadata"]["semantic_workspace"]["coverage"]["evaluation"]
        self.assertEqual(coverage["unknown_candidate_contexts"], 1)
        self.assertEqual(coverage["unknown_history_occurrences"], 2)
        self.assertEqual(coverage["unknown_contexts"], 1)

    def test_validation_outcomes_and_latest_user_profile_do_not_change_evidence(self):
        original = self.project()
        changed = copy.deepcopy(self.data)
        changed["tests"][0]["labels"] = {"item_E": 0, "item_U": 1}
        changed["tests"][0]["relevant"] = ["item_U"]
        changed["users"]["u"]["history"] = ["item_A", "item_B", "item_E"]
        result = self.project(changed)
        self.assertEqual(result["semantic_workspace_model"], original["semantic_workspace_model"])
        self.assertEqual(result["tests"][0]["candidate_context"], original["tests"][0]["candidate_context"])
        self.assertEqual(result["events"], original["events"])

    def test_stale_article_text_and_wrong_provenance_fail(self):
        changed = copy.deepcopy(self.data)
        changed["articles"][0]["title"] = "Revised article with a stale embedding"
        with self.assertRaisesRegex(TextEmbeddingError, "article content mismatch"):
            self.project(changed)
        wrong = self.root / "wrong-corpus.json"
        wrong.write_text(json.dumps({"articles": changed["articles"][:-1]}), encoding="utf-8")
        with self.assertRaisesRegex(TextEmbeddingError, "content provenance"):
            self.project(corpus=wrong)

    def test_source_is_unchanged_and_legacy_evidence_is_removed(self):
        before = json.dumps(self.data, sort_keys=True)
        result = self.project()
        self.assertEqual(json.dumps(self.data, sort_keys=True), before)
        self.assertNotIn("article_entity_vectors", result)
        self.assertNotIn("article_text_vectors", result)
        self.assertNotIn("entity_overlap", result["events"][0])
        self.assertNotEqual(result["events"][0]["text_semantic_top1_similarity"], 0.123)
        self.assertFalse(result["metadata"]["semantic_workspace"]["nl2pln_used"])

    def test_fresh_source_selection_provenance_is_preserved_without_aliasing(self):
        upstream = {
            "fresh_holdout": True, "previous_impression_overlap": 0,
            "excluded_previous_impression_ids": ["dev-1", "dev-2"],
            "validation_selection": {"seed": 47, "policy": "whole-slate ID hash"},
        }
        self.data["metadata"]["symbolic_projection"] = copy.deepcopy(upstream)
        before = json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result = self.project()
        audit = result["metadata"]["semantic_workspace"]
        self.assertEqual(audit["source_projection"], upstream)
        self.assertEqual(audit["source_dataset_sha256_kind"], "canonical-json-mapping")
        self.assertEqual(audit["source_dataset_sha256"], hashlib.sha256(before.encode()).hexdigest())
        self.assertFalse(result["metadata"]["symbolic_projection"]["fresh_holdout"])
        audit["source_projection"]["excluded_previous_impression_ids"].append("another")
        self.assertEqual(self.data["metadata"]["symbolic_projection"], upstream)
        self.assertEqual(json.dumps(self.data, ensure_ascii=False, sort_keys=True, separators=(",", ":")), before)

    def test_file_source_hash_is_exact_compressed_bytes(self):
        source = self.root / "source.json.gz"
        with gzip.open(source, "wt", encoding="utf-8") as handle:
            json.dump(self.data, handle)
        result = self.project(source)
        audit = result["metadata"]["semantic_workspace"]
        self.assertEqual(audit["source_dataset_sha256_kind"], "file-bytes")
        self.assertEqual(audit["source_dataset_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())

    def test_missing_training_vectors_and_histories_fail(self):
        changed = copy.deepcopy(self.data)
        for event in changed["events"]:
            event["article"] = "item_U"
            event["history"] = []
        with self.assertRaisesRegex(ValueError, "at least one training vector"):
            self.project(changed)
        del changed["events"][0]["history"]
        with self.assertRaisesRegex(ValueError, "lack explicit histories"):
            self.project(changed)

    def test_cli_atomic_reproducible_gzip_and_source_overwrite_rejection(self):
        source = self.root / "data.json"
        source.write_text(json.dumps(self.data), encoding="utf-8")
        before = source.read_bytes()
        outputs = [self.root / "first.json.gz", self.root / "second.json.gz"]
        base = ["--data", str(source), "--embeddings", str(self.sidecar), "--provenance-corpus", str(self.corpus)]
        for output in outputs:
            with redirect_stdout(StringIO()):
                self.assertEqual(main([*base, "--output", str(output)]), 0)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                self.assertIn("semantic_workspace_model", json.load(handle))
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".first.json.gz.*")))
        for protected in (source, self.sidecar, self.corpus):
            with self.subTest(protected=protected), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main([*base, "--output", str(protected)])


if __name__ == "__main__":
    unittest.main()
