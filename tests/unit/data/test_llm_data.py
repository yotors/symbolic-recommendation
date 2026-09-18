import copy
import gzip
import hashlib
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from recommendation.pipelines.llm_data import build_llm_projection, main
from recommendation.features.concept_canonicalization import canonicalize_annotations
from recommendation.features.llm_workspace import LLM_WORKSPACE_FEATURES, build_llm_workspace_facts
from recommendation.features.text_embeddings import article_text


class LLMProjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {
            "articles": [
                {"id": "A", "source_id": "source_A", "title": "Robot research", "abstract": "Scientific progress", "entities": ["robot"]},
                {"id": "B", "source_id": "source_B", "title": "Forest birds", "abstract": "Natural history"},
                {"id": "H", "source_id": "source_H", "title": "Earlier robot discovery", "abstract": "Prior reading"},
            ],
            "users": {"u": {"history": ["B", "A"]}},
            "events": [
                {"user": "u", "article": "A", "action": "click", "impression": "train", "entity_overlap": "high", "text_semantic_attention_t8_similarity": 0.123},
                {"user": "u", "article": "B", "action": "skip", "impression": "train"},
            ],
            "tests": [{"id": "test", "user": "u", "candidates": ["A", "B"],
                       "history": ["H", "missing"], "relevant": ["A"], "labels": {"A": 1, "B": 0},
                       "candidate_context": {"A": {"entity_overlap": "high"}, "B": {"prior": 0.2}}}],
            "article_entity_vectors": {"A": [1, 2]},
            "article_text_vectors": {"A": [3, 4]},
            "title_idf_model": {"robot": 2.3},
            "config": {"pair_feature_profile": "text_semantic_attention_t8"},
            "metadata": {"dataset": "portable-fixture", "original": {"recorded": True}},
        }
        self.histories = copy.deepcopy(self.source)
        for event in self.histories["events"]:
            event["history"] = ["H", "missing"]
        self.histories["events"][0]["entity_overlap"] = "none"
        self.annotations = {}
        for article in self.source["articles"]:
            if article["id"] == "B":
                continue
            self.annotations[article["id"]] = {
                "concepts": ["robotics"], "format": "news_report",
                "event_types": ["discovery"], "intents": ["inform"], "audiences": ["general"],
                "provenance": {"article_content_sha256": hashlib.sha256(article_text(article["title"], article["abstract"]).encode()).hexdigest(),
                               "source_id": article["source_id"], "model": "test-fixture"},
            }

    def project(self, source=None, histories=None, annotations=None):
        return build_llm_projection(self.source if source is None else source,
                                    self.histories if histories is None else histories,
                                    self.annotations if annotations is None else annotations)

    def test_original_evidence_labels_and_input_objects_are_unchanged(self):
        originals = copy.deepcopy((self.source, self.histories, self.annotations))
        result = self.project()
        self.assertEqual((self.source, self.histories, self.annotations), originals)
        restored = copy.deepcopy(result)
        restored.pop("llm_article_annotations")
        restored["metadata"].pop("llm_workspace")
        for event in restored["events"]:
            event.pop("history")
            for name in LLM_WORKSPACE_FEATURES:
                event.pop(name)
        for case in restored["tests"]:
            for context in case["candidate_context"].values():
                for name in LLM_WORKSPACE_FEATURES:
                    context.pop(name)
        self.assertEqual(restored, self.source)
        result["llm_article_annotations"]["A"]["concepts"].append("changed")
        self.assertEqual(self.annotations, originals[2])

    def test_live_helper_and_projected_contexts_match_with_partial_annotation_coverage(self):
        result = self.project()
        expected = build_llm_workspace_facts("A", ["H", "missing"], self.annotations)
        for name, value in expected.items():
            self.assertEqual(result["events"][0][name], value)
            self.assertEqual(result["tests"][0]["candidate_context"]["A"][name], value)
        unknown = build_llm_workspace_facts("B", ["H", "missing"], self.annotations)
        for name, value in unknown.items():
            self.assertEqual(result["events"][1][name], value)
        audit = result["metadata"]["llm_workspace"]
        self.assertEqual(audit["annotated_articles"], 2)
        self.assertEqual(audit["unannotated_articles"], 1)
        self.assertEqual(audit["coverage"]["training"]["unannotated_candidate_contexts"], 1)
        self.assertEqual(audit["coverage"]["training"]["history_occurrences"], 4)
        self.assertEqual(audit["coverage"]["training"]["annotated_history_occurrences"], 2)
        self.assertGreater(audit["coverage"]["training"]["observations_missing"], 0)

    def test_future_profiles_and_validation_labels_do_not_construct_facts(self):
        before = self.project()
        original, companion = copy.deepcopy(self.source), copy.deepcopy(self.histories)
        original["users"]["u"]["history"] = ["A"] * 50
        for data in (original, companion):
            data["tests"][0]["labels"] = {"A": 0, "B": 1}
            data["tests"][0]["relevant"] = ["B"]
        after = self.project(original, companion)
        self.assertEqual(after["events"], before["events"])
        self.assertEqual(after["tests"][0]["candidate_context"], before["tests"][0]["candidate_context"])

    def test_history_identity_order_and_slates_must_match(self):
        for field in ("user", "article", "action", "impression"):
            companion = copy.deepcopy(self.histories)
            companion["events"][0][field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "training row identity"):
                self.project(histories=companion)
        for field, value in (("candidates", ["B", "A"]), ("history", ["A"]), ("id", "other")):
            companion = copy.deepcopy(self.histories)
            companion["tests"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "evaluation slate"):
                self.project(histories=companion)
        companion = copy.deepcopy(self.histories)
        del companion["events"][0]["history"]
        with self.assertRaisesRegex(ValueError, "explicit ordered history"):
            self.project(histories=companion)

    def test_stale_unknown_and_rule_shaped_annotations_are_rejected(self):
        annotations = copy.deepcopy(self.annotations)
        annotations["A"]["provenance"]["article_content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stale annotation"):
            self.project(annotations=annotations)
        annotations = copy.deepcopy(self.annotations)
        annotations["A"]["provenance"]["source_id"] = "other"
        with self.assertRaisesRegex(ValueError, "source ID mismatch"):
            self.project(annotations=annotations)
        annotations = copy.deepcopy(self.annotations)
        annotations["unknown"] = annotations["A"]
        with self.assertRaisesRegex(ValueError, "absent from source"):
            self.project(annotations=annotations)
        annotations = copy.deepcopy(self.annotations)
        annotations["A"]["recommendation_rule"] = "prefer article A"
        with self.assertRaisesRegex(ValueError, "invalid article annotation"):
            self.project(annotations=annotations)

    def test_empty_annotation_map_is_explicitly_missing_and_existing_features_cannot_be_overwritten(self):
        result = self.project(annotations={})
        self.assertEqual(result["metadata"]["llm_workspace"]["annotated_articles"], 0)
        self.assertEqual(result["events"][0]["llm_history_coverage"], 0.0)
        self.assertTrue(all(result["events"][0][name] is None
                            for name in LLM_WORKSPACE_FEATURES if name != "llm_history_coverage"))
        original = copy.deepcopy(self.source)
        original["events"][0][LLM_WORKSPACE_FEATURES[0]] = 0.5
        with self.assertRaisesRegex(ValueError, "already contains LLM observation"):
            self.project(source=original)

    def test_cli_atomic_new_path_and_exact_input_fingerprints(self):
        paths = [self.root / name for name in ("source.json", "histories.json", "annotations.json")]
        for path, data in zip(paths, (self.source, self.histories, self.annotations)):
            path.write_text(json.dumps(data), encoding="utf-8")
        arguments = ["--data", str(paths[0]), "--histories", str(paths[1]), "--annotations", str(paths[2])]
        outputs = [self.root / "first.json.gz", self.root / "second.json.gz"]
        for output in outputs:
            with redirect_stdout(StringIO()):
                self.assertEqual(main([*arguments, "--output", str(output)]), 0)
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        with gzip.open(outputs[0], "rt") as stream:
            audit = json.load(stream)["metadata"]["llm_workspace"]
        for name, path in zip(("source_dataset", "histories_dataset", "annotations"), paths):
            self.assertEqual(audit[f"{name}_sha256_kind"], "file-bytes")
            self.assertEqual(audit[f"{name}_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (*paths, outputs[0]):
            before = path.read_bytes()
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                main([*arguments, "--output", str(path)])
            self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".first.json.gz.*")))

    def test_extractor_envelope_cli_preserves_file_and_extracted_record_fingerprints(self):
        envelope = {
            "schema": "mindplex-llm-article-facts-v1", "model": "test-fixture",
            "prompt_sha256": "a" * 64, "progress": {"available_articles": 2, "unprocessed_articles": 1},
            "extraction_records": self.annotations,
        }
        files = [self.root / name for name in ("source.json", "histories.json", "export.json")]
        for path, value in zip(files, (self.source, self.histories, envelope)):
            path.write_text(json.dumps(value), encoding="utf-8")
        output = self.root / "from-export.json.gz"
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["--data", str(files[0]), "--histories", str(files[1]),
                                   "--annotations", str(files[2]), "--output", str(output)]), 0)
        with gzip.open(output, "rt") as stream:
            result = json.load(stream)
        audit = result["metadata"]["llm_workspace"]
        self.assertEqual(result["llm_article_annotations"], self.annotations)
        self.assertEqual(audit["annotation_source"]["model"], "test-fixture")
        self.assertEqual(audit["annotations_sha256_kind"], "file-bytes")
        self.assertEqual(audit["annotations_sha256"], hashlib.sha256(files[2].read_bytes()).hexdigest())
        canonical = json.dumps(self.annotations, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(audit["annotation_records_sha256"], hashlib.sha256(canonical.encode()).hexdigest())
        direct = self.project()
        self.assertEqual(result["events"], direct["events"])
        self.assertEqual(result["tests"], direct["tests"])
        for invalid in ({"schema": "mindplex-llm-article-cache-v1", "entries": {}},
                        {**envelope, "schema": "unknown"}, {**envelope, "unexpected": True}):
            with self.subTest(schema=invalid["schema"]), self.assertRaisesRegex(ValueError, "article-facts snapshot"):
                self.project(annotations=invalid)

    def test_canonical_envelope_preserves_registry_provenance_and_rejects_tampering(self):
        canonical = canonicalize_annotations(
            self.annotations, self.source, include_entity_concepts=False,
        )
        result = self.project(annotations=canonical)
        audit = result["metadata"]["llm_workspace"]
        self.assertEqual(
            audit["annotation_source"]["kind"],
            "canonical-annotation-snapshot",
        )
        self.assertEqual(
            audit["annotation_source"]["provenance"]["registry_sha256"],
            canonical["provenance"]["registry_sha256"],
        )
        self.assertTrue(
            result["llm_article_annotations"]["A"]["concepts"][0]
            .startswith("concept:")
        )
        corrupted = copy.deepcopy(canonical)
        corrupted["annotations"]["A"]["concepts"].append("concept:tampered")
        with self.assertRaisesRegex(ValueError, "provenance hash"):
            self.project(annotations=corrupted)


if __name__ == "__main__":
    unittest.main()
