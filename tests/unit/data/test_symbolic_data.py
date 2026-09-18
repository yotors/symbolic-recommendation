from __future__ import annotations

import csv
import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from recommendation.pipelines.symbolic_data import build_symbolic_projection, strip_neural_evidence


def _csv_text(header, rows, delimiter=","):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


class SymbolicProjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = {
            "articles": [
                {"id": "article_H", "source_id": "H", "title": "robot learning history", "abstract": "", "topic": "tech", "subcategory": "robots", "format": "short", "entities": ["Q1"]},
                {"id": "article_A", "source_id": "A", "title": "robot learning tools", "abstract": "", "topic": "tech", "subcategory": "robots", "format": "short"},
                {"id": "article_B", "source_id": "B", "title": "forest bird songs", "abstract": "", "topic": "nature", "subcategory": "birds", "format": "short"},
            ],
            "users": {"user_U": {"history": ["article_A"], "topics": ["tech"]}},
            "events": [
                {"impression": "impression_train_T", "source_impression_id": "T", "user": "user_U", "article": "article_A", "action": "click", "recent_affinity": "high", "recent_subcategory_transition_score": 0.123, "entity_overlap": "high", "text_semantic_top1_similarity": 0.9},
                {"impression": "impression_train_T", "source_impression_id": "T", "user": "user_U", "article": "article_B", "action": "skip", "recent_affinity": "none", "recent_subcategory_transition_score": 0.012},
            ],
            "tests": [{
                "id": "impression_valid_OLD", "source_impression_id": "OLD", "user": "user_U",
                "history": ["article_H"], "candidates": ["article_A", "article_B"],
                "relevant": ["article_A"], "labels": {"article_A": 1, "article_B": 0},
                "candidate_context": {"article_A": {"long_affinity": "high"}, "article_B": {"long_affinity": "none"}},
            }],
            "title_idf_model": {"version": "title-idf-v1", "document_count": 2, "idf": {"robot": 1.5, "entity": 2.0}, "default_idf": 2.0},
            "subcategory_transition_model": {"global": 0.1, "candidate": {"robots": 0.2}, "transition": {"robots": {"robots": 0.3}}},
            "article_entity_vectors": {"article_A": [0.1, 0.2]},
            "article_text_vectors": {"article_A": [0.3, 0.4]},
            "metadata": {"text_embedding_sidecar": "never-read.npz", "dataset": "fixture"},
        }

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name="source.zip", *, reverse_labels=False):
        header = ["imp_id", "click", "hour", "user_id", "news_id", "news_his"]
        train = [["T", 1, 8, "U", "A", "H^UNKNOWN"], ["T", 0, 8, "U", "B", "H^UNKNOWN"]]
        valid = []
        for impression in ["OLD", "F1", "F2", "F3", "F4"]:
            for article, label in (("C", 1), ("B", 0)):
                valid.append([impression, 1 - label if reverse_labels else label, 9, "NEW", article, "H^UNKNOWN"])
        news = [
            ["H", "tech", "robots", "robot learning history", "", "FORBIDDEN_Q"],
            ["A", "tech", "robots", "robot learning tools", "", "FORBIDDEN_Q"],
            ["B", "nature", "birds", "forest bird songs", "", "FORBIDDEN_Q"],
            ["C", "tech", "robots", "robot learning holdoutonlyword", "", "FORBIDDEN_Q"],
        ]
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("train.csv", _csv_text(header, train))
            archive.writestr("valid.csv", _csv_text(header, valid))
            archive.writestr("news_corpus.tsv", _csv_text(
                ["news_id", "cat", "sub_cat", "title", "abstract", "title_entities"], news, delimiter="\t",
            ))
        return path

    def test_deep_purge_preserves_statistical_models_and_source(self):
        self.cache["title_idf_model"]["idf"]["neural"] = 1.75
        self.cache["users"]["user_vector"] = {"history": [], "semantic_vector": [1, 2]}
        self.cache["tests"][0]["labels"]["article_entity"] = 0
        self.cache["tests"][0]["candidate_context"]["article_vector"] = {
            "long_affinity": "none", "text_semantic_top1_similarity": 0.8,
        }
        original = json.dumps(self.cache, sort_keys=True)
        clean = strip_neural_evidence(self.cache)
        self.assertNotIn("article_entity_vectors", clean)
        self.assertNotIn("article_text_vectors", clean)
        self.assertNotIn("entities", clean["articles"][0])
        self.assertNotIn("text_semantic_top1_similarity", clean["events"][0])
        self.assertNotIn("entity_overlap", clean["events"][0])
        self.assertNotIn("text_embedding_sidecar", clean["metadata"])
        self.assertEqual(clean["title_idf_model"], self.cache["title_idf_model"])
        self.assertIn("user_vector", clean["users"])
        self.assertNotIn("semantic_vector", clean["users"]["user_vector"])
        self.assertIn("article_entity", clean["tests"][0]["labels"])
        self.assertEqual(clean["tests"][0]["candidate_context"]["article_vector"], {"long_affinity": "none"})
        clean["events"][0]["action"] = "skip"
        self.assertEqual(original, json.dumps(self.cache, sort_keys=True))

    def test_fresh_projection_is_disjoint_causal_and_training_only(self):
        original = json.dumps(self.cache, sort_keys=True)
        result = build_symbolic_projection(self.cache, self.archive(), fresh_eval_impressions=2, seed=11)
        audit = result["metadata"]["symbolic_projection"]
        self.assertEqual(len(result["tests"]), 2)
        self.assertEqual(audit["previous_impression_overlap"], 0)
        self.assertEqual(audit["auc_eligible_impressions"], 2)
        self.assertEqual(audit["training_history_recovery"]["recovered_events"], 2)
        for event in result["events"]:
            self.assertEqual(event["history"], ["article_H", "article_UNKNOWN"])
            self.assertIn("lexical_peak_match", event)
        self.assertEqual(result["events"][0]["recent_subcategory_transition_score"], 0.123)
        self.assertEqual(result["events"][1]["recent_subcategory_transition_score"], 0.012)
        self.assertNotIn("holdoutonlyword", result["lexical_idf_model"]["idf"])
        self.assertIn("robot", result["lexical_idf_model"]["idf"])
        self.assertEqual(result["users"]["user_NEW"]["history"], ["article_H", "article_UNKNOWN"])
        self.assertTrue(any(article["id"] == "article_C" for article in result["articles"]))
        for case in result["tests"]:
            self.assertEqual(set(case["candidates"]), {"article_C", "article_B"})
            for context in case["candidate_context"].values():
                self.assertNotIn("entity_overlap", context)
                self.assertNotIn("text_semantic_top1_similarity", context)
                self.assertIn("lexical_peak_match", context)
                self.assertNotIn("ctr_bucket", context)
        self.assertEqual(original, json.dumps(self.cache, sort_keys=True))

    def test_selection_and_facts_do_not_depend_on_validation_labels(self):
        left = build_symbolic_projection(self.cache, self.archive("left.zip"), fresh_eval_impressions=2, seed=11)
        right = build_symbolic_projection(self.cache, self.archive("right.zip", reverse_labels=True), fresh_eval_impressions=2, seed=11)
        self.assertEqual([case["id"] for case in left["tests"]], [case["id"] for case in right["tests"]])
        self.assertEqual(left["lexical_idf_model"], right["lexical_idf_model"])
        for first, second in zip(left["tests"], right["tests"]):
            self.assertEqual(first["candidate_context"], second["candidate_context"])
            self.assertNotEqual(first["labels"], second["labels"])

    def test_additional_evaluation_exclusions_preserve_training_and_frozen_model(self):
        archive = self.archive()
        first = build_symbolic_projection(self.cache, archive, fresh_eval_impressions=2, seed=11)
        first_ids = {case["source_impression_id"] for case in first["tests"]}
        path = self.root / "already-evaluated.json"
        path.write_text(json.dumps(first), encoding="utf-8")
        for source in (first, path):
            with self.subTest(source_type=type(source).__name__):
                second = build_symbolic_projection(
                    self.cache, archive, fresh_eval_impressions=2, seed=11,
                    exclude_evaluation_sources=[source],
                )
                second_ids = {case["source_impression_id"] for case in second["tests"]}
                self.assertTrue(second_ids.isdisjoint(first_ids | {"OLD"}))
                self.assertEqual(first["events"], second["events"])
                self.assertEqual(first["lexical_idf_model"], second["lexical_idf_model"])
                audit = second["metadata"]["symbolic_projection"]
                self.assertEqual(set(audit["excluded_previous_impression_ids"]), first_ids | {"OLD"})
                self.assertEqual(audit["previous_impression_overlap"], 0)
                self.assertEqual(audit["additional_evaluation_exclusions"][0]["impressions"], 2)
        with self.assertRaisesRegex(ValueError, "only 0 available"):
            build_symbolic_projection(
                self.cache, archive, fresh_eval_impressions=1, seed=11,
                exclude_evaluation_sources=[first, second],
            )

    def test_portable_json_needs_no_reczoo_if_histories_are_explicit(self):
        for event in self.cache["events"]:
            event["history"] = ["article_H"]
        result = build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        self.assertEqual([case["id"] for case in result["tests"]], ["impression_valid_OLD"])
        self.assertFalse(result["metadata"]["symbolic_projection"]["fresh_holdout"])
        self.assertIn("lexical_peak_match", result["tests"][0]["candidate_context"]["article_A"])

    def test_portable_missing_context_uses_explicit_history_not_latest_profile(self):
        for event in self.cache["events"]:
            event["history"] = []
            event.pop("recent_affinity", None)
            event.pop("recent_subcategory_transition_score", None)
        case = self.cache["tests"][0]
        case["history"] = []
        case.pop("candidate_context")
        self.cache["users"]["user_U"]["history"] = ["article_H", "article_A"]
        result = build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        for event in result["events"]:
            self.assertEqual(event["history_size_bucket"], "cold")
            self.assertEqual(event["long_affinity"], "none")
            self.assertIsNone(event["recent_subcategory_transition_score"])
            self.assertIsNone(event["lexical_peak_match"])
        for context in result["tests"][0]["candidate_context"].values():
            self.assertEqual(context["history_size_bucket"], "cold")
            self.assertEqual(context["long_affinity"], "none")
            self.assertIsNone(context["lexical_peak_match"])

    def test_portable_training_pairs_reject_mixed_users_or_prior_histories(self):
        for event in self.cache["events"]:
            event["history"] = ["article_H", "article_UNKNOWN"]
        mixed_user = copy.deepcopy(self.cache)
        mixed_user["events"][1]["user"] = "user_OTHER"
        with self.assertRaisesRegex(ValueError, "one user.*identical ordered"):
            build_symbolic_projection(mixed_user, None, fresh_eval_impressions=0)
        mixed_history = copy.deepcopy(self.cache)
        mixed_history["events"][1]["history"] = ["article_UNKNOWN", "article_H"]
        with self.assertRaisesRegex(ValueError, "one user.*identical ordered"):
            build_symbolic_projection(mixed_history, None, fresh_eval_impressions=0)
        # A different impression is allowed to have a different preceding
        # state, even for the same user; source order remains unchanged.
        mixed_history["events"][1]["impression"] = "another_impression"
        result = build_symbolic_projection(mixed_history, None, fresh_eval_impressions=0)
        self.assertEqual(result["events"][1]["history"], ["article_UNKNOWN", "article_H"])

    def test_portable_evaluation_ids_are_stable_unique_and_outcome_independent(self):
        for event in self.cache["events"]:
            event["history"] = ["article_H"]
        case = self.cache["tests"][0]
        case.pop("id")
        case.pop("source_impression_id")
        second = copy.deepcopy(case)
        # Reserve the ordinary generated name to verify collision avoidance.
        second["id"] = "symbolic_eval_00000000"
        self.cache["tests"].append(second)
        original = json.dumps(self.cache, sort_keys=True)
        first = build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        repeated = build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        ids = [item["id"] for item in first["tests"]]
        self.assertEqual(ids, [item["id"] for item in repeated["tests"]])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(original, json.dumps(self.cache, sort_keys=True))
        case["relevant"] = ["article_B"]
        changed = build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        self.assertEqual(ids, [item["id"] for item in changed["tests"]])

    def test_portable_duplicate_evaluation_ids_or_effective_identities_fail(self):
        for event in self.cache["events"]:
            event["history"] = ["article_H"]
        duplicate = copy.deepcopy(self.cache["tests"][0])
        duplicate["source_impression_id"] = "different_source"
        self.cache["tests"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate evaluation id"):
            build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        duplicate["id"] = "unique_id"
        duplicate["source_impression_id"] = "OLD"
        with self.assertRaisesRegex(ValueError, "duplicate evaluation impression identity"):
            build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)

    def test_missing_causal_histories_and_insufficient_holdout_fail(self):
        with self.assertRaisesRegex(ValueError, "lack explicit histories"):
            build_symbolic_projection(self.cache, None, fresh_eval_impressions=0)
        with self.assertRaisesRegex(ValueError, "only 4 available"):
            build_symbolic_projection(self.cache, self.archive(), fresh_eval_impressions=5)


if __name__ == "__main__":
    unittest.main()
