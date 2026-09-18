from __future__ import annotations

import io
import json
import math
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

from recommendation.adapters.mind import (
    MindDataError,
    entity_long_mean_similarity,
    fit_title_idf_model,
    history_feature_context,
    load_mind,
    safe_metta_symbol,
    subcategory_transition_score,
    title_history_idf_jaccard,
)

try:
    import h5py
    import numpy as np
except ImportError:  # The adapter itself still supports archives without HDF5.
    h5py = None
    np = None


def news_row(news_id, category, subcategory, title):
    return "\t".join(
        [news_id, category, subcategory, title, f"Abstract for {title}",
         f"https://example.test/{news_id}", "[]", "[]"]
    )


class MindAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.dev = self.root / "dev"
        self.train.mkdir(); self.dev.mkdir()

        common_news = [
            news_row("N1", "Sports", "Football", "Home team wins"),
            news_row("N2", "Tech", "AI", 'Models say "hello"'),
            news_row("N3", "Tech", "Gadgets", "New device"),
        ]
        (self.train / "news.tsv").write_text("\n".join(common_news) + "\n", encoding="utf-8")
        (self.dev / "news.tsv").write_text(
            "\n".join(common_news + [news_row("N4", "Sports", "Tennis", "Final result")]) + "\n",
            encoding="utf-8",
        )
        (self.train / "behaviors.tsv").write_text(
            "\n".join([
                "I2\tU1\t11/14/2019 10:00:00 AM\tN1\tN2-1 N3-0",
                "I1\tU2\t11/13/2019 09:00:00 AM\tN2\tN1-1 N3-0",
            ]) + "\n",
            encoding="utf-8",
        )
        # Reverse chronological file order deliberately: output must be temporal.
        (self.dev / "behaviors.tsv").write_text(
            "\n".join([
                "D2\tU1\t11/16/2019 11:00:00 AM\tN1 N2\tN3-0 N4-1",
                "D1\tU1\t11/16/2019 09:00:00 AM\tN1\tN2-1 N4-0",
            ]) + "\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_real_schema_fixture_contract_and_temporal_impressions(self):
        data = load_mind(self.root, max_train_cases=4, max_eval_impressions=2, seed=19)

        self.assertEqual(len(data["events"]), 4)
        self.assertEqual([test["source_impression_id"] for test in data["tests"]], ["D1", "D2"])
        self.assertTrue(all(set(("user", "article", "action")) <= event.keys()
                            for event in data["events"]))
        self.assertTrue(all(set(("id", "title", "topic", "format")) <= article.keys()
                            for article in data["articles"]))

        first = data["tests"][0]
        source_by_id = {article["id"]: article["source_id"] for article in data["articles"]}
        self.assertEqual([source_by_id[item] for item in first["candidates"]], ["N2", "N4"])
        self.assertEqual([source_by_id[item] for item in first["relevant"]], ["N2"])
        contexts = {source_by_id[key]: value for key, value in first["candidate_context"].items()}
        self.assertEqual(contexts["N2"]["affinity"], "low")
        self.assertEqual(contexts["N4"]["affinity"], "high")
        self.assertEqual(contexts["N4"]["topic_affinity"], 1.0)

        u1 = safe_metta_symbol("U1", "user")
        # Training history N1 plus the training click N2 are legal pre-dev profile data.
        self.assertEqual(set(data["users"][u1]["topics"]), {"sports", "tech"})
        self.assertTrue(data["users"][u1]["history"])
        self.assertEqual(data["metadata"]["format_source"], "title length tier (short/medium/long)")

        feature_names = {
            "subcategory", "affinity_level", "recent_affinity", "long_affinity",
            "history_size_bucket", "entity_overlap", "entity_overlap_detail",
            "history_topic_count_bucket", "recent_topic_count_bucket", "topic_rank_bucket",
            "subcategory_affinity", "time_bucket", "ctr_bucket", "freshness_bucket",
            "position_bucket", "title_overlap_detail", "entity_recent_top1_similarity",
            "title_history_idf_jaccard", "entity_long_mean_similarity",
            "recent_subcategory_transition_score",
            "recent_subcategory_affinity_score", "topic_recency_score",
            "subcategory_recency_score", "topic_recency_bucket",
            "subcategory_recency_bucket",
        }
        self.assertTrue(all(feature_names <= event.keys() for event in data["events"]))
        self.assertTrue(all(
            feature_names <= context.keys()
            for test in data["tests"] for context in test["candidate_context"].values()
        ))
        self.assertEqual(first["candidate_context"][first["candidates"][0]]["time_bucket"], "morning")
        self.assertEqual(first["candidate_context"][first["candidates"][0]]["position_bucket"], "top")

    def test_native_profile_preserves_ordered_history_for_live_sequence_parity(self):
        data = load_mind(
            self.root, max_train_cases=None, max_eval_impressions=2, seed=19
        )
        test = data["tests"][-1]
        user = test["user"]
        profile = data["users"][user]
        articles = {article["id"]: article for article in data["articles"]}
        source_by_id = {
            article["id"]: article["source_id"] for article in data["articles"]
        }

        # The final selected native-MIND impression is D2 with N1 then N2 in
        # its causal history.  The live profile must retain that exact order,
        # rather than a topic set or a sorted collection.
        self.assertEqual(
            [source_by_id[article_id] for article_id in profile["history"]],
            ["N1", "N2"],
        )
        self.assertEqual(profile["history"], test["history"])

        candidate_id = test["candidates"][0]
        live = history_feature_context(
            articles[candidate_id],
            profile["history"],
            articles,
            entity_vectors=data["article_entity_vectors"],
            title_idf_model=data["title_idf_model"],
            transition_model=data["subcategory_transition_model"],
            hour=test["hour"],
        )
        replay = test["candidate_context"][candidate_id]
        sequence_keys = {
            "recent_subcategory_affinity_score",
            "topic_recency_score",
            "subcategory_recency_score",
            "topic_recency_bucket",
            "subcategory_recency_bucket",
        }
        self.assertEqual(
            {key: live[key] for key in sequence_keys},
            {key: replay[key] for key in sequence_keys},
        )

    def test_raw_features_are_temporal_and_use_only_preceding_history(self):
        def entity_news(news_id, category, subcategory, title, title_entities):
            return "\t".join([
                news_id, category, subcategory, title, f"Abstract for {title}",
                f"https://example.test/{news_id}", json.dumps(title_entities), "[]",
            ])

        rows = [
            entity_news("N1", "Sports", "Football", "History", [
                {"WdId": "Q1"}, {"WdId": "Q2"},
            ]),
            entity_news("N2", "Tech", "AI", "Candidate", [{"WdId": "Q1"}]),
            entity_news("N3", "Tech", "Gadgets", "Other", []),
            entity_news("N4", "Sports", "Tennis", "Overlap", [
                {"WdId": "Q1"}, {"WdId": "Q2"},
            ]),
        ]
        (self.train / "news.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (self.dev / "news.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        # Later impression deliberately appears first in the file. Its click
        # must not influence the earlier event's CTR feature.
        (self.train / "behaviors.tsv").write_text("\n".join([
            "LATE\tU1\t11/14/2019 08:00:00 PM\tN1\tN2-1 N3-0",
            "EARLY\tU1\t11/13/2019 01:00:00 AM\tN1\tN2-0 N3-1",
        ]) + "\n", encoding="utf-8")
        (self.dev / "behaviors.tsv").write_text(
            "D\tU1\t11/15/2019 02:00:00 PM\tN1\tN4-1 N2-0\n",
            encoding="utf-8",
        )
        data = load_mind(self.root, max_train_cases=None, max_eval_impressions=1)
        n2_events = [event for event in data["events"] if event["source_article_id"] == "N2"]
        self.assertEqual([event["source_impression_id"] for event in n2_events], ["EARLY", "LATE"])
        self.assertEqual(n2_events[0]["ctr_bucket"], "cold")
        self.assertEqual(n2_events[0]["freshness_bucket"], "new")
        self.assertEqual(n2_events[1]["ctr_bucket"], "medium")  # (0+1)/(1+10)
        self.assertEqual(n2_events[1]["freshness_bucket"], "recent")
        # N3 is in the same impression as N2, so neither label can affect the
        # other's pre-impression context.
        early = [event for event in data["events"] if event["source_impression_id"] == "EARLY"]
        self.assertEqual({event["ctr_bucket"] for event in early}, {"cold"})
        self.assertEqual({event["freshness_bucket"] for event in early}, {"new"})

        articles = {article["source_id"]: article["id"] for article in data["articles"]}
        context = data["tests"][0]["candidate_context"]
        self.assertEqual(context[articles["N4"]]["entity_overlap"], "high")
        self.assertEqual(context[articles["N4"]]["subcategory"], "tennis")
        self.assertEqual(context[articles["N4"]]["history_size_bucket"], "light")
        self.assertEqual(context[articles["N4"]]["time_bucket"], "afternoon")
        self.assertEqual(context[articles["N2"]]["ctr_bucket"], "medium")

    def test_recent_and_long_affinity_have_distinct_bounded_windows(self):
        (self.train / "behaviors.tsv").write_text(
            "T\tU1\t11/14/2019 10:00:00 AM\tN1 N1 N2 N2 N2 N2 N2\tN3-1\n",
            encoding="utf-8",
        )
        (self.dev / "behaviors.tsv").write_text(
            "D\tU1\t11/15/2019 10:00:00 AM\tN1 N1 N2 N2 N2 N2 N2\tN1-1 N3-0\n",
            encoding="utf-8",
        )
        data = load_mind(self.root, max_train_cases=None, max_eval_impressions=1)
        source_by_id = {article["id"]: article["source_id"] for article in data["articles"]}
        contexts = {
            source_by_id[article_id]: context
            for article_id, context in data["tests"][0]["candidate_context"].items()
        }
        self.assertEqual(contexts["N1"]["recent_affinity"], "none")
        self.assertEqual(contexts["N1"]["long_affinity"], "medium")  # 2/7
        self.assertEqual(contexts["N1"]["affinity_level"], "medium")
        self.assertEqual(contexts["N3"]["recent_affinity"], "high")
        self.assertEqual(contexts["N3"]["long_affinity"], "high")
        self.assertAlmostEqual(contexts["N1"]["topic_recency_score"], 1 / 6)
        self.assertEqual(contexts["N1"]["topic_recency_bucket"], "older")
        self.assertEqual(contexts["N3"]["topic_recency_score"], 1.0)
        self.assertEqual(contexts["N3"]["topic_recency_bucket"], "immediate")
        self.assertEqual(contexts["N3"]["recent_subcategory_affinity_score"], 0.0)
        self.assertEqual(data["metadata"]["feature_thresholds"]["history_size_bucket"],
                         "cold=0; light=1-5; regular=6-20; heavy=21+")

    def test_sequence_evidence_handles_empty_and_missing_metadata(self):
        unknown = {
            "id":"unknown_candidate", "title":"candidate", "topic":"unknown",
            "category":"unknown", "subcategory":"unknown", "format":"short",
        }
        articles = {
            "unknown_candidate": unknown,
            "unknown_history": {
                "id":"unknown_history", "title":"history", "topic":"unknown",
                "category":"unknown", "subcategory":"unknown", "format":"short",
            },
        }
        missing = history_feature_context(
            unknown, ["unknown_history"], articles
        )
        self.assertEqual(missing["topic_affinity"], 0.0)
        self.assertEqual(missing["subcategory_affinity_score"], 0.0)
        self.assertIsNone(missing["topic_recency_score"])
        self.assertIsNone(missing["subcategory_recency_score"])
        self.assertIsNone(missing["recent_subcategory_affinity_score"])
        self.assertEqual(missing["topic_recency_bucket"], "unknown")
        self.assertEqual(missing["subcategory_recency_bucket"], "unknown")

        known = {
            "id":"known", "title":"known", "topic":"news",
            "category":"news", "subcategory":"world", "format":"short",
        }
        empty = history_feature_context(known, [], {"known":known})
        self.assertEqual(empty["topic_recency_score"], 0.0)
        self.assertEqual(empty["subcategory_recency_score"], 0.0)
        self.assertEqual(empty["recent_subcategory_affinity_score"], 0.0)
        self.assertEqual(empty["topic_recency_bucket"], "none")
        self.assertEqual(empty["subcategory_recency_bucket"], "none")

    def test_seeded_sampling_is_bounded_reproducible_and_keeps_whole_impressions(self):
        # Add enough rows to force both reservoirs to replace entries.
        train_rows = []
        dev_rows = []
        for index in range(12):
            train_rows.append(
                f"T{index}\tU{index % 3}\t11/14/2019 {index % 12 + 1:02d}:00:00 AM\tN1\tN2-1 N3-0"
            )
        for index in range(8):
            dev_rows.append(
                f"V{index}\tU{index % 2}\t11/17/2019 {index % 12 + 1:02d}:00:00 AM\tN1\tN2-1 N3-0 N4-0"
            )
        (self.train / "behaviors.tsv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
        (self.dev / "behaviors.tsv").write_text("\n".join(dev_rows) + "\n", encoding="utf-8")

        first = load_mind(self.root, max_train_cases=5, max_eval_impressions=3, seed=41)
        second = load_mind(self.root, max_train_cases=5, max_eval_impressions=3, seed=41)
        self.assertEqual(first, second)
        self.assertEqual(len(first["events"]), 5)
        self.assertEqual(len(first["tests"]), 3)
        self.assertTrue(all(len(test["candidates"]) == 3 for test in first["tests"]))
        self.assertEqual(
            [event["timestamp"] for event in first["events"]],
            sorted(event["timestamp"] for event in first["events"]),
        )
        self.assertEqual(
            [test["timestamp"] for test in first["tests"]],
            sorted(test["timestamp"] for test in first["tests"]),
        )

    def test_archive_style_split_names_are_discovered(self):
        train = self.root / "MINDsmall_train"
        valid = self.root / "MINDsmall_valid"
        self.train.rename(train); self.dev.rename(valid)
        data = load_mind(self.root, max_train_cases=2, max_eval_impressions=1)
        self.assertEqual(data["metadata"]["train_split"], "MINDsmall_train")
        self.assertEqual(data["metadata"]["eval_split"], "MINDsmall_valid")

    def test_metta_normalization_and_schema_errors(self):
        symbol_a = safe_metta_symbol("user / one", "user")
        symbol_b = safe_metta_symbol("user : one", "user")
        self.assertRegex(symbol_a, r"^[A-Za-z_][A-Za-z0-9_]*$")
        self.assertNotEqual(symbol_a, symbol_b)
        (self.train / "news.tsv").write_text("bad\trow\n", encoding="utf-8")
        with self.assertRaisesRegex(MindDataError, "expected 8 news columns"):
            load_mind(self.root, max_train_cases=1, max_eval_impressions=1)

    def test_limits_must_be_positive(self):
        with self.assertRaises(ValueError):
            load_mind(self.root, max_train_cases=0)
        with self.assertRaises(ValueError):
            load_mind(self.root, max_eval_impressions=-1)

    def test_portable_semantic_evidence_math_and_missing_values(self):
        articles = {
            "A": {"title": "alpha beta"},
            "B": {"title": "alpha gamma"},
            "C": {"title": "alpha beta"},
            "EMPTY": {"title": "the and"},
        }
        model = fit_title_idf_model(articles, ["A", "B"])
        self.assertEqual(model["document_count"], 2)
        self.assertEqual(model["idf"]["alpha"], 1.0)
        expected = 1.0 / (
            1.0 + model["idf"]["beta"] + model["idf"]["gamma"]
        )
        self.assertAlmostEqual(
            title_history_idf_jaccard(articles["C"], ["B"], articles, model),
            expected,
            places=7,
        )
        # Maximum is per history item, rather than Jaccard against one merged
        # history vocabulary.
        self.assertEqual(
            title_history_idf_jaccard(
                articles["C"], ["B", "A"], articles, model
            ),
            1.0,
        )
        self.assertIsNone(
            title_history_idf_jaccard(articles["EMPTY"], ["A"], articles, model)
        )
        self.assertIsNone(
            title_history_idf_jaccard(articles["C"], ["missing"], articles, model)
        )

        vectors = {
            "C": [2.0, 0.0],
            "same": [3.0, 0.0],
            "orthogonal": [0.0, 4.0],
            "bad": [0.0, 0.0],
        }
        self.assertEqual(
            entity_long_mean_similarity(
                "C", ["same", "missing", "orthogonal"], vectors
            ),
            0.5,
        )
        self.assertIsNone(entity_long_mean_similarity("missing", ["same"], vectors))
        self.assertIsNone(entity_long_mean_similarity("C", ["bad"], vectors))

    def test_live_context_projects_text_vectors_into_semantic_facts(self):
        articles={
            "candidate":{"id":"candidate","title":"Candidate","topic":"news",
                         "subcategory":"world","format":"short"},
            "near":{"id":"near","title":"Near","topic":"news",
                    "subcategory":"world","format":"short"},
            "far":{"id":"far","title":"Far","topic":"sports",
                   "subcategory":"other","format":"short"},
        }
        context=history_feature_context(
            articles["candidate"],["far","near"],articles,
            text_semantic_vectors={
                "candidate":[1.0,0.0],"near":[1.0,0.0],"far":[0.0,1.0],
            },
        )
        self.assertEqual(context["text_semantic_top1_similarity"],1.0)
        self.assertEqual(context["text_semantic_top3_mean_similarity"],0.75)
        self.assertEqual(context["text_semantic_coverage"],1.0)
        self.assertEqual(
            context["text_semantic_attention_t8_similarity"],
            round((math.exp(8.0) + 0.5) / (math.exp(8.0) + 1.0), 8),
        )
        self.assertEqual(
            context["text_semantic_attention_t12_similarity"],
            round((math.exp(12.0) + 0.5) / (math.exp(12.0) + 1.0), 8),
        )

    def test_streamable_reczoo_archive_preserves_impression_groups(self):
        archive_path = self.root / "MIND_small_x1.zip"
        news_header = "news_id\tcat\tsub_cat\ttitle_entities\tabstract_entities\ttitle\tabstract"
        csv_header = "imp_id,click,hour,user_id,news_id,cat,sub_cat,title_entities,abstract_entities,news_his,cat_his,subcat_his"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("news_corpus.tsv", "\n".join([
                news_header,
                "N1\tsports\tfootball\t\t\tA short title\tOne",
                "N2\ttech\tai\t\t\tA medium sized article title for testing\tTwo",
                "N3\tnews\tworld\t\t\tA deliberately rather long article title for the length tier test case\tThree",
            ]) + "\n")
            archive.writestr("train.csv", "\n".join([
                csv_header,
                "1,1,9AM,U1,N1,sports,football,,,N2,tech,ai",
                "1,0,9AM,U1,N2,tech,ai,,,N2,tech,ai",
                "2,1,8PM,U1,N1,sports,football,,,N2,tech,ai",
            ]) + "\n")
            archive.writestr("valid.csv", "\n".join([
                csv_header,
                "9,0,10AM,U1,N1,sports,football,,,N2,tech,ai",
                "9,1,10AM,U1,N3,news,world,,,N2,tech,ai",
                "10,1,11AM,U2,N2,tech,ai,,,N1,sports,football",
            ]) + "\n")
        data = load_mind(archive_path, max_train_cases=3, max_eval_impressions=1)
        self.assertEqual(len(data["events"]), 3)
        self.assertEqual(len(data["tests"]), 1)
        self.assertEqual(len(data["tests"][0]["candidates"]), 2)
        self.assertEqual(len(data["tests"][0]["relevant"]), 1)
        self.assertEqual(data["metadata"]["projection"], "RecZoo MIND_small_x1")
        self.assertEqual(data["title_idf_model"]["document_count"], 2)
        self.assertEqual(data["metadata"]["title_idf_documents"], 2)
        # N3 exists only as a validation exposure, so its distinctive token is
        # never allowed into the training-fitted IDF vocabulary.
        self.assertNotIn("deliberately", data["title_idf_model"]["idf"])
        self.assertEqual(data["events"][1]["affinity"], "high")
        self.assertEqual(data["events"][0]["ctr_bucket"], "cold")
        self.assertEqual(data["events"][1]["ctr_bucket"], "cold")
        self.assertEqual(data["events"][2]["ctr_bucket"], "medium")
        self.assertEqual(data["events"][2]["freshness_bucket"], "recent")
        self.assertEqual(data["events"][2]["time_bucket"], "evening")
        eval_context = data["tests"][0]["candidate_context"]
        n1_id = next(article["id"] for article in data["articles"]
                     if article["source_id"] == "N1")
        self.assertEqual(eval_context[n1_id]["ctr_bucket"], "high")
        model = data["subcategory_transition_model"]
        self.assertEqual(data["metadata"]["subcategory_transition_cells"], 2)
        expected_candidate = (2 + 20 * (2 / 3)) / 22
        expected_transition = (2 + 10 * expected_candidate) / 12
        self.assertAlmostEqual(
            subcategory_transition_score("football", ["ai"], model),
            expected_transition,
            places=7,
        )
        self.assertAlmostEqual(
            eval_context[n1_id]["recent_subcategory_transition_score"],
            expected_transition,
            places=7,
        )
        training_n1=next(event for event in data["events"]
                         if event["source_article_id"]=="N1")
        self.assertNotEqual(
            training_n1["recent_subcategory_transition_score"],
            eval_context[n1_id]["recent_subcategory_transition_score"],
        )
        first_impression=[event for event in data["events"]
                          if event["source_impression_id"]=="1"]
        first_by_source={event["source_article_id"]:event for event in first_impression}
        self.assertGreater(first_by_source["N1"]["title_history_idf_jaccard"], 0.0)
        self.assertEqual(first_by_source["N2"]["title_history_idf_jaccard"], 1.0)
        # Before the first impression there is no outcome evidence.  Both
        # opposite-label candidates must therefore receive the same cold
        # posterior from a model frozen for that complete impression.
        self.assertEqual(
            {event["recent_subcategory_transition_score"]
             for event in first_impression},
            {0.0},
        )
        self.assertEqual(data["metadata"]["transition_training_encoding"],
                         "prequential-impression-v1")

    @unittest.skipIf(h5py is None or np is None, "h5py is not installed")
    def test_reczoo_entity_similarity_uses_normalized_mean_top1_and_recent5(self):
        archive_path = self.root / "entity_mind.zip"
        news_header = "news_id\tcat\tsub_cat\ttitle_entities\tabstract_entities\ttitle\tabstract"
        csv_header = "imp_id,click,hour,user_id,news_id,cat,sub_cat,title_entities,abstract_entities,news_his,cat_his,subcat_his"

        def unit_vector(cosine):
            vector = np.zeros(100, dtype=np.float64)
            vector[0] = cosine
            vector[1] = math.sqrt(1.0 - cosine * cosine)
            return vector

        keys = ["QC", "QH1", "QH2", "QO", "QH4"]
        values = np.stack([
            unit_vector(1.0), unit_vector(0.65), unit_vector(0.6),
            unit_vector(0.0), unit_vector(0.2),
        ])
        hdf5 = io.BytesIO()
        with h5py.File(hdf5, "w") as embeddings:
            embeddings.create_dataset(
                "key", data=np.asarray([key.encode("utf-8") for key in keys])
            )
            embeddings.create_dataset("value", data=values)

        history = "H0^H1^H2^H3^H4^H5"
        history_topics = "^".join(["news"] * 6)
        history_subcategories = "^".join(["test"] * 6)
        rows = [
            news_header,
            "C\tnews\ttest\tQC\t\tCandidate\tCandidate abstract",
            "H0\tnews\ttest\tQC\t\tOutside recent window\tHistory",
            "H1\tnews\ttest\tQH1\t\tRecent one\tHistory",
            "H2\tnews\ttest\tQH2\t\tRecent two\tHistory",
            # Its mean [1, 1] vector must be normalized before cosine use.
            "H3\tnews\ttest\tQC\tQO\tRecent combined\tHistory",
            "H4\tnews\ttest\tQH4\t\tRecent four\tHistory",
            "H5\tnews\ttest\tQO\t\tRecent five\tHistory",
            "X\tnews\ttest\t\t\tNo entities\tHistory",
        ]
        train = [
            csv_header,
            f"T,1,9AM,U,C,news,test,QC,,{history},{history_topics},{history_subcategories}",
            f"T,0,9AM,U,X,news,test,,,{history},{history_topics},{history_subcategories}",
        ]
        valid = [
            csv_header,
            f"V,1,10AM,U,C,news,test,QC,,{history},{history_topics},{history_subcategories}",
            f"V,0,10AM,U,X,news,test,,,{history},{history_topics},{history_subcategories}",
        ]
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("news_corpus.tsv", "\n".join(rows) + "\n")
            archive.writestr("train.csv", "\n".join(train) + "\n")
            archive.writestr("valid.csv", "\n".join(valid) + "\n")
            archive.writestr("entity_emb_dim100.h5", hdf5.getvalue())

        data = load_mind(
            archive_path, max_train_cases=2, max_eval_impressions=1, seed=3
        )
        events = {event["source_article_id"]: event for event in data["events"]}
        # The outside-window similarity is 1.0. Inside the last five, the
        # normalized title+abstract mean vector is the maximum at sqrt(.5).
        expected = math.sqrt(0.5)
        self.assertAlmostEqual(
            events["C"]["entity_recent_top1_similarity"], expected, places=7
        )
        expected_long = (1.0 + 0.65 + 0.6 + math.sqrt(0.5) + 0.2 + 0.0) / 6.0
        self.assertAlmostEqual(
            events["C"]["entity_long_mean_similarity"], expected_long, places=7
        )
        self.assertIsNone(events["X"]["entity_recent_top1_similarity"])
        self.assertIsNone(events["X"]["entity_long_mean_similarity"])
        article_ids = {
            article["source_id"]: article["id"] for article in data["articles"]
        }
        contexts = data["tests"][0]["candidate_context"]
        self.assertAlmostEqual(
            contexts[article_ids["C"]]["entity_recent_top1_similarity"],
            expected,
            places=7,
        )
        self.assertAlmostEqual(
            contexts[article_ids["C"]]["entity_long_mean_similarity"],
            expected_long,
            places=7,
        )
        self.assertIsNone(
            contexts[article_ids["X"]]["entity_recent_top1_similarity"]
        )
        self.assertEqual(data["metadata"]["entity_vector_articles"], 7)

        articles={article["id"]:article for article in data["articles"]}
        test=data["tests"][0]; candidate_id=article_ids["C"]
        live_context=history_feature_context(
            articles[candidate_id],data["users"][test["user"]]["history"],articles,
            entity_vectors=data["article_entity_vectors"],
            title_idf_model=data["title_idf_model"],
            transition_model=data["subcategory_transition_model"],hour="10AM",
        )
        replay_context=contexts[candidate_id]
        parity_keys={
            "recent_affinity","long_affinity","topic_affinity",
            "recent_topic_affinity","subcategory_affinity_score",
            "history_topic_count_bucket",
            "recent_topic_count_bucket","subcategory_affinity",
            "entity_overlap_detail","title_overlap_detail",
            "title_history_idf_jaccard", "entity_recent_top1_similarity",
            "entity_long_mean_similarity", "recent_subcategory_transition_score",
            "recent_subcategory_affinity_score", "topic_recency_score",
            "subcategory_recency_score", "topic_recency_bucket",
            "subcategory_recency_bucket",
        }
        self.assertEqual({key:live_context[key] for key in parity_keys},
                         {key:replay_context[key] for key in parity_keys})

        cached = load_mind(
            archive_path, max_train_cases=2, max_eval_impressions=1, seed=3
        )
        self.assertEqual(cached, data)

    @unittest.skipIf(h5py is None or np is None, "h5py is not installed")
    def test_live_replay_preserve_unknown_history_positions_and_id_alignment(self):
        archive_path = self.root / "unknown_history_mind.zip"
        news_header = "news_id\tcat\tsub_cat\ttitle_entities\tabstract_entities\ttitle\tabstract"
        csv_header = "imp_id,click,hour,user_id,news_id,cat,sub_cat,title_entities,abstract_entities,news_his,cat_his,subcat_his"

        values = np.zeros((3, 100), dtype=np.float64)
        values[0, 0] = 1.0  # Candidate.
        values[1, 0] = 1.0  # Strong but outside the raw recent-five window.
        values[2, 1] = 1.0  # Inside the window and orthogonal.
        hdf5 = io.BytesIO()
        with h5py.File(hdf5, "w") as embeddings:
            embeddings.create_dataset(
                "key", data=np.asarray([b"QC", b"QOUT", b"QKNOWN"])
            )
            embeddings.create_dataset("value", data=values)

        raw_history = "OUT^MISSING1^MISSING2^MISSING3^MISSING4^KNOWN"
        # These parallel columns are deliberately wrong. Context construction
        # must align metadata through news_his IDs instead of trusting them.
        wrong_categories = "wrong^wrong^wrong^wrong^wrong^wrong"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("news_corpus.tsv", "\n".join([
                news_header,
                "C\tcandidate\tcand\tQC\t\tShared lexical signal\tCandidate",
                "OUT\tcandidate\tout\tQOUT\t\tShared lexical signal\tOutside",
                "KNOWN\tother\tknown\tQKNOWN\t\tUnrelated phrase\tKnown",
            ]) + "\n")
            archive.writestr("train.csv", "\n".join([
                csv_header,
                "T1,1,9AM,U,C,candidate,cand,QC,,OUT,wrong,wrong",
                "T2,0,9AM,U,C,candidate,cand,QC,,KNOWN,wrong,wrong",
            ]) + "\n")
            archive.writestr("valid.csv", "\n".join([
                csv_header,
                f"V,1,10AM,U,C,candidate,cand,QC,,{raw_history},{wrong_categories},{wrong_categories}",
            ]) + "\n")
            archive.writestr("entity_emb_dim100.h5", hdf5.getvalue())

        data = load_mind(
            archive_path, max_train_cases=2, max_eval_impressions=1, seed=5
        )
        test = data["tests"][0]
        self.assertEqual(len(test["history"]), 6)
        candidate_id = test["candidates"][0]
        replay = test["candidate_context"][candidate_id]
        self.assertEqual(replay["entity_recent_top1_similarity"], 0.0)
        self.assertEqual(replay["entity_long_mean_similarity"], 0.5)
        self.assertEqual(replay["recent_topic_affinity"], 0.0)
        self.assertEqual(replay["topic_affinity"], 0.5)
        self.assertEqual(replay["recent_history_subcategories"], ["known"])
        self.assertEqual(replay["title_history_idf_jaccard"], 1.0)
        self.assertAlmostEqual(
            replay["recent_subcategory_transition_score"],
            subcategory_transition_score(
                "cand", ["known"], data["subcategory_transition_model"]
            ),
            places=7,
        )

        articles = {article["id"]: article for article in data["articles"]}
        live = history_feature_context(
            articles[candidate_id],
            test["history"],
            articles,
            entity_vectors=data["article_entity_vectors"],
            title_idf_model=data["title_idf_model"],
            transition_model=data["subcategory_transition_model"],
            hour="10AM",
        )
        parity_keys = {
            "recent_affinity", "long_affinity", "topic_affinity",
            "recent_topic_affinity", "subcategory_affinity_score",
            "recent_history_subcategories", "title_history_idf_jaccard",
            "entity_recent_top1_similarity", "entity_long_mean_similarity",
            "recent_subcategory_transition_score",
            "recent_subcategory_affinity_score", "topic_recency_score",
            "subcategory_recency_score", "topic_recency_bucket",
            "subcategory_recency_bucket",
        }
        self.assertEqual(
            {key: live[key] for key in parity_keys},
            {key: replay[key] for key in parity_keys},
        )

    def test_reczoo_hash_sampling_scans_population_and_preserves_causal_context(self):
        archive_path = self.root / "representative_mind.zip"
        news_header = "news_id\tcat\tsub_cat\ttitle_entities\tabstract_entities\ttitle\tabstract"
        csv_header = "imp_id,click,hour,user_id,news_id,cat,sub_cat,title_entities,abstract_entities,news_his,cat_his,subcat_his"
        train_rows = [csv_header]
        for index in range(12):
            train_rows.extend([
                f"T{index},0,9AM,U{index},N1,sports,football,,,N2,tech,ai",
                f"T{index},1,9AM,U{index},N2,tech,ai,,,N2,tech,ai",
            ])
        valid_rows = [csv_header]
        for index in range(8):
            valid_rows.extend([
                f"V{index},0,10AM,VU{index},N1,sports,football,,,N2,tech,ai",
                f"V{index},1,10AM,VU{index},N3,news,world,,,N2,tech,ai",
            ])
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("news_corpus.tsv", "\n".join([
                news_header,
                "N1\tsports\tfootball\t\t\tSports title\tOne",
                "N2\ttech\tai\t\t\tTechnology title\tTwo",
                "N3\tnews\tworld\t\t\tWorld news title\tThree",
            ]) + "\n")
            archive.writestr("train.csv", "\n".join(train_rows) + "\n")
            archive.writestr("valid.csv", "\n".join(valid_rows) + "\n")

        first = load_mind(
            archive_path, max_train_cases=4, max_eval_impressions=2, seed=0
        )
        repeated = load_mind(
            archive_path, max_train_cases=4, max_eval_impressions=2, seed=0
        )
        other_seed = load_mind(
            archive_path, max_train_cases=4, max_eval_impressions=2, seed=1
        )

        self.assertEqual(first, repeated)
        self.assertEqual(first["metadata"]["sampling"],
                         "seeded hash-priority whole impressions")
        self.assertEqual(first["metadata"]["train_interactions_scanned"], 24)
        self.assertEqual(first["metadata"]["train_impressions_scanned"], 12)
        self.assertEqual(first["metadata"]["eval_interactions_scanned"], 16)
        self.assertEqual(first["metadata"]["eval_impressions_scanned"], 8)
        self.assertEqual(first["metadata"]["train_cases_loaded"], 4)
        self.assertEqual(first["metadata"]["train_impressions_loaded"], 2)

        train_ids = [event["source_impression_id"] for event in first["events"]]
        eval_ids = [test["source_impression_id"] for test in first["tests"]]
        self.assertEqual(train_ids, ["T6", "T6", "T11", "T11"])
        self.assertEqual(eval_ids, ["V0", "V3"])
        self.assertNotEqual(
            {event["source_impression_id"] for event in first["events"]},
            {event["source_impression_id"] for event in other_seed["events"]},
        )
        self.assertEqual(
            [event["timestamp"] for event in first["events"]],
            sorted(event["timestamp"] for event in first["events"]),
        )

        # T6 and T11 retain priors from every preceding source impression,
        # including impressions that the bounded sampler did not retain.
        n1_train = {
            event["source_impression_id"]: event["ctr_bucket"]
            for event in first["events"] if event["source_article_id"] == "N1"
        }
        self.assertEqual(n1_train, {"T6": "medium", "T11": "low"})

        # Validation sees all twelve scanned N1 skips, not only the two sampled
        # training impressions: Beta(1, 9) gives 1 / (12 + 10) < 0.05.
        n1_id = next(
            article["id"] for article in first["articles"]
            if article["source_id"] == "N1"
        )
        self.assertTrue(all(
            test["candidate_context"][n1_id]["ctr_bucket"] == "low"
            for test in first["tests"]
        ))


if __name__ == "__main__":
    unittest.main()
