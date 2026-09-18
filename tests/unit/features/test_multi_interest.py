import math
import unittest
from unittest.mock import patch

import recommendation.core.multi_interest as multi_interest

from recommendation.core.multi_interest import (
    MultiInterestConfig,
    build_multi_interest_facts,
    build_semantic_match_facts,
)


class MultiInterestFactsTest(unittest.TestCase):
    def setUp(self):
        self.articles = {
            "h1": {"id":"h1", "topic":"Sports", "subcategory":"Football",
                   "entities":["Q_TEAM"], "semantic_vector":[1.0, 0.0]},
            "h2": {"id":"h2", "topic":"Tech", "subcategory":"AI",
                   "entities":["Q_AI"], "semantic_vector":[0.0, 1.0]},
            "h3": {"id":"h3", "topic":"Sports", "subcategory":"Football",
                   "entities":["Q_TEAM", "Q_PLAYER"], "semantic_vector":[0.98, 0.02]},
            "future": {"id":"future", "topic":"ForbiddenFuture",
                       "subcategory":"Leak", "entities":["Q_LEAK"],
                       "semantic_vector":[-1.0, 0.0], "action":"click"},
        }
        self.candidate = {
            "id":"candidate-secret", "topic":"Sports", "subcategory":"Football",
            "entities":["Q_TEAM"], "semantic_vector":[1.0, 0.0],
            "action":"skip", "clicked":False,
        }

    def test_fixed_identifier_free_mining_vocabulary(self):
        facts = build_multi_interest_facts(
            self.candidate, ["h1", "h2", "h3"], self.articles
        )
        serialized = repr(facts).casefold()
        for forbidden in (
            "candidate-secret", "h1", "h2", "h3", "sports", "football",
            "q_team", "q_ai", "click", "skip",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(facts["mi_topic_candidate_match_rank"], "top")
        self.assertEqual(facts["mi_entity_candidate_match_rank"], "top")
        self.assertIn("mi_topic_slot3_candidate_match", facts)
        self.assertIn("mi_entity_slot4_candidate_match", facts)
        for value in facts.values():
            if isinstance(value, float):
                self.assertTrue(0.0 <= value <= 1.0)
            else:
                self.assertLessEqual(len(value), 16)

    def test_outcomes_and_unseen_future_articles_cannot_change_facts(self):
        before = build_multi_interest_facts(
            self.candidate, ["h1", "h2", "h3"], self.articles
        )
        changed_candidate = {**self.candidate, "action":"click", "clicked":True,
                             "label":1}
        changed_articles = {**self.articles, "future":{
            **self.articles["future"], "action":"skip", "label":0,
        }}
        after = build_multi_interest_facts(
            changed_candidate, ["h1", "h2", "h3"], changed_articles
        )
        self.assertEqual(before, after)

    def test_support_recency_and_candidate_match_are_candidate_aware(self):
        facts = build_multi_interest_facts(
            self.candidate, ["h1", "h2", "h3"], self.articles,
            config=MultiInterestConfig(recency_decay=0.5),
        )
        self.assertEqual(facts["mi_topic_slot1_candidate_match"], "yes")
        self.assertEqual(facts["mi_topic_slot1_recency"], "latest")
        self.assertAlmostEqual(facts["mi_topic_slot1_support_score"], 2 / 3)
        tech_candidate = {**self.candidate, "topic":"Tech", "subcategory":"AI",
                          "entities":["Q_AI"], "semantic_vector":[0.0, 1.0]}
        tech = build_multi_interest_facts(
            tech_candidate, ["h1", "h2", "h3"], self.articles,
            config=MultiInterestConfig(recency_decay=0.5),
        )
        self.assertEqual(tech["mi_topic_candidate_match_rank"], "secondary")
        self.assertEqual(tech["mi_entity_candidate_match_rank"], "secondary")

    def test_multiple_semantic_prototypes_preserve_distinct_interests(self):
        vectors = {
            article_id: article["semantic_vector"]
            for article_id, article in self.articles.items()
        }
        vectors[self.candidate["id"]] = self.candidate["semantic_vector"]
        metadata_only = {
            article_id: {
                key: value for key, value in article.items()
                if key != "semantic_vector"
            }
            for article_id, article in self.articles.items()
        }
        candidate_metadata = {
            key: value for key, value in self.candidate.items()
            if key != "semantic_vector"
        }
        facts = build_multi_interest_facts(
            candidate_metadata, ["h1", "h2", "h3"], metadata_only,
            semantic_vectors=vectors,
            config=MultiInterestConfig(semantic_cluster_threshold=0.8),
        )
        self.assertEqual(facts["mi_topic_prototype_count"], "few")
        self.assertEqual(facts["mi_subcategory_prototype_count"], "few")
        self.assertEqual(facts["mi_entity_prototype_count"], "few")
        self.assertEqual(facts["mi_semantic_prototype_count"], "few")
        self.assertEqual(facts["mi_semantic_slot1_similarity"], "high")
        self.assertGreater(facts["mi_semantic_top1_similarity"], 0.99)
        self.assertLess(
            facts["mi_semantic_slot2_similarity_score"],
            facts["mi_semantic_slot1_similarity_score"],
        )
        self.assertGreater(facts["mi_semantic_topk_mean_similarity"], 0.70)

    def test_semantic_top1_searches_beyond_exported_slots(self):
        tech_candidate = {
            **self.candidate,
            "id":"tech-candidate",
            "topic":"Tech",
            "subcategory":"AI",
            "entities":["Q_AI"],
            "semantic_vector":[0.0, 1.0],
        }
        facts = build_multi_interest_facts(
            tech_candidate,
            ["h1", "h2", "h3"],
            self.articles,
            config=MultiInterestConfig(
                max_semantic_prototypes=1,
                semantic_cluster_threshold=0.8,
            ),
        )
        # The single exported slot is the larger sports cluster, while the
        # exact tech match is a smaller second cluster. Top-1 is an aggregate
        # and must still find that candidate-aware secondary interest.
        self.assertLess(facts["mi_semantic_slot1_similarity_score"], 0.6)
        self.assertAlmostEqual(facts["mi_semantic_top1_similarity"], 1.0)

    def test_missing_candidate_vector_keeps_semantic_aggregates_unavailable(self):
        candidate = {
            key:value for key,value in self.candidate.items()
            if key != "semantic_vector"
        }
        facts = build_multi_interest_facts(
            candidate, ["h1", "h2", "h3"], self.articles
        )
        self.assertEqual(facts["mi_semantic_available"], "no")
        self.assertEqual(facts["mi_semantic_top1_similarity"], 0.0)
        self.assertEqual(facts["mi_semantic_topk_mean_similarity"], 0.0)

    def test_unknown_history_keeps_original_recency_positions(self):
        facts = build_multi_interest_facts(
            self.candidate, ["h1", "missing", "missing2", "h3"], self.articles,
            config=MultiInterestConfig(recency_decay=0.5),
        )
        # h3 is newest and h1 remains lag three; missing metadata is not removed
        # before lags are calculated.
        self.assertEqual(facts["mi_topic_slot1_recency"], "latest")
        self.assertEqual(facts["mi_history_size"], "few")

    def test_recent_window_is_applied_before_unknown_id_resolution(self):
        facts = build_multi_interest_facts(
            self.candidate, ["h1", "missing", "missing2", "h3"], self.articles,
            config=MultiInterestConfig(recent_window=3, recency_decay=0.5),
        )
        # h1 is outside the last three interactions and cannot inflate support.
        self.assertEqual(facts["mi_history_size"], "one")
        self.assertEqual(facts["mi_topic_slot1_support_score"], 1.0)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            MultiInterestConfig(max_topic_prototypes=0)
        with self.assertRaises(ValueError):
            MultiInterestConfig(recency_decay=0.0)

    def test_generic_semantic_match_is_causal_bounded_and_identifier_free(self):
        vectors = {
            "candidate-secret":[1.0, 0.0],
            "opposite-secret":[-1.0, 0.0],
            "orthogonal-secret":[0.0, 1.0],
            "same-secret":[1.0, 0.0],
        }
        facts = build_semantic_match_facts(
            "candidate-secret",
            ["missing-secret", "opposite-secret", "orthogonal-secret", "same-secret"],
            vectors,
            recency_decay=0.5,
        )
        self.assertEqual(facts["semantic_match_match_available"], "yes")
        self.assertAlmostEqual(facts["semantic_match_coverage"], 0.75)
        self.assertAlmostEqual(facts["semantic_match_top1_similarity"], 1.0)
        self.assertAlmostEqual(facts["semantic_match_top3_mean_similarity"], 0.5)
        # Missing vector remains in its original history position: weights are
        # 0.25, 0.5, and 1.0 for the three available trailing observations.
        self.assertAlmostEqual(
            facts["semantic_match_last20_recency_decayed_similarity"],
            1.25 / 1.75,
            places=7,
        )
        serialized = repr(facts)
        self.assertNotIn("candidate-secret", serialized)
        self.assertNotIn("same-secret", serialized)
        self.assertTrue(all(
            not isinstance(value, float) or 0.0 <= value <= 1.0
            for value in facts.values()
        ))

    def test_generic_semantic_match_reports_missing_and_rejects_bad_options(self):
        facts = build_semantic_match_facts(
            "missing-candidate", ["known"], {"known":[1.0, 0.0]}
        )
        self.assertEqual(facts["semantic_match_candidate_available"], "no")
        self.assertEqual(facts["semantic_match_match_available"], "no")
        self.assertEqual(facts["semantic_match_top1_similarity"], 0.0)
        self.assertEqual(facts["semantic_match_attention_t8_similarity"], 0.0)
        self.assertEqual(facts["semantic_match_attention_t12_similarity"], 0.0)
        with self.assertRaises(ValueError):
            build_semantic_match_facts("c", [], {}, prefix="bad prefix")
        with self.assertRaises(ValueError):
            build_semantic_match_facts("c", [], {}, recency_decay=0.0)

    def test_generic_semantic_softmax_attention_is_exact_and_causal(self):
        vectors = {
            "candidate": [1.0, 0.0],
            "same": [1.0, 0.0],       # cosine 1, mapped similarity 1
            "orthogonal": [0.0, 1.0], # cosine 0, mapped similarity 1/2
            "opposite": [-1.0, 0.0],  # cosine -1, mapped similarity 0
            "unseen-future": [1.0, 0.0],
        }
        history = ["opposite", "orthogonal", "same"]
        facts = build_semantic_match_facts("candidate", history, vectors)

        def expected(temperature):
            # Stable softmax after subtracting max cosine=1.
            middle = math.exp(-temperature)
            lowest = math.exp(-2.0 * temperature)
            return round((1.0 + 0.5 * middle) / (1.0 + middle + lowest), 8)

        self.assertEqual(
            facts["semantic_match_attention_t8_similarity"], expected(8.0)
        )
        self.assertEqual(
            facts["semantic_match_attention_t12_similarity"], expected(12.0)
        )
        self.assertGreater(
            facts["semantic_match_attention_t12_similarity"],
            facts["semantic_match_attention_t8_similarity"],
        )
        # A vector present in the global map but absent from prior history is
        # never inspected; adding/removing it cannot affect causal facts.
        without_future = {key:value for key,value in vectors.items()
                          if key != "unseen-future"}
        self.assertEqual(
            facts,
            build_semantic_match_facts("candidate", history, without_future),
        )

    def test_softmax_attention_stdlib_and_numpy_paths_agree(self):
        vectors = {
            "candidate": [1.0, 0.0],
            "near": [0.8, 0.6],
            "far": [-0.6, 0.8],
        }
        accelerated = build_semantic_match_facts(
            "candidate", ["far", "near"], vectors
        )
        with patch.object(multi_interest, "_np", None):
            portable = build_semantic_match_facts(
                "candidate", ["far", "near"], vectors
            )
        self.assertEqual(
            accelerated["semantic_match_attention_t8_similarity"],
            portable["semantic_match_attention_t8_similarity"],
        )
        self.assertEqual(
            accelerated["semantic_match_attention_t12_similarity"],
            portable["semantic_match_attention_t12_similarity"],
        )


if __name__ == "__main__":
    unittest.main()
