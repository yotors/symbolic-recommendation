import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from recommendation.core.multi_interest import build_semantic_match_facts
from recommendation.features.semantic_workspace import (
    SEMANTIC_WORKSPACE_FEATURES,
    build_semantic_workspace_facts,
    fit_semantic_workspace_model,
)


class SemanticWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.vectors = {
            "candidate": [1.0, 0.0],
            "same": [1.0, 0.0],
            "other": [0.0, 1.0],
            "opposite": [-1.0, 0.0],
            "future": [0.5, -0.5],
        }
        self.model = fit_semantic_workspace_model({"train-a": [1.0, 0.0], "train-b": [0.0, 1.0]})

    def facts(self, history, candidate="candidate", vectors=None, model=None):
        return build_semantic_workspace_facts(candidate, history, vectors or self.vectors, model or self.model)

    def test_fit_normalizes_inputs_then_averages_without_normalizing_mean(self):
        model = fit_semantic_workspace_model([[10.0, 0.0], [0.0, 2.0]])
        self.assertEqual(model["mean_vector"], [0.5, 0.5])
        self.assertEqual(model["training_vector_count"], 2)
        self.assertEqual(model["dimensions"], 2)
        self.assertAlmostEqual(np.linalg.norm(model["mean_vector"]), math.sqrt(0.5))
        self.assertEqual(model, json.loads(json.dumps(model, allow_nan=False)))

    def test_fit_is_independent_of_ids_order_and_vector_scale(self):
        renamed = fit_semantic_workspace_model({"skip-secret": [0.0, 5.0], "click-secret": [3.0, 0.0]})
        self.assertEqual(self.model, renamed)
        self.assertNotIn("secret", repr(renamed))
        self.assertEqual(self.model, fit_semantic_workspace_model(iter([[0.0, 1.0], [1.0, 0.0]])))

    def test_distinct_articles_with_same_vector_preserve_multiplicity(self):
        model = fit_semantic_workspace_model({"a": [1, 0], "b": [1, 0], "c": [0, 1]})
        self.assertEqual(model["training_vector_count"], 3)
        self.assertEqual(model["mean_vector"], [2 / 3, 1 / 3])

    def test_invalid_training_vectors_are_rejected(self):
        for values in ([], [[0, 0]], [[1, 0], [1, 0, 0]], [[float("nan"), 0]],
                       [[float("inf"), 0]], [[True, False]], [[1j, 0]], [["x", "y"]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_semantic_workspace_model(values)

    def test_large_finite_coordinates_normalize_without_overflow(self):
        model = fit_semantic_workspace_model([[1e308, 1e308], [-1e308, -1e308]])
        self.assertEqual(model["mean_vector"], [0.0, 0.0])

    def test_centered_formulas_have_explicit_expected_values(self):
        facts = self.facts(["same", "other"])
        # Mean [.5,.5] turns the two unit axes into opposite centered vectors.
        self.assertAlmostEqual(facts["text_semantic_centered_top3_similarity"], 0.5)
        expected = 1.0 / (1.0 + math.exp(-16.0))
        self.assertAlmostEqual(facts["text_semantic_centered_attention_t8_similarity"], expected, places=7)
        self.assertEqual(facts["text_semantic_centered_recent5_max_similarity"], 1.0)

    def test_zero_mean_preserves_existing_cosine_observations(self):
        model = fit_semantic_workspace_model([[1.0, 0.0], [-1.0, 0.0]])
        facts = self.facts(["same", "other", "opposite"], model=model)
        self.assertEqual(facts["text_semantic_centered_top3_similarity"], facts["text_semantic_top3_mean_similarity"])
        self.assertEqual(facts["text_semantic_centered_attention_t8_similarity"], facts["text_semantic_attention_t8_similarity"])

    def test_existing_observations_are_emitted_unchanged_when_available(self):
        expected = build_semantic_match_facts("candidate", ["same", "other"], self.vectors, prefix="text_semantic")
        actual = self.facts(["same", "other"])
        for key, value in expected.items():
            self.assertEqual(actual[key], value, key)

    def test_effective_support_distinguishes_single_match_and_consensus(self):
        concentrated = self.facts(["same", "other"])
        diffuse = self.facts(["same", "same"])
        weight = math.exp(-16.0)
        expected = (1 + weight) ** 2 / (2 * (1 + weight ** 2))
        self.assertAlmostEqual(concentrated["text_semantic_effective_support_ratio"], expected, places=8)
        self.assertEqual(diffuse["text_semantic_effective_support_ratio"], 1.0)
        self.assertEqual(self.facts(["same"])["text_semantic_effective_support_ratio"], 1.0)

    def test_missing_evidence_is_none_and_keeps_recency_positions(self):
        facts = self.facts(["same"] + ["unknown"] * 5)
        self.assertEqual(facts["text_semantic_centered_top3_similarity"], 1.0)
        self.assertAlmostEqual(facts["text_semantic_centered_coverage"], 1 / 6)
        self.assertIsNone(facts["text_semantic_centered_recent5_max_similarity"])
        self.assertIsNone(facts["text_semantic_recent5_max_similarity"])
        self.assertIsNone(facts["text_semantic_recent5_centroid_similarity"])
        self.assertEqual(facts["text_semantic_centered_recent5_available"], "no")

    def test_missing_candidate_or_history_masks_every_similarity(self):
        for facts in (self.facts(["same"], candidate="missing"), self.facts([]), self.facts(["missing"])):
            for feature in SEMANTIC_WORKSPACE_FEATURES:
                self.assertIsNone(facts[feature], feature)
            for key, value in facts.items():
                if key.endswith("_similarity"):
                    self.assertIsNone(value, key)

    def test_invalid_observations_are_missing_not_fake_low_scores(self):
        for bad in ([float("nan"), 1], [float("inf"), 1], [0, 0], [1, 0, 0], [True, False]):
            facts = self.facts(["same"], vectors={**self.vectors, "candidate": bad})
            self.assertEqual(facts["text_semantic_candidate_available"], "no")
            self.assertIsNone(facts["text_semantic_attention_t8_similarity"])
            self.assertIsNone(facts["text_semantic_centered_attention_t8_similarity"])

    def test_zero_centered_norm_is_unavailable_even_when_raw_vector_exists(self):
        model = fit_semantic_workspace_model([[1.0, 0.0]])
        facts = self.facts(["same", "other"], model=model)
        self.assertEqual(facts["text_semantic_candidate_available"], "yes")
        self.assertEqual(facts["text_semantic_centered_candidate_available"], "no")
        self.assertIsNone(facts["text_semantic_centered_top3_similarity"])
        self.assertIsNone(facts["text_semantic_effective_support_ratio"])
        self.assertIsNotNone(facts["text_semantic_top3_mean_similarity"])

    def test_cancelled_recent_centroid_is_missing_not_orthogonal(self):
        facts = self.facts(["same", "opposite"])
        self.assertEqual(facts["text_semantic_recent5_available"], "yes")
        self.assertEqual(facts["text_semantic_recent5_centroid_available"], "no")
        self.assertIsNone(facts["text_semantic_recent5_centroid_similarity"])

    def test_roundoff_in_identical_training_vectors_does_not_create_direction(self):
        vector = [1.0, 2.0, 3.0]
        model = fit_semantic_workspace_model([vector] * 7)
        facts = build_semantic_workspace_facts("a", ["b"], {"a": vector, "b": vector}, model)
        self.assertEqual(facts["text_semantic_candidate_available"], "yes")
        self.assertEqual(facts["text_semantic_centered_candidate_available"], "no")
        self.assertIsNone(facts["text_semantic_centered_attention_t8_similarity"])

    def test_future_vectors_metadata_labels_and_renamed_ids_do_not_change_facts(self):
        expected = self.facts(["same", "other"])
        changed = {**self.vectors, "future": [99.0, -77.0], "unseen": [-1.0, -1.0]}
        actual = self.facts([{"id": "same", "label": 0}, {"id": "other", "action": "click"}],
                            candidate={"id": "candidate", "label": 1}, vectors=changed)
        self.assertEqual(expected, actual)
        self.assertEqual(expected, build_semantic_workspace_facts(
            "x", ["y", "z"], {"x": [1, 0], "y": [1, 0], "z": [0, 1]}, self.model))

    def test_model_vectors_and_history_are_not_mutated_or_refit(self):
        history = ["same", "other"]
        before = copy.deepcopy((self.model, self.vectors, history))
        with patch("recommendation.features.semantic_workspace.fit_semantic_workspace_model", side_effect=AssertionError("refit")):
            self.facts(history)
            self.facts(["future"])
        self.assertEqual((self.model, self.vectors, history), before)

    def test_all_numeric_observations_are_finite_and_bounded(self):
        for history in ([], ["same"], ["same", "other", "opposite", "missing"], ["future"] * 30):
            facts = self.facts(history)
            json.dumps(facts, allow_nan=False)
            for name, value in facts.items():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value), name)
                    self.assertGreaterEqual(value, 0, name)
                    self.assertLessEqual(value, 1, name)

    def test_invalid_frozen_models_fail_explicitly(self):
        for change in ({"schema": "unknown"}, {"dimensions": 3}, {"mean_vector": [float("nan"), 0]},
                       {"mean_vector": [2, 0]}, {"training_vector_count": 0}, {"attention_temperature": 12}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.facts(["same"], model={**self.model, **change})


if __name__ == "__main__":
    unittest.main()
