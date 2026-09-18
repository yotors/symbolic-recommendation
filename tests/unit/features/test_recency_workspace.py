import copy
import json
import math
import unittest

import numpy as np

from recommendation.features.recency_workspace import (
    RECENCY_HALF_LIVES,
    RECENCY_WORKSPACE_FEATURES,
    build_recency_workspace_facts,
)


class RecencyWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.vectors = {
            "candidate": [1.0, 0.0],
            "same": [1.0, 0.0],
            "other": [0.0, 1.0],
            "opposite": [-1.0, 0.0],
            "near": [0.2, math.sqrt(0.96)],
        }

    def facts(self, history, candidate="candidate", vectors=None):
        return build_recency_workspace_facts(
            candidate, history, self.vectors if vectors is None else vectors
        )

    def test_formula_uses_raw_cosine_logits_and_mapped_observations(self):
        actual = self.facts(["near", "other"])
        for half_life, feature in zip(RECENCY_HALF_LIVES, RECENCY_WORKSPACE_FEATURES):
            # Older cosine .2 has mapped observation .6; newest cosine 0 has .5.
            older_weight = math.exp(8.0 * 0.2) * 2.0 ** (-1.0 / half_life)
            expected = (older_weight * 0.6 + 0.5) / (older_weight + 1.0)
            self.assertAlmostEqual(actual[feature], expected, places=8)

    def test_newest_evidence_has_greater_weight_and_shorter_half_life_is_stronger(self):
        old_match = self.facts(["near"] + ["missing"] * 15 + ["other"])
        new_match = self.facts(["other"] + ["missing"] * 15 + ["near"])
        for feature in RECENCY_WORKSPACE_FEATURES:
            self.assertGreater(new_match[feature], old_match[feature])
        h8, h16 = RECENCY_WORKSPACE_FEATURES
        self.assertGreater(new_match[h8], new_match[h16])
        self.assertLess(old_match[h8], old_match[h16])

    def test_missing_slots_keep_the_original_lags(self):
        actual = self.facts(["near"] + ["missing"] * 7 + ["other"])
        compacted = self.facts(["near", "other"])
        for half_life, feature in zip(RECENCY_HALF_LIVES, RECENCY_WORKSPACE_FEATURES):
            older_weight = math.exp(1.6) * 2.0 ** (-8.0 / half_life)
            expected = (older_weight * 0.6 + 0.5) / (older_weight + 1.0)
            self.assertAlmostEqual(actual[feature], expected, places=8)
            self.assertNotEqual(actual[feature], compacted[feature])

    def test_repeated_interactions_keep_multiplicity_and_lags(self):
        actual = self.facts(["near", "near", "other"])
        deduplicated = self.facts(["near", "other"])
        for half_life, feature in zip(RECENCY_HALF_LIVES, RECENCY_WORKSPACE_FEATURES):
            older_weight = math.exp(1.6) * (
                2.0 ** (-2.0 / half_life) + 2.0 ** (-1.0 / half_life)
            )
            expected = (older_weight * 0.6 + 0.5) / (older_weight + 1.0)
            self.assertAlmostEqual(actual[feature], expected, places=8)
            self.assertGreater(actual[feature], deduplicated[feature])

    def test_very_old_available_observations_do_not_underflow_to_missing_or_zero(self):
        actual = self.facts(["near", "other"] + ["missing"] * 20000)
        # A common age offset cancels in softmax, even when naive decay is zero.
        self.assertEqual(actual, self.facts(["near", "other"]))
        self.assertEqual(self.facts(["same"] + ["missing"] * 20000),
                         dict.fromkeys(RECENCY_WORKSPACE_FEATURES, 1.0))

    def test_single_opposite_vector_is_valid_zero_evidence(self):
        self.assertEqual(self.facts(["opposite"]),
                         dict.fromkeys(RECENCY_WORKSPACE_FEATURES, 0.0))

    def test_absent_candidate_or_compatible_history_returns_none(self):
        vectors = {**self.vectors, "wrong": [1, 0, 0]}
        cases = [self.facts([]), self.facts(["missing"]),
                 self.facts(["same"], candidate="missing"),
                 self.facts(["wrong"], vectors=vectors), self.facts(["same"], vectors={})]
        for actual in cases:
            self.assertEqual(actual, dict.fromkeys(RECENCY_WORKSPACE_FEATURES))

    def test_malformed_vectors_are_missing_without_poisoning_valid_evidence(self):
        for invalid in ([0, 0], [], [float("nan"), 1], [float("inf"), 0],
                        [[1, 0]], [True, False], [1j, 0], ["x", "y"],
                        "10", {"id": "same"}, [1, 0, 0]):
            with self.subTest(invalid=invalid):
                vectors = {**self.vectors, "invalid": invalid}
                self.assertEqual(self.facts(["invalid", "same"], vectors=vectors),
                                 self.facts(["missing", "same"]))
                if invalid != [1, 0, 0]:
                    self.assertEqual(self.facts(["same"], candidate="invalid", vectors=vectors),
                                     dict.fromkeys(RECENCY_WORKSPACE_FEATURES))

    def test_opaque_identifier_renaming_and_mapping_order_do_not_change_facts(self):
        names = {key: index + 10 for index, key in enumerate(self.vectors)}
        renamed = {names[key]: value for key, value in reversed(list(self.vectors.items()))}
        actual = build_recency_workspace_facts(
            names["candidate"], [names["near"], names["other"], names["near"]], renamed
        )
        self.assertEqual(actual, self.facts(["near", "other", "near"]))

    def test_outcome_metadata_and_unrelated_vectors_are_ignored(self):
        history = [{"id": "near", "clicked": True, "score": 1000},
                   {"id": "other", "clicked": False, "topic": "irrelevant"}]
        vectors = {**self.vectors, "unrelated": [1, 0]}
        actual = self.facts(history, candidate={"id": "candidate", "label": 1}, vectors=vectors)
        self.assertEqual(actual, self.facts(["near", "other"]))
        history[0]["clicked"] = False
        history[1]["clicked"] = True
        self.assertEqual(actual, self.facts(history, candidate={"id": "candidate", "label": 0}))

    def test_unhashable_missing_ids_do_not_fail_resolution(self):
        self.assertEqual(self.facts([[], {"id": []}, None, {"label": 1}, "same"]),
                         self.facts([None] * 4 + ["same"]))
        self.assertEqual(self.facts(["same"], candidate={"id": []}),
                         dict.fromkeys(RECENCY_WORKSPACE_FEATURES))

    def test_finite_extreme_coordinates_and_positive_rescaling_are_supported(self):
        vectors = {key: np.asarray(vector) * 1e308 for key, vector in self.vectors.items()}
        self.assertEqual(self.facts(["near", "other", "same"], vectors=vectors),
                         self.facts(["near", "other", "same"]))
        tiny = {key: np.asarray(vector) * 1e-300 for key, vector in self.vectors.items()}
        self.assertEqual(self.facts(["near", "other", "same"], vectors=tiny),
                         self.facts(["near", "other", "same"]))

    def test_history_iterables_are_supported_and_inputs_are_not_mutated(self):
        history = [{"id": "near"}, {"id": "other"}, {"id": "near"}]
        before_history = copy.deepcopy(history)
        before_vectors = copy.deepcopy(self.vectors)
        actual = self.facts(iter(history))
        self.assertEqual(actual, self.facts(history))
        self.assertEqual(history, before_history)
        self.assertEqual(self.vectors, before_vectors)
        self.assertEqual(actual, json.loads(json.dumps(actual, allow_nan=False)))


if __name__ == "__main__":
    unittest.main()
