import copy
import math
import unittest
from unittest.mock import patch

from recommendation.features.lexical_workspace import (
    LEXICAL_FEATURES,
    LexicalWorkspaceConfig,
    build_lexical_workspace_facts,
    fit_lexical_idf_model,
    lexical_tokens,
)


class LexicalWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"title": "Space telescope discovers distant planets", "abstract": "Astronomy research"},
            {"title": "Local football team wins championship", "abstract": "Sports results"},
            {"title": "New telescope measures planetary atmosphere", "abstract": "Space research"},
        ]
        self.model = fit_lexical_idf_model(self.history)
        self.candidate = {"title": "Telescope research finds new planets", "abstract": "Space astronomy"}

    def test_strong_literal_match_beats_disjoint_content(self):
        matched = build_lexical_workspace_facts(self.candidate, self.history, self.model)
        disjoint = build_lexical_workspace_facts({"title": "Cooking recipes kitchen meals"}, self.history, self.model)
        self.assertEqual(set(matched), set(LEXICAL_FEATURES))
        for name in LEXICAL_FEATURES:
            self.assertGreater(matched[name], disjoint[name])
            self.assertEqual(disjoint[name], 0.0)
            self.assertLessEqual(matched[name], 1.0)

    def test_absent_evidence_is_not_zero_overlap(self):
        for history, candidate, model in (
            ([], self.candidate, self.model),
            ([{}], self.candidate, self.model),
            (self.history, {}, self.model),
            (self.history, self.candidate, None),
            (self.history, self.candidate, fit_lexical_idf_model([])),
        ):
            self.assertEqual(build_lexical_workspace_facts(candidate, history, model), dict.fromkeys(LEXICAL_FEATURES))

    def test_ids_labels_vectors_and_other_fields_are_ignored(self):
        expected = build_lexical_workspace_facts(self.candidate, self.history, self.model)
        extra = {"id": "candidate_42", "user": "u3", "label": 1, "click": True,
                 "action": "skip", "semantic_vector": [1, 2], "topic": "football"}
        changed = build_lexical_workspace_facts(
            {**self.candidate, **extra}, [{**item, **extra} for item in self.history], self.model)
        self.assertEqual(expected, changed)
        self.assertEqual(self.model, fit_lexical_idf_model([{**item, **extra} for item in self.history]))
        self.assertEqual(lexical_tokens({**extra, "title": None, "abstract": {"label": "leak"}}), ())

    def test_case_unicode_and_token_order_invariance(self):
        def changed(article):
            return {key: " ".join(reversed(value.upper().split())) for key, value in article.items()}
        expected = build_lexical_workspace_facts(self.candidate, self.history, self.model)
        actual = build_lexical_workspace_facts(changed(self.candidate), [changed(item) for item in self.history], self.model)
        self.assertEqual(expected, actual)
        self.assertEqual(lexical_tokens({"title": "Ｔｅｌｅｓｃｏｐｅ"}), ("telescope",))

    def test_fit_order_repeated_records_and_ids_do_not_change_model(self):
        reordered = list(reversed(self.history)) + self.history
        self.assertEqual(self.model, fit_lexical_idf_model(reordered))

    def test_frozen_model_is_neither_refit_nor_mutated_at_scoring(self):
        before = copy.deepcopy(self.model)
        with patch("recommendation.features.lexical_workspace.fit_lexical_idf_model", side_effect=AssertionError("refit")):
            build_lexical_workspace_facts(self.candidate, self.history, self.model)
            build_lexical_workspace_facts({"title": "Entirely novel unheardof terminology"}, self.history, self.model)
        self.assertEqual(self.model, before)
        self.assertNotIn("unheardof", self.model["idf"])

    def test_training_rarity_gives_rare_matches_more_weight(self):
        model = fit_lexical_idf_model([
            {"title": "common rare"}, {"title": "common second"}, {"title": "common third"},
        ])
        rare = build_lexical_workspace_facts({"title": "common rare"}, [{"title": "rare"}], model)
        common = build_lexical_workspace_facts({"title": "common rare"}, [{"title": "common"}], model)
        self.assertGreater(rare["lexical_history_coverage"], common["lexical_history_coverage"])

    def test_full_history_is_order_invariant_but_recent_is_causal(self):
        history = [{"title": "telescope planets"}] + [{"title": "football games"}] * 5
        old = build_lexical_workspace_facts(self.candidate, history, self.model)
        recent = build_lexical_workspace_facts(self.candidate, list(reversed(history)), self.model)
        self.assertEqual(old["lexical_history_coverage"], recent["lexical_history_coverage"])
        self.assertEqual(old["lexical_peak_match"], recent["lexical_peak_match"])
        self.assertEqual(old["lexical_top3_match"], recent["lexical_top3_match"])
        self.assertEqual(old["lexical_recent_coverage"], 0.0)
        self.assertGreater(recent["lexical_recent_coverage"], 0.0)

    def test_unknown_documents_preserve_recency_positions(self):
        history = [self.history[0]] + [{}] * 5
        facts = build_lexical_workspace_facts(self.candidate, history, self.model)
        self.assertGreater(facts["lexical_history_coverage"], 0)
        self.assertIsNone(facts["lexical_recent_coverage"])

    def test_abstract_without_title_works(self):
        facts = build_lexical_workspace_facts({"abstract": "Space research"}, self.history, self.model)
        self.assertGreater(facts["lexical_peak_match"], 0)

    def test_history_is_bounded_and_generator_supported(self):
        config = LexicalWorkspaceConfig(max_history=2)
        history = [self.history[0], {"title": "football games"}, {"title": "cooking meals"}]
        facts = build_lexical_workspace_facts(self.candidate, iter(history), self.model, config=config)
        self.assertEqual(facts["lexical_peak_match"], 0.0)

    def test_tokenization_limits_must_match_fitted_model(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            build_lexical_workspace_facts(self.candidate, self.history, self.model,
                                          config=LexicalWorkspaceConfig(max_tokens=12))

    def test_nonfinite_model_values_do_not_emit_invalid_facts(self):
        for value in (float("nan"), float("inf"), -1, True):
            changed = {**self.model, "default_idf": value}
            facts = build_lexical_workspace_facts(self.candidate, self.history, changed)
            self.assertEqual(facts, dict.fromkeys(LEXICAL_FEATURES))

    def test_outputs_bounded_with_repeated_tokens_and_long_documents(self):
        facts = build_lexical_workspace_facts(
            {"title": "space " * 1000}, [{"title": "space " * 1000}], self.model)
        for value in facts.values():
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
