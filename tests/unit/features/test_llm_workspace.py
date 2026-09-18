"""Content-annotation observations stay label-free, causal and explicit."""

import copy
import unittest

from recommendation.features.llm_workspace import (
    LLM_NUMERIC_FEATURES, LLM_WORKSPACE_FEATURES,
    build_llm_workspace_facts,
)


class LLMWorkspaceTest(unittest.TestCase):
    def test_schema_contains_eight_numeric_observations_plus_coverage_and_format(self):
        self.assertEqual(len(LLM_NUMERIC_FEATURES), 8)
        facts = build_llm_workspace_facts("candidate", [], {})
        self.assertEqual(set(facts), set(LLM_WORKSPACE_FEATURES))
        self.assertEqual(set(facts) - set(LLM_NUMERIC_FEATURES),
                         {"llm_history_coverage", "llm_format"})

    def test_missing_annotations_are_unknown_not_zero_similarity(self):
        for history in ([], ["unknown"], ["unknown", "unknown"]):
            with self.subTest(history=history):
                facts = build_llm_workspace_facts("candidate", history, {})
                self.assertEqual(facts["llm_history_coverage"], 0.0)
                self.assertTrue(all(value is None for key, value in facts.items()
                                    if key != "llm_history_coverage"))

    def test_empty_history_keeps_candidate_format_but_no_preference_observations(self):
        facts = build_llm_workspace_facts("c", [], {
            "c": {"concepts": ["astronomy"], "format": "  Explainer "},
        })
        self.assertEqual(facts["llm_format"], "explainer")
        self.assertEqual(facts["llm_history_coverage"], 0.0)
        self.assertTrue(all(facts[key] is None for key in LLM_NUMERIC_FEATURES))

    def test_affinity_peak_and_novelty_have_distinct_formulas(self):
        facts = build_llm_workspace_facts("c", ["h1", "h2"], {
            "c": {"concepts": ["a", "b", "c", "d"]},
            "h1": {"concepts": ["a", "b"]},
            "h2": {"concepts": ["b"]},
        })
        self.assertEqual(facts["llm_concept_affinity"], 0.375)
        self.assertEqual(facts["llm_recent_concept_affinity"], 0.375)
        self.assertEqual(facts["llm_concept_peak_overlap"], 0.5)
        self.assertEqual(facts["llm_concept_novelty"], 0.5)

    def test_known_disjoint_concepts_are_zero_match_and_complete_novelty(self):
        facts = build_llm_workspace_facts("c", ["h"], {
            "c": {"concepts": ["astronomy"]},
            "h": {"concepts": ["football"]},
        })
        for feature in ("llm_concept_affinity", "llm_recent_concept_affinity",
                        "llm_concept_peak_overlap"):
            self.assertEqual(facts[feature], 0.0)
        self.assertEqual(facts["llm_concept_novelty"], 1.0)

    def test_field_denominators_exclude_unavailable_field_values(self):
        facts = build_llm_workspace_facts("c", ["h1", "h2", "h3", "missing"], {
            "c": {"concepts": ["a"], "format": "report",
                  "event_types": ["launch"], "intents": ["learn"],
                  "audiences": ["developer"]},
            "h1": {"concepts": ["a"], "format": "report",
                   "event_types": ["launch"], "intents": ["learn"]},
            "h2": {"concepts": ["b"], "format": "opinion",
                   "audiences": ["developer"]},
            "h3": {"concepts": [], "event_types": ["election"]},
        })
        self.assertEqual(facts["llm_history_coverage"], 0.75)
        self.assertEqual(facts["llm_concept_affinity"], 0.5)
        self.assertEqual(facts["llm_format_affinity"], 0.5)
        self.assertEqual(facts["llm_event_affinity"], 0.5)
        self.assertEqual(facts["llm_intent_affinity"], 1.0)
        self.assertEqual(facts["llm_audience_affinity"], 1.0)

    def test_recent_window_preserves_original_missing_positions(self):
        records = {"c": {"concepts": ["a"]}, "old": {"concepts": ["a"]}}
        facts = build_llm_workspace_facts("c", ["old"] + ["missing"] * 5, records)
        self.assertEqual(facts["llm_concept_affinity"], 1.0)
        self.assertIsNone(facts["llm_recent_concept_affinity"])
        self.assertEqual(facts["llm_history_coverage"], 0.16666667)

    def test_recent_window_uses_newest_five_not_first_five(self):
        facts = build_llm_workspace_facts("c", ["old"] * 5 + ["new"], {
            "c": {"concepts": ["a"]}, "old": {"concepts": ["b"]},
            "new": {"concepts": ["a"]},
        })
        self.assertEqual(facts["llm_concept_affinity"], 0.16666667)
        self.assertEqual(facts["llm_recent_concept_affinity"], 0.2)

    def test_repeated_reads_keep_their_original_multiplicity(self):
        records = {"c": {"concepts": ["a"]}, "a": {"concepts": ["a"]},
                   "b": {"concepts": ["b"]}}
        facts = build_llm_workspace_facts("c", ["a", "a", "b"], records)
        self.assertEqual(facts["llm_concept_affinity"], 0.66666667)
        self.assertEqual(facts["llm_concept_novelty"], 0.0)

    def test_labels_are_normalized_and_duplicate_values_do_not_add_weight(self):
        facts = build_llm_workspace_facts("c", ["h"], {
            "c": {"concepts": [" ＡＩ ", "ai", " Space   Research ", None, 42],
                  "format": " ＲＥＰＯＲＴ "},
            "h": {"concepts": ["AI", "space research"], "format": "report"},
        })
        self.assertEqual(facts["llm_concept_affinity"], 1.0)
        self.assertEqual(facts["llm_format_affinity"], 1.0)
        self.assertEqual(facts["llm_format"], "report")

    def test_opaque_id_renaming_and_record_ids_preserve_observations(self):
        records = {"c": {"concepts": ["a", "b"]}, "h": {"concepts": ["b"]}}
        expected = build_llm_workspace_facts("c", ["h", "missing"], records)
        renamed = {"item_900": records["c"], "prior_12": records["h"]}
        actual = build_llm_workspace_facts({"id": "item_900"},
                                         [{"id": "prior_12"}, {"id": "absent"}], renamed)
        self.assertEqual(actual, expected)

    def test_labels_outcomes_and_unrelated_records_cannot_change_facts(self):
        records = {"c": {"concepts": ["a"]}, "h": {"concepts": ["a"]}}
        expected = build_llm_workspace_facts("c", ["h"], records)
        changed = copy.deepcopy(records)
        changed["c"].update(label=0, action="skip", score=0.0, clicks=999)
        changed["h"].update(label=1, action="click", score=1.0)
        changed["future"] = {"concepts": ["b"], "label": 1}
        self.assertEqual(build_llm_workspace_facts("c", ["h"], changed), expected)

    def test_missing_candidate_malformed_records_and_unhashable_ids_are_safe(self):
        records = {"h": {"concepts": ["a"]}, "bad": "not a record"}
        for candidate in ("missing", "bad", ["unhashable"], {"id": []}):
            with self.subTest(candidate=candidate):
                facts = build_llm_workspace_facts(candidate, ["h", "bad", []], records)
                self.assertEqual(facts["llm_history_coverage"], 0.33333333)
                self.assertTrue(all(facts[key] is None for key in LLM_NUMERIC_FEATURES))

    def test_input_is_not_mutated_and_annotations_require_a_mapping(self):
        records = {"c": {"concepts": [" A ", "a"]}, "h": {"concepts": ["a"]}}
        history = ["h", "missing"]
        before = copy.deepcopy((records, history))
        build_llm_workspace_facts("c", history, records)
        self.assertEqual((records, history), before)
        for invalid in (None, [], "records"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    build_llm_workspace_facts("c", [], invalid)


if __name__ == "__main__":
    unittest.main()
