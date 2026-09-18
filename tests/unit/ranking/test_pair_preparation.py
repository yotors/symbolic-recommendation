import itertools
import unittest

from recommendation.app.server import (
    CONTEXT_FEATURES,
    Lab,
    NUMERIC_PAIR_EVIDENCE,
    PAIR_EVIDENCE_ALIASES,
    PAIR_FEATURE_PROFILES,
    _proof_channel_preparation_profile,
)
from recommendation.core.symbolic import QuantileNumericEvidence


def _numeric_encoders():
    return {
        source: QuantileNumericEvidence.fit(
            source, [0.0, 0.25, 0.75, 1.0],
            [(0.0, 0.25), (0.0, 0.75), (0.0, 1.0)], bins=4,
        )
        for source in NUMERIC_PAIR_EVIDENCE
    }


def _contexts():
    left = {
        key: (0.2 if any(token in key for token in (
            "score", "similarity", "affinity", "llm_", "lexical_", "semantic_"
        )) else "low")
        for key in CONTEXT_FEATURES
    }
    right = dict(left)
    for key in left:
        if isinstance(left[key], float):
            right[key] = 0.8
    left.update({
        "history_size_bucket":"regular", "affinity":"high",
        "recent_affinity":"high", "long_affinity":"medium",
        "entity_overlap_detail":"one", "history_topic_count_bucket":"two",
        "recent_topic_count_bucket":"one", "topic_rank_bucket":"top",
        "subcategory_affinity":"high", "title_overlap_detail":"two",
        "ctr_bucket":"medium", "freshness_bucket":"new", "format":"long",
        "llm_format":"analysis",
    })
    right.update({
        "history_size_bucket":"regular", "affinity":"low",
        "recent_affinity":"low", "long_affinity":"none",
        "entity_overlap_detail":"none", "history_topic_count_bucket":"zero",
        "recent_topic_count_bucket":"zero", "topic_rank_bucket":"secondary",
        "subcategory_affinity":"low", "title_overlap_detail":"one",
        "ctr_bucket":"low", "freshness_bucket":"recent", "format":"short",
        "llm_format":"brief",
    })
    return left, right


class PairPreparationTest(unittest.TestCase):
    def setUp(self):
        self.lab=object.__new__(Lab)
        self.lab._pair_categorical_labels={}
        self.lab._numeric_pair_encoders=_numeric_encoders()

    def test_derived_reverse_equals_explicit_reverse_for_every_profile(self):
        left,right=_contexts()
        left_article={"id":"left","topic":"news","subcategory":"world"}
        right_article={"id":"right","topic":"sports","subcategory":"soccer"}
        for profile,needed in PAIR_FEATURE_PROFILES.items():
            with self.subTest(profile=profile):
                forward=self.lab._pair_features(
                    left,right,left_article,right_article,needed=needed
                )
                explicit=self.lab._pair_features(
                    right,left,right_article,left_article,needed=needed
                )
                self.assertEqual(
                    self.lab._reverse_pair_features(forward),explicit
                )

    def test_categorical_reserved_tokens_are_opaque_during_reverse(self):
        for token in ("left", "right", "left_known", "right_known",
                      "left_q1", "right_q4"):
            with self.subTest(token=token):
                attrs={
                    "pair_left_topic":token,
                    "pair_right_topic":"news",
                    "pair_long_affinity":"left",
                }
                reversed_attrs=self.lab._reverse_pair_features(attrs)
                self.assertEqual(reversed_attrs["pair_right_topic"],token)
                self.assertEqual(reversed_attrs["pair_left_topic"],"news")
                self.assertEqual(reversed_attrs["pair_long_affinity"],"right")
                self.assertEqual(
                    self.lab._reverse_pair_features(reversed_attrs),attrs
                )

    def test_relational_scope_is_antisymmetric_and_keeps_one_lineage(self):
        predicate="pair_rel_concept_continuity_scope"
        forward=self.lab._pair_features(
            {"rel_concept_continuity_scope":"recent"},
            {"rel_concept_continuity_scope":"older"},
            {},{},needed={predicate},
        )
        reverse=self.lab._pair_features(
            {"rel_concept_continuity_scope":"older"},
            {"rel_concept_continuity_scope":"recent"},
            {},{},needed={predicate},
        )

        self.assertEqual(forward[predicate],"left")
        self.assertEqual(reverse[predicate],"right")
        self.assertEqual(
            PAIR_EVIDENCE_ALIASES[predicate],"pair_text_semantic_top3_mean"
        )
        self.assertEqual(
            self.lab._pair_rule_family({"premises":[(predicate,"left")]}),
            "text_semantic",
        )

    def test_relational_unknown_abstains_instead_of_becoming_known_side(self):
        predicate="pair_rel_concept_continuity_scope"
        left=self.lab._pair_features(
            {"rel_concept_continuity_scope":"recent"},{},{},{},
            needed={predicate},
        )
        right=self.lab._pair_features(
            {},{"rel_concept_continuity_scope":"recent"},{},{},
            needed={predicate},
        )
        explicit_unknown=self.lab._pair_features(
            {"rel_concept_continuity_scope":"unknown"},
            {"rel_concept_continuity_scope":"recent"},{},{},
            needed={predicate},
        )
        self.assertNotIn(predicate,left)
        self.assertNotIn(predicate,right)
        self.assertNotIn(predicate,explicit_unknown)

    def test_proof_channel_profile_excludes_query_and_enclosing_total(self):
        raw={
            "proof_topology_validation_seconds":0.1,
            "proof_activation_join_seconds":0.2,
            "proof_template_serialization_seconds":0.3,
            "proof_atomspace_insertion_seconds":0.4,
            "proof_query_seconds":5.0,
            "proof_template_atoms_inserted":12,
            "total_seconds":6.25,
        }
        profile=_proof_channel_preparation_profile(raw)
        self.assertNotIn("total_seconds",profile)
        self.assertNotIn("proof_query_seconds",profile)
        self.assertEqual(profile["proof_query_seconds_excluded"],5.0)
        self.assertEqual(profile["pre_query_component_sum_seconds"],1.0)
        self.assertEqual(
            sum(profile[key] for key in (
                "proof_topology_validation_seconds",
                "proof_activation_join_seconds",
                "proof_template_serialization_seconds",
                "proof_atomspace_insertion_seconds",
            )),
            profile["pre_query_component_sum_seconds"],
        )

    def test_optimized_plan_is_identical_to_explicit_two_pass_plan(self):
        left,right=_contexts()
        articles={
            "a":{"id":"a","topic":"news","subcategory":"world"},
            "b":{"id":"b","topic":"sports","subcategory":"soccer"},
            "c":{"id":"c","topic":"news","subcategory":"local"},
        }
        contexts={"a":left,"b":right,"c":{**left,"long_affinity":"high"}}
        self.lab.article=articles.__getitem__
        self.lab.pair_rules=[
            {"id":"r1","dependency_id":"r1",
             "premises":[("pair_long_affinity","left")]},
            {"id":"r2","dependency_id":"r2",
             "premises":[("pair_same_topic","different")]},
        ]
        self.lab._pair_feature_vocabulary={
            "pair_long_affinity":{"left","right","equal","other"},
            "pair_same_topic":{"same","different","other"},
        }
        self.lab.config={"pairwise_opponents":0,"max_pair_comparisons":10}
        rows=[{"article":articles[aid]} for aid in articles]
        specs=[(aid,"unused",{},contexts[aid]) for aid in articles]
        expected_comparisons=[]; expected_specs=[]
        for left_index,right_index in itertools.combinations(range(len(rows)),2):
            left_id=rows[left_index]["article"]["id"]
            right_id=rows[right_index]["article"]["id"]
            forward=self.lab._pair_spec(
                (articles[left_id],contexts[left_id]),
                (articles[right_id],contexts[right_id]),
            )
            reverse=self.lab._pair_spec(
                (articles[right_id],contexts[right_id]),
                (articles[left_id],contexts[left_id]),
            )
            expected_comparisons.append(
                (left_index,right_index,forward[0],reverse[0])
            )
            expected_specs.extend((forward,reverse))
        actual_comparisons,actual_specs=self.lab._pairwise_plan(rows,specs)
        self.assertEqual(actual_comparisons,expected_comparisons)
        self.assertEqual(actual_specs,expected_specs)
        self.assertTrue(self.lab._last_pair_plan_profile["candidate_feature_reuse"])
        profile=self.lab._last_pair_plan_profile
        components=(
            "comparison_graph_seconds", "pair_feature_derivation_seconds",
            "reverse_orientation_derivation_seconds",
            "pair_feature_bounding_seconds", "pair_activation_join_seconds",
            "pair_activation_case_cache_seconds",
            "pair_case_serialization_seconds", "pair_plan_assembly_seconds",
        )
        self.assertAlmostEqual(
            sum(profile[key] for key in components),profile["total_seconds"],
            places=9,
        )
        self.assertIn("unique_activation_cases_in_slate",profile)
        self.assertNotIn("unique_activation_cases",profile)


if __name__ == "__main__":
    unittest.main()
