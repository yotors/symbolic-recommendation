"""Entity-continuity plans are causal, stable, and proof gated."""

import copy
import inspect
import unittest

from recommendation.features.relational_workspace import (
    CANONICAL_CONCEPT_BRIDGE_RULE_ID,
    CONCEPT_CONTINUITY_RULE_ID,
    ENGAGED_ENTITY_RULE_ID,
    ENGAGED_CONCEPT_RULE_ID,
    ENTITY_CONTINUITY_RULE_ID,
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
    build_concept_relational_proof_plan,
    build_relational_proof_plan,
    canonical_annotation_concepts,
    mind_wikidata_entities,
    reduce_concept_relational_proofs,
    reduce_relational_proofs,
)


def articles():
    return {
        "candidate": {
            "id": "candidate",
            "title_entities": ["Q1", {"WikidataId": "q2"}, "Q1"],
            "abstract_entities": [],
        },
        "old": {
            "id": "old", "title_entities": ["Q1", "Q2"],
            "abstract_entities": [],
        },
        "recent": {
            "id": "recent", "title_entities": [],
            "abstract_entities": [{"WikidataId": "Q2"}],
        },
        "different": {
            "id": "different", "title_entities": ["Q9"],
            "abstract_entities": [],
        },
        "empty": {
            "id": "empty", "title_entities": [], "abstract_entities": [],
        },
        "unknown": {"id": "unknown", "title": "No entity columns"},
    }


def proof(plan, origin, path_index=0, suffix=""):
    candidate_fact = origin.candidate_entity_fact_ids[path_index]
    history_fact = origin.history_entity_fact_ids[path_index]
    entity_atom = f"rel_entity_{origin.matched_entity_ids[path_index].lower()}"
    return (
        f"(: (merge/revision "
        f"(by {ENTITY_CONTINUITY_RULE_ID} {plan.case_candidate_fact_id} "
        f"(by {ENGAGED_ENTITY_RULE_ID} {origin.observed_click_fact_id} "
        f"{history_fact}) {candidate_fact}){suffix}) "
        f"(RelEntityContinuity {plan.case_id} {origin.origin_id} {entity_atom}) "
        "(STV 1.0 0.999))"
    )


def annotation(article_id, concept_id, lexical):
    anchor = f"anchor_{article_id.lower()}"
    return {
        "concepts": [concept_id],
        "provenance": {
            "article_id": article_id,
            "article_id": article_id,
            "article_content_sha256": "a" * 64,
            "anchored_statements": [
                f'(: {anchor} (HasConcept "{article_id}" "{lexical}") (STV 1 1))'
            ],
            "canonicalization": {
                "registry_sha256": "b" * 64,
                "mappings": {"concepts": [{
                    "canonical_id": concept_id, "lexical": lexical,
                }]},
            },
        },
    }


def concept_proof(plan, origin, path_index=0):
    candidate = origin.candidate_bridge_paths[path_index]
    history = next(
        path for path in origin.history_bridge_paths
        if path.canonical_concept_id == candidate.canonical_concept_id
    )
    return (
        f"(: (by {CONCEPT_CONTINUITY_RULE_ID} {plan.case_candidate_fact_id} "
        f"(by {ENGAGED_CONCEPT_RULE_ID} {origin.observed_click_fact_id} "
        f"(by {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
        f"{history.annotation_anchor_fact_id} {history.canonical_mapping_fact_id})) "
        f"(by {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
        f"{candidate.annotation_anchor_fact_id} {candidate.canonical_mapping_fact_id})) "
        f"(RelConceptContinuity {plan.case_id} {origin.origin_id} "
        f"{candidate.concept_atom}) "
        "(STV 1.0 0.999))"
    )


class RelationalWorkspaceTest(unittest.TestCase):
    def test_canonical_concept_plan_requires_named_anchors_and_is_proof_gated(self):
        concept = "concept:space~" + "c" * 64
        annotations = {
            "candidate": annotation("candidate", concept, "space"),
            "recent": annotation("recent", concept, "space"),
        }
        self.assertEqual(canonical_annotation_concepts(annotations["candidate"]), (concept,))
        plan = build_concept_relational_proof_plan(
            "candidate", ["recent"], annotations, user_id="u",
        )
        self.assertTrue(plan.requires_query)
        facts, ledger = reduce_concept_relational_proofs(
            plan, [concept_proof(plan, plan.origins[0])],
        )
        self.assertEqual(facts[REL_CONCEPT_CONTINUITY_SCOPE], "recent")
        self.assertEqual(len(facts[REL_CONCEPT_CONTINUITY_PROOF_IDS]), 1)
        record = next(iter(ledger.values()))
        self.assertEqual(record["matched_canonical_concept_ids"], [concept])
        self.assertEqual(record["annotation_anchor_fact_ids"], [
            "anchor_candidate", "anchor_recent",
        ])
        self.assertEqual(len(record["canonical_mapping_fact_ids"]), 2)
        retained = record["proof_alternatives"][0]
        self.assertEqual(
            retained["annotation_anchor_fact_ids"],
            ["anchor_candidate", "anchor_recent"],
        )
        self.assertTrue(all(
            fact_id in retained["proof_metta"]
            for fact_id in retained["source_fact_ids"]
        ))
        joined = "\n".join(plan.statements)
        self.assertIn(annotations["candidate"]["provenance"]["anchored_statements"][0], joined)
        self.assertIn("RelCanonicalConceptMapping", joined)
        self.assertNotIn("(RelHasCanonicalConcept", joined)
        entity_plan = build_relational_proof_plan(
            "candidate", ["recent"], articles(), user_id="u",
        )
        self.assertEqual(
            plan.origins[0].dependency_key,
            entity_plan.origins[0].dependency_key,
        )

        damaged = copy.deepcopy(annotations["candidate"])
        damaged["provenance"]["anchored_statements"] = []
        self.assertIsNone(canonical_annotation_concepts(damaged))

    def test_multiple_canonical_values_retain_the_proof_for_each_value(self):
        concepts = [
            "concept:space~" + "c" * 64,
            "concept:science~" + "d" * 64,
        ]

        def record(article_id):
            result = annotation(article_id, concepts[0], "space")
            result["concepts"].append(concepts[1])
            result["provenance"]["anchored_statements"].append(
                f'(: anchor_{article_id}_science '
                f'(HasConcept "{article_id}" "science") (STV 1 1))'
            )
            result["provenance"]["canonicalization"]["mappings"]["concepts"].append({
                "canonical_id": concepts[1], "lexical": "science",
            })
            return result

        annotations = {
            "candidate": record("candidate"),
            "history": record("history"),
        }
        plan = build_concept_relational_proof_plan(
            "candidate", ["history"], annotations, user_id="u",
        )
        origin = plan.origins[0]
        self.assertEqual(len(plan.proof_queries), 2)
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            reduce_concept_relational_proofs(
                plan, [concept_proof(plan, origin, 0)],
            )
        facts, ledger = reduce_concept_relational_proofs(
            plan, [concept_proof(plan, origin, 0), concept_proof(plan, origin, 1)],
        )
        self.assertEqual(facts[REL_CONCEPT_CONTINUITY_SCOPE], "recent")
        record = next(iter(ledger.values()))
        self.assertEqual(record["matched_canonical_concept_ids"], sorted(concepts))
        self.assertEqual(record["proof_alternative_count"], 2)
        self.assertEqual(len(record["proof_alternatives"]), 2)

    def test_exact_wikidata_ids_only_and_missing_is_not_empty(self):
        self.assertEqual(mind_wikidata_entities(articles()["candidate"]), ("Q1", "Q2"))
        self.assertEqual(mind_wikidata_entities(articles()["empty"]), ())
        self.assertIsNone(mind_wikidata_entities(articles()["unknown"]))
        self.assertIsNone(mind_wikidata_entities({
            "title_entities": [{"Label": "not a stable ID"}],
            "abstract_entities": [],
        }))

    def test_builder_surface_cannot_receive_an_outcome(self):
        signature = inspect.signature(build_relational_proof_plan)
        for forbidden in ("action", "click", "engagement", "label", "outcome", "score"):
            self.assertNotIn(forbidden, signature.parameters)
        with self.assertRaises(TypeError):
            build_relational_proof_plan(
                "candidate", ["old"], articles(), user_id="u", label=1,
            )

    def test_stable_plan_uses_original_last_five_positions(self):
        history = ["old", "different", "different", "different", "different", "recent"]
        first = build_relational_proof_plan(
            "candidate", history, articles(), user_id="u",
        )
        second = build_relational_proof_plan(
            "candidate", list(history), copy.deepcopy(articles()), user_id="u",
        )
        self.assertEqual(first, second)
        self.assertEqual(first.origins[0].recency, "older")
        self.assertEqual(first.origins[-1].recency, "recent")
        self.assertEqual(first.origins[0].matched_entity_ids, ("Q1", "Q2"))
        self.assertEqual(first.origins[-1].matched_entity_ids, ("Q2",))
        self.assertIn("RelObservedClick", "\n".join(first.statements))
        self.assertIn("RelCaseCandidate", "\n".join(first.statements))
        self.assertIsNotNone(first.query)
        self.assertEqual(len(first.proof_roots), 3)
        self.assertEqual(len(first.proof_queries), 3)
        self.assertEqual(len(set(first.proof_queries)), 3)
        self.assertTrue(all(
            "$origin" not in query and "$entity" not in query
            for query in first.proof_queries
        ))

    def test_repeated_article_positions_are_distinct_origins(self):
        plan = build_relational_proof_plan(
            "candidate", ["old", "old"], articles(), user_id="u",
        )
        self.assertNotEqual(plan.origins[0].origin_id, plan.origins[1].origin_id)
        self.assertNotEqual(
            plan.origins[0].observed_click_fact_id,
            plan.origins[1].observed_click_fact_id,
        )

    def test_multiple_entity_paths_from_one_click_are_one_origin(self):
        plan = build_relational_proof_plan(
            "candidate", ["old"], articles(), user_id="u",
        )
        origin = plan.origins[0]
        facts, ledger = reduce_relational_proofs(
            plan, [proof(plan, origin, 0), proof(plan, origin, 1)],
        )
        self.assertEqual(facts[REL_ENTITY_CONTINUITY_SCOPE], "recent")
        self.assertEqual(len(facts[REL_ENTITY_CONTINUITY_PROOF_IDS]), 1)
        self.assertEqual(len(ledger), 1)
        record = next(iter(ledger.values()))
        self.assertEqual(record["dependency_key"], origin.dependency_key)
        self.assertEqual(record["matched_wikidata_entity_ids"], ["Q1", "Q2"])
        self.assertEqual(record["proof_alternative_count"], 2)
        self.assertEqual(len(record["proof_alternatives"]), 2)
        self.assertTrue(all(
            all(fact_id in alternative["proof_metta"]
                for fact_id in alternative["source_fact_ids"])
            for alternative in record["proof_alternatives"]
        ))

    def test_recent_dominates_older_but_each_origin_remains_once(self):
        plan = build_relational_proof_plan(
            "candidate",
            ["old", "different", "different", "different", "different", "recent"],
            articles(), user_id="u",
        )
        older, recent = plan.origins[0], plan.origins[-1]
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            reduce_relational_proofs(
                plan, [proof(plan, older, 0), proof(plan, recent)],
            )
        facts, ledger = reduce_relational_proofs(
            plan, [
                proof(plan, older, 0), proof(plan, older, 1),
                proof(plan, recent), proof(plan, recent),
            ],
        )
        self.assertEqual(facts[REL_ENTITY_CONTINUITY_SCOPE], "recent")
        self.assertEqual(len(facts[REL_ENTITY_CONTINUITY_PROOF_IDS]), 2)
        self.assertEqual(len(ledger), 2)

    def test_none_and_unknown_are_distinct_and_have_no_fake_proof(self):
        known = build_relational_proof_plan(
            "candidate", ["different", "empty"], articles(), user_id="u",
        )
        unknown = build_relational_proof_plan(
            "candidate", ["different", "unknown"], articles(), user_id="u",
        )
        self.assertFalse(known.requires_query)
        self.assertFalse(unknown.requires_query)
        known_facts, known_ledger = reduce_relational_proofs(known, [])
        unknown_facts, unknown_ledger = reduce_relational_proofs(unknown, [])
        self.assertEqual(known_facts[REL_ENTITY_CONTINUITY_SCOPE], "none")
        self.assertEqual(unknown_facts[REL_ENTITY_CONTINUITY_SCOPE], "unknown")
        self.assertEqual(known_ledger, {})
        self.assertEqual(unknown_ledger, {})

    def test_older_entity_proof_abstains_when_recent_window_is_incomplete(self):
        plan = build_relational_proof_plan(
            "candidate",
            ["old", "different", "different", "different", "different", "unknown"],
            articles(), user_id="u",
        )
        self.assertFalse(plan.complete_recent_entity_evidence)
        older = plan.origins[0]
        facts, ledger = reduce_relational_proofs(
            plan, [proof(plan, older, 0), proof(plan, older, 1)],
        )
        self.assertEqual(facts[REL_ENTITY_CONTINUITY_SCOPE], "unknown")
        self.assertEqual(facts[REL_ENTITY_CONTINUITY_PROOF_IDS], [])
        self.assertEqual(ledger, {})

    def test_older_concept_proof_abstains_when_recent_window_is_incomplete(self):
        space = "concept:space~" + "c" * 64
        sports = "concept:sports~" + "d" * 64
        annotations = {
            "candidate": annotation("candidate", space, "space"),
            "old": annotation("old", space, "space"),
            "different": annotation("different", sports, "sports"),
        }
        plan = build_concept_relational_proof_plan(
            "candidate",
            ["old", "different", "different", "different", "different", "missing"],
            annotations, user_id="u",
        )
        self.assertFalse(plan.complete_recent_concept_evidence)
        facts, ledger = reduce_concept_relational_proofs(
            plan, [concept_proof(plan, plan.origins[0])],
        )
        self.assertEqual(facts[REL_CONCEPT_CONTINUITY_SCOPE], "unknown")
        self.assertEqual(facts[REL_CONCEPT_CONTINUITY_PROOF_IDS], [])
        self.assertEqual(ledger, {})

    def test_asserted_or_foreign_roots_cannot_pass_as_two_hop_proofs(self):
        plan = build_relational_proof_plan(
            "candidate", ["old"], articles(), user_id="u",
        )
        origin = plan.origins[0]
        asserted = [
            f"(: directly_asserted_{entity.lower()} "
            f"(RelEntityContinuity {plan.case_id} {origin.origin_id} "
            f"rel_entity_{entity.lower()}) (STV 1.0 1.0))"
            for entity in origin.matched_entity_ids
        ]
        with self.assertRaisesRegex(ValueError, "two-hop dependency"):
            reduce_relational_proofs(plan, asserted)
        foreign = proof(plan, origin).replace(origin.origin_id, "rel_origin_foreign")
        with self.assertRaisesRegex(ValueError, "coverage is incomplete or unexpected"):
            reduce_relational_proofs(plan, [foreign])
        extra_value = proof(plan, origin).replace("rel_entity_q1", "rel_entity_q9")
        with self.assertRaisesRegex(ValueError, "extra=1"):
            reduce_relational_proofs(plan, [
                extra_value, proof(plan, origin, 1),
            ])


if __name__ == "__main__":
    unittest.main()
