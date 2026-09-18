"""The generated relational plans execute as real two-hop PeTTa proofs."""

import hashlib
import unittest

from recommendation.features.relational_workspace import (
    CANONICAL_CONCEPT_BRIDGE_RULE_ID,
    RELATIONAL_STRUCTURAL_RULES,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_SCOPE,
    build_relational_plans,
    reduce_concept_relational_proofs,
    reduce_relational_proofs,
)
from recommendation.pipelines.relational_data import _new_reasoner
from recommendation.pipelines.relational_data import build_relational_projection


def annotation(article_id, *lexicals):
    concepts = [
        f"concept:{lexical}~" + hashlib.sha256(lexical.encode()).hexdigest()
        for lexical in lexicals
    ]
    return {
        "concepts": concepts,
        "provenance": {
            "article_id": article_id,
            "article_content_sha256": "a" * 64,
            "anchored_statements": [
                f'(: anchor_{article_id.lower()} '
                f'(HasConcept "{article_id}" "{lexical}") (STV 1 1))'
                if index == 0 else
                f'(: anchor_{article_id.lower()}_{index} '
                f'(HasConcept "{article_id}" "{lexical}") (STV 1 1))'
                for index, lexical in enumerate(lexicals)
            ],
            "canonicalization": {
                "registry_sha256": "b" * 64,
                "mappings": {"concepts": [
                    {
                        "canonical_id": concept,
                        "lexical": lexical,
                        "raw": lexical,
                    }
                    for concept, lexical in zip(concepts, lexicals)
                ]},
            },
        },
    }


class RelationalPeTTaProofTest(unittest.TestCase):
    def test_spawned_shards_match_one_workspace_with_real_petta(self):
        source = {
            "articles": [
                {"id": "candidate", "title_entities": ["Q42"], "abstract_entities": []},
                {"id": "history", "title_entities": ["Q42"], "abstract_entities": []},
            ],
            "users": {"user": {"history": ["history"]}},
            "events": [{
                "user": "user", "article": "candidate", "history": ["history"],
                "action": "click", "impression": "train",
            }],
            "tests": [{
                "id": "eval", "user": "user", "history": ["history"],
                "candidates": ["candidate"], "labels": {"candidate": 1},
                "relevant": ["candidate"], "candidate_context": {"candidate": {}},
            }],
            "llm_article_annotations": {
                "candidate": annotation("candidate", "reasoning"),
                "history": annotation("history", "reasoning"),
            },
        }
        one = build_relational_projection(
            source, query_batch_size=1, query_steps=64, shard_root_size=100,
        )
        many = build_relational_projection(
            source, query_batch_size=1, query_steps=64, shard_root_size=1,
            shard_workers=2,
        )
        self.assertEqual(
            one["relational_proof_ledger"], many["relational_proof_ledger"],
        )
        for split in ("events", "tests"):
            self.assertEqual(one[split], many[split])
        self.assertEqual(
            one["metadata"]["relational_workspace"]["context_observations_sha256"],
            many["metadata"]["relational_workspace"]["context_observations_sha256"],
        )
        self.assertEqual(
            one["metadata"]["relational_workspace"]["workspace_shard_count"], 1,
        )
        self.assertEqual(
            many["metadata"]["relational_workspace"]["workspace_shard_count"], 2,
        )

    def test_entity_and_concept_continuity_are_real_nested_proofs(self):
        articles = {
            "candidate": {"title_entities": ["Q42"], "abstract_entities": []},
            "history": {"title_entities": [], "abstract_entities": ["Q42"]},
        }
        annotations = {
            "candidate": annotation("candidate", "reasoning", "symbolic"),
            "history": annotation("history", "reasoning", "symbolic"),
        }
        entity_plan, concept_plan = build_relational_plans(
            "candidate", ["history"], articles, annotations, user_id="user",
        )
        engine = _new_reasoner()
        engine.add_atoms_no_check(sorted({
            *RELATIONAL_STRUCTURAL_RULES,
            *entity_plan.statements,
            *concept_plan.statements,
        }))
        results = engine.query_many(
            [entity_plan.query, concept_plan.query], steps=2_000, timeout_sec=0,
        )
        entity_facts, entity_ledger = reduce_relational_proofs(
            entity_plan, results[0],
        )
        concept_facts, concept_ledger = reduce_concept_relational_proofs(
            concept_plan, results[1],
        )
        self.assertEqual(entity_facts[REL_ENTITY_CONTINUITY_SCOPE], "recent")
        self.assertEqual(concept_facts[REL_CONCEPT_CONTINUITY_SCOPE], "recent")
        self.assertEqual(len(entity_ledger), 1)
        self.assertEqual(len(concept_ledger), 1)
        self.assertEqual(
            next(iter(entity_ledger.values()))["dependency_key"],
            next(iter(concept_ledger.values()))["dependency_key"],
        )
        concept_record = next(iter(concept_ledger.values()))
        self.assertIn(CANONICAL_CONCEPT_BRIDGE_RULE_ID, concept_record["proof_metta"])
        self.assertEqual(
            concept_record["annotation_anchor_fact_ids"],
            [
                "anchor_candidate", "anchor_candidate_1",
                "anchor_history", "anchor_history_1",
            ],
        )
        self.assertEqual(len(concept_record["canonical_mapping_fact_ids"]), 4)
        self.assertTrue(all(
            source_id in alternative["proof_metta"]
            for alternative in concept_record["proof_alternatives"]
            for source_id in alternative["source_fact_ids"]
        ))
        self.assertEqual(
            concept_record["matched_canonical_concept_ids"],
            sorted(annotations["candidate"]["concepts"]),
        )
        self.assertGreaterEqual(concept_record["proof_alternative_count"], 2)

    def test_twenty_specific_roots_are_complete_and_wildcard_prefix_fails_closed(self):
        history_ids = [f"history_{index}" for index in range(20)]
        articles = {
            "candidate": {"title_entities": ["Q42"], "abstract_entities": []},
            **{
                article_id: {
                    "title_entities": ["Q42"], "abstract_entities": [],
                }
                for article_id in history_ids
            },
        }
        from recommendation.features.relational_workspace import (
            build_relational_proof_plan,
        )

        plan = build_relational_proof_plan(
            "candidate", history_ids, articles, user_id="user",
        )
        self.assertEqual(len(plan.proof_roots), 20)
        self.assertEqual(len(plan.proof_queries), 20)
        self.assertEqual(len(set(plan.proof_queries)), 20)

        engine = _new_reasoner()
        engine.add_atoms_no_check(sorted({
            *RELATIONAL_STRUCTURAL_RULES, *plan.statements,
        }))
        results = engine.query_many(
            plan.proof_queries, steps=32 * len(plan.proof_queries), timeout_sec=0,
        )
        self.assertEqual(len(results), 20)
        self.assertTrue(all(results))
        facts, ledger = reduce_relational_proofs(
            plan, [proof for result in results for proof in result],
        )
        self.assertEqual(facts[REL_ENTITY_CONTINUITY_SCOPE], "recent")
        self.assertEqual(len(ledger), 20)
        self.assertEqual(
            sum(record["recency"] == "older" for record in ledger.values()), 15,
        )
        self.assertEqual(
            sum(record["recency"] == "recent" for record in ledger.values()), 5,
        )

        prefix = None
        for steps in (8, 16, 32, 64, 128, 256):
            candidate = engine.query(plan.query, steps=steps, timeout_sec=0)
            if 0 < len(candidate) < len(plan.proof_roots):
                prefix = candidate
                break
        self.assertIsNotNone(prefix, "expected a nonempty low-budget wildcard prefix")
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            reduce_relational_proofs(plan, prefix)


if __name__ == "__main__":
    unittest.main()
