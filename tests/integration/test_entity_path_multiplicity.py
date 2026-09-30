"""PeTTa proves categorical degree over distinct clicked-origin nodes."""

import unittest

from recommendation.features.relational_workspace import (
    ENTITY_PATH_CASE_NODE_RULE_ID,
    ENTITY_PATH_COUNT_RULE_ID,
    ENTITY_PATH_NONE_RULE_ID,
    ENTITY_PATH_ONE_RULE_ID,
    ENTITY_PATH_ORIGIN_NODE_RULE_ID,
    ENTITY_PATH_TWO_PLUS_RULE_ID,
    RELATIONAL_STRUCTURAL_RULES,
    REL_ENTITY_PATH_MULTIPLICITY,
    REL_ENTITY_PATH_MULTIPLICITY_PROOF_IDS,
    build_relational_proof_plan,
    reduce_entity_path_multiplicity_proofs,
)
from recommendation.pipelines.relational_data import (
    _run_isolated_query_workspace,
)


def _articles():
    return {
        "candidate": {
            "title_entities": ["Q1", "Q2"], "abstract_entities": [],
        },
        "history": {
            "title_entities": ["Q1", "Q2"], "abstract_entities": [],
        },
        "different": {
            "title_entities": ["Q9"], "abstract_entities": [],
        },
        "unknown": {"title": "entity annotations are unavailable"},
    }


class EntityPathMultiplicityPeTTaTest(unittest.TestCase):
    def test_petta_classifies_zero_one_and_two_distinct_origins(self):
        articles = _articles()
        plans = {
            "none": build_relational_proof_plan(
                "candidate", ["different"], articles, user_id="user",
            ),
            # Two shared entities are alternate paths from one click origin,
            # and therefore remain one graph neighbor.
            "one": build_relational_proof_plan(
                "candidate", ["history"], articles, user_id="user",
            ),
            # Repeated article IDs are two distinct historical interactions.
            "two_plus": build_relational_proof_plan(
                "candidate", ["history", "history"], articles, user_id="user",
            ),
        }
        statements=sorted({
            *RELATIONAL_STRUCTURAL_RULES,
            *(statement for plan in plans.values() for statement in plan.statements),
        })
        query_results=_run_isolated_query_workspace(
            statements,
            [((plan.entity_path_multiplicity_query,),2_000)
             for plan in plans.values()],
        )

        for (expected, plan),result in zip(plans.items(),query_results):
            with self.subTest(expected=expected):
                self.assertTrue(plan.requires_entity_path_multiplicity_query)
                proofs=result[0]
                facts, ledger = reduce_entity_path_multiplicity_proofs(
                    plan, proofs,
                )
                self.assertEqual(facts[REL_ENTITY_PATH_MULTIPLICITY], expected)
                self.assertEqual(
                    len(facts[REL_ENTITY_PATH_MULTIPLICITY_PROOF_IDS]), 1,
                )
                record = next(iter(ledger.values()))
                expected_origins = {"none": 0, "one": 1, "two_plus": 2}[expected]
                self.assertEqual(len(record["origin_ids"]), expected_origins)
                self.assertEqual(
                    len(record["origin_ids"]),
                    len(set(record["dependency_keys"])),
                )
                self.assertIn("foldall-proof", record["proof_metta"])

        incomplete = build_relational_proof_plan(
            "candidate", ["unknown"], articles, user_id="user",
        )
        self.assertIsNone(incomplete.entity_path_multiplicity_query)
        self.assertEqual(
            reduce_entity_path_multiplicity_proofs(incomplete, ()), ({}, {}),
        )

    def test_classifier_rules_are_certain_and_forged_category_fails_closed(self):
        rules = "\n".join(RELATIONAL_STRUCTURAL_RULES)
        for rule_id in (
            ENTITY_PATH_CASE_NODE_RULE_ID,
            ENTITY_PATH_ORIGIN_NODE_RULE_ID,
            ENTITY_PATH_COUNT_RULE_ID,
            ENTITY_PATH_NONE_RULE_ID,
            ENTITY_PATH_ONE_RULE_ID,
            ENTITY_PATH_TWO_PLUS_RULE_ID,
        ):
            statement = next(
                rule for rule in RELATIONAL_STRUCTURAL_RULES if rule_id in rule
            )
            self.assertIn("(CTV (STV 1.0 1.0) (STV 0.0 1.0))", statement)
        self.assertIn("(FoldAll (RelEntityPathNode", rules)

        plan = build_relational_proof_plan(
            "candidate", ["history", "history"], _articles(), user_id="user",
        )
        partial_result,complete_result=_run_isolated_query_workspace([
            *RELATIONAL_STRUCTURAL_RULES, *plan.statements,
        ],[
            ((plan.entity_path_multiplicity_query,),32),
            ((plan.entity_path_multiplicity_query,),2_000),
        ])
        partial=partial_result[0]
        self.assertEqual(len(partial), 1)
        self.assertIn(
            f"(RelEntityPathMultiplicity {plan.case_id} none)", partial[0],
        )
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            reduce_entity_path_multiplicity_proofs(plan, partial)

        proofs=complete_result[0]
        with self.assertRaisesRegex(ValueError, "exactly one"):
            reduce_entity_path_multiplicity_proofs(plan, [])
        with self.assertRaises(ValueError):
            reduce_entity_path_multiplicity_proofs(plan, [proofs[0][:-40]])
        forged = proofs[0].replace("two_plus", "one")
        with self.assertRaises(ValueError):
            reduce_entity_path_multiplicity_proofs(plan, [forged])


if __name__ == "__main__":
    unittest.main()
