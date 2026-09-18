import copy
from contextlib import redirect_stderr, redirect_stdout
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
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
    RELATIONAL_PROJECTION_SCHEMA,
    RELATIONAL_WORKSPACE_SCHEMA,
    build_relational_plans,
    relational_context_observations_sha256,
    relational_safety_audit,
    validate_relational_projection,
)
from recommendation.pipelines.relational_data import build_relational_projection, main


class StructuralProofReasoner:
    """Tiny executor fixture that emits nested proofs from submitted facts."""

    def __init__(self, expected):
        self.expected = expected
        self.added = []
        self.queries = []
        self.query_calls = []

    def add_atoms_no_check(self, atoms):
        self.added.extend(atoms)

    def query_many(self, queries, *, steps, timeout_sec):
        self.queries.extend(queries)
        self.query_calls.append((tuple(queries), steps, timeout_sec))
        results = []
        for query in queries:
            plan, origin, root = self.expected.get(query, (None, None, None))
            if plan is None:
                results.append([])
                continue
            if hasattr(origin, "matched_entity_ids"):
                path_index = origin.matched_entity_ids.index(root.matched_value_id)
                entity_atom = root.matched_value_atom
                results.append([(
                    f"(: (by {ENTITY_CONTINUITY_RULE_ID} "
                    f"{plan.case_candidate_fact_id} "
                    f"(by {ENGAGED_ENTITY_RULE_ID} {origin.observed_click_fact_id} "
                    f"{origin.history_entity_fact_ids[path_index]}) "
                    f"{origin.candidate_entity_fact_ids[path_index]}) "
                    f"(RelEntityContinuity {plan.case_id} {origin.origin_id} "
                    f"{entity_atom}) "
                    "(STV 1.0 0.999))"
                )])
            else:
                history = next(
                    path for path in origin.history_bridge_paths
                    if path.canonical_concept_id == root.matched_value_id
                )
                candidate = next(
                    path for path in origin.candidate_bridge_paths
                    if path.canonical_concept_id == root.matched_value_id
                )
                results.append([(
                    f"(: (by {CONCEPT_CONTINUITY_RULE_ID} "
                    f"{plan.case_candidate_fact_id} "
                    f"(by {ENGAGED_CONCEPT_RULE_ID} {origin.observed_click_fact_id} "
                    f"(by {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
                    f"{history.annotation_anchor_fact_id} "
                    f"{history.canonical_mapping_fact_id})) "
                    f"(by {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
                    f"{candidate.annotation_anchor_fact_id} "
                    f"{candidate.canonical_mapping_fact_id})) "
                    f"(RelConceptContinuity {plan.case_id} {origin.origin_id} "
                    f"{candidate.concept_atom}) "
                    "(STV 1.0 0.999))"
                )])
        return results


class RelationalProjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {
            "articles": [
                {"id": "C", "title": "candidate", "title_entities": ["Q1"], "abstract_entities": []},
                {"id": "H", "title": "history", "title_entities": ["Q1"], "abstract_entities": []},
                {"id": "D", "title": "different", "title_entities": ["Q9"], "abstract_entities": []},
            ],
            "users": {"u": {"history": ["future-mutable-profile"]}},
            "events": [
                {"user": "u", "article": "C", "history": ["H"], "action": "click", "impression": "i"},
                {"user": "u", "article": "D", "history": ["H"], "action": "skip", "impression": "i"},
            ],
            "tests": [{
                "id": "test", "user": "u", "history": ["H"],
                "candidates": ["C", "D"], "labels": {"C": 1, "D": 0},
                "relevant": ["C"],
                "candidate_context": {"C": {"prior": 0.2}, "D": {}},
            }],
            "llm_article_annotations": {
                "C": self.annotation("C", "space"),
                "H": self.annotation("H", "space"),
                "D": self.annotation("D", "sports"),
            },
            "metadata": {"dataset": "relational-fixture"},
        }

    @staticmethod
    def annotation(article_id, lexical):
        concept = f"concept:{lexical}~" + hashlib.sha256(lexical.encode()).hexdigest()
        return {
            "concepts": [concept],
            "provenance": {
                "article_id": article_id,
                "article_id": article_id,
                "article_content_sha256": "a" * 64,
                "anchored_statements": [
                    f'(: anchor_{article_id.lower()} (HasConcept "{article_id}" "{lexical}") (STV 1 1))'
                ],
                "canonicalization": {
                    "registry_sha256": "b" * 64,
                    "mappings": {"concepts": [{
                        "canonical_id": concept, "lexical": lexical,
                    }]},
                },
            },
        }

    def reasoner(self, source=None):
        data = source or self.source
        article_map = {article["id"]: article for article in data["articles"]}
        expected = {}
        for user, candidate, history in (("u", "C", ["H"]),):
            plans = build_relational_plans(
                candidate, history, article_map, data["llm_article_annotations"],
                user_id=user,
            )
            for plan in plans:
                origins = {origin.origin_id: origin for origin in plan.origins}
                for root in plan.proof_roots:
                    expected[root.query] = (plan, origins[root.origin_id], root)
        return StructuralProofReasoner(expected)

    def project(self, source=None):
        data = self.source if source is None else source
        return build_relational_projection(data, reasoner=self.reasoner(data), query_batch_size=1)

    @staticmethod
    def rehash_projection(result):
        encoded = json.dumps(
            result["relational_proof_ledger"], ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        audit = result["metadata"]["relational_workspace"]
        audit["proof_ledger_sha256"] = hashlib.sha256(encoded).hexdigest()
        audit["context_observations_sha256"] = relational_context_observations_sha256(result)

    def test_projection_preserves_input_and_labels_and_attaches_proof_refs(self):
        original = copy.deepcopy(self.source)
        result = self.project()
        self.assertEqual(self.source, original)
        self.assertEqual(result["events"][0]["action"], "click")
        self.assertEqual(result["tests"][0]["labels"], {"C": 1, "D": 0})
        self.assertEqual(result["events"][0][REL_ENTITY_CONTINUITY_SCOPE], "recent")
        self.assertEqual(result["events"][0][REL_CONCEPT_CONTINUITY_SCOPE], "recent")
        self.assertEqual(result["events"][1][REL_ENTITY_CONTINUITY_SCOPE], "none")
        self.assertEqual(result["events"][1][REL_CONCEPT_CONTINUITY_SCOPE], "none")
        self.assertEqual(
            result["tests"][0]["candidate_context"]["C"][REL_ENTITY_CONTINUITY_SCOPE],
            "recent",
        )
        refs = result["events"][0][REL_ENTITY_CONTINUITY_PROOF_IDS]
        self.assertEqual(len(refs), 1)
        self.assertIn(refs[0], result["relational_proof_ledger"])
        self.assertNotIn("label", result["relational_proof_ledger"][refs[0]])
        self.assertEqual(len(result["events"][0][REL_CONCEPT_CONTINUITY_PROOF_IDS]), 1)
        validate_relational_projection(result)

    def test_projection_queries_every_specific_root_with_per_root_budget(self):
        reasoner = self.reasoner()
        result = build_relational_projection(
            self.source, reasoner=reasoner, query_batch_size=2, query_steps=17,
        )
        queries = [query for call, _steps, _timeout in reasoner.query_calls for query in call]
        self.assertEqual(len(queries), 2)
        self.assertEqual(len(set(queries)), 2)
        self.assertTrue(all("$origin" not in query for query in queries))
        self.assertEqual(reasoner.query_calls[0][1], 34)
        audit = result["metadata"]["relational_workspace"]
        self.assertEqual(audit["queryable_plans"], 2)
        self.assertEqual(audit["expected_entity_proof_roots"], 1)
        self.assertEqual(audit["expected_concept_proof_roots"], 1)
        self.assertEqual(audit["expected_proof_roots"], 2)
        self.assertEqual(audit["queries_submitted"], 2)
        self.assertEqual(audit["complete_proof_roots"], 2)
        self.assertEqual(audit["wildcard_queries_submitted"], 0)
        self.assertEqual(audit["query_steps_per_root"], 17)
        self.assertEqual(audit["total_query_step_budget"], 34)

    def test_root_batch_shards_are_proof_equivalent_to_reference_workspace(self):
        reference = self.project()
        import recommendation.pipelines.relational_data as module
        previous = module._run_isolated_reasoner_shard
        shard_root_indices = []

        def execute(roots, **kwargs):
            shard_root_indices.append(tuple(index for index, _plan, _root in roots))
            return module._run_reasoner_shard(
                roots, self.reasoner(), **kwargs,
            )

        module._run_isolated_reasoner_shard = execute
        self.addCleanup(
            setattr, module, "_run_isolated_reasoner_shard", previous,
        )
        sharded = build_relational_projection(
            self.source, query_batch_size=1, query_steps=17,
            shard_root_size=1,
        )
        self.assertEqual(shard_root_indices, [(0,), (1,)])
        self.assertEqual(
            sharded["relational_proof_ledger"],
            reference["relational_proof_ledger"],
        )
        for before, after in zip(reference["events"], sharded["events"]):
            for field in (
                REL_ENTITY_CONTINUITY_SCOPE, REL_ENTITY_CONTINUITY_PROOF_IDS,
                REL_CONCEPT_CONTINUITY_SCOPE, REL_CONCEPT_CONTINUITY_PROOF_IDS,
            ):
                self.assertEqual(after[field], before[field])
        audit = sharded["metadata"]["relational_workspace"]
        self.assertEqual(audit["workspace_shard_count"], 2)
        self.assertEqual(audit["query_batches"], 2)
        validate_relational_projection(sharded)

    def test_statement_name_collision_is_rejected_before_sharding(self):
        import recommendation.pipelines.relational_data as module
        with self.assertRaisesRegex(ValueError, "statement name collision"):
            module._require_unique_statement_names((
                "(: fact_x (P A) (STV 1 1))",
                "(: fact_x (P B) (STV 1 1))",
            ))

    def test_projection_rejects_one_missing_specific_root(self):
        reasoner = self.reasoner()
        reasoner.expected.pop(next(iter(reasoner.expected)))
        with self.assertRaisesRegex(RuntimeError, "required relational proof root"):
            build_relational_projection(
                self.source, reasoner=reasoner, query_batch_size=2,
            )

    def test_projection_abstains_when_an_older_proof_has_unknown_recent_evidence(self):
        changed = copy.deepcopy(self.source)
        changed["articles"].append({
            "id": "N", "title": "neutral", "title_entities": ["Q8"],
            "abstract_entities": [],
        })
        changed["llm_article_annotations"]["N"] = self.annotation("N", "neutral")
        history = ["H", "N", "N", "N", "N", "missing"]
        for event in changed["events"]:
            event["history"] = list(history)
        changed["tests"][0]["history"] = list(history)

        article_map = {article["id"]: article for article in changed["articles"]}
        plans = build_relational_plans(
            "C", history, article_map, changed["llm_article_annotations"],
            user_id="u",
        )
        expected = {}
        for plan in plans:
            origins = {origin.origin_id: origin for origin in plan.origins}
            for root in plan.proof_roots:
                expected[root.query] = (plan, origins[root.origin_id], root)
        result = build_relational_projection(
            changed, reasoner=StructuralProofReasoner(expected),
            query_batch_size=2,
        )
        for context in (
            result["events"][0],
            result["tests"][0]["candidate_context"]["C"],
        ):
            self.assertEqual(context[REL_ENTITY_CONTINUITY_SCOPE], "unknown")
            self.assertEqual(context[REL_CONCEPT_CONTINUITY_SCOPE], "unknown")
            self.assertEqual(context[REL_ENTITY_CONTINUITY_PROOF_IDS], [])
            self.assertEqual(context[REL_CONCEPT_CONTINUITY_PROOF_IDS], [])
        self.assertEqual(result["relational_proof_ledger"], {})
        validate_relational_projection(result)

    def test_validator_rejects_stale_or_non_exact_projection_protocols(self):
        mutations = (
            ("schema", "mindplex-relational-continuity-proofs-v2", "metadata"),
            ("projection_schema", "mindplex-preserved-relational-projection-v1", "projection schema"),
            ("wildcard_queries_submitted", 1, "wildcard"),
            ("complete_proof_roots", 1, "incomplete"),
            ("expected_proof_roots", 1, "label-free proof plans"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                result = self.project()
                result["metadata"]["relational_workspace"][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_relational_projection(result)
        result = self.project()
        audit = result["metadata"]["relational_workspace"]
        self.assertEqual(audit["schema"], RELATIONAL_WORKSPACE_SCHEMA)
        self.assertEqual(audit["projection_schema"], RELATIONAL_PROJECTION_SCHEMA)

    def test_future_profile_and_outcomes_do_not_change_projected_observations(self):
        before = self.project()
        changed = copy.deepcopy(self.source)
        changed["users"]["u"]["history"] = ["D"] * 50
        changed["events"][0]["action"] = "skip"
        changed["events"][1]["action"] = "click"
        changed["tests"][0]["labels"] = {"C": 0, "D": 1}
        changed["tests"][0]["relevant"] = ["D"]
        after = self.project(changed)
        for event_before, event_after in zip(before["events"], after["events"]):
            self.assertEqual(
                {key: event_before[key] for key in (
                    REL_ENTITY_CONTINUITY_SCOPE, REL_ENTITY_CONTINUITY_PROOF_IDS,
                    REL_CONCEPT_CONTINUITY_SCOPE, REL_CONCEPT_CONTINUITY_PROOF_IDS,
                )},
                {key: event_after[key] for key in (
                    REL_ENTITY_CONTINUITY_SCOPE, REL_ENTITY_CONTINUITY_PROOF_IDS,
                    REL_CONCEPT_CONTINUITY_SCOPE, REL_CONCEPT_CONTINUITY_PROOF_IDS,
                )},
            )
        self.assertEqual(before["relational_proof_ledger"], after["relational_proof_ledger"])

    def test_exclusive_gzip_cli_records_source_and_timings(self):
        source = self.root / "source.json"
        source.write_text(json.dumps(self.source), encoding="utf-8")
        outputs = [self.root / "first.json.gz", self.root / "second.json.gz"]

        # main() uses spawned PeTTa shards by default. Keep this unit test
        # process-local by replacing only the shard execution boundary.
        import recommendation.pipelines.relational_data as module
        previous = module._run_isolated_reasoner_shard
        module._run_isolated_reasoner_shard = lambda roots, **kwargs: (
            module._run_reasoner_shard(roots, self.reasoner(), **kwargs)
        )
        self.addCleanup(
            setattr, module, "_run_isolated_reasoner_shard", previous,
        )
        for output in outputs:
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["--data", str(source), "--output", str(output)]), 0)
        decoded = []
        for output in outputs:
            with gzip.open(output, "rt", encoding="utf-8") as stream:
                decoded.append(json.load(stream))
        result = decoded[0]
        timing_keys = set(
            result["metadata"]["relational_workspace"]["timings_seconds"]
        )
        for item in decoded:
            item["metadata"]["relational_workspace"].pop("timings_seconds")
        self.assertEqual(decoded[0], decoded[1])
        self.assertEqual(
            result["metadata"]["relational_workspace"]["source_dataset_sha256"],
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            timing_keys,
            {
                "planning", "atomspace_insertion", "petta_queries",
                "shard_execution_wall",
                "reduction_and_ledger_serialization", "total_projection",
            },
        )
        before = outputs[0].read_bytes()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(["--data", str(source), "--output", str(outputs[0])])
        self.assertEqual(outputs[0].read_bytes(), before)
        self.assertFalse(list(self.root.glob(".first.json.gz.*")))

    def test_inconsistent_impression_histories_and_tampering_fail_closed(self):
        changed = copy.deepcopy(self.source)
        changed["events"][1]["history"] = ["D"]
        with self.assertRaisesRegex(ValueError, "different preceding histories"):
            self.project(changed)

        result = self.project()
        proof_id = next(iter(result["relational_proof_ledger"]))
        result["relational_proof_ledger"][proof_id]["proof_metta"] += " tampered"
        with self.assertRaisesRegex(ValueError, "ledger disagrees"):
            validate_relational_projection(result)

        result = self.project()
        result["events"][1][REL_ENTITY_CONTINUITY_SCOPE] = "unknown"
        with self.assertRaisesRegex(ValueError, "context observations disagree"):
            validate_relational_projection(result)

        result = self.project()
        result["metadata"]["relational_workspace"]["structural_rules_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "structural rules disagree"):
            validate_relational_projection(result)

    def test_safety_is_recomputed_and_every_reference_is_context_bound(self):
        for field, replacement in (
            ("user", "another-user"),
            ("article", "D"),
            ("history", ["D"]),
        ):
            result = self.project()
            result["events"][0][field] = replacement
            self.rehash_projection(result)
            computed = relational_safety_audit(result)
            self.assertGreater(computed["future_history_references"], 0)
            with self.assertRaisesRegex(
                ValueError, "was not computed|label-free proof plans",
            ):
                validate_relational_projection(result)

            # Even a caller that copies the computed nonzero value into the
            # metadata cannot turn a causal violation into a passing audit.
            result["metadata"]["relational_workspace"]["safety"] = computed
            with self.assertRaisesRegex(
                ValueError, "computed.*nonzero|label-free proof plans",
            ):
                validate_relational_projection(result)

        result = self.project()
        references = result["events"][0][REL_ENTITY_CONTINUITY_PROOF_IDS]
        references.append(references[0])
        self.assertEqual(
            relational_safety_audit(result)["duplicate_origin_contributions"], 1,
        )

    def test_retained_alternative_cannot_claim_an_absent_anchor(self):
        result = self.project()
        record = next(
            item for item in result["relational_proof_ledger"].values()
            if item["relation_family"] == "canonical_concept_continuity"
        )
        alternative = record["proof_alternatives"][0]
        anchor_id = alternative["annotation_anchor_fact_ids"][0]
        alternative["proof_metta"] = alternative["proof_metta"].replace(
            anchor_id, "foreign_anchor",
        )
        alternative["proof_sha256"] = hashlib.sha256(
            alternative["proof_metta"].encode("utf-8")
        ).hexdigest()
        record["proof_metta"] = alternative["proof_metta"]
        record["proof_sha256"] = alternative["proof_sha256"]
        self.rehash_projection(result)
        with self.assertRaisesRegex(ValueError, "dependency absent"):
            validate_relational_projection(result)


if __name__ == "__main__":
    unittest.main()
