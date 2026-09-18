import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recommendation.app.server import (
    FEATURE_PROFILES,
    Lab,
    PAIR_FEATURE_PROFILES,
    RELATIONAL_CONCEPT_SCORE_PREDICATES,
    RELATIONAL_ENTITY_SCORE_PREDICATES,
    _validate_relational_score_dependencies,
)
from recommendation.evaluation.relational_ablation import (
    ARM_SPECS,
    RelationalAblationError,
    _atomic_publish_json,
    _digest,
    _evaluation_point_activation_audit,
    _rule_audit,
    _runtime_dependency_provenance,
    _source_hashes,
    assemble_report,
    cohort_audit,
    relational_dataset_audit,
)
from recommendation.features.relational_workspace import RELATIONAL_WORKSPACE_SCHEMA


def _dataset():
    return {
        "metadata": {
            "relational_workspace": {
                "schema": RELATIONAL_WORKSPACE_SCHEMA,
                "graph_sha256": "a" * 64,
                "rule_sha256": "b" * 64,
                "timings": {"projection_seconds": 1.25},
                "checks": {
                    "duplicate_proof_count": 0,
                    "future_history_reference_count": 0,
                },
            },
        },
        "relational_proof_ledger": {
            "p1": {
                "case_id": "case_1", "origin_id": "origin_1",
                "rule": "entity_continuity",
            },
            "p2": {
                "case_id": "case_2", "origin_id": "origin_2",
                "rule": "entity_continuity",
            },
        },
        "events": [
            {
                "id": "train_1", "user": "u1", "article": "h1",
                "history": [],
                "rel_entity_continuity_scope": "none",
                "rel_entity_continuity_proof_ids": [],
                "rel_concept_continuity_scope": "none",
                "rel_concept_continuity_proof_ids": [],
            },
        ],
        "evaluation": [
            {
                "id": "i1", "user": "u1", "history": ["h1"],
                "candidates": ["a", "b", "c"], "relevant": ["b"],
                "candidate_context": {
                    "a": {
                        "rel_entity_continuity_scope": "none",
                        "rel_entity_continuity_proof_ids": [],
                        "rel_concept_continuity_scope": "none",
                        "rel_concept_continuity_proof_ids": [],
                    },
                    "b": {
                        "rel_entity_continuity_scope": "recent",
                        "rel_entity_continuity_proof_ids": [],
                        "rel_concept_continuity_scope": "recent",
                        "rel_concept_continuity_proof_ids": [],
                    },
                    "c": {
                        "rel_entity_continuity_scope": "older",
                        "rel_entity_continuity_proof_ids": [],
                        "rel_concept_continuity_scope": "older",
                        "rel_concept_continuity_proof_ids": [],
                    },
                },
            },
        ],
    }


def _artifact(dataset_digest, *, arm, auc=0.5, order=("a", "b", "c")):
    modes = {"A": "disabled", "B": "facts_only", "C": "chained"}
    point = "accuracy_detail_relational" if arm == "C" else "accuracy_detail"
    pair = (
        "llm_conditional_quantile_relational"
        if arm == "C" else "llm_conditional_quantile"
    )
    ranked = [
        {
            "article": article,
            "ranking_score": float(3 - index),
            "score": float(3 - index),
            "stv": [0.5, 0.5],
            "point_rule_ids": (
                ["rel_rule"] if arm == "C" and article == "b" else []
            ),
            "relational_evidence": {
                "scopes": ({
                    "rel_concept_continuity_scope": "recent",
                } if article == "b" else {}),
                "proof_ids": ({
                    "rel_concept_continuity_proof_ids": ["p2"],
                } if article == "b" else {}),
            },
        }
        for index, article in enumerate(order)
    ]
    return {
        "dataset_sha256": dataset_digest,
        "result": {
            "config": {
                "random_seed": 7,
                "feature_profile": point,
                "pair_feature_profile": pair,
                "relational_evidence_mode": modes[arm],
            },
            "auc": auc,
            "auc_proof_only": auc,
            "mrr": 0.5,
            "ndcg_at_5": 0.6,
            "ndcg_at_10": 0.6,
            "cases": 1,
            "candidates": 3,
            "auc_cases": 1,
            "proof_cache": {
                "requested_mode": "force_cold",
                "observed_mode": "cold",
            },
            "auc_per_impression": [
                {
                    "index": 0, "id": "i1", "user": "u1",
                    "candidates": 3, "positives": 1, "negatives": 2,
                    "auc": auc, "auc_proof_only": auc,
                },
            ],
        },
        "point_rules": ([{"id": "rel_rule",
            "premises": [["rel_concept_continuity_scope", "recent"]],
        }] if arm == "C" else [{"premises": [["topic", "news"]]}]),
        "pair_rules": [],
        "compiled_point_channels": {},
        "compiled_pair_rules": {},
        "ranked_impressions": [ranked],
    }


class RelationalDatasetAuditTest(unittest.TestCase):
    def test_workspace_ledger_and_categorical_counts_are_bound(self):
        with patch(
            "recommendation.evaluation.relational_ablation.validate_relational_projection"
        ):
            audit = relational_dataset_audit(_dataset())
        self.assertEqual(audit["schema"], RELATIONAL_WORKSPACE_SCHEMA)
        self.assertEqual(audit["ledger"]["records"], 2)
        self.assertEqual(audit["ledger"]["duplicate_exact_records"], 0)
        self.assertEqual(audit["ledger"]["duplicate_primary_ids"], 0)
        self.assertEqual(
            audit["derived_field_counts"]["evaluation"]["fields"]
            ["rel_concept_continuity_scope"]["value_counts"],
            {"none": 1, "older": 1, "recent": 1},
        )
        self.assertEqual(
            audit["reported_audit_fields"][
                "checks.future_history_reference_count"
            ], 0,
        )

    def test_missing_workspace_or_ledger_fails_before_experiment(self):
        for key in ("workspace", "ledger"):
            data = _dataset()
            if key == "workspace":
                del data["metadata"]["relational_workspace"]
            else:
                del data["relational_proof_ledger"]
            with self.subTest(key=key), self.assertRaises(RelationalAblationError):
                relational_dataset_audit(data)


class RelationalCohortAuditTest(unittest.TestCase):
    def test_input_order_reconstructed_independently_of_rank_order(self):
        data = _dataset()
        digest = "d" * 64
        artifact = _artifact(digest, arm="A", order=("c", "a", "b"))
        audit = cohort_audit(artifact, data)
        row = audit["per_impression"][0]
        self.assertEqual(row["ordered_slate_sha256"], _digest(["a", "b", "c"]))
        self.assertEqual(row["ordered_relevant_sha256"], _digest(["b"]))
        self.assertEqual(row["ranked_output_sha256"], _digest(["c", "a", "b"]))

    def test_changed_candidate_set_is_rejected(self):
        artifact = _artifact("d" * 64, arm="A", order=("a", "b", "outside"))
        with self.assertRaisesRegex(RelationalAblationError, "candidate set"):
            cohort_audit(artifact, _dataset())


class RelationalReportTest(unittest.TestCase):
    def _assemble(self, root, *, mutate_b=False):
        data = _dataset()
        dataset_path = root / "data.json"
        dataset_path.write_text(json.dumps(data), encoding="utf-8")
        config_path = root / "config.json"
        config_path.write_text("{}", encoding="utf-8")
        digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        a = _artifact(digest, arm="A", auc=0.5)
        b = _artifact(
            digest, arm="B", auc=0.5,
            order=("b", "a", "c") if mutate_b else ("a", "b", "c"),
        )
        c = _artifact(digest, arm="C", auc=0.7, order=("b", "a", "c"))
        execution = {"wall_seconds": 1.0, "artifact_sha256": "f" * 64}
        with patch(
            "recommendation.evaluation.relational_ablation.validate_relational_projection"
        ):
            return assemble_report(
                data=data, dataset_path=dataset_path, dataset_sha256=digest,
                config_path=config_path,
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                arm_runs={
                    "A": (a, execution), "B": (b, execution), "C": (c, execution),
                },
                comparison_seed=37, bootstrap_repetitions=100,
                runtime={"python": "test"},
            )

    def test_three_arm_controls_and_paired_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = self._assemble(Path(temporary))
        self.assertTrue(report["controls"]["A_B_only_mode_differs"])
        self.assertTrue(report["controls"]["A_B_ranked_output_exact"])
        self.assertAlmostEqual(
            report["comparisons"]["C_minus_B"]["metrics"]["served"]["delta"],
            0.2,
        )
        self.assertNotIn(
            "ranked_impressions",
            report["arms"]["C"]["symbolic_artifact_without_ranked_rows"],
        )
        self.assertEqual(report["arms"]["C"]["rules"]["relational_point_rules"], 1)

    def test_inert_control_must_be_output_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RelationalAblationError, "inert"):
                self._assemble(Path(temporary), mutate_b=True)

    def test_atomic_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result.json"
            _atomic_publish_json(target, {"ok": True})
            self.assertEqual(json.loads(target.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                _atomic_publish_json(target, {"ok": False})
            self.assertEqual(json.loads(target.read_text()), {"ok": True})

    def test_arm_contract_names_modes_and_profiles(self):
        arms = {spec["id"]: spec["overrides"] for spec in ARM_SPECS}
        self.assertEqual(arms["A"]["relational_evidence_mode"], "disabled")
        self.assertEqual(arms["B"]["relational_evidence_mode"], "facts_only")
        self.assertEqual(arms["C"]["relational_evidence_mode"], "chained")
        self.assertEqual(
            arms["C"]["pair_feature_profile"],
            "llm_conditional_quantile_relational",
        )

    def test_selected_relational_arm_activates_only_one_dependency_family(self):
        arm = next(spec["overrides"] for spec in ARM_SPECS
                   if spec["id"] == "C")
        active = set(FEATURE_PROFILES[arm["feature_profile"]]) | set(
            PAIR_FEATURE_PROFILES[arm["pair_feature_profile"]]
        )
        self.assertTrue(active & RELATIONAL_CONCEPT_SCORE_PREDICATES)
        self.assertFalse(active & RELATIONAL_ENTITY_SCORE_PREDICATES)
        _validate_relational_score_dependencies(
            FEATURE_PROFILES[arm["feature_profile"]],
            PAIR_FEATURE_PROFILES[arm["pair_feature_profile"]],
        )

    def test_relational_dependency_guard_rejects_same_origin_double_vote(self):
        with self.assertRaisesRegex(
                ValueError, "shared-origin dependency-aware fusion"):
            _validate_relational_score_dependencies(
                ("rel_entity_continuity_scope",),
                ("pair_rel_concept_continuity_scope",),
            )
        with self.assertRaisesRegex(
                ValueError, "shared-origin dependency-aware fusion"):
            _validate_relational_score_dependencies(
                FEATURE_PROFILES["all"], PAIR_FEATURE_PROFILES["all"]
            )

        lab = object.__new__(Lab)
        lab.config = {
            "conjunctions": 2,
            "pair_conjunctions": 3,
            "pair_numeric_bins": 4,
            "max_rules": 30,
            "pair_max_rules": 40,
            "max_proof_cache_entries": 250_000,
            "max_pair_comparisons": 8_192,
            "max_total_pair_comparisons": 1_000_000,
            "mining_retention_max_units": 10_000,
            "mining_retention_max_cases": 100_000,
            "miner_strategy": "fixed_combinations",
            "feature_profile": "accuracy_detail",
            "pair_feature_profile": "llm_conditional_quantile",
            "relational_evidence_mode": "disabled",
            "max_feature_values": 24,
            "pair_aggregation": "proof_margin",
            "pair_ctv_mode": "raw_pairs",
            "pair_dependency_mode": "clustered",
        }
        with self.assertRaisesRegex(
                ValueError, "shared-origin dependency-aware fusion"):
            lab.configure({
                "feature_profile": "all",
                "relational_evidence_mode": "chained",
            })

    def test_rule_audit_reports_relational_premise_values(self):
        concept = "rel_concept_continuity_scope"
        entity = "rel_entity_continuity_scope"
        audit = _rule_audit({
            "point_rules": [
                {"premises": [[concept, "recent"], ["topic", "news"]]},
                {"premises": [[concept, "none"]]},
                {"premises": [[entity, "unknown"]]},
                {"premises": [["topic", "sports"]]},
            ],
            "pair_rules": [
                {"premises": [[f"pair_{concept}", "left"]]},
                {"premises": [[f"pair_{concept}", "equal"]]},
                {"premises": [[f"pair_{concept}", "right_known"]]},
                {"premises": [[f"pair_{entity}", "right"]]},
            ],
            "compiled_point_channels": {},
            "compiled_pair_rules": {},
        })
        concept_audit = audit["by_derived_field"][concept]
        self.assertEqual(
            concept_audit["point"]["rules_by_premise_value"],
            {"none": 1, "recent": 1},
        )
        self.assertEqual(
            concept_audit["pair"]["rules_by_premise_value"],
            {"equal": 1, "left": 1, "right_known": 1},
        )
        self.assertEqual(concept_audit["point"]["proof_positive_mined_rules"], 1)
        self.assertEqual(concept_audit["pair"]["proof_positive_mined_rules"], 1)
        self.assertEqual(audit["proof_positive_relational_point_rules"], 1)
        self.assertEqual(audit["proof_positive_relational_pair_rules"], 2)

    def test_point_activation_audit_requires_proof_ids_and_fired_rule(self):
        audit = _evaluation_point_activation_audit(
            _artifact("d" * 64, arm="C")
        )
        self.assertEqual(audit["proof_positive_candidate_rows"], 1)
        self.assertEqual(
            audit["fired_proof_positive_relational_point_rule_uses"], 1,
        )
        self.assertEqual(
            audit["fired_proof_positive_relational_point_rule_ids"],
            ["rel_rule"],
        )

    def test_runtime_audit_hashes_actual_imported_dependencies(self):
        runtime = _runtime_dependency_provenance()
        self.assertEqual(set(runtime), {"petta", "janus_swi", "pettachainer"})
        for package in runtime.values():
            self.assertTrue(package["version"])
            self.assertTrue(package["distribution"])
            for module in package["modules"].values():
                origin = Path(module["origin"])
                self.assertTrue(origin.is_file())
                self.assertEqual(module["bytes"], origin.stat().st_size)
                self.assertEqual(
                    module["sha256"], hashlib.sha256(origin.read_bytes()).hexdigest(),
                )
        self.assertNotIn(
            "/PeTTa/python/petta/",
            runtime["petta"]["modules"]["petta"]["origin"],
        )
        workspace_root = Path(__file__).resolve().parents[4]
        self.assertNotIn(
            "PeTTa/python/petta/__init__.py", _source_hashes(workspace_root),
        )


if __name__ == "__main__":
    unittest.main()
