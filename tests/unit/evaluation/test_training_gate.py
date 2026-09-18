import copy
import json
import threading
import unittest
from collections import OrderedDict

import recommendation.evaluation.training_gate as training_gate
from recommendation.evaluation.training_gate import (
    TrainingGateProtocol,
    compare_confirmation_runs,
    confirmation_slate_digest,
    prepare_training_confirmation,
    run_training_confirmation,
)


CONTEXT = ("history_size_bucket", "topic")


TEST_PROTOCOL = TrainingGateProtocol(
    bootstrap_repetitions=20,
    minimum_confirmation_impressions=1,
    minimum_confirmation_users=2,
    enforce_single_attempt=False,
)


def _data(actions=None):
    actions = actions or {
        "g1": ("click", "skip"),
        "g2": ("skip", "click"),
        "g3": ("click", "skip"),
        "g4": ("skip", "click"),
        "g5": ("click", "skip"),
        "g6": ("skip", "click"),
    }
    articles = [
        {"id": f"a{index}", "topic": "news"}
        for index in range(1, 2 * len(actions) + 1)
    ]
    events = []
    row = 1
    for group_index, (impression, outcomes) in enumerate(actions.items(), 1):
        user = "u1" if group_index % 2 else "u2"
        for candidate_index, action in enumerate(outcomes):
            article = f"a{2 * group_index - 1 + candidate_index}"
            events.append({
                "user": user, "article": article, "action": action,
                "impression": impression,
                "source_impression_id": f"source-{impression}",
                "timestamp": f"train-row-{row:09d}",
                "history_size_bucket": "small", "topic": "news",
            })
            row += 1
    return {
        "articles": articles,
        "users": {"u1": ["news"], "u2": ["news"]},
        "events": events,
        "eval_impressions": [{"id": "public-dev", "label": "must-not-be-read"}],
        "metadata": {"dataset": "gate-test"},
    }


def _run(rows):
    return {
        "cases": len(rows), "auc_cases": len(rows),
        "sampled_without_positive": 0, "auc_per_impression": rows,
    }


class TrainingSplitTest(unittest.TestCase):
    def test_split_is_chronological_whole_and_filters_only_after_cutoff(self):
        data = _data({
            "g1": ("click", "skip"),
            "g2": ("skip", "click"),
            "g3": ("click", "click"),  # Tail, but not AUC-eligible.
            "g4": ("skip", "click"),
        })
        # Source container order is not trusted; timestamps define chronology.
        by_group = {}
        for event in data["events"]:
            by_group.setdefault(event["impression"], []).append(event)
        data["events"] = by_group["g3"] + by_group["g1"] + by_group["g4"] + by_group["g2"]

        split = prepare_training_confirmation(
            data, context_features=CONTEXT, positive_actions={"click"},
            build_fraction=0.5,
        )
        self.assertEqual(
            [event["impression"] for event in split.data["events"]],
            ["g1", "g1", "g2", "g2"],
        )
        self.assertEqual([case["id"] for case in split.data["tests"]],
                         ["training_confirmation_g4"])
        self.assertEqual([row["id"] for row in split.expected], ["source-g4"])
        self.assertEqual(split.audit["confirmation_excluded_after_cutoff"], ["g3"])
        self.assertEqual(split.audit["build_impressions"], 2)
        self.assertNotIn("eval_impressions", split.data)

    def test_duplicate_effective_source_identity_fails_closed(self):
        data = _data()
        for event in data["events"]:
            if event["impression"] in {"g5", "g6"}:
                event["source_impression_id"] = "duplicate-tail-source"
        with self.assertRaisesRegex(ValueError, "duplicate effective impression"):
            prepare_training_confirmation(
                data, context_features=CONTEXT, positive_actions={"click"}
            )

    def test_context_is_causal_whitelist_and_public_dev_is_irrelevant(self):
        first = _data()
        second = copy.deepcopy(first)
        second["eval_impressions"][0]["label"] = "changed"
        left = prepare_training_confirmation(
            first, context_features=CONTEXT, positive_actions={"click"}
        )
        right = prepare_training_confirmation(
            second, context_features=CONTEXT, positive_actions={"click"}
        )
        self.assertEqual(left.audit["cohort_fingerprint"],
                         right.audit["cohort_fingerprint"])
        for case in left.data["tests"]:
            for context in case["candidate_context"].values():
                self.assertEqual(set(context), set(CONTEXT))
                self.assertNotIn("action", context)

        changed_context = copy.deepcopy(first)
        changed_context["events"][-1]["topic"] = "sports"
        changed = prepare_training_confirmation(
            changed_context, context_features=CONTEXT, positive_actions={"click"}
        )
        self.assertNotEqual(
            left.audit["cohort_fingerprint"], changed.audit["cohort_fingerprint"]
        )

        changed_user = copy.deepcopy(first)
        changed_user["users"]["u1"] = ["sports"]
        changed = prepare_training_confirmation(
            changed_user, context_features=CONTEXT, positive_actions={"click"}
        )
        self.assertNotEqual(
            left.audit["cohort_fingerprint"], changed.audit["cohort_fingerprint"]
        )

        changed_representation = copy.deepcopy(first)
        changed_representation["articles"][-1]["topic"] = "sports"
        changed = prepare_training_confirmation(
            changed_representation,
            context_features=CONTEXT,
            positive_actions={"click"},
        )
        self.assertNotEqual(
            left.audit["cohort_fingerprint"], changed.audit["cohort_fingerprint"]
        )

    def test_missing_snapshot_and_overlapping_impressions_fail_closed(self):
        missing = _data()
        missing["events"][0].pop("history_size_bucket")
        with self.assertRaisesRegex(ValueError, "authoritative causal"):
            prepare_training_confirmation(
                missing, context_features=CONTEXT, positive_actions={"click"}
            )

        overlap = _data()
        overlap["events"][0]["timestamp"] = "train-row-000000001"
        overlap["events"][1]["timestamp"] = "train-row-000000004"
        overlap["events"][2]["timestamp"] = "train-row-000000003"
        with self.assertRaisesRegex(ValueError, "chronology overlaps"):
            prepare_training_confirmation(
                overlap, context_features=CONTEXT, positive_actions={"click"}
            )

        tied = _data()
        tied["events"][2]["timestamp"] = tied["events"][1]["timestamp"]
        with self.assertRaisesRegex(ValueError, "overlaps or ties"):
            prepare_training_confirmation(
                tied, context_features=CONTEXT, positive_actions={"click"}
            )

    def test_inconsistent_history_snapshot_within_impression_fails_closed(self):
        inconsistent = _data()
        inconsistent["events"][1]["history_size_bucket"] = "large"

        with self.assertRaisesRegex(
            ValueError,
            "inconsistent authoritative causal context marker 'history_size_bucket'",
        ):
            prepare_training_confirmation(
                inconsistent,
                context_features=CONTEXT,
                positive_actions={"click"},
            )

    def test_comparison_uses_raw_paired_values_and_exact_identity(self):
        digest = confirmation_slate_digest(
            "u1", ("a", "b"), ("a",),
            {"a": {"topic": "news"}, "b": {"topic": "news"}},
        )
        expected = ({
            "index": 0, "id": "i1", "candidates": 2,
            "user": "u1",
            "positives": 1, "negatives": 1,
            "candidate_ids": ("a", "b"), "relevant_ids": ("a",),
            "slate_digest": digest,
        },)
        champion_row = {
            "index": 0, "id": "i1", "candidates": 2,
            "user": "u1",
            "positives": 1, "negatives": 1,
            "auc": 0.50001, "auc_proof_only": 0.4,
            "slate_digest": digest,
        }
        challenger_row = {**champion_row, "auc": 0.50002,
                          "auc_proof_only": 0.40001}
        result = compare_confirmation_runs(
            _run([champion_row]), _run([challenger_row]), expected,
            min_delta=0.0, bootstrap_seed=7, bootstrap_repetitions=20,
        )
        self.assertEqual(result["status"], "pass")
        self.assertAlmostEqual(result["mean_auc_delta"], 0.00001)
        malformed = {**challenger_row, "candidates": 3}
        with self.assertRaisesRegex(ValueError, "different confirmation"):
            compare_confirmation_runs(
                _run([champion_row]), _run([malformed]), expected,
                min_delta=0.0, bootstrap_seed=7, bootstrap_repetitions=20,
            )

        wrong_slate = {**challenger_row, "slate_digest": "0" * 64}
        with self.assertRaisesRegex(ValueError, "ordered slate"):
            compare_confirmation_runs(
                _run([champion_row]), _run([wrong_slate]), expected,
                min_delta=0.0, bootstrap_seed=7, bootstrap_repetitions=20,
            )

        wrong_user = {**challenger_row, "user": "u2"}
        with self.assertRaisesRegex(ValueError, "different confirmation user"):
            compare_confirmation_runs(
                _run([champion_row]), _run([wrong_user]), expected,
                min_delta=0.0, bootstrap_seed=7, bootstrap_repetitions=20,
            )

    def test_user_cluster_bootstrap_moves_repeated_user_impressions_together(self):
        expected = []
        champion_rows = []
        challenger_rows = []
        for index in range(100):
            user = "heavy-user" if index < 99 else "outlier-user"
            candidates = (f"p{index}", f"n{index}")
            context = {
                candidates[0]: {"topic": "news"},
                candidates[1]: {"topic": "news"},
            }
            digest = confirmation_slate_digest(
                user, candidates, (candidates[0],), context
            )
            identity = {
                "index": index, "id": f"i{index}", "user": user,
                "candidates": 2, "positives": 1, "negatives": 1,
                "slate_digest": digest,
            }
            expected.append({
                **identity, "candidate_ids": candidates,
                "relevant_ids": (candidates[0],),
            })
            champion_rows.append({
                **identity, "auc": 0.5, "auc_proof_only": 0.5,
            })
            delta = 0.1 if user == "heavy-user" else -0.1
            challenger_rows.append({
                **identity, "auc": 0.5 + delta,
                "auc_proof_only": 0.5 + delta,
            })

        result = compare_confirmation_runs(
            _run(champion_rows), _run(challenger_rows), expected,
            min_delta=0.0, bootstrap_seed=37, bootstrap_repetitions=5_000,
        )
        self.assertGreater(result["delta_95_ci"][0], 0.0)
        self.assertLess(result["user_cluster_delta_95_ci"][0], 0.0)
        self.assertTrue(result["impression_served_criterion_passed"])
        self.assertFalse(result["user_cluster_served_criterion_passed"])
        self.assertTrue(result["impression_proof_noninferiority_passed"])
        self.assertFalse(result["user_cluster_proof_noninferiority_passed"])
        self.assertEqual(result["user_cluster_bootstrap"]["clusters"], 2)
        self.assertEqual(result["status"], "hold")


class _FakeEngine:
    created = []

    def __init__(self):
        self.pid = len(self.created) + 100
        self.closed = False
        self.created.append(self)

    def close(self):
        self.closed = True
        self.pid = None


class _FakeLab:
    def __init__(self, data, *, symbolic_only=False, config=None):
        self.data = data
        self.symbolic_only = symbolic_only
        self.config = dict(config or {"random_seed": 7, "pair_margin_power": 1.0})
        self.lock = threading.RLock()
        self.engine = _FakeEngine()
        self._online_events = []
        self.version = 1
        self.pending_events = 0
        self.feed_cache = OrderedDict({"old": True})
        self._feed_sessions = {"old": True}
        self._feature_vocabulary = {"point": {"x"}}
        self._pair_feature_vocabulary = {"pair": {"left"}}
        self._pair_categorical_labels = {
            "left": f"margin={self.config.get('pair_margin_power')}"
        }
        self._numeric_pair_encoders = {}
        self.mined_rules = [{
            "id": "mined_1", "premises": (("topic", "news"),),
            "target": "click", "strength": 0.7, "confidence": 0.4,
            "negative_strength": 0.2, "negative_confidence": 0.3,
            "support": 11, "antecedent_support": 17, "joint_support": 11,
            "source": "recommendation/miner/fpMiner.metta",
            # Deliberately dangerous internal fields: an artifact projection
            # must never expose these row-level values.
            "coverage": {"secret-point-match-index"},
            "covered_row_indexes": ["secret-point-row-index"],
            "case_data": "secret-point-case-data",
        }]
        self.mined_output = ["real-fpminer-output"]
        self._point_rule_sources = ["point-shared-rule"]
        self._point_channel_sources = ["point-channel-rule"]
        self.pair_rules = [{
            "id": "pair_mined_cluster_1",
            "dependency_id": "pair_mined_cluster_1",
            "variant_id": "pair_mined_cluster_1_v1",
            "proof_channel_id": "pair_mined_cluster_1_v1",
            "premises": (("pair_topic", "left"),),
            "strength": 0.8, "confidence": 0.5,
            "negative_strength": 0.1, "negative_confidence": 0.4,
            "support": 9, "antecedent_support": 12, "joint_support": 9,
            "temporal_fold_effects": [0.2, 0.15, 0.1],
            "temporal_fold_weighted_supports": [3.0, 3.0, 3.0],
            "required_temporal_folds": 3,
            "source": "recommendation/miner/fpMiner.metta",
            "coverage": {"secret-pair-match-index"},
            "covered_row_indexes": ["secret-pair-row-index"],
            "outcomes": "secret-pair-outcomes",
        }]
        self._pair_rule_sources = ["pettachainer-rule"]
        self._active_rule_ids = {"point"}
        self._active_pair_rule_ids = {"pair"}
        self.last_pair_mining = {
            "rules": 1, "cases": 24, "source_cases": 24,
            "wins": 12, "losses": 12, "miner_calls": 2,
            "miner_strategy": "fixed_combinations",
            "fpminer_min_support": 4,
            "categorical_search": {
                "actual_miner": "recommendation/miner/fpMiner.metta",
                "host_generated_rules": 0,
                "emitted_with_side_predicate": 3,
                "eligible_scoped_rules": 2,
                "selected_scoped_rules": 1,
                "support_gate": "equal-impression mass",
                "eligible_scoped_rule_records": [
                    {"case_data": "secret-category-case-data"}
                ],
            },
            "target_search": {
                "candidate_patterns": 4,
                "audit": {"covered_rows": "secret-target-row-index"},
            },
        }
        self._click_base_rate = 0.5
        self._tie_break_stats = {"topic": 0.5}
        self.last_mining = {
            "rules": 1, "version": 1, "miner_calls": 1,
            "miner_strategy": "fixed_combinations", "source_cases": 8,
            "fpminer_min_support": 2,
            "events": "secret-point-events",
        }
        self.last_mined_at = 1.0
        self._popularity = {"a": 1}
        self._proof_cache = {"old": True}
        self._loaded_candidates = {"old"}
        self._loaded_point_channels = {"old"}
        self._point_channel_proof_cache = {"old": True}
        self._point_channel_templates = {"old": True}
        self._point_query_calls = 1
        self._point_query_roots = 1
        self._point_pruned_query_roots = 1
        self._point_channel_activations = 1
        self._point_reused_channel_activations = 1
        self._last_point_completeness = {"old": True}
        self._last_point_cache_stats = {"old": True}
        self._loaded_pairs = {"old"}
        self._loaded_pair_channels = {"old"}
        self._pair_proof_cache = {"old": True}
        self._pair_channel_proof_cache = {"old": True}
        self._pair_margin_cache = {"old": True}
        self._pair_case_attrs = {"old": True}
        self._pair_channel_templates = {"old": True}
        self._pair_proof_origins = {"old": True}
        self._pair_query_calls = 1
        self._pair_query_roots = 1
        self._pair_pruned_query_roots = 1
        self._pair_channel_activations = 1
        self._pair_reused_channel_activations = 1

    def benchmark(self, _config):
        boost = 0.2 if self.config.get("pair_margin_power") == 2.0 else 0.0
        rows = []
        for index, case in enumerate(self.data["tests"]):
            positives = len(case["relevant"])
            # Match Lab.benchmark's externally visible identity contract.
            # This prevents the gate tests from drifting back to synthetic
            # case IDs when source impression IDs are available.
            case_identity = case.get("source_impression_id") or case.get("id")
            rows.append({
                "index": index, "id": str(case_identity),
                "user": case["user"],
                "candidates": len(case["candidates"]), "positives": positives,
                "negatives": len(case["candidates"]) - positives,
                "auc": 0.4 + boost, "auc_proof_only": 0.4 + boost,
                "slate_digest": case["training_confirmation_slate_digest"],
            })
        return {
            **_run(rows), "id": "fake", "auc": 0.4 + boost,
            "auc_proof_only": 0.4 + boost, "candidates": sum(
                len(case["candidates"]) for case in self.data["tests"]
            ),
            "proof_coverage": 1.0, "pairwise_proof_coverage": 1.0,
            "rules": 1, "pair_rules": 1, "seconds": 0.01,
            "config": self.config.copy(),
        }


class _FailingFullStageLab(_FakeLab):
    def __init__(self, data, *, symbolic_only=False, config=None):
        if (len(data.get("events", ())) == 12
                and (config or {}).get("pair_margin_power") == 2.0):
            raise RuntimeError("simulated full-training stage failure")
        super().__init__(data, symbolic_only=symbolic_only, config=config)


class TrainingGateOrchestrationTest(unittest.TestCase):
    def setUp(self):
        _FakeEngine.created = []

    def _host(self):
        return _FakeLab(
            _data(), config={"random_seed": 7, "pair_margin_power": 1.0}
        )

    def test_hold_does_not_mutate_live_model_and_closes_ephemeral_workers(self):
        host = self._host()
        original_engine = host.engine
        original_config = host.config.copy()
        original_version = host.version
        result = run_training_confirmation(
            host, {
                "challenger_config": {"pair_margin_power": 2.0},
                "min_delta": 0.3, "promote": True,
            },
            context_features=CONTEXT, positive_actions={"click"},
            challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            _protocol=TEST_PROTOCOL,
        )
        self.assertEqual(result["status"], "hold")
        self.assertFalse(result["promoted"])
        self.assertNotIn("per_impression", result)
        self.assertIn("user_cluster_delta_95_ci", result)
        self.assertIn("user_cluster_proof_only_delta_95_ci", result)
        self.assertEqual(result["user_cluster_bootstrap"]["clusters"], 2)
        self.assertNotIn("per_user", result)
        self.assertTrue(result["require_proof_noninferiority"])
        self.assertEqual(result["protocol"]["bootstrap_seed"], 37)
        self.assertEqual(result["protocol"]["minimum_confirmation_users"], 2)
        self.assertIs(host.engine, original_engine)
        self.assertFalse(original_engine.closed)
        self.assertEqual(host.config, original_config)
        self.assertEqual(host.version, original_version)
        self.assertTrue(all(engine.closed for engine in _FakeEngine.created[1:]))

    def test_public_protocol_rejects_overrides_and_requires_enough_tail_data(self):
        host = self._host()
        self.assertEqual(training_gate.PUBLIC_MIN_CONFIRMATION_USERS, 30)
        self.assertEqual(TrainingGateProtocol().minimum_confirmation_users, 30)
        with self.assertRaisesRegex(ValueError, "public confirmation protocol is fixed"):
            run_training_confirmation(
                host, {
                    "challenger_config": {"pair_margin_power": 2.0},
                    "bootstrap_repetitions": 20,
                },
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            )
        with self.assertRaisesRegex(ValueError, "public confirmation protocol is fixed"):
            run_training_confirmation(
                host, {
                    "challenger_config": {"pair_margin_power": 2.0},
                    "minimum_confirmation_users": 2,
                },
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            )
        with self.assertRaisesRegex(ValueError, "at least 100 eligible"):
            run_training_confirmation(
                host, {"challenger_config": {"pair_margin_power": 2.0}},
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            )
        with self.assertRaisesRegex(ValueError, "mandatory"):
            run_training_confirmation(
                host, {
                    "challenger_config": {"pair_margin_power": 2.0},
                    "require_proof_noninferiority": False,
                },
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
                _protocol=TEST_PROTOCOL,
            )

        one_user_data = _data()
        for event in one_user_data["events"]:
            event["user"] = "u1"
        one_user_host = _FakeLab(
            one_user_data,
            config={"random_seed": 7, "pair_margin_power": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "at least 2 distinct tail users"):
            run_training_confirmation(
                one_user_host,
                {"challenger_config": {"pair_margin_power": 2.0}},
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"},
                lab_factory=_FakeLab, _protocol=TEST_PROTOCOL,
            )

        many_actions = OrderedDict(
            (f"g{index:03d}", ("click", "skip")) for index in range(300)
        )
        few_user_host = _FakeLab(
            _data(many_actions),
            config={"random_seed": 7, "pair_margin_power": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "at least 30 distinct tail users"):
            run_training_confirmation(
                few_user_host,
                {"challenger_config": {"pair_margin_power": 2.0}},
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            )

    def test_immutable_confirmation_cohort_is_single_use(self):
        host = self._host()
        registry = {}
        protocol = TrainingGateProtocol(
            bootstrap_repetitions=10,
            minimum_confirmation_impressions=1,
            minimum_confirmation_users=2,
            enforce_single_attempt=True,
        )
        request = {"challenger_config": {"pair_margin_power": 2.0}}
        first = run_training_confirmation(
            host, request,
            context_features=CONTEXT, positive_actions={"click"},
            challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            _protocol=protocol, _attempt_registry=registry,
            _attempt_registry_lock=threading.Lock(),
        )
        self.assertEqual(first["status"], "pass")
        with self.assertRaisesRegex(ValueError, "already been used"):
            run_training_confirmation(
                host, request,
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
                _protocol=protocol, _attempt_registry=registry,
                _attempt_registry_lock=threading.Lock(),
            )

    def test_model_fingerprint_binds_learned_and_representation_state(self):
        baseline_lab = self._host()
        baseline = training_gate._model_fingerprint(baseline_lab)
        mutations = (
            lambda lab: setattr(lab, "_feature_vocabulary", {"point": {"y"}}),
            lambda lab: setattr(lab, "_pair_feature_vocabulary", {"pair": {"right"}}),
            lambda lab: setattr(lab, "_pair_categorical_labels", {"right": "topic=sports"}),
            lambda lab: setattr(lab, "_numeric_pair_encoders", {"score": {"cuts": [0.2]}}),
            lambda lab: setattr(lab, "_click_base_rate", 0.25),
            lambda lab: setattr(lab, "_tie_break_stats", {"topic": 0.25}),
            lambda lab: lab.data["articles"][0].update(topic="sports"),
            lambda lab: lab.last_pair_mining["categorical_search"].update(
                support_gate="changed aggregate calibration policy"
            ),
        )
        try:
            for mutate in mutations:
                candidate = self._host()
                try:
                    mutate(candidate)
                    self.assertNotEqual(
                        baseline, training_gate._model_fingerprint(candidate)
                    )
                finally:
                    candidate.engine.close()
        finally:
            baseline_lab.engine.close()

    def test_summaries_persist_compact_build_mining_audit_without_rows(self):
        host = self._host()
        result = run_training_confirmation(
            host, {
                "challenger_config": {"pair_margin_power": 2.0},
                "min_delta": 0.3,
            },
            context_features=CONTEXT, positive_actions={"click"},
            challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            _protocol=TEST_PROTOCOL,
        )
        for model in ("champion", "challenger"):
            audit = result[model]["build_mining_audit"]
            self.assertEqual(
                audit["schema"], "recommendation-build-mining-audit-v1"
            )
            self.assertEqual(
                audit["fit_scope"], "chronological training build prefix only"
            )
            self.assertEqual(
                audit["actual_pattern_miner"],
                "recommendation/miner/fpMiner.metta",
            )
            self.assertEqual(audit["host_generated_categorical_rules"], 0)
            point = audit["point"]["selected_rules"][0]
            self.assertEqual(point["premises"][0]["value"], "news")
            self.assertEqual(point["conclusion"]["action"], "click")
            self.assertEqual(point["stv"]["positive"]["strength"], 0.7)
            self.assertEqual(point["support"]["joint_support"], 11)
            pair = audit["pair"]["selected_rules"][0]
            self.assertEqual(
                pair["provenance"]["dependency_id"], "pair_mined_cluster_1"
            )
            self.assertEqual(
                pair["temporal_validation"]["required_temporal_folds"], 3
            )
            category = audit["pair"]["categorical_search"]
            self.assertEqual(category["eligible_scoped_rules"], 2)
            self.assertEqual(category["support_gate"], "equal-impression mass")
            self.assertNotIn("eligible_scoped_rule_records", category)

            serialized = json.dumps(audit, sort_keys=True)
            for secret in (
                "secret-point-match-index", "secret-point-row-index",
                "secret-point-case-data", "secret-point-events",
                "secret-pair-match-index", "secret-pair-row-index",
                "secret-pair-outcomes", "secret-category-case-data",
                "secret-target-row-index",
            ):
                self.assertNotIn(secret, serialized)

            forbidden_keys = {
                "coverage", "covered_row_indexes", "case_data", "outcomes",
                "events", "per_impression", "candidate_context", "relevant",
            }

            def visit(value):
                if isinstance(value, dict):
                    self.assertFalse(forbidden_keys.intersection(value))
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(audit)

        self.assertNotEqual(
            result["champion"]["model_fingerprint"],
            result["challenger"]["model_fingerprint"],
        )

    def test_pass_stages_full_training_model_then_atomically_promotes(self):
        host = self._host()
        old_engine = host.engine
        result = run_training_confirmation(
            host, {
                "challenger_config": {"pair_margin_power": 2.0},
                "promote": True,
            },
            context_features=CONTEXT, positive_actions={"click"},
            challenger_config_keys={"pair_margin_power"}, lab_factory=_FakeLab,
            _protocol=TEST_PROTOCOL,
        )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["promoted"])
        self.assertTrue(old_engine.closed)
        self.assertEqual(host.config["pair_margin_power"], 2.0)
        self.assertEqual(host._pair_categorical_labels, {"left": "margin=2.0"})
        self.assertEqual(host.version, 2)
        self.assertIsNotNone(host.engine.pid)
        self.assertFalse(host.feed_cache)
        self.assertFalse(host._proof_cache)
        self.assertEqual(host._point_channel_sources,["point-channel-rule"])
        self.assertFalse(host._loaded_point_channels)
        self.assertFalse(host._point_channel_proof_cache)
        self.assertFalse(host._point_channel_templates)
        self.assertFalse(host._last_point_completeness)
        # Two evaluation workers are closed; the staged worker now belongs to host.
        self.assertTrue(_FakeEngine.created[1].closed)
        self.assertTrue(_FakeEngine.created[2].closed)
        self.assertFalse(host.engine.closed)

    def test_full_training_stage_failure_preserves_live_state(self):
        host = self._host()
        old_engine = host.engine
        old_config = host.config.copy()
        old_version = host.version
        old_cache = host.feed_cache
        with self.assertRaisesRegex(RuntimeError, "stage failure"):
            run_training_confirmation(
                host, {
                    "challenger_config": {"pair_margin_power": 2.0},
                    "promote": True,
                },
                context_features=CONTEXT, positive_actions={"click"},
                challenger_config_keys={"pair_margin_power"},
                lab_factory=_FailingFullStageLab,
                _protocol=TEST_PROTOCOL,
            )
        self.assertIs(host.engine, old_engine)
        self.assertFalse(old_engine.closed)
        self.assertEqual(host.config, old_config)
        self.assertEqual(host.version, old_version)
        self.assertIs(host.feed_cache, old_cache)
        self.assertTrue(all(engine.closed for engine in _FakeEngine.created[1:]))


if __name__ == "__main__":
    unittest.main()
