"""Grounded annotations are observations; real mined PeTTa proofs rank them."""

import copy
import hashlib
import unittest
from unittest.mock import patch

from recommendation.app.server import (
    Lab, LLM_PAIR_PREDICATES, LLM_QUANTILE_PAIR_PREDICATES,
    PAIR_CATEGORICAL_SIDE_PREDICATES,
    PAIR_FEATURE_PROFILES,
)
from recommendation.pipelines.llm_data import build_llm_projection
from recommendation.features.llm_workspace import (
    LLM_NUMERIC_FEATURES, LLM_WORKSPACE_FEATURES, build_llm_workspace_facts,
)
from recommendation.core.ctv_calibration import CTVObservation, calibrate_ctv
from recommendation.core.symbolic import QuantileNumericEvidence
from recommendation.features.text_embeddings import article_text
from recommendation.tests.fixtures import (
    conditional_annotation_fixture,
    lexical_fixture,
)


def _annotation_fixture():
    source = lexical_fixture()
    source.pop("article_text_vectors")
    source["metadata"].pop("text_embedding_sidecar")
    source["tests"][0]["id"] = "evaluation_1"
    histories = copy.deepcopy(source)
    for event in histories["events"]:
        event["history"] = ["past"]
    annotations = {}
    for article in source["articles"]:
        annotations[article["id"]] = {
            "concepts": ["astronomy" if article["id"] != "unrelated" else "football"],
            "format": "report", "event_types": [], "intents": [], "audiences": [],
            "provenance": {"article_content_sha256": hashlib.sha256(
                article_text(article["title"], article["abstract"]).encode("utf-8")
            ).hexdigest()},
        }
    return source, histories, annotations


def _scoped_categorical_fixture():
    """Three-fold fixture whose only directional pair evidence is taxonomy."""
    articles = [
        {
            "id": "preferred", "title": "Evening television bulletin",
            "topic": "television", "subcategory": "bulletin",
            "format": "article",
        },
        {
            "id": "other", "title": "Daily world digest",
            "topic": "world", "subcategory": "digest",
            "format": "article",
        },
    ]
    events = []
    for index in range(18):
        for article_id, action in (("preferred", "click"), ("other", "skip")):
            events.append({
                "user": "reader", "article": article_id, "action": action,
                "impression": f"taxonomy_{index:02d}",
                # Persisted pre-outcome scope keeps the mined conjunction
                # causal and gives every source-order third identical support.
                "history_size_bucket": "light",
            })
    return {
        "articles": articles,
        "users": {"reader": {"topics": [], "history": []}},
        "events": events,
        "tests": [{
            "id": "taxonomy_eval", "user": "reader",
            "candidates": ["preferred", "other"],
            "relevant": ["preferred"],
            "candidate_context": {
                "preferred": {"history_size_bucket": "light"},
                "other": {"history_size_bucket": "light"},
            },
        }],
        "metadata": {"dataset": "scoped-categorical-fixture"},
    }


class LLMLabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, histories, cls.annotations = _annotation_fixture()
        cls.data = build_llm_projection(cls.source, histories, cls.annotations)
        cls.lab = Lab(cls.data, config={
            "pair_feature_profile": "llm_only", "pair_family_fusion": "balanced_rank",
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
        })
        cls.addClassCleanup(cls.lab.engine.close)

    def test_llm_quantile_fit_is_action_independent_and_train_only(self):
        def fit(actions, evaluation_value):
            lab = Lab.__new__(Lab)
            lab.config = {
                "pair_feature_profile": "llm_content_quantile",
                "pair_numeric_bins": 4,
            }
            lab.data = {
                "events": [
                    {"impression": "train", "score": score, "action": action}
                    for score, action in zip((0.0, 0.1, 0.4, 0.9), actions)
                ],
                # Evaluation values are intentionally invisible to fitting.
                "tests": [{"candidate_context": {"future": {
                    "llm_concept_affinity": evaluation_value,
                }}}],
            }
            lab.event_features = lambda event: {
                "llm_concept_affinity": event["score"],
            }
            lab._fit_pair_numeric_encoders()
            return lab._numeric_pair_encoders["llm_concept_affinity"].to_json()

        first = fit(("click", "skip", "skip", "click"), 1e20)
        permuted = fit(("skip", "click", "click", "skip"), -1e20)
        self.assertEqual(first, permuted)

    def test_llm_quantile_missing_is_incomparable_and_swap_is_antisymmetric(self):
        lab = Lab.__new__(Lab)
        source = "llm_concept_affinity"
        predicate = "pair_llm_concept_affinity_quantile"
        lab._numeric_pair_encoders = {
            source: QuantileNumericEvidence.fit(
                source, (0.0, 0.2, 0.5, 1.0),
                ((0.0, 0.1), (0.0, 0.3), (0.0, 0.7), (0.0, 1.0)),
                bins=4,
            )
        }
        for malformed in (None, "nan", float("inf"), True):
            missing = lab._pair_features(
                {source: malformed}, {source: 0.5}, {}, {}, needed={predicate}
            )
            reverse_missing = lab._pair_features(
                {source: 0.5}, {source: malformed}, {}, {}, needed={predicate}
            )
            self.assertEqual(missing[predicate], "incomparable")
            self.assertEqual(reverse_missing[predicate], "incomparable")
        forward = lab._pair_features(
            {source: 0.9}, {source: 0.1}, {}, {}, needed={predicate}
        )[predicate]
        reverse = lab._pair_features(
            {source: 0.1}, {source: 0.9}, {}, {}, needed={predicate}
        )[predicate]

        self.assertTrue(forward.startswith("left_q"))
        self.assertEqual(reverse, "right_" + forward.removeprefix("left_"))

    def test_llm_quantile_profiles_and_mirror_closed_cap_validate(self):
        self.assertTrue(set(LLM_QUANTILE_PAIR_PREDICATES) <= set(
            PAIR_FEATURE_PROFILES["llm_content_quantile"]
        ))
        self.assertIn(
            "pair_history_scope",
            PAIR_FEATURE_PROFILES["llm_conditional_quantile"],
        )
        lab = Lab(conditional_annotation_fixture(), config={
            "miner_strategy": "conditional_llm_seed_only",
            "pair_feature_profile": "llm_conditional_quantile",
            "pair_conjunctions": 3,
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
            "pair_negative_ratio": 0,
        })
        try:
            lab.configure({"pair_conjunctions": 4})
            self.assertEqual(lab.config["pair_conjunctions"], 4)
            with self.assertRaisesRegex(ValueError, "must be <= 4"):
                lab.configure({"pair_conjunctions": 5})
            with self.assertRaisesRegex(ValueError, "mirror-closed"):
                lab.configure({"pair_numeric_bins": 8, "max_feature_values": 10})
            with self.assertRaisesRegex(ValueError, "requires pair_feature_profile"):
                lab.configure({"pair_feature_profile": "llm_content_quantile"})
            with self.assertRaisesRegex(ValueError, "require pair_conjunctions >= 3"):
                lab.configure({
                    "miner_strategy": "fixed_combinations",
                    "pair_feature_profile": "scoped_taxonomy",
                    "pair_conjunctions": 2,
                })
        finally:
            lab.engine.close()

    def test_categorical_pair_facts_swap_namespace_and_abstain_on_oov(self):
        lab = Lab.__new__(Lab)
        lab._numeric_pair_encoders = {}
        wanted = set(PAIR_CATEGORICAL_SIDE_PREDICATES)
        left_attrs = {"llm_format": "analysis"}
        right_attrs = {"llm_format": "report"}
        left_article = {"topic": "news", "subcategory": "general"}
        right_article = {"topic": "sports", "subcategory": "general"}
        forward = lab._pair_features(
            left_attrs, right_attrs, left_article, right_article, needed=wanted
        )
        reverse = lab._pair_features(
            right_attrs, left_attrs, right_article, left_article, needed=wanted
        )

        for family in ("topic", "subcategory", "llm_format"):
            left_key = f"pair_left_{family}"
            right_key = f"pair_right_{family}"
            self.assertEqual(forward[left_key], reverse[right_key])
            self.assertEqual(forward[right_key], reverse[left_key])
        self.assertEqual(
            forward["pair_left_subcategory"],
            "topic=news|subcategory=general",
        )
        self.assertEqual(
            forward["pair_right_subcategory"],
            "topic=sports|subcategory=general",
        )

        missing = lab._pair_features(
            {}, right_attrs, left_article, right_article,
            needed={"pair_left_llm_format", "pair_right_llm_format"},
        )
        self.assertEqual(missing, {})
        lab._pair_feature_vocabulary = {
            "pair_left_topic": {"news"},
            "pair_right_topic": {"news"},
        }
        self.assertEqual(lab._bounded_pair_features({
            "pair_left_topic": "news", "pair_right_topic": "unseen",
        }), {})
        self.assertEqual(
            Lab._pair_rule_family({
                "categorical_fact_family": "llm_format",
                "dependency_owner": "pair_text_semantic_top3_mean",
                "premises": (("pair_left_llm_format", "report"),),
            }),
            "text_semantic",
        )
        self.assertEqual(Lab._category_symbol("  News  "), ("news", "news"))
        unsafe_symbol, unsafe_label = Lab._category_symbol('News "quoted" \\ path')
        self.assertNotIn('"', unsafe_symbol)
        self.assertNotIn("\\", unsafe_symbol)
        self.assertEqual(unsafe_label, 'news "quoted" \\ path')
        collision_probe = {
            lab._pair_features(
                {}, {},
                {"topic": topic, "subcategory": subcategory},
                {"topic": "z", "subcategory": "other"},
                needed={"pair_left_subcategory", "pair_right_subcategory"},
            )["pair_left_subcategory"]
            for topic, subcategory in (("a/b", "c"), ("a", "b/c"))
        }
        self.assertEqual(len(collision_probe), 2)

    def test_scoped_categorical_rules_are_actual_fpminer_rules(self):
        lab = Lab(conditional_annotation_fixture(), config={
            "miner_strategy": "conditional_llm_seed_only",
            "pair_feature_profile": "llm_conditional_scoped_taxonomy",
            "pair_conjunctions": 3,
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 12, "pair_max_rules": 16,
            "pair_negative_ratio": 0,
        })
        try:
            categorical = [
                rule for rule in lab.pair_rules
                if any(predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                       for predicate, _value in rule["premises"])
            ]
            # A categorical rule may be removed as a coverage-equivalent
            # duplicate of a stronger semantic rule. Discovery provenance is
            # still required even when the final dependency-aware selector
            # correctly avoids compiling the duplicate as another vote.
            audit = lab.last_pair_mining["categorical_search"]
            self.assertGreater(audit["eligible_scoped_rules"], 0)
            records = audit["eligible_scoped_rule_records"]
            self.assertTrue(records)
            self.assertTrue(all(
                record["source"].endswith("fpMiner.metta")
                for record in records
            ))
            for rule in categorical:
                self.assertTrue(rule["source"].endswith("fpMiner.metta"))
                self.assertTrue(rule["scoped_categorical_prior"])
                self.assertEqual(
                    rule["evidence_relationship"],
                    "dependent_variant_not_independent_vote",
                )
                self.assertEqual(len(rule["premises"]), 2)
                self.assertEqual(
                    sum(predicate == "pair_history_scope"
                        for predicate, _value in rule["premises"]),
                    1,
                )
                self.assertEqual(
                    sum(predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                        for predicate, _value in rule["premises"]),
                    1,
                )
            self.assertGreater(audit["emitted_with_side_predicate"], 0)
            self.assertEqual(audit["host_generated_rules"], 0)
            self.assertEqual(
                audit["calibration"],
                "mandatory equal-impression CTV regardless of global pair_ctv_mode",
            )
            self.assertEqual(audit["temporal_gate"], "all three source-order folds")
        finally:
            lab.engine.close()

    def test_selected_scoped_category_is_macro_calibrated_and_proved(self):
        lab = Lab(_scoped_categorical_fixture(), config={
            "miner_strategy": "fixed_combinations",
            "pair_feature_profile": "scoped_taxonomy",
            "pair_conjunctions": 3,
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
            "pair_negative_ratio": 0,
        })
        try:
            categorical = [
                rule for rule in lab.pair_rules
                if rule.get("scoped_categorical_prior") is True
            ]
            self.assertTrue(categorical)
            rule = categorical[0]
            self.assertTrue(rule["source"].endswith("fpMiner.metta"))
            self.assertEqual(
                rule["confidence_basis"],
                "petta_kish_effective_impressions_k800_scoped_categorical",
            )
            self.assertEqual(
                rule["calibrated_support_unit"],
                "equal_impression_weighted_activation_mass",
            )
            self.assertEqual(rule["required_temporal_folds"], 3)
            self.assertEqual(len(rule["temporal_fold_effects"]), 3)
            self.assertEqual(
                len(rule["temporal_fold_weighted_supports"]), 3
            )
            self.assertEqual(
                len(rule["temporal_fold_kish_effective_impressions"]), 3
            )
            calibration = rule["ctv_calibration"]
            selection_calibration = rule["selection_calibration"]
            self.assertIsNotNone(calibration)
            self.assertIsNotNone(selection_calibration)
            self.assertEqual(calibration["evidence_k"], 800.0)
            self.assertEqual(selection_calibration["evidence_k"], 20.0)
            self.assertAlmostEqual(
                rule["calibrated_support"],
                calibration["positive"]["weighted_support"],
            )
            self.assertGreaterEqual(
                rule["calibrated_support"],
                rule["required_calibrated_support"],
            )

            side_predicate = next(
                predicate for predicate, _value in rule["premises"]
                if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
            )
            family = rule["categorical_fact_family"]
            left_predicate, right_predicate = (
                f"pair_left_{family}", f"pair_right_{family}"
            )
            # Selection may keep only one directional rule, but serving must
            # retain both observations to enforce whole-family abstention.
            self.assertIn(side_predicate, {left_predicate, right_predicate})
            self.assertIn(left_predicate, lab._pair_feature_vocabulary)
            self.assertIn(right_predicate, lab._pair_feature_vocabulary)
            self.assertEqual(
                lab._pair_feature_vocabulary[left_predicate],
                lab._pair_feature_vocabulary[right_predicate],
            )

            scope = next(
                value for predicate, value in rule["premises"]
                if predicate == "pair_history_scope"
            )
            attrs = {"history_size_bucket": scope}
            articles = [lab.article("preferred"), lab.article("other")]
            oriented = None
            for left, right in ((articles[0], articles[1]),
                                (articles[1], articles[0])):
                forward = lab._pair_spec((left, attrs), (right, attrs))
                reverse = lab._pair_spec((right, attrs), (left, attrs))
                forward_matches = all(
                    forward[1].get(predicate) == value
                    for predicate, value in rule["premises"]
                )
                reverse_matches = all(
                    reverse[1].get(predicate) == value
                    for predicate, value in rule["premises"]
                )
                if forward_matches and not reverse_matches:
                    oriented = forward, reverse
                    break
            self.assertIsNotNone(oriented)
            forward, reverse = oriented
            lab._ensure_pair_specs([forward, reverse])
            proofs, _calls = lab._proofs_for_pair_specs([forward, reverse])
            self.assertTrue(proofs[forward[0]])
            self.assertFalse(proofs[reverse[0]])
            forward_vote = lab._proof_vote_margin(
                forward[0], proofs[forward[0]]
            )
            reverse_vote = lab._proof_vote_margin(
                reverse[0], proofs[reverse[0]]
            )
            self.assertGreater(forward_vote, reverse_vote)
            self.assertEqual(forward[1][left_predicate], reverse[1][right_predicate])
            self.assertEqual(forward[1][right_predicate], reverse[1][left_predicate])

            # Exercise the serving tournament twice with the candidate input
            # order genuinely swapped. The mined direction must rank the same
            # article first and assign the same signed margin to each article;
            # this is not the algebraic identity ``b-a == -(a-b)``.
            def ranked(order):
                rows=[{
                    "article":article,
                    "score":0.5,
                    "stv":{"strength":0.5,"confidence":1.0},
                    "tie_break":{
                        "topic_prior":0.0,"format_prior":0.0,
                        "subcategory_prior":0.0,
                    },
                } for article in order]
                point_specs=[
                    (article["id"],"unused",{},attrs) for article in order
                ]
                return lab._pairwise_rank(rows,point_specs)

            ranked_forward=ranked((left,right))
            ranked_swapped=ranked((right,left))
            expected_ids=[left["id"],right["id"]]
            self.assertEqual(
                [row["article"]["id"] for row in ranked_forward],expected_ids
            )
            self.assertEqual(
                [row["article"]["id"] for row in ranked_swapped],expected_ids
            )
            forward_by_id={row["article"]["id"]:row for row in ranked_forward}
            swapped_by_id={row["article"]["id"]:row for row in ranked_swapped}
            for article_id in expected_ids:
                self.assertEqual(
                    forward_by_id[article_id]["pairwise_margin_score"],
                    swapped_by_id[article_id]["pairwise_margin_score"],
                )
            self.assertGreater(
                forward_by_id[left["id"]]["pairwise_margin_score"],0.0
            )
            self.assertAlmostEqual(
                forward_by_id[left["id"]]["pairwise_margin_score"],
                -forward_by_id[right["id"]]["pairwise_margin_score"],
            )
            audit = lab.last_pair_mining["categorical_search"]
            self.assertGreater(audit["selected_scoped_rules"], 0)
            self.assertEqual(
                audit["calibration"],
                "mandatory equal-impression CTV regardless of global pair_ctv_mode",
            )
            self.assertEqual(
                audit["temporal_gate"], "all three source-order folds"
            )
        finally:
            lab.engine.close()

    def test_impression_macro_calibration_does_not_let_large_slate_dominate(self):
        rows = [
            CTVObservation("large", matched=True, target=False)
            for _index in range(100)
        ] + [
            CTVObservation("small", matched=True, target=True)
            for _index in range(2)
        ]
        result = calibrate_ctv(rows, evidence_k=1.0)

        self.assertEqual(result.positive.raw_support, 102)
        self.assertEqual(result.positive.raw_target_support, 2)
        self.assertAlmostEqual(result.positive.weighted_support, 2.0)
        self.assertAlmostEqual(result.positive.weighted_target_support, 1.0)
        # Each impression contributes mass one: 100 raw losses cannot outweigh
        # the two-row positive impression in the conditional strength.
        self.assertAlmostEqual(result.positive.strength, 0.5)

    def test_concept_preference_is_discovered_by_real_miner_and_proved(self):
        rules = [rule for rule in self.lab.pair_rules
                 if ("pair_llm_concept_affinity", "left") in rule["premises"]]
        self.assertTrue(rules, "Actual miner did not discover concept affinity")
        self.assertTrue(all(rule["source"].endswith("fpMiner.metta") for rule in rules))
        self.assertTrue(all(rule["strength"] > 0.5 for rule in rules))
        result = self.lab.benchmark({"remine": False})
        self.assertEqual(result["auc_proof_only"], 1.0)
        self.assertEqual(result["pairwise_proof_coverage"], 1.0)
        first, second = self.lab.data["events"][:2]
        spec = self.lab._pair_spec(
            (self.lab.article(first["article"]), self.lab.event_features(first)),
            (self.lab.article(second["article"]), self.lab.event_features(second)),
        )
        self.lab._ensure_pair_specs([spec])
        proof_map, _ = self.lab._proofs_for_pair_specs([spec])
        proof = " ".join(proof_map[spec[0]])
        self.assertIn("(STV", proof)
        self.assertTrue(any(rule["dependency_id"] in proof for rule in rules))

    def test_annotations_cannot_rank_when_preference_proofs_are_removed(self):
        with patch.object(self.lab, "_proofs_for_pair_specs",
                          side_effect=lambda specs, **_kwargs: (
                              {spec[0]: [] for spec in specs}, 0
                          )):
            result = self.lab.benchmark({"remine": False})
        self.assertEqual(result["auc_proof_only"], 0.5)
        self.assertEqual(result["pairwise_proof_coverage"], 0.0)

    def test_live_training_and_replay_use_the_same_content_observation_builder(self):
        expected = build_llm_workspace_facts("matched", ["past"], self.annotations)
        live = self.lab.features("reader", self.lab.article("matched"))
        event = self.lab.data["events"][0]
        context = self.lab.data["tests"][0]["candidate_context"]["matched"]
        historical = self.lab.event_features(event)
        replay = self.lab.contextual_features("reader", self.lab.article("matched"), context)
        for feature, value in expected.items():
            self.assertEqual(live[feature], value, feature)
            self.assertEqual(event[feature], value, feature)
            self.assertEqual(context[feature], value, feature)
            if value is None:
                self.assertNotIn(feature, historical)
                self.assertNotIn(feature, replay)
            elif isinstance(value, float):
                self.assertEqual(float(historical[feature]), value, feature)
                self.assertEqual(float(replay[feature]), value, feature)
            else:
                self.assertEqual(historical[feature], value, feature)
                self.assertEqual(replay[feature], value, feature)

    def test_historical_missing_observations_never_read_future_live_profiles(self):
        article = self.lab.article("matched")
        self.assertEqual(self.lab.features("reader", article)["llm_concept_affinity"], 1.0)
        # No legacy MIND snapshot markers: an explicit new-format None is
        # sufficient to make the persisted observation authoritative.
        event = {"user": "reader", "article": "matched", "llm_concept_affinity": None}
        context = {"llm_concept_affinity": None}
        with patch.object(self.lab, "features", side_effect=AssertionError("read future profile")):
            historical = self.lab.event_features(event)
            replay = self.lab.contextual_features("reader", article, context)
        self.assertTrue(set(LLM_WORKSPACE_FEATURES).isdisjoint(historical))
        self.assertTrue(set(LLM_WORKSPACE_FEATURES).isdisjoint(replay))

    def test_known_zero_is_not_conflated_with_missing_in_pair_workspace(self):
        predicate = "pair_llm_concept_affinity"
        missing = {"llm_concept_affinity": None}
        zero = {"llm_concept_affinity": 0.0}
        self.assertEqual(self.lab._pair_features(missing, zero, {}, {})[predicate], "incomparable")
        self.assertEqual(self.lab._pair_features(zero, missing, {}, {})[predicate], "incomparable")
        self.assertEqual(self.lab._pair_features(zero, {"llm_concept_affinity": 1.0}, {}, {})[predicate], "right")
        self.assertEqual(self.lab._pair_features(zero, zero, {}, {})[predicate], "equal")

    def test_projection_retains_existing_entity_and_lexical_evidence(self):
        self.assertEqual(self.lab.data["article_entity_vectors"], self.source["article_entity_vectors"])
        self.assertEqual(self.lab.data["lexical_idf_model"], self.source["lexical_idf_model"])
        for original, enriched in zip(self.source["events"], self.lab.data["events"]):
            for key, value in original.items():
                self.assertEqual(enriched[key], value, key)

    def test_strict_symbolic_strips_every_llm_field_and_still_mines_and_reasons(self):
        with patch("recommendation.app.server.build_llm_workspace_facts",
                   side_effect=AssertionError("strict mode used LLM observations")):
            strict = Lab(self.data, symbolic_only=True, config={
                "min_support": 2, "pair_min_support": 2, "max_rules": 8, "pair_max_rules": 8,
            })
        try:
            self.assertFalse(strict._llm_workspace)
            self.assertFalse(strict._llm_article_annotations)

            def check(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        self.assertFalse(str(key).startswith("llm_"), key)
                        check(child)
                elif isinstance(value, list):
                    for child in value:
                        check(child)

            check(strict.data)
            self.assertTrue(strict.mined_rules)
            self.assertEqual(strict.benchmark({"remine": False})["cases"], 1)
            self.assertTrue(self.data["llm_article_annotations"])
            self.assertIn("llm_concept_affinity", self.data["events"][0])
            for profile in ("llm_content", "llm_content_no_lexical", "llm_only",
                            "llm_conditional"):
                with self.assertRaisesRegex(ValueError, "symbolic-only"):
                    strict.configure({"pair_feature_profile": profile})
        finally:
            strict.engine.close()

    def test_corrupt_annotation_hash_or_schema_is_rejected_before_mining(self):
        for corruption in ("records", "hash", "schema"):
            with self.subTest(corruption=corruption):
                bad = copy.deepcopy(self.data)
                if corruption == "records":
                    bad["llm_article_annotations"]["matched"]["concepts"] = ["changed"]
                elif corruption == "hash":
                    bad["metadata"]["llm_workspace"]["annotation_records_sha256"] = "0" * 64
                else:
                    bad["metadata"]["llm_workspace"]["observation_schema"] = "unverified-v2"
                with patch("recommendation.app.server.PeTTa", side_effect=AssertionError("mining started")):
                    with self.assertRaisesRegex(ValueError, "frozen workspace provenance"):
                        Lab(bad)

    def test_changed_article_text_is_rejected_before_mining(self):
        bad = copy.deepcopy(self.data)
        bad["articles"][0]["title"] += " changed after extraction"
        with patch("recommendation.app.server.PeTTa", side_effect=AssertionError("mining started")):
            with self.assertRaisesRegex(ValueError, "frozen corpus"):
                Lab(bad)

    def test_llm_profiles_support_a_matched_no_lexical_ablation(self):
        llm = {f"pair_{feature}" for feature in LLM_NUMERIC_FEATURES}
        full = set(PAIR_FEATURE_PROFILES["llm_content"])
        no_lexical = set(PAIR_FEATURE_PROFILES["llm_content_no_lexical"])
        self.assertTrue(llm <= full)
        self.assertTrue(llm <= no_lexical)
        self.assertEqual(full - no_lexical, {"pair_title_overlap"})
        self.assertEqual(no_lexical - full, set())
        self.assertEqual(set(PAIR_FEATURE_PROFILES["llm_only"]), llm)
        self.assertEqual(
            set(PAIR_FEATURE_PROFILES["llm_conditional"]),
            full | {"pair_history_scope"},
        )
        self.assertEqual(
            no_lexical - llm,
            set(PAIR_FEATURE_PROFILES["text_semantic_attention_t8_no_lexical"]),
        )

    def test_profiles_without_annotations_and_residual_mode_fail_without_config_mutation(self):
        plain = object.__new__(Lab)
        plain.data = self.source
        plain.symbolic_only = False
        plain._llm_workspace = {}
        plain._llm_article_annotations = {}
        plain.config = {**self.lab.config, "pair_feature_profile": "stable_multi_interest"}
        before = plain.config.copy()
        for profile in ("llm_content", "llm_content_no_lexical", "llm_only",
                        "llm_conditional"):
            with self.subTest(profile=profile):
                with self.assertRaisesRegex(ValueError, "prepared annotation workspace"):
                    plain.configure({"pair_feature_profile": profile})
                self.assertEqual(plain.config, before)
        before = self.lab.config.copy()
        with self.assertRaisesRegex(ValueError, "clustered dependencies"):
            self.lab.configure({"pair_dependency_mode": "residual_hypergraph"})
        self.assertEqual(self.lab.config, before)
        with self.assertRaisesRegex(ValueError, "isolated PeTTa proof channels"):
            self.lab.configure({"pair_aggregation": "posterior"})
        self.assertEqual(self.lab.config, before)
        with self.assertRaisesRegex(ValueError, "pair_conjunctions"):
            self.lab.configure({"miner_strategy": "conditional_llm"})
        self.assertEqual(self.lab.config, before)
        with self.assertRaisesRegex(ValueError, "llm_conditional"):
            self.lab.configure({
                "miner_strategy": "conditional_llm", "pair_conjunctions": 3,
            })
        self.assertEqual(self.lab.config, before)

    def test_all_annotation_views_share_text_lineage_and_one_evidence_family(self):
        expected = Lab._pair_evidence_lineage([("pair_text_semantic_attention_t8", "left")])
        self.assertTrue(expected)
        for feature in LLM_NUMERIC_FEATURES:
            premises = [(f"pair_{feature}", "left")]
            with self.subTest(feature=feature):
                self.assertEqual(Lab._pair_evidence_lineage(premises), expected)
                self.assertEqual(Lab._pair_rule_family({"premises": premises}), "text_semantic")
        self.assertEqual(len({rule["dependency_id"] for rule in self.lab.pair_rules}), 1)

    def test_repeated_correlated_proof_views_do_not_receive_independent_votes(self):
        dependency = self.lab.pair_rules[0]["dependency_id"]
        proof = f'(by {dependency} (: p (PairSignal example "{dependency}") (STV 0.8 0.9)))'
        one = self.lab._proof_dependency_margins("one_llm_view", [proof])
        many = self.lab._proof_dependency_margins("eight_llm_views", [proof] * 8)
        self.assertEqual(one, many)
        self.assertEqual(set(one), {dependency})
        self.assertGreater(one[dependency], 0.0)

    def test_conditional_strategy_expands_real_miner_seed_and_petta_proves_it(self):
        lab = Lab(conditional_annotation_fixture(), config={
            "miner_strategy": "conditional_llm",
            "pair_feature_profile": "llm_conditional",
            "pair_conjunctions": 3,
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
            "pair_negative_ratio": 0,
        })
        try:
            conditional = [rule for rule in lab.pair_rules
                           if rule["source"].endswith("conditional_llm_mining.py")]
            self.assertTrue(conditional)
            self.assertTrue(all(rule["specificity"] == 2 for rule in conditional))
            self.assertTrue(all(rule["conditional_fpminer_seed_rule_ids"]
                                for rule in conditional))
            self.assertEqual(len({rule["dependency_id"] for rule in conditional}), 1)
            audit = lab.last_pair_mining["target_search"]
            self.assertEqual(audit["kind"], "conditional_llm")
            self.assertTrue(audit["requires_real_fpminer_seed"])
            self.assertGreater(audit["audit"]["eligible_patterns"], 0)
            result = lab.benchmark({"remine": False})
            self.assertEqual(result["auc_proof_only"], 1.0)
            self.assertEqual(result["pairwise_proof_coverage"], 1.0)
            pair = lab._pair_spec(
                (lab.article("same_match"), lab.contextual_features(
                    "reader", lab.article("same_match"),
                    lab.data["tests"][0]["candidate_context"]["same_match"],
                )),
                (lab.article("same_other"), lab.contextual_features(
                    "reader", lab.article("same_other"),
                    lab.data["tests"][0]["candidate_context"]["same_other"],
                )),
            )
            lab._ensure_pair_specs([pair])
            proofs, _ = lab._proofs_for_pair_specs([pair])
            proof = " ".join(proofs[pair[0]])
            self.assertTrue(any(rule["variant_id"] in proof
                                for rule in conditional))
        finally:
            lab.engine.close()

    def test_seed_only_conditional_strategy_keeps_seed_provenance_not_unary_vote(self):
        lab = Lab(conditional_annotation_fixture(), config={
            "miner_strategy": "conditional_llm_seed_only",
            "pair_feature_profile": "llm_conditional",
            "pair_conjunctions": 3,
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
            "pair_negative_ratio": 0,
        })
        try:
            conditional = [
                rule for rule in lab.pair_rules
                if rule["source"].endswith("conditional_llm_mining.py")
            ]
            llm_unaries = [
                rule for rule in lab.pair_rules
                if len(rule["premises"]) == 1
                and rule["premises"][0][0] in LLM_PAIR_PREDICATES
            ]
            audit = lab.last_pair_mining["target_search"]

            self.assertTrue(conditional)
            self.assertEqual(llm_unaries, [])
            self.assertEqual(audit["semantic_seed_policy"], "discovery_only")
            self.assertGreater(audit["discovery_only_seed_count"], 0)
            self.assertGreater(audit["candidate_conditional_children"], 0)
            self.assertEqual(
                audit["compiled_conditional_children"], len(conditional)
            )
            removed_ids = {
                row["rule_id"] for row in audit["discovery_only_seed_rules"]
            }
            removed_by_id = {
                row["rule_id"]: row for row in audit["discovery_only_seed_rules"]
            }
            self.assertTrue(all(
                set(rule["conditional_fpminer_seed_rule_ids"]) <= removed_ids
                for rule in conditional
            ))
            for rule in conditional:
                premises = {tuple(item) for item in rule["premises"]}
                for seed_id in rule["conditional_fpminer_seed_rule_ids"]:
                    seed = removed_by_id[seed_id]
                    self.assertTrue(seed["source"].endswith("fpMiner.metta"))
                    self.assertIn(tuple(seed["premises"][0]), premises)
                    self.assertIn("positive", seed["discovery_ctv"])
            self.assertEqual(lab.benchmark({"remine": False})["auc_proof_only"], 1.0)
        finally:
            lab.engine.close()

    def test_conditional_effective_backoff_changes_children_not_baseline(self):
        lab = Lab(conditional_annotation_fixture(), config={
            "miner_strategy": "conditional_llm_seed_only",
            "pair_feature_profile": "llm_conditional_quantile",
            "pair_conjunctions": 3,
            "pair_ctv_mode": "conditional_effective_backoff",
            "min_support": 2, "pair_min_support": 2,
            "max_rules": 8, "pair_max_rules": 8,
            "pair_negative_ratio": 0,
        })
        try:
            children = [
                rule for rule in lab.pair_rules
                if rule["source"].endswith("conditional_llm_mining.py")
            ]
            baseline = [rule for rule in lab.pair_rules if rule not in children]
            self.assertTrue(children)
            self.assertTrue(all(
                rule["confidence_basis"] ==
                    "petta_kish_effective_impressions_k800_conditional_backoff"
                and rule["ctv_calibration"] is not None
                for rule in children
            ))
            self.assertTrue(all(
                rule["confidence_basis"] ==
                    "petta_raw_pair_count_k800"
                and rule["ctv_calibration"] is None
                for rule in baseline
            ))
            self.assertTrue(all(
                rule["dependency_id"] == children[0]["dependency_id"]
                for rule in children
            ))
            self.assertFalse(
                lab.last_pair_mining["conditional_effective_backoff"]
                ["independent_vote_added"]
            )
            with self.assertRaisesRegex(
                ValueError, "requires miner_strategy"
            ):
                lab.configure({"miner_strategy": "fixed_combinations"})
        finally:
            lab.engine.close()


if __name__ == "__main__":
    unittest.main()
