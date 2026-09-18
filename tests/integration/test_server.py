import copy
import math
import unittest
from unittest.mock import patch
from recommendation.app.server import (
    LAB, NUMERIC_PAIR_EVIDENCE, PAIR_EVIDENCE_ALIASES, PAIR_FEATURES,
    PAIR_FEATURE_PROFILES, PAIR_INTERACTIONS, SEMANTIC_CACHE_TTL_SECONDS, STV_RE,
    Lab, balanced_forms, fixture, parse_rules,
)
from recommendation.adapters.mind import history_feature_context
from recommendation.core.ctv_calibration import (
    CTVObservation, calibrate_ctv, reencode_ctv_confidence,
)


class LabTest(unittest.TestCase):
    def test_mrr_averages_reciprocal_ranks_for_multiple_clicked_items(self):
        data=fixture()
        data["tests"]=[{
            "id":"multi_positive",
            "user":"u1",
            "candidates":["n1","n2","n3","n4"],
            "relevant":["n2","n4"],
        }]
        lab=Lab(data=data,config={"ranking_mode":"pointwise"})
        self.addCleanup(lab.engine.close)
        original_rank=lab._rank

        def known_order(*args,**kwargs):
            rows=original_rank(*args,**kwargs)
            by_id={row["article"]["id"]:row for row in rows}
            return [by_id[article_id] for article_id in ("n1","n2","n3","n4")]

        lab._rank=known_order
        result=lab.benchmark({"remine":False,"max_candidates":0})

        # Clicked items occupy ranks 2 and 4: (1/2 + 1/4) / 2.
        self.assertEqual(result["mrr"],0.375)

    def test_real_miner_output_is_used(self):
        self.assertTrue(LAB.mined_output)
        self.assertTrue(balanced_forms(" ".join(LAB.mined_output)))
        self.assertTrue(all("(CTV" in form for form in balanced_forms(
            " ".join(LAB.mined_output)
        )))
        self.assertTrue(LAB.mined_rules)
        self.assertTrue(all(rule["source"].endswith("fpMiner.metta") for rule in LAB.mined_rules))
        self.assertTrue(all(rule["discovery_ctv"]["complete"]
                            for rule in LAB.mined_rules))

    def test_parser_retains_both_discovery_ctv_branches(self):
        rules=parse_rules([
            '(supportOf (((topic $case "ai") '
            '(engagement $case "click")) '
            '(CTV (STV 0.75 0.2) (STV 0.25 0.4))) 3)'
        ])
        self.assertEqual(len(rules),1)
        self.assertEqual(rules[0]["discovery_ctv"],{
            "positive":{"strength":0.75,"confidence":0.2},
            "negative":{"strength":0.25,"confidence":0.4},
            "complete":True,
        })

    def test_native_ctv_counts_cases_and_enforces_requested_target(self):
        space="&recommendation_ctv_test_cases"
        LAB.petta.process_metta_string(f"!(bind! {space} (new-space))")
        LAB.petta.process_metta_string(
            f'!(superpose ('
            f'(add-atom {space} (topic ctv_c1 "ai")) '
            f'(add-atom {space} (extra ctv_c1 "one")) '
            f'(add-atom {space} (extra_two ctv_c1 "two")) '
            f'(add-atom {space} (engagement ctv_c1 "click")) '
            f'(add-atom {space} (topic ctv_c2 "ai")) '
            f'(add-atom {space} (engagement ctv_c2 "skip")) '
            f'(add-atom {space} (topic ctv_c3 "science")) '
            f'(add-atom {space} (engagement ctv_c3 "click"))))'
        )
        query=(
            '!(recommendation-ctv-with-support '
            '(, (topic $case "ai") (engagement $case "click")) '
            f'{space} 1 "click" 800.0)'
        )
        result=" ".join(str(value) for value in
                        LAB.petta.process_metta_string(query))
        values=[tuple(map(float,pair)) for pair in STV_RE.findall(result)]
        self.assertEqual(len(values),2)
        self.assertAlmostEqual(values[0][0],0.5)
        self.assertAlmostEqual(values[0][1],2/802)
        self.assertAlmostEqual(values[1][0],1.0)
        self.assertAlmostEqual(values[1][1],1/801)
        rejected=LAB.petta.process_metta_string(
            '!(formatter (conjunct (, (topic $case "ai") '
            f'(engagement $case "skip"))) {space} 1 "click" 800.0)'
        )
        self.assertFalse(rejected)
        invalid_k=LAB.petta.process_metta_string(
            '!(recommendation-ctv-with-support '
            '(, (topic $case "ai") (engagement $case "click")) '
            f'{space} 1 "click" -1.0)'
        )
        self.assertFalse(invalid_k)

    def test_ctv_evidence_k_is_configurable_and_used_by_calibration(self):
        lab=Lab(data=fixture())
        try:
            lab.configure({"ctv_evidence_k":400})
            lab.mine()
            self.assertEqual(lab.last_mining["ctv_evidence_k"],400.0)
            for rule in lab.mined_rules:
                support=rule["antecedent_support"]
                self.assertAlmostEqual(
                    rule["confidence"],support/(support+400.0)
                )
            for rule in lab.pair_rules:
                support=rule["antecedent_support"]
                self.assertAlmostEqual(
                    rule["count_confidence"],support/(support+800.0)
                )
                self.assertAlmostEqual(
                    rule["confidence"],rule["count_confidence"],
                )
                self.assertAlmostEqual(
                    rule["selection_count_confidence"],support/(support+20.0)
                )
                self.assertAlmostEqual(
                    rule["selection_confidence"],
                    rule["selection_count_confidence"]
                    *rule["activation_coverage"],
                )
                self.assertEqual(rule["petta_evidence_k"],800.0)
            lab.configure({
                "pair_ctv_mode":"raw_strength_effective_confidence",
                "pair_ctv_evidence_k":20.0,
            })
            lab.mine()
            self.assertTrue(lab.pair_rules)
            for rule in lab.pair_rules:
                calibration=rule["ctv_calibration"]
                selection=rule["selection_calibration"]
                self.assertIsNotNone(calibration)
                self.assertIsNotNone(selection)
                self.assertAlmostEqual(
                    rule["strength"],
                    rule["joint_support"]/rule["antecedent_support"],
                )
                self.assertAlmostEqual(
                    rule["confidence"],calibration["positive"]["confidence"]
                )
                self.assertAlmostEqual(
                    rule["selection_confidence"],
                    selection["positive"]["confidence"],
                )
                self.assertEqual(calibration["evidence_k"],800.0)
                self.assertEqual(selection["evidence_k"],20.0)
                self.assertEqual(
                    rule["confidence_basis"],
                    "petta_kish_effective_impressions_k800",
                )
            lab.configure({"pair_rule_selection_k":30.0})
            self.assertEqual(lab.config["pair_rule_selection_k"],30.0)
            self.assertEqual(lab.config["pair_ctv_evidence_k"],30.0)
            with self.assertRaisesRegex(ValueError,"deprecated alias"):
                lab.configure({
                    "pair_rule_selection_k":30.0,
                    "pair_ctv_evidence_k":20.0,
                })
            for invalid in (0,-1,float("nan"),float("inf"),1e9+1):
                with self.assertRaisesRegex(ValueError,"ctv_evidence_k"):
                    lab.configure({"ctv_evidence_k":invalid})
            with self.assertRaisesRegex(ValueError,"pair_ctv_mode"):
                lab.configure({"pair_ctv_mode":"invented"})
        finally:
            lab.engine.close()

    def test_feed_is_proof_scored_and_cached(self):
        first = LAB.score("u1"); second = LAB.score("u1")
        self.assertIs(first, second)
        self.assertTrue(any(row["proofs"] and row["rules"] for row in first))
        self.assertTrue(all(row["engine"] == "PeTTaChainer" for row in first))
        self.assertTrue(all("inference_stv" in row and "score_method" in row for row in first))

    def test_feed_cache_observability_distinguishes_scoring_from_cache_hit(self):
        lab=Lab(data=fixture())
        self.addCleanup(lab.engine.close)
        initial=lab.state()["engine"]["feed_rank_cache"]
        self.assertEqual(initial,{"entries":0,"hits":0,"misses":0})

        first=lab.score("u1")
        after_miss=lab.state()["engine"]["feed_rank_cache"]
        second=lab.score("u1")
        after_hit=lab.state()["engine"]["feed_rank_cache"]

        self.assertIs(first,second)
        self.assertEqual(after_miss,{"entries":1,"hits":0,"misses":1})
        self.assertEqual(after_hit,{"entries":1,"hits":1,"misses":1})

    def test_semantic_preview_is_content_cached_and_does_not_mutate_ranker(self):
        article=dict(LAB.article("n1"))
        before=(len(LAB.data["events"]),len(LAB._online_events),LAB.version,
                copy.deepcopy(LAB.mined_rules),copy.deepcopy(LAB.pair_rules))
        converted={
            "article_id":"n1","conversion":"nl2pln","committed":False,
            "statement_count":1,
            "statements":["(: semantic_1 (HasTopic n1 ai) (STV 1 1))"],
            "converter":{},"converter_provenance":{},
            "predicate_schema":[],
        }
        LAB._semantic_preview_cache.clear()
        LAB._semantic_preview_cache_bytes=0
        with patch.object(LAB.semantic_client,"parse_article",return_value=converted) as parse:
            first=LAB.preview_semantics(article)
            second=LAB.preview_semantics(article)
        self.assertEqual(parse.call_count,1)
        self.assertFalse(first["cached"]); self.assertTrue(second["cached"])
        self.assertEqual(first["content_hash"],second["content_hash"])
        self.assertEqual(
            (len(LAB.data["events"]),len(LAB._online_events),LAB.version,
             LAB.mined_rules,LAB.pair_rules),before
        )

    def test_semantic_preview_never_returns_expired_reordered_entry(self):
        first_article=dict(LAB.article("n1"))
        second_article=dict(LAB.article("n2"))
        converted={
            "article_id":"source","conversion":"nl2pln","committed":False,
            "statement_count":0,"statements":[],"converter":{},
            "converter_provenance":{},"predicate_schema":[],
        }
        LAB._semantic_preview_cache.clear()
        LAB._semantic_preview_cache_bytes=0
        with (
            patch.object(LAB.semantic_client,"parse_article",return_value=converted) as parse,
            patch("recommendation.app.server.time.monotonic",
                  side_effect=[0.0,10.0,20.0,SEMANTIC_CACHE_TTL_SECONDS+5.0]),
        ):
            LAB.preview_semantics(first_article)
            LAB.preview_semantics(second_article)
            self.assertTrue(LAB.preview_semantics(first_article)["cached"])
            expired=LAB.preview_semantics(first_article)
        self.assertFalse(expired["cached"])
        self.assertEqual(parse.call_count,3)

    def test_point_scores_shrink_toward_the_training_base_rate(self):
        lab = Lab(data=fixture())
        rows = lab.score("u1", candidates=[article["id"] for article in lab.data["articles"]], limit=0)
        self.assertTrue(rows)
        self.assertTrue(all("format_prior" in row["tie_break"] for row in rows))
        self.assertTrue(all("topic_prior" in row["tie_break"] for row in rows))
        self.assertTrue(all(row["score"] == round(
            lab._click_base_rate + row["stv"]["confidence"]
            * (row["stv"]["strength"] - lab._click_base_rate), 8
        ) for row in rows))
        result = lab.benchmark({"remine": False})
        self.assertIn("score_tie_groups", result)
        self.assertIn("score_unique", result)

    def test_feedback_does_not_remine_before_interval(self):
        before = LAB.version
        LAB.event("u1", "n1", "click")
        self.assertEqual(LAB.version, before)
        self.assertGreater(LAB.pending_events, 0)

    def test_cursor_feed_returns_ranked_unseen_pages(self):
        first = LAB.feed_page("u1", limit=3)
        second = LAB.feed_page("u1", cursor=first["next_cursor"], limit=3)
        first_ids = {row["article"]["id"] for row in first["feed"]}
        second_ids = {row["article"]["id"] for row in second["feed"]}
        self.assertEqual(len(first_ids), 3)
        self.assertEqual(len(second_ids), 3)
        self.assertTrue(first_ids.isdisjoint(second_ids))
        self.assertTrue(all(row["engine"] == "PeTTaChainer" for row in first["feed"] + second["feed"]))
        self.assertTrue(all(row["score"] == round(
                            LAB._click_base_rate + row["stv"]["confidence"]
                            * (row["stv"]["strength"] - LAB._click_base_rate), 8)
                            for row in first["feed"] + second["feed"]))
        with self.assertRaisesRegex(ValueError, "stale feed cursor"):
            LAB.feed_page("u1", cursor=first["next_cursor"], limit=3)

    def test_feed_session_token_is_bound_to_owner_for_pages_and_feedback(self):
        lab = Lab(data=fixture())
        first = lab.feed_page("u1", limit=1)
        session = first["session"]
        cursor = first["next_cursor"]
        served = first["feed"][0]

        with self.assertRaisesRegex(ValueError, "belongs to another user"):
            lab.feed_page("u2", session=session, limit=1)

        # Ownership must be checked before a stale token is reset into a new
        # session for the requesting user.
        lab._feed_sessions[session]["version"] = lab.version - 1
        with self.assertRaisesRegex(ValueError, "belongs to another user"):
            lab.feed_page("u2", cursor=cursor, limit=1)
        lab._feed_sessions[session]["version"] = lab.version

        event_count = len(lab.data["events"])
        with self.assertRaisesRegex(ValueError, "belongs to another user"):
            lab.event("u2", served["article"]["id"], "skip",
                      served["context"], served["impression"])
        self.assertEqual(len(lab.data["events"]), event_count)

        result = lab.event("u1", served["article"]["id"], "skip",
                           served["context"], served["impression"])
        self.assertEqual(len(lab.data["events"]), event_count + 1)
        self.assertEqual(result["recorded_context"], served["context"])
        with self.assertRaisesRegex(ValueError, "stale feed cursor revision"):
            lab.feed_page("u1", cursor=cursor, limit=1)
        continuation = lab.feed_page(
            "u1", cursor=result["queue_revision"]["next_cursor"], limit=1
        )
        self.assertEqual(continuation["session"], session)
        lab._feed_sessions[session]["version"] = lab.version - 1
        replacement = lab.feed_page("u1", session=session, limit=1)
        self.assertTrue(replacement["reset"])
        self.assertNotEqual(replacement["session"], session)

    def test_feedback_recovers_page_delivered_after_last_browser_acceptance(self):
        lab = Lab(data=fixture(), config={"mine_interval": 100})
        try:
            accepted = lab.feed_page("u1", limit=1)
            session = accepted["session"]
            served = accepted["feed"][0]
            self.assertEqual(accepted["position"], 1)

            # Model an infinite-scroll GET that completed on the server after
            # the browser had already started feedback and aborted its fetch.
            unaccepted = lab.feed_page(
                "u1", cursor=accepted["next_cursor"], limit=1
            )
            unaccepted_row = unaccepted["feed"][0]
            self.assertEqual(unaccepted["position"], 2)
            self.assertNotIn(
                unaccepted_row["article"]["id"],
                {row["article"]["id"]
                 for row in lab._feed_sessions[session]["queue"]},
            )

            result = lab.event(
                "u1", served["article"]["id"], "skip",
                served["context"], served["impression"],
                feed_session=session,
                queue_revision=accepted["queue_revision"],
                feed_position=accepted["position"],
            )

            recovery = result["queue_revision"]["delivery_recovery"]
            self.assertEqual(recovery, {
                "accepted_position": 1,
                "server_position_before_recovery": 2,
                "recovered_rows": 1,
            })
            self.assertEqual(lab._feed_sessions[session]["position"], 1)
            self.assertIn(
                unaccepted_row["article"]["id"],
                {row["article"]["id"]
                 for row in lab._feed_sessions[session]["queue"]},
            )
            unaccepted_key=(
                unaccepted_row["impression"],
                unaccepted_row["article"]["id"],
            )
            self.assertNotIn(
                unaccepted_key,
                lab._feed_sessions[session]["feedback_contexts"],
            )
            self.assertFalse(lab._feed_sessions[session]["deliveries"])
        finally:
            lab.close()

    def test_feedback_acceptance_fields_are_owned_current_and_complete(self):
        lab = Lab(data=fixture(), config={"mine_interval": 100})
        try:
            accepted = lab.feed_page("u1", limit=1)
            session = accepted["session"]
            served = accepted["feed"][0]
            unaccepted = lab.feed_page(
                "u1", cursor=accepted["next_cursor"], limit=1
            )["feed"][0]
            event_count = len(lab.data["events"])

            with self.assertRaisesRegex(ValueError, "requires session"):
                lab.event(
                    "u1", served["article"]["id"], "skip",
                    served["context"], served["impression"],
                    feed_session=session,
                )
            with self.assertRaisesRegex(ValueError, "does not match"):
                lab.event(
                    "u1", served["article"]["id"], "skip",
                    served["context"], served["impression"],
                    feed_session="0" * 32,
                    queue_revision=accepted["queue_revision"],
                    feed_position=accepted["position"],
                )
            with self.assertRaisesRegex(ValueError, "stale feedback"):
                lab.event(
                    "u1", served["article"]["id"], "skip",
                    served["context"], served["impression"],
                    feed_session=session,
                    queue_revision=accepted["queue_revision"] + 1,
                    feed_position=accepted["position"],
                )
            with self.assertRaisesRegex(ValueError, "not accepted"):
                lab.event(
                    "u1", unaccepted["article"]["id"], "skip",
                    unaccepted["context"], unaccepted["impression"],
                    feed_session=session,
                    queue_revision=accepted["queue_revision"],
                    feed_position=accepted["position"],
                )
            self.assertEqual(len(lab.data["events"]), event_count)
            self.assertEqual(lab._feed_sessions[session]["position"], 2)
        finally:
            lab.close()

    def test_live_skip_is_negative_petta_evidence_and_reranks_same_session(self):
        # One step and one root per batch reproduce the finite-budget setting
        # in which generalized feedback used to be starved behind Engagement
        # roots.  The feedback policy now has a dedicated PeTTa conclusion.
        lab = Lab(data=fixture(), config={
            "mine_interval":100,"chain_steps":1,"query_batch_size":1,
        })
        try:
            # Emit only the first item so a same-topic sibling remains in the
            # unserved queue and can carry the generalized skip proof.
            first = lab.feed_page("u1", limit=1)
            session = first["session"]
            cursor = first["next_cursor"]
            skipped = first["feed"][0]
            before_queue = [
                row["article"]["id"] for row in lab._feed_sessions[session]["queue"]
            ]

            result = lab.event(
                "u1", skipped["article"]["id"], "skip",
                skipped["context"], skipped["impression"],
            )

            revision = result["queue_revision"]
            self.assertEqual(revision["session"], session)
            self.assertEqual(revision["revision"], 1)
            self.assertEqual(revision["reranked_candidates"], len(before_queue))
            self.assertIn(skipped["article"]["id"], result["negative_profile"]["articles"])
            self.assertEqual(
                result["negative_profile"]["provenance"],
                "bounded_online_feedback_policy_not_mined",
            )
            self.assertNotIn("feedback_skip_exact", {
                rule["id"] for rule in lab.mined_rules
            })

            exact = lab.score(
                "u1", candidates=[skipped["article"]["id"]], limit=0
            )[0]
            self.assertEqual(exact["feedback_evidence"]["match"], "exact")
            self.assertIn(
                "feedback_skip_exact", exact["feedback_evidence"]["rule_ids"]
            )
            self.assertEqual(
                exact["score_method"], "pettachainer_live_feedback_revision"
            )
            self.assertEqual(
                exact["feedback_evidence"]["proof_stv"]["strength"],0.0
            )
            self.assertTrue(any(
                "feedback_skip_exact" in proof for proof in exact["proofs"]
            ))
            self.assertLess(exact["score"], skipped["score"])

            with self.assertRaisesRegex(ValueError, "stale feed cursor revision"):
                lab.feed_page("u1", cursor=cursor, limit=1)
            continuation = lab.feed_page(
                "u1", cursor=revision["next_cursor"], limit=1
            )
            self.assertEqual(continuation["session"], session)
            self.assertEqual(continuation["queue_revision"], 1)
            self.assertTrue(all(
                row["queue_revision"] == 1 for row in continuation["feed"]
            ))
            causal=revision["causal_demotion"]
            self.assertIsNotNone(causal)
            self.assertEqual(
                causal["score_method"],"pettachainer_live_feedback_revision"
            )
            self.assertEqual(causal["match"],"topic")
            self.assertIn("feedback_skip_topic",causal["rule_ids"])
            self.assertLess(causal["score_delta"],0)
            self.assertTrue(causal["feedback_evidence_introduced"])
            self.assertTrue(causal["feedback_evidence_changed"])
            self.assertIsNone(causal["before_feedback_signature"])
            self.assertTrue(causal["after_feedback_signature"])
            self.assertTrue(causal["own_score_decreased"])
            self.assertTrue(any(
                "feedback_skip_topic" in proof for proof in causal["proofs"]
            ))
            self.assertTrue(any(
                "(STV 0.0 " in proof for proof in causal["proofs"]
            ))
            self.assertTrue(any(
                rule_id in proof
                for rule_id in causal["rule_ids"]
                for proof in causal["proofs"]
            ))
        finally:
            lab.close()

    def test_rank_only_movement_with_unchanged_feedback_is_not_causal(self):
        lab=object.__new__(Lab)
        lab.version=1
        proof=(
            '(by feedback_skip_topic fact_candidate_n1_topic '
            '(Live_Feedback_Click candidate_n1) (STV 0.0 0.65))'
        )
        evidence={
            "match":"topic",
            "rule_ids":["feedback_skip_topic"],
            "proof_stv":{"strength":0.0,"confidence":0.65},
        }
        before_negative={
            "article":{"id":"n1"},"score":0.2,"ranking_score":0.4,
            "score_method":"pettachainer_live_feedback_revision",
            "proofs":[proof],"feedback_evidence":evidence,
        }
        neutral={
            "article":{"id":"n2"},"score":0.3,"ranking_score":0.5,
            "score_method":"proof_gated_base_rate_posterior",
            "proofs":[],"feedback_evidence":{"match":"none","rule_ids":[]},
        }
        # n1 moves down only because n2 moves above it. Its own proof and both
        # own scores remain byte-for-byte unchanged.
        lab.score=lambda *_args,**_kwargs: [neutral,before_negative]
        lab.contextual_features=lambda *_args,**_kwargs: {}
        state={
            "user":"u1","queue":[before_negative,neutral],"position":1,
            "source_position":2,"queue_revision":0,"version":1,
            "last_feedback_revision":None,"last_access":0.0,
        }

        revision=lab._rerank_unserved_queue(
            "session",state,action="skip",article="n0"
        )

        self.assertEqual(revision["changed_positions"],2)
        self.assertIsNone(revision["causal_demotion"])
        self.assertEqual(revision["negative_proof_candidates"],[])
        self.assertEqual(revision["unchanged_negative_proof_candidates"],1)
        observation=revision["negative_proof_observations"][0]
        self.assertFalse(observation["feedback_evidence_changed"])
        self.assertFalse(observation["own_score_decreased"])
        self.assertFalse(observation["own_ranking_score_decreased"])

    def test_live_event_keeps_non_symbolic_tie_priors_frozen_until_mining(self):
        lab=Lab(data=fixture(),config={"mine_interval":100})
        try:
            before=copy.deepcopy(lab._tie_break_stats)
            page=lab.feed_page("u1",limit=1)
            served=page["feed"][0]
            lab.event("u1",served["article"]["id"],"skip",
                      served["context"],served["impression"])
            self.assertEqual(lab._tie_break_stats,before)
            lab.mine()
            self.assertNotEqual(lab._tie_break_stats,before)
        finally:
            lab.close()

    def test_only_emitted_live_rows_accept_feedback(self):
        lab = Lab(data=fixture(), config={"mine_interval":100})
        try:
            first = lab.feed_page("u1", limit=1)
            session = first["session"]
            queued = lab._feed_sessions[session]["queue"][0]
            with self.assertRaisesRegex(
                    ValueError,"active served feed item"):
                lab.event(
                    "u1",queued["article"]["id"],"skip",
                    queued["context"],queued["impression"],
                )
            self.assertNotIn(
                (queued["impression"],queued["article"]["id"]),
                lab._feed_sessions[session]["feedback_contexts"],
            )
        finally:
            lab.engine.close()

    def test_feed_pool_excludes_complete_current_and_replay_history(self):
        data=fixture()
        data["users"]["u1"]={"topics":["ai"],"history":["n1"]}
        data["tests"]=[
            {"user":"u1","candidates":["n2","n3"],"relevant":["n3"],
             "history":[]},
            {"user":"u1","candidates":["n4"],"relevant":["n4"],
             "history":["n2"]},
        ]
        lab=Lab(data=data)
        pool={article for article,_context in lab._feed_pool("u1")}
        self.assertNotIn("n1",pool)
        self.assertNotIn("n2",pool)
        self.assertIn("n3",pool)

    def test_benchmark_has_ranking_metrics(self):
        result = LAB.benchmark({"top_k": 5})
        self.assertIn("mrr", result); self.assertIn("ndcg", result)
        self.assertIn("Official MIND convention",result["metric_definitions"]["mrr"])
        self.assertIn("auc_primary_score", result)
        self.assertIn("auc_signal", result)
        self.assertIn(result["pairwise_pointwise_diagnostic"]["status"],
                      {"pass","hold","not_applicable"})
        self.assertEqual(
            result["pairwise_pointwise_diagnostic"]["delta_95_ci"],
            result["auc_pairwise_delta_95_ci"],
        )
        self.assertEqual(
            result["pairwise_pointwise_diagnostic"]["kind"],
            "diagnostic_not_architecture_gate",
        )
        self.assertTrue(result["promotion_gate"]["deprecated_alias"])
        self.assertEqual(result["mining"]["version"], LAB.version)
        self.assertFalse(result["candidate_retrieval"]["included"])
        self.assertEqual(
            result["ranking_workload"]["oriented_pair_cases"],
            2*result["ranking_workload"]["unordered_pair_comparisons"],
        )
        self.assertEqual(
            result["ranking_workload"]["comparison_policy"],
            "complete_all_pairs",
        )
        self.assertEqual(
            result["ranking_workload"]
                  ["configured_max_unordered_pair_comparisons_per_slate"],
            LAB.config["max_pair_comparisons"],
        )
        reliability=result["heldout_pair_proof_reliability"]
        self.assertEqual(reliability["status"],"computed")
        self.assertGreater(reliability["proof_observations"],0)
        self.assertIn(
            "macro_impression_brier",
            reliability["confidence_shrunk_posterior"],
        )
        self.assertEqual(
            reliability["evaluated_probability"]["definition"],
            "q = 0.5 + confidence * (strength - 0.5)",
        )
        self.assertFalse(
            reliability["evaluated_probability"]["ordinal_ranking_used"]
        )
        self.assertEqual(
            reliability["constant_probability_baselines"]
                       ["constant_0_5_balanced_pair_target"]["probability"],
            0.5,
        )
        self.assertIn(
            "confidence_shrunk_minus_constant_0_5",
            reliability["paired_deltas"],
        )
        self.assertGreaterEqual(result["timings"]["pair_reasoning_seconds"],0.0)
        self.assertGreaterEqual(
            result["timings"]["pair_proof_reliability_seconds"],0.0
        )
        self.assertGreaterEqual(result["timings"]["parity_audit_seconds"],0.0)
        baselines=result["same_cohort_baselines"]
        self.assertEqual(baselines["random_expected_auc"],0.5)
        self.assertIsNotNone(
            baselines["training_click_count_popularity_auc"]
        )
        self.assertEqual(
            baselines["pointwise_symbolic_auc"],
            result["auc_pointwise_ranking"],
        )
        self.assertGreaterEqual(
            result["timings"]["total_seconds"],
            result["timings"]["evaluation_seconds"],
        )
        self.assertIn(
            result["reasoner_semantic_parity"]["status"],
            {"full_model_match","sampled_match"},
        )
        self.assertTrue(
            result["reasoner_semantic_parity"]["sampled_semantic_parity"]
        )
        self.assertEqual(
            result["reasoner_semantic_parity"]["summary"]["error_count"],0
        )
        parity_summary=result["reasoner_semantic_parity"]["summary"]
        self.assertEqual(
            parity_summary["worker_kind"],
            "disposable_isolated_pettachainer",
        )
        self.assertEqual(parity_summary["host_optimized_proof_path_calls"],0)
        self.assertFalse(parity_summary["live_worker_mutated"])
        self.assertEqual(
            parity_summary["queried_pair_signal_roots"],
            parity_summary["cases"]*parity_summary["compiled_channels"],
        )

    def test_benchmark_opt_in_matched_direct_reasoner_ablation(self):
        lab=Lab(data=fixture(),config={
            "min_support":2,"pair_min_support":2,
            "max_rules":8,"pair_max_rules":8,
            "conjunctions":2,"pair_conjunctions":2,
            "pair_aggregation":"proof_margin",
            "pair_margin_transform":"linear","pair_margin_power":1.0,
        })
        self.addCleanup(lab.engine.close)

        result=lab.benchmark({
            "remine":False,"top_k":5,
            "matched_direct_reasoner_ablation":True,
        })
        ablation=result["matched_direct_reasoner_ablation"]

        self.assertEqual(ablation["status"],"completed_match")
        self.assertEqual(
            ablation["scope"],"full_point_and_pair_scoring_output_path"
        )
        self.assertEqual(ablation["normal_ranking_path"],"PeTTaChainer proofs")
        self.assertFalse(ablation["direct_path_used_for_serving"])
        self.assertEqual(ablation["pettachainer_auc"],ablation["direct_auc"])
        self.assertEqual(ablation["auc_delta_direct_minus_pettachainer"],0.0)
        self.assertEqual(ablation["exact_slate_order_match_rate"],1.0)
        self.assertEqual(ablation["exact_point_slate_order_match_rate"],1.0)
        self.assertEqual(ablation["exact_point_signature_match_rate"],1.0)
        self.assertEqual(
            ablation["exact_candidate_signature_match_rate"],1.0
        )
        self.assertEqual(
            ablation["exact_slate_order_and_signature_match_rate"],1.0
        )
        self.assertEqual(ablation["mismatch_impression_ids"],[])
        self.assertGreater(ablation["reconstructed_point_cases"],0)
        self.assertGreater(ablation["reconstructed_pair_cases"],0)
        self.assertGreater(ablation["compiled_point_rules"],0)
        self.assertGreater(ablation["compiled_pair_channels"],0)
        self.assertGreaterEqual(
            result["timings"]["matched_direct_ablation_seconds"],0.0
        )

    def test_benchmark_rejects_non_boolean_direct_ablation_flag(self):
        with self.assertRaisesRegex(
                ValueError,"matched_direct_reasoner_ablation must be a boolean"):
            LAB.benchmark({"matched_direct_reasoner_ablation":"yes"})

    def test_matched_direct_ablation_fails_closed_for_hybrid_point_merge(self):
        lab=Lab(data=fixture(),config={
            "aggregation":"hybrid","min_support":2,"pair_min_support":2,
            "max_rules":8,"pair_max_rules":8,
        })
        self.addCleanup(lab.engine.close)
        result=lab.benchmark({
            "remine":False,"eval_case_limit":1,
            "matched_direct_reasoner_ablation":True,
        })
        ablation=result["matched_direct_reasoner_ablation"]
        self.assertEqual(ablation["status"],"unsupported")
        self.assertIn("weighted point aggregation",ablation["reason"])

    def test_benchmark_reports_matched_cold_and_warm_proof_cache_runs(self):
        lab=Lab(data=fixture())
        self.addCleanup(lab.engine.close)

        cold=lab.benchmark({
            "remine":False,"force_cold_proof_cache":True,
        })
        warm=lab.benchmark({"remine":False})
        cold_again=lab.benchmark({
            "remine":False,"force_cold_proof_cache":True,
        })

        self.assertEqual(cold["proof_cache"]["requested_mode"],"force_cold")
        self.assertEqual(cold["proof_cache"]["observed_mode"],"cold")
        self.assertEqual(cold["proof_cache"]["hits"],0)
        self.assertGreater(cold["proof_cache"]["misses"],0)
        self.assertEqual(warm["proof_cache"]["requested_mode"],"reuse_existing")
        self.assertEqual(warm["proof_cache"]["observed_mode"],"warm")
        self.assertGreater(warm["proof_cache"]["hits"],0)
        self.assertEqual(warm["proof_cache"]["misses"],0)
        self.assertEqual(cold_again["proof_cache"]["observed_mode"],"cold")
        self.assertEqual(cold["auc_proof_only"],warm["auc_proof_only"])
        self.assertEqual(cold["auc_proof_only"],cold_again["auc_proof_only"])
        self.assertEqual(cold["ranking_workload"],warm["ranking_workload"])
        self.assertGreaterEqual(cold["timings"]["ranking_pipeline_seconds"],0.0)
        self.assertNotIn(
            "parity",cold["proof_cache"]["accounting_unit"].lower()
        )

    def test_reasoner_deadlines_are_separate_finite_configuration(self):
        lab=Lab(data=fixture())
        self.addCleanup(lab.engine.close)
        lab.configure({
            "serving_reasoner_timeout_seconds":2.5,
            "benchmark_reasoner_timeout_seconds":45.0,
        })
        result=lab.benchmark({"remine":False,"eval_case_limit":1})
        self.assertEqual(result["reasoner_deadlines"]["serving_seconds"],2.5)
        self.assertEqual(result["reasoner_deadlines"]["benchmark_seconds"],45.0)
        self.assertIn("per scorer RPC",result["reasoner_deadlines"]["scope"])
        for key in (
            "serving_reasoner_timeout_seconds",
            "benchmark_reasoner_timeout_seconds",
        ):
            with self.assertRaisesRegex(ValueError,"finite number greater than 0"):
                lab.configure({key:0})

    def test_hybrid_aggregation_keeps_engine_proof_fields(self):
        lab = Lab(data=fixture())
        lab.configure({"aggregation": "hybrid"})
        rows = lab.score("u1", limit=0)
        self.assertTrue(rows)
        self.assertTrue(all(row["score_method"] == "pettachainer_hybrid_base_rate_posterior"
                            for row in rows))
        self.assertTrue(all(row["engine"] == "PeTTaChainer" for row in rows))

    def test_pair_rule_cap_matches_bounded_parity_contract(self):
        lab = Lab(data=fixture())
        self.addCleanup(lab.engine.close)
        with self.assertRaisesRegex(ValueError,"max_rules must be <= 1024"):
            lab.configure({"max_rules":1025})
        with self.assertRaisesRegex(ValueError,"pair_max_rules must be <= 1024"):
            lab.configure({"pair_max_rules":1025})
        with self.assertRaisesRegex(
                ValueError,"max_pair_comparisons must be <= 32768"):
            lab.configure({"max_pair_comparisons":32769})
        with self.assertRaisesRegex(
                ValueError,"max_total_pair_comparisons must be <= 5000000"):
            lab.configure({"max_total_pair_comparisons":5000001})
        with self.assertRaisesRegex(
                ValueError,"max_proof_cache_entries must be <= 1000000"):
            lab.configure({"max_proof_cache_entries":1000001})


class OptimizationTest(unittest.TestCase):
    def test_empty_history_live_features_match_replay_sequence_snapshot(self):
        data = fixture()
        for article in data["articles"]:
            article["subcategory"] = f'{article["topic"]}_detail'
        data["users"]["u1"] = {
            "topics": ["ai", "science"],
            "history": [],
            "recent_subcategories": [],
        }
        lab = Lab(data=data)
        try:
            article = lab.article("n1")
            replay = history_feature_context(
                article,
                [],
                lab._articles,
                entity_vectors=lab._article_entity_vectors,
                title_idf_model=lab._title_idf_model,
                transition_model=data.get("subcategory_transition_model"),
            )
            live = lab.features("u1", article)
            sequence_keys = {
                "recent_subcategory_affinity_score",
                "topic_recency_score",
                "subcategory_recency_score",
                "topic_recency_bucket",
                "subcategory_recency_bucket",
            }
            self.assertEqual(
                {key: live[key] for key in sequence_keys},
                {key: replay[key] for key in sequence_keys},
            )
            self.assertEqual(live["topic_recency_bucket"], "none")
            self.assertEqual(live["subcategory_recency_bucket"], "none")
        finally:
            lab.engine.close()

    def test_sequence_lineage_aliases_are_recursively_canonicalized(self):
        topic = Lab._pair_evidence_lineage((
            ("pair_topic_recency_quantile", "left_q4"),
            ("pair_recent_topic_share_quantile", "left_q4"),
        ))
        self.assertEqual(topic, frozenset({"topic_interest"}))
        self.assertEqual(
            topic,
            Lab._pair_evidence_lineage((("pair_long_affinity", "left"),)),
        )

        subcategory = Lab._pair_evidence_lineage((
            ("pair_recent_subcategory_affinity_quantile", "left_q4"),
            ("pair_subcategory_recency_quantile", "left_q4"),
            ("pair_subcategory_share_quantile", "left_q4"),
        ))
        self.assertEqual(
            subcategory, frozenset({"pair_subcategory_affinity"})
        )
        self.assertEqual(
            subcategory,
            Lab._pair_evidence_lineage((
                ("pair_subcategory_affinity", "left"),
            )),
        )

    def test_dependency_clustering_ignores_shared_context_gate_but_keeps_owner(self):
        shared_coverage={"pair-1","pair-2"}
        semantic={
            "premises":(
                ("pair_text_semantic_attention_t8","left"),
                ("pair_history_scope","light"),
            ),
            "coverage":shared_coverage,
        }
        taxonomy={
            "premises":(
                ("pair_left_topic","news"),
                ("pair_history_scope","light"),
            ),
            "coverage":shared_coverage,
        }

        self.assertEqual(
            Lab._pair_dependency_lineage(semantic["premises"]),
            frozenset({"pair_text_semantic_top3_mean"}),
        )
        self.assertEqual(
            Lab._pair_dependency_lineage(taxonomy["premises"]),
            frozenset({"topic_interest"}),
        )
        self.assertFalse(Lab._pair_rules_share_dependency(semantic,taxonomy))

        semantic["dependency_owner"]="shared_directional_source"
        taxonomy["dependency_owner"]="shared_directional_source"
        self.assertTrue(Lab._pair_rules_share_dependency(semantic,taxonomy))

    def test_residual_hypergraph_keeps_only_positive_incremental_evidence(self):
        parent=("pair_topic", "left")
        rules=[
            {
                "premises":(parent,), "specificity":1,
                "strength":0.8, "confidence":1.0,
            },
            {
                # Positive in isolation, but weaker than its selected parent.
                "premises":(parent,("pair_format", "left")),
                "specificity":2, "strength":0.7, "confidence":1.0,
            },
            {
                "premises":(parent,("pair_entity", "left")),
                "specificity":2, "strength":0.9, "confidence":1.0,
            },
            {
                # Zero reliability contributes exactly zero residual evidence.
                "premises":(("pair_history", "left"),), "specificity":1,
                "strength":0.9, "confidence":0.0,
            },
        ]

        selected,audit=Lab._select_positive_residual_hyperedges(reversed(rules))

        self.assertEqual(audit,{
            "candidates":4,"selected_positive":2,"rejected_nonpositive":2,
        })
        self.assertEqual(
            [rule["premises"] for rule in selected],
            [(parent,),(parent,("pair_entity", "left"))],
        )
        self.assertTrue(all(rule["residual_delta"]>0.0 for rule in selected))
        self.assertTrue(all(rule["proof_strength"]>0.5 for rule in selected))
        self.assertEqual(selected[1]["residual_parent_count"],1)
        self.assertEqual(
            [rule["dependency_id"] for rule in selected],
            ["pair_mined_cluster_1","pair_mined_cluster_2"],
        )

    def test_sequence_profile_mines_petta_proof_and_ranks_clicked_side(self):
        data = fixture()
        # Remove every non-sequence pair discriminator, then make the three
        # ordered-history summaries consistently favor the clicked item in
        # every impression and temporal fold.
        for article in data["articles"]:
            article.update(topic="shared", subcategory="shared", format="article")
        for event in data["events"]:
            positive = event["action"] == "click"
            event.update({
                "recent_affinity": "none",
                "long_affinity": "none",
                "history_size_bucket": "regular",
                "history_topic_count_bucket": "zero",
                "recent_topic_count_bucket": "zero",
                "subcategory_affinity": "none",
                "title_overlap_detail": "none",
                "entity_recent_top1_similarity": 0.5,
                "recent_subcategory_transition_score": 0.5,
                "recent_subcategory_affinity_score": 0.9 if positive else 0.1,
                "topic_recency_score": 1.0 if positive else 0.0,
                "subcategory_recency_score": 1.0 if positive else 0.0,
            })

        lab = Lab(data=data)
        try:
            lab.configure({
                "pair_feature_profile": "sequence_multi_interest",
                "pair_min_support": 4,
                "pair_min_effect": 0.01,
                "pair_max_rules": 10,
            })
            lab.mine()
            sequence_predicates = {
                "pair_recent_subcategory_affinity",
                "pair_topic_recency",
                "pair_subcategory_recency",
            }
            sequence_rules = [
                rule for rule in lab.pair_rules
                if any(
                    predicate in sequence_predicates and value == "left"
                    for predicate, value in rule["premises"]
                )
            ]
            self.assertTrue(sequence_rules)

            groups = {}
            for event in data["events"]:
                groups.setdefault(event["impression"], []).append(event)
            impression = next(
                events for events in groups.values()
                if any(event["action"] == "click" for event in events)
                and any(event["action"] == "skip" for event in events)
            )
            clicked = next(
                event for event in impression if event["action"] == "click"
            )
            skipped = next(
                event for event in impression if event["action"] == "skip"
            )
            clicked_pair = (
                lab.article(clicked["article"]), lab.event_features(clicked)
            )
            skipped_pair = (
                lab.article(skipped["article"]), lab.event_features(skipped)
            )
            forward = lab._pair_spec(clicked_pair, skipped_pair)
            reverse = lab._pair_spec(skipped_pair, clicked_pair)
            lab._ensure_pair_specs([forward, reverse])
            proof_groups, _calls = lab._proofs_for_pair_specs([forward, reverse])
            proof_text = " ".join(proof_groups[forward[0]])
            self.assertTrue(any(
                rule["dependency_id"] in proof_text for rule in sequence_rules
            ))
            self.assertGreater(
                lab._proof_vote_margin(forward[0], proof_groups[forward[0]]),
                lab._proof_vote_margin(reverse[0], proof_groups[reverse[0]]),
            )

            ranked = lab.score(
                clicked["user"],
                [clicked["article"], skipped["article"]],
                contexts={
                    clicked["article"]: clicked,
                    skipped["article"]: skipped,
                },
                limit=0,
            )
            self.assertEqual(ranked[0]["article"]["id"], clicked["article"])
            self.assertGreater(
                ranked[0]["pairwise_margin_score"],
                ranked[1]["pairwise_margin_score"],
            )
        finally:
            lab.engine.close()

    def test_proof_only_signature_ignores_editorial_tie_breaks(self):
        common={
            "ranking_score":0.5,"pairwise_score":0.5,"score":0.4,
            "stv":{"strength":0.6,"confidence":0.7},
        }
        left={**common,"tie_break":{"topic_prior":0.9}}
        right={**common,"tie_break":{"topic_prior":0.1}}
        self.assertEqual(
            Lab._proof_ranking_signature(left),
            Lab._proof_ranking_signature(right),
        )

    def test_benchmark_rejects_stale_rules_after_mining_config_change(self):
        lab=Lab(data=fixture())
        original_bins=lab.config["pair_numeric_bins"]
        with self.assertRaisesRegex(ValueError,"requires conjunctions"):
            lab.configure({"miner_strategy":"target_aware"})
        self.assertEqual(lab.config["miner_strategy"],"fixed_combinations")
        with self.assertRaisesRegex(ValueError,"remine=false"):
            lab.benchmark({"pair_numeric_bins":2,"remine":False})
        self.assertEqual(lab.config["pair_numeric_bins"],original_bins)
        with self.assertRaisesRegex(ValueError,"ctv_evidence_k"):
            lab.benchmark({"ctv_evidence_k":400,"remine":False})
        with self.assertRaisesRegex(ValueError,"remine=false"):
            lab.benchmark({"miner_strategy":"target_aware","conjunctions":3,
                           "remine":False})
        self.assertEqual(lab.config["miner_strategy"],"fixed_combinations")
        with self.assertRaisesRegex(ValueError,"miner_strategy"):
            lab.configure({"miner_strategy":"not_a_miner"})
        result=lab.benchmark({"pairwise_weight":0.5,"remine":False})
        self.assertEqual(result["config"]["pairwise_weight"],0.5)
        tempered=lab.benchmark({"pair_margin_power":0.5,"remine":False})
        self.assertEqual(tempered["config"]["pair_margin_power"],0.5)
        previous_config=lab.config.copy()
        previous_rules=copy.deepcopy(lab.mined_rules)
        with patch.object(lab,"mine",side_effect=RuntimeError("compile failed")):
            with self.assertRaisesRegex(RuntimeError,"compile failed"):
                lab.benchmark({"miner_strategy":"target_aware",
                               "conjunctions":3})
        self.assertEqual(lab.config,previous_config)
        self.assertEqual(lab.mined_rules,previous_rules)

    def test_architecture_gate_remines_both_and_restores_champion_on_hold(self):
        lab=Lab(data=fixture())
        champion=lab.config["pair_feature_profile"]
        champion_power=lab.config["pair_margin_power"]
        result=lab.compare_profiles({
            "challenger_config":{
                "pair_feature_profile":"semantic_consensus",
                "pair_margin_power":0.33,
            },
            # An AUC delta cannot exceed one, making this a deterministic hold
            # while still exercising both real miner/reasoner pipelines.
            "min_delta":1.0,
            "bootstrap_repetitions":64,
        })
        self.assertEqual(result["kind"],"champion_challenger_architecture_gate")
        self.assertEqual(result["status"],"hold")
        self.assertEqual(result["decision"],"retain_champion")
        self.assertTrue(result["full_cohort_eligible"])
        self.assertTrue(result["champion_restored"])
        self.assertEqual(result["paired_impressions"],len(lab.evaluation_cases()))
        self.assertEqual(result["champion"]["profile"],champion)
        self.assertEqual(result["challenger"]["profile"],"semantic_consensus")
        self.assertEqual(result["champion"]["pair_margin_power"],champion_power)
        self.assertEqual(result["challenger"]["pair_margin_power"],0.33)
        self.assertEqual(
            result["challenger"]["architecture_config"],
            {
                "conjunctions":2,
                "ctv_evidence_k":800.0,
                "miner_strategy":"fixed_combinations",
                "pair_conjunctions":2,
                "pair_ctv_evidence_k":20.0,
                "pair_ctv_mode":"raw_pairs",
                "pair_dependency_mode":"clustered",
                "pair_family_fusion":"flat_margin",
                "pair_feature_profile":"semantic_consensus",
                "pair_margin_power":0.33,
                "pair_max_rules":40,
                "pair_rule_selection_k":20.0,
                "relational_evidence_mode":"disabled",
            },
        )
        self.assertEqual(result["champion"]["engine"],"fpMiner -> PeTTaChainer")
        self.assertIsNotNone(result["champion"]["mining"])
        self.assertIsNotNone(result["challenger"]["mining"])
        self.assertGreater(result["champion"]["reasoner_batches"],0)
        self.assertGreater(result["challenger"]["reasoner_batches"],0)
        self.assertEqual(lab.config["pair_feature_profile"],champion)
        self.assertEqual(result["active_pair_feature_profile"],champion)
        self.assertEqual(lab.config["pair_margin_power"],champion_power)
        self.assertEqual(result["active_pair_margin_power"],champion_power)
        self.assertEqual(result["distinct_from"],"pairwise_pointwise_diagnostic")

    def test_architecture_gate_seals_challenger_overrides_and_keeps_legacy_alias(self):
        lab=Lab(data=fixture())
        with self.assertRaisesRegex(ValueError,"unsupported keys"):
            lab.compare_profiles({
                "challenger_config":{"pair_margin_power":0.33,"max_candidates":1}
            })
        with self.assertRaisesRegex(ValueError,"conflicts"):
            lab.compare_profiles({
                "challenger_pair_feature_profile":"semantic_consensus",
                "challenger_config":{"pair_feature_profile":"stable_multi_interest"},
            })
        # Reaching bootstrap validation proves the historical top-level profile
        # alias is still accepted, without running another full comparison.
        with self.assertRaisesRegex(ValueError,"bootstrap_repetitions"):
            lab.compare_profiles({
                "challenger_pair_feature_profile":"semantic_consensus",
                "bootstrap_repetitions":0,
            })

    def test_recent_entity_similarity_is_a_bounded_numeric_pair_predicate(self):
        compare = Lab._relative_numeric_value
        self.assertEqual(compare(0.9, 0.2), "left")
        self.assertEqual(compare("0.2", "0.9"), "right")
        self.assertEqual(compare(0.3, 0.3), "equal")
        self.assertEqual(compare(0.3, None), "left_known")
        self.assertEqual(compare(None, 0.3), "right_known")
        self.assertEqual(compare("nan", "not-a-number"), "unknown")

        lab = Lab(data=fixture())
        left = lab.contextual_features(
            "u1", lab.article("n1"), {"entity_recent_top1_similarity": 0.75}
        )
        right = lab.contextual_features(
            "u1", lab.article("n2"), {"entity_recent_top1_similarity": 0.25}
        )
        attrs = lab._pair_features(left, right, lab.article("n1"), lab.article("n2"))
        self.assertEqual(attrs["pair_entity_recent_top1_similarity"], "left")

        transition_left = lab.contextual_features(
            "u1", lab.article("n1"), {"recent_subcategory_transition_score": 0.8}
        )
        transition_right = lab.contextual_features(
            "u1", lab.article("n2"), {"recent_subcategory_transition_score": 0.3}
        )
        transition_attrs = lab._pair_features(
            transition_left, transition_right, lab.article("n1"), lab.article("n2")
        )
        self.assertEqual(
            transition_attrs["pair_recent_subcategory_transition"], "left"
        )

        semantic_left=lab.contextual_features(
            "u1",lab.article("n1"),{
                "entity_long_mean_similarity":0.7,
                "title_history_idf_jaccard":0.4,
            }
        )
        semantic_right=lab.contextual_features(
            "u1",lab.article("n2"),{
                "entity_long_mean_similarity":0.2,
                "title_history_idf_jaccard":0.1,
            }
        )
        semantic_attrs=lab._pair_features(
            semantic_left,semantic_right,lab.article("n1"),lab.article("n2")
        )
        self.assertEqual(semantic_attrs["pair_entity_long_mean_similarity"],"left")
        self.assertEqual(semantic_attrs["pair_title_history_idf_jaccard"],"left")

        sequence_left=lab.contextual_features(
            "u1",lab.article("n1"),{
                "recent_subcategory_affinity_score":0.6,
                "topic_recency_score":1.0,
                "subcategory_recency_score":0.5,
            }
        )
        sequence_right=lab.contextual_features(
            "u1",lab.article("n2"),{
                "recent_subcategory_affinity_score":0.2,
                "topic_recency_score":0.25,
                "subcategory_recency_score":0.0,
            }
        )
        sequence_attrs=lab._pair_features(
            sequence_left,sequence_right,lab.article("n1"),lab.article("n2")
        )
        self.assertEqual(sequence_attrs["pair_recent_subcategory_affinity"],"left")
        self.assertEqual(sequence_attrs["pair_topic_recency"],"left")
        self.assertEqual(sequence_attrs["pair_subcategory_recency"],"left")

        text_left=lab.contextual_features(
            "u1",lab.article("n1"),{
                "text_semantic_top3_mean_similarity":0.91,
                "text_semantic_attention_t8_similarity":0.87,
                "text_semantic_attention_t12_similarity":0.93,
            }
        )
        text_right=lab.contextual_features(
            "u1",lab.article("n2"),{
                "text_semantic_top3_mean_similarity":0.42,
                "text_semantic_attention_t8_similarity":0.38,
                "text_semantic_attention_t12_similarity":0.44,
            }
        )
        text_attrs=lab._pair_features(
            text_left,text_right,lab.article("n1"),lab.article("n2")
        )
        self.assertEqual(text_attrs["pair_text_semantic_top3_mean"],"left")
        self.assertEqual(text_attrs["pair_text_semantic_attention_t8"],"left")
        self.assertEqual(text_attrs["pair_text_semantic_attention_t12"],"left")
        self.assertIn("pair_text_semantic_attention_t8", PAIR_FEATURES)
        self.assertIn("pair_text_semantic_attention_t12", PAIR_FEATURES)
        self.assertIn(
            "pair_text_semantic_attention_t8",
            PAIR_FEATURE_PROFILES["text_semantic_attention"],
        )
        self.assertEqual(
            NUMERIC_PAIR_EVIDENCE["text_semantic_attention_t12_similarity"],
            "pair_text_semantic_attention_t12_quantile",
        )
        self.assertEqual(
            PAIR_EVIDENCE_ALIASES["pair_text_semantic_attention_t8"],
            "pair_text_semantic_top3_mean",
        )
        self.assertIn(
            ("pair_text_semantic_attention_t12", "pair_long_affinity"),
            PAIR_INTERACTIONS,
        )

    def test_recent_entity_similarity_is_mined_and_petta_proof_gated(self):
        data = fixture()
        for event in data["events"]:
            event["entity_recent_top1_similarity"] = (
                0.9 if event["action"] == "click" else 0.1
            )
        lab = Lab(data=data)
        entity_rules = [
            rule for rule in lab.pair_rules
            if ("pair_entity_recent_top1_similarity", "left") in rule["premises"]
        ]
        self.assertTrue(entity_rules)
        rule = entity_rules[0]
        self.assertGreater(rule["strength"], 0.5)
        self.assertGreater(rule["vote_weight"], 0.0)
        self.assertGreaterEqual(rule["stable_effect"], lab.config["pair_min_effect"])

        clicked = next(event for event in data["events"] if event["action"] == "click")
        skipped = next(
            event for event in data["events"]
            if event["impression"] == clicked["impression"] and event["action"] == "skip"
        )
        spec = lab._pair_spec(
            (lab.article(clicked["article"]), lab.event_features(clicked)),
            (lab.article(skipped["article"]), lab.event_features(skipped)),
        )
        lab._ensure_pair_specs([spec])
        groups, _calls = lab._proofs_for_pair_specs([spec])
        proof = " ".join(groups[spec[0]])
        self.assertIn(rule["dependency_id"], proof)
        self.assertGreater(lab._proof_vote_margin(spec[0], groups[spec[0]]), 0.0)

    def test_recent_subcategory_transition_is_mined_and_petta_proof_gated(self):
        data = fixture()
        for event in data["events"]:
            event["recent_subcategory_transition_score"] = (
                0.9 if event["action"] == "click" else 0.1
            )
        lab = Lab(data=data)
        transition_rules = [
            rule for rule in lab.pair_rules
            if ("pair_recent_subcategory_transition", "left") in rule["premises"]
        ]
        self.assertTrue(transition_rules)
        clicked = next(event for event in data["events"] if event["action"] == "click")
        skipped = next(
            event for event in data["events"]
            if event["impression"] == clicked["impression"]
            and event["action"] == "skip"
        )
        spec = lab._pair_spec(
            (lab.article(clicked["article"]), lab.event_features(clicked)),
            (lab.article(skipped["article"]), lab.event_features(skipped)),
        )
        lab._ensure_pair_specs([spec])
        groups, _calls = lab._proofs_for_pair_specs([spec])
        proof = " ".join(groups[spec[0]])
        self.assertIn(transition_rules[0]["dependency_id"], proof)
        self.assertGreater(lab._proof_vote_margin(spec[0], groups[spec[0]]), 0.0)

    def test_pair_training_is_symmetric_balanced_and_mined(self):
        lab = Lab(data=fixture())
        lab.configure({
            "pair_feature_profile": "stable_multi_interest",
            "pair_conjunctions": 3,
        })
        lab.mine()
        cases = lab._pair_training_cases(negative_ratio=0)
        self.assertTrue(cases)
        self.assertEqual(sum(case["positive"] for case in cases) * 2, len(cases))
        self.assertTrue(lab.pair_rules)
        self.assertTrue(any(len(rule["premises"]) == 2 for rule in lab.pair_rules))
        self.assertTrue(all(rule["source"].endswith("fpMiner.metta")
                            for rule in lab.pair_rules))
        self.assertTrue(all(rule["variant_id"].startswith(rule["dependency_id"])
                            for rule in lab.pair_rules))
        self.assertLessEqual(len({rule["dependency_id"] for rule in lab.pair_rules}),
                             len(lab.pair_rules))
        self.assertEqual(
            Lab._pair_evidence_lineage((
                ("pair_long_affinity","left"),
                ("pair_entity_recent_top1_similarity_quantile","left_q4"),
            )),
            frozenset({"topic_interest","pair_entity_recent_top1_similarity"}),
        )
        self.assertEqual(
            Lab._pair_evidence_lineage((
                ("pair_recent_affinity","left"),
                ("pair_long_topic_share_quantile","left_q4"),
            )),
            frozenset({"topic_interest"}),
        )
        self.assertEqual(
            Lab._pair_evidence_lineage(((
                "pair_entity_long_mean_similarity","left"
            ),)),
            Lab._pair_evidence_lineage(((
                "pair_entity_recent_top1_similarity","left"
            ),)),
        )

    def test_target_aware_expansion_keeps_real_miner_and_real_petta_proofs(self):
        data=fixture()
        # Three contexts make the long-affinity unary useful but imperfect,
        # while (long=left & same_topic=same) is a non-redundant stable target
        # interaction absent from PAIR_INTERACTIONS. Interleave the contexts so
        # every source-order fold observes the same relationship.
        pair_contexts=(
            ("high","none","n1",("n2",)),
            ("high","none","n1",("n3","n7")),
            ("none","high","n1",("n3","n7")),
        )
        data["events"]=[]
        for repeat in range(12):
            for context_index,(positive_long,negative_long,
                               positive_article,negative_articles) in enumerate(pair_contexts):
                impression=f"target_{repeat}_{context_index}"
                selected_negatives=(negative_articles if repeat%2
                                    else negative_articles[:1])
                rows=[(positive_article,"click",positive_long)]
                rows.extend((article,"skip",negative_long)
                            for article in selected_negatives)
                for article,action,long_affinity in rows:
                    data["events"].append({
                        "user":"u1","article":article,"action":action,
                        "impression":impression,
                        "long_affinity":long_affinity,
                    })
        lab=Lab(data=data)
        lab.configure({
            "miner_strategy":"target_aware",
            "min_support":2,"conjunctions":3,"max_rules":80,
            "pair_min_support":1,"pair_conjunctions":3,"pair_negative_ratio":1,
            "pair_max_rules":80,"pair_min_effect":0.0,
        })
        mining=lab.mine()

        # fpMiner remains the live unary discovery path; target-aware search
        # contributes only deeper structures over the same bounded facts.
        self.assertEqual(mining["miner_strategy"],"target_aware")
        self.assertEqual(mining["miner_calls"],1)
        self.assertTrue(lab.mined_output)
        self.assertTrue(any(rule["source"].endswith("fpMiner.metta")
                            and len(rule["premises"])==1
                            for rule in lab.mined_rules))
        self.assertGreater(mining["target_search"]["deeper_candidate_patterns"],0)
        self.assertGreater(mining["target_search"]["audit"]["nodes_evaluated"],0)

        pair_mining=mining["pairwise"]
        self.assertEqual(pair_mining["miner_strategy"],"target_aware")
        self.assertEqual(pair_mining["miner_calls"],1)
        self.assertEqual(pair_mining["fpminer_support_unit"],
                         "raw_oriented_pair_case")
        self.assertEqual(pair_mining["target_support_unit"],
                         "equal_impression_mass")
        self.assertEqual(pair_mining["min_support_unit"],
                         "equal_impression_mass_target_search_only")
        self.assertEqual(pair_mining["search_weight_unit"],
                         "equal_impression_mass_target_search_only")
        def impression_weights(cases):
            weights={}; counts={}
            for case in cases:
                impression=case["impression"]
                weights[impression]=weights.get(impression,0.0)+case["search_weight"]
                counts[impression]=counts.get(impression,0)+1
            self.assertTrue(all(math.isclose(weight,1.0)
                                for weight in weights.values()))
            return counts
        sampled_counts=impression_weights(lab._pair_training_cases())
        full_counts=impression_weights(lab._pair_training_cases(negative_ratio=0))
        self.assertEqual(set(sampled_counts.values()),{2})
        self.assertGreater(len(set(full_counts.values())),1)
        target_pair_rules=[rule for rule in lab.pair_rules
                           if rule["source"].endswith("target_miner.py")]
        self.assertTrue(target_pair_rules)
        self.assertGreater(len(lab.pair_rules),2)
        for rule in target_pair_rules:
            weighted=rule["target_weighted_contingency"]
            counts=rule["target_count_contingency"]
            self.assertAlmostEqual(weighted["tp"]+weighted["fp"],
                                   rule["target_weighted_support"])
            self.assertEqual(counts["tp"],rule["mined_support"])
            self.assertTrue(rule["target_fold_statistics"])
            self.assertIn("target_wracc",rule)
        whitelisted={frozenset(pair) for pair in PAIR_INTERACTIONS}
        self.assertTrue(any(
            frozenset(predicate for predicate,_value in rule["premises"])
            not in whitelisted
            for rule in target_pair_rules
        ))
        self.assertTrue(any(rule["source"].endswith("fpMiner.metta")
                            and len(rule["premises"])==1
                            for rule in lab.pair_rules))

        # Find a grounded training pair that activates one target-expanded
        # rule, then require its dependency ID in an actual PeTTa proof.
        by_impression={}
        for event in lab.data["events"]:
            by_impression.setdefault(event["impression"],[]).append(event)
        proved=False
        for events in by_impression.values():
            positives=[event for event in events if event["action"]=="click"]
            negatives=[event for event in events if event["action"]=="skip"]
            for positive in positives:
                for negative in negatives:
                    left=(lab.article(positive["article"]),lab.event_features(positive))
                    right=(lab.article(negative["article"]),lab.event_features(negative))
                    attrs=lab._bounded_pair_features(lab._pair_features(
                        left[1],right[1],left[0],right[0]
                    ))
                    matching=next((rule for rule in target_pair_rules if all(
                        attrs.get(predicate)==value
                        for predicate,value in rule["premises"]
                    )),None)
                    if matching is None:
                        continue
                    spec=lab._pair_spec(left,right)
                    lab._ensure_pair_specs([spec])
                    groups,_calls=lab._proofs_for_pair_specs([spec])
                    if matching["dependency_id"] in " ".join(groups[spec[0]]):
                        proved=True
                        break
                if proved: break
            if proved: break
        self.assertTrue(proved,"target-expanded pair rule lacked a PeTTa proof")

        # Saturate a tiny shared cap. Deterministic family quotas must preserve
        # both the real-fpMiner unary backoff and target-expanded evidence.
        lab.configure({"pair_max_rules":2})
        saturated=lab.mine()["pairwise"]
        self.assertEqual(saturated["rules"],2)
        self.assertTrue(any(rule["source"].endswith("fpMiner.metta")
                            and len(rule["premises"])==1
                            for rule in lab.pair_rules))
        self.assertTrue(any(rule["source"].endswith("target_miner.py")
                            and len(rule["premises"])>1
                            for rule in lab.pair_rules))

    def test_pair_decision_is_a_real_two_hop_pettachainer_proof(self):
        lab = Lab(data=fixture())
        articles = lab.data["articles"]
        proof = ""
        for left in articles:
            for right in articles:
                if left["id"] == right["id"]:
                    continue
                spec = lab._pair_spec(
                    (left, lab.contextual_features("u1", left)),
                    (right, lab.contextual_features("u1", right)),
                )
                lab._ensure_pair_specs([spec])
                groups, _calls = lab._proofs_for_pair_specs([spec])
                proof = " ".join(groups[spec[0]])
                if proof:
                    break
            if proof:
                break
        self.assertIn("pair_mined_cluster_", proof)
        self.assertIn("pair_decision_rule", proof)
        self.assertGreater(lab._proof_vote_margin(spec[0], groups[spec[0]]), 0.0)

    def test_pair_margin_uses_inferred_stv_confidence_once_per_dependency(self):
        lab = Lab(data=fixture())
        self.assertEqual(lab.config["pair_margin_power"],1.0)
        lab.configure({"pair_margin_transform": "linear"})
        low = lab._proof_vote_margin(
            "synthetic_low", ["(pair_mined_cluster_1 (STV 0.8 0.4))"]
        )
        high = lab._proof_vote_margin(
            "synthetic_high", ["(pair_mined_cluster_1 (STV 0.8 0.9))"]
        )
        duplicate = lab._proof_vote_margin(
            "synthetic_duplicate",
            ["(pair_mined_cluster_1 (STV 0.8 0.4))",
             "(pair_mined_cluster_1 (STV 0.8 0.9))"],
        )
        self.assertAlmostEqual(low, 0.24)
        self.assertAlmostEqual(high, 0.54)
        self.assertAlmostEqual(duplicate, high)

        self.assertTrue(lab._pair_margin_cache)
        lab.configure({"pair_margin_power":0.5})
        self.assertFalse(lab._pair_margin_cache)
        tempered=lab._proof_vote_margin(
            "synthetic_tempered",["(pair_mined_cluster_1 (STV 0.8 0.4))"]
        )
        negative=lab._proof_vote_margin(
            "synthetic_tempered_negative",
            ["(pair_mined_cluster_1 (STV 0.2 0.4))"],
        )
        self.assertAlmostEqual(tempered,math.sqrt(0.24))
        self.assertAlmostEqual(negative,-math.sqrt(0.24))
        for invalid in (0.0,4.01,float("nan"),float("inf")):
            with self.assertRaisesRegex(ValueError,"pair_margin_power"):
                lab.configure({"pair_margin_power":invalid})

        lab.configure({"pair_margin_transform": "log_odds","pair_margin_power":1.0})
        log_odds = lab._proof_vote_margin(
            "synthetic_log_odds", ["(pair_mined_cluster_1 (STV 0.8 0.4))"]
        )
        self.assertAlmostEqual(log_odds, 0.4895482253187058)

        decision_sources = [source for source in lab._pair_rule_sources
                            if "(: pair_decision_rule_" in source]
        clusters = {rule["dependency_id"] for rule in lab.pair_rules}
        self.assertEqual(len(decision_sources), len(clusters))
        self.assertEqual(len(decision_sources), len(set(decision_sources)))

    def test_pair_rules_are_recalibrated_on_all_training_pairs(self):
        lab = Lab(data=fixture())
        lab.configure({"pair_negative_ratio": 1})
        mining = lab.mine()
        population = lab._pair_training_cases(negative_ratio=0)
        pair_mining=mining["pairwise"]
        self.assertGreater(pair_mining["source_cases"], pair_mining["cases"])
        self.assertEqual(pair_mining["source_cases"], len(population))
        for rule in lab.pair_rules:
            matches = [case for case in population if all(
                case["attrs"].get(predicate) == value
                for predicate, value in rule["premises"]
            )]
            self.assertTrue(matches)
            expected = sum(case["positive"] for case in matches) / len(matches)
            self.assertAlmostEqual(rule["strength"], expected)
            self.assertTrue(any(
                source.startswith(f'(: {rule["variant_id"]} ')
                for source in lab._pair_rule_sources
            ))

    def test_pair_impression_macro_ctv_uses_full_unsampled_training_population(self):
        lab = Lab(data=fixture())
        try:
            lab.configure({
                "pair_negative_ratio": 1,
                "pair_ctv_mode": "impression_macro",
                "pair_ctv_evidence_k": 2.0,
                "pair_min_support": 1,
            })
            mining = lab.mine()["pairwise"]
            population = [
                {**case, "attrs": lab._bounded_pair_features(case["attrs"])}
                for case in lab._pair_training_cases(negative_ratio=0)
            ]

            self.assertGreater(
                mining["ctv_estimation_population"]["oriented_pair_cases"],
                mining["discovery_sample"]["oriented_pair_cases"],
            )
            self.assertFalse(
                mining["ctv_estimation_population"]
                      ["held_out_evaluation_labels_used"]
            )
            self.assertFalse(
                mining["discovery_sample"]
                      ["discovery_statistics_reused_for_final_ctv_estimation"]
            )
            self.assertTrue(mining["full_population_ctv_estimation"])
            self.assertEqual(
                mining["selected_ctv_min_support_unit"],
                "equal_impression_weighted_activation_mass",
            )
            self.assertTrue(lab.pair_rules)
            unavailable = {"incomparable", "unknown", "left_known", "right_known"}
            for rule in lab.pair_rules:
                expected = calibrate_ctv((
                    CTVObservation(
                        impression_id=case["impression"],
                        matched=all(
                            case["attrs"].get(predicate) == value
                            for predicate, value in rule["premises"]
                        ),
                        target=bool(case["positive"]),
                        applicable=all(
                            case["attrs"].get(predicate) is not None
                            and str(case["attrs"].get(predicate)).lower()
                                not in unavailable
                            for predicate, _value in rule["premises"]
                        ),
                        weight=float(case["search_weight"]),
                    )
                    for case in population
                ), evidence_k=2.0, rule_kind="pair_preference")
                petta_expected = reencode_ctv_confidence(
                    expected, evidence_k=800.0
                )
                self.assertAlmostEqual(rule["strength"], expected.positive.strength)
                self.assertAlmostEqual(
                    rule["confidence"], petta_expected.positive.confidence
                )
                self.assertAlmostEqual(
                    rule["selection_confidence"],
                    expected.positive.confidence,
                )
                self.assertAlmostEqual(
                    rule["calibrated_support"],
                    expected.positive.weighted_support,
                )
                self.assertEqual(
                    rule["ctv_estimation_mode"], "impression_macro_kish"
                )
                self.assertEqual(rule["ctv_calibration"]["evidence_k"],800.0)
                self.assertEqual(
                    rule["selection_calibration"]["evidence_k"],2.0
                )
                self.assertEqual(rule["petta_evidence_k"],800.0)
                self.assertEqual(rule["ctv_support"], rule["calibrated_support"])
                self.assertLessEqual(
                    expected.positive.effective_impressions,
                    expected.positive.distinct_impressions,
                )
                self.assertEqual(
                    rule["calibrated_support_unit"],
                    "equal_impression_weighted_activation_mass",
                )
                self.assertEqual(
                    rule["ctv_support_unit"], rule["calibrated_support_unit"]
                )
            target = mining["pair_target_semantics"]
            self.assertEqual(
                target["training_unit"],
                "one closed observed training impression",
            )
            self.assertIn("both left/right orientations", target["pair_construction"])
            self.assertIn("not a universal", target["interpretation"])
            self.assertEqual(mining["petta_ctv_evidence_k"],800.0)
            self.assertEqual(mining["pair_rule_selection_k"],2.0)
        finally:
            lab.engine.close()

    def test_pairwise_order_matches_reported_ranking_signature(self):
        lab = Lab(data=fixture())
        rows = lab.score("u1", limit=0)
        signatures = [lab._ranking_signature(row) for row in rows]
        self.assertEqual(signatures, sorted(signatures, reverse=True))
        self.assertTrue(all("pairwise_margin_score" in row for row in rows))

    def test_remine_atomically_replaces_an_isolated_deterministic_scorer(self):
        lab = Lab(data=fixture())
        candidates=[article["id"] for article in lab.data["articles"]]
        before=lab.score("u1",candidates=candidates,limit=0)
        before_signature=[lab._ranking_signature(row) for row in before]
        before_proofs=[row["proofs"] for row in before]
        before_rules=[(rule["premises"],rule["strength"],rule["confidence"])
                      for rule in lab.mined_rules]
        old_pid=lab.engine.pid

        lab.mine()
        after=lab.score("u1",candidates=candidates,limit=0)

        self.assertNotEqual(old_pid,lab.engine.pid)
        self.assertEqual(before_rules,[(rule["premises"],rule["strength"],rule["confidence"])
                                       for rule in lab.mined_rules])
        self.assertEqual(before_signature,[lab._ranking_signature(row) for row in after])
        self.assertEqual([bool(proofs) for proofs in before_proofs],
                         [bool(row["proofs"]) for row in after])

    def test_feedback_invalidates_pre_mining_ranking_cache(self):
        lab = Lab(data=fixture())
        first = lab.score("u1", limit=0)
        lab.event("u1", "n4", "click")
        second = lab.score("u1", limit=0)
        self.assertIsNot(first, second)
        self.assertFalse(lab.state()["engine"]["benchmark_clean"])
        with self.assertRaisesRegex(ValueError, "reload the dataset before benchmarking"):
            lab.benchmark({"remine": False})

    def test_pointwise_mode_makes_no_pair_queries(self):
        lab = Lab(data=fixture())
        lab.configure({"ranking_mode": "pointwise"})
        lab._pair_query_calls = 0
        lab.score("u1", limit=0)
        self.assertEqual(lab._pair_query_calls, 0)

    def test_sampling_is_calibrated_on_full_population_and_depth_is_cumulative(self):
        data = fixture()
        for index, event in enumerate(data["events"]):
            event["impression"] = f"impression_{index // len(data['articles'])}"
        lab = Lab(data=data)
        lab.configure({"feature_profile": "core", "negative_ratio": 1,
                       "conjunctions": 3, "max_rules": 100})
        mining = lab.mine()
        source_positives = sum(event["action"] == "click" for event in data["events"])
        self.assertLess(mining["cases"], mining["source_cases"])
        self.assertEqual(mining["source_positives"], source_positives)
        self.assertAlmostEqual(mining["base_rate"], source_positives / len(data["events"]))
        self.assertGreater(mining["sample_base_rate"], mining["base_rate"])
        self.assertEqual(mining["depths"], [2, 3])
        self.assertGreater(mining["miner_calls"], 1)
        self.assertTrue(any(len(rule["premises"]) == 1 for rule in lab.mined_rules))
        self.assertTrue(any(len(rule["premises"]) == 2 for rule in lab.mined_rules))
        rule = lab.mined_rules[0]
        matches = [event for event in data["events"]
                   if all(lab.event_features(event).get(key) == value
                          for key, value in rule["premises"])]
        expected = sum(event["action"] == "click" for event in matches) / len(matches)
        self.assertAlmostEqual(rule["strength"], expected)
        lab.score("u1")
        self.assertTrue(lab._proof_cache)
        self.assertTrue(lab._point_channel_proof_cache)
        lab.configure({"chain_steps": lab.config["chain_steps"] + 1})
        self.assertFalse(lab._proof_cache)
        self.assertFalse(lab._point_channel_proof_cache)
        capped = lab.benchmark({"max_candidates": 1, "remine": False})
        self.assertEqual(capped["cases"], len(data["tests"]))
        self.assertGreater(capped["sampled_without_positive"], 0)
        admission=capped["candidate_retrieval"]["logged_slate_admission"]
        self.assertEqual(admission["candidate_cap"],1)
        self.assertEqual(
            admission["supplied_candidates"],
            sum(len(case["candidates"]) for case in data["tests"]),
        )
        self.assertEqual(admission["admitted_candidates"],len(data["tests"]))
        self.assertLess(admission["relevant_item_inclusion_recall"],1.0)
        self.assertFalse(
            capped["candidate_retrieval"]["production_retrieval_recall_measured"]
        )

    def test_live_feedback_keeps_display_context_and_candidate_cap_is_label_blind(self):
        lab = Lab(data=fixture())
        result = lab.event("u1", "n4", "click",
                           {"subcategory": "environment", "time_bucket": "evening"},
                           "feed_window_7")
        recorded = lab.data["events"][-1]
        self.assertEqual(recorded["subcategory"], "environment")
        self.assertEqual(recorded["time_bucket"], "evening")
        self.assertEqual(recorded["impression"], "feed_window_7")
        self.assertEqual(result["recorded_context"]["subcategory"], "environment")
        self.assertIn("climate", lab.user_topics("u1"))
        candidates = ["n1", "n2", "n3", "n4", "n5"]
        first = Lab._bounded_candidates({"id": "same", "candidates": candidates,
                                         "relevant": ["n1"]}, 3, 7)
        second = Lab._bounded_candidates({"id": "same", "candidates": candidates,
                                          "relevant": ["n5"]}, 3, 7)
        self.assertEqual(first, second)
        page = lab.feed_page("u1", limit=1)
        served = page["feed"][0]
        lab.event("u1", served["article"]["id"], "skip",
                  {"topic": "forged"}, served["impression"])
        self.assertEqual(lab.data["events"][-1]["topic"], served["context"]["topic"])
        self.assertNotEqual(lab.data["events"][-1]["topic"], "forged")

    def test_tuner_uses_disjoint_validation_and_restores_full_evaluation(self):
        data = fixture()
        data["tests"] = [
            {**copy.deepcopy(case), "id": f"eval_{repeat}_{case['user']}"}
            for repeat in range(5) for case in data["tests"]
        ]
        lab = Lab(data=data)
        result = lab.tune({"support_grid": [4], "depth_grid": [2],
                           "negative_ratio_grid": [0],
                           "aggregation_grid": ["max", "weighted"]})
        self.assertEqual(result["split"], {"tuning": 10, "heldout": 10})
        self.assertIn(result["best_config"]["aggregation"], {"max", "weighted"})
        self.assertEqual(result["heldout_result"]["cases"], 10)
        self.assertEqual(len(lab.evaluation_cases()), 20)


if __name__ == "__main__": unittest.main()
