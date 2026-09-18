import copy
import math
import unittest

from recommendation.app.server import Lab, MINING_CONFIG_KEYS


def _rule(dependency, predicate, strength=0.8):
    return {
        "id": dependency,
        "dependency_id": dependency,
        "premises": [(predicate, "left")],
        "strength": strength,
        "confidence": 1.0,
    }


def _rows():
    return [
        {
            "article": {"id": article},
            "score": 0.5,
            "stv": {"strength": 0.5, "confidence": 0.0},
            "tie_break": {
                "topic_prior": 0.0,
                "format_prior": 0.0,
                "subcategory_prior": 0.0,
            },
        }
        for article in ("a", "b", "c")
    ]


def _bare_lab(fusion, rules):
    lab=object.__new__(Lab)
    lab.config={
        "ranking_mode":"pairwise",
        "pair_aggregation":"proof_margin",
        "pair_family_fusion":fusion,
        "pairwise_weight":1.0,
        "pairwise_fusion":"rank",
        "pair_margin_transform":"linear",
        "pair_margin_power":1.0,
        "pairwise_opponents":0,
        "max_rules":30,
        "pair_max_rules":40,
        "max_pair_comparisons":8192,
        "max_total_pair_comparisons":1_000_000,
        "max_proof_cache_entries":250_000,
        "mining_retention_max_units":10_000,
        "mining_retention_max_cases":100_000,
    }
    lab.pair_rules=rules
    lab.version=1
    lab._pair_margin_cache={}
    return lab


def _proof(dependency, strength=0.8, confidence=1.0):
    return f"(by {dependency} (STV {strength} {confidence}))"


PLAN=(
    [(0,1,"ab","ba"),(0,2,"ac","ca"),(1,2,"bc","cb")],
    [],
)


class PairFamilyFusionTest(unittest.TestCase):
    def test_pair_proof_reliability_uses_one_total_weight_per_impression(self):
        prepared=[
            ({"id":"i1","relevant":["a"]},[],0,0),
            ({"id":"i2","relevant":["c"]},[],0,0),
        ]
        planned=[
            ([{"article":{"id":"a"}},{"article":{"id":"b"}}],
             ([(0,1,"ab","ba")],[])),
            ([{"article":{"id":"c"}},{"article":{"id":"d"}}],
             ([(0,1,"cd","dc")],[])),
        ]
        proof_map={
            "ab":["(proof (STV 1.0 0.5))"],
            "ba":["(proof (STV 0.0 0.5))"],
            # The second impression has four times as many active channels in
            # each orientation. It must still have the same macro weight as i1.
            "cd":["(proof (STV 0.5 1.0))"]*4,
            "dc":["(proof (STV 0.5 1.0))"]*4,
        }
        result=Lab._pair_proof_reliability(
            prepared,planned,proof_map
        )
        self.assertEqual(result["status"],"computed")
        self.assertEqual(result["impressions"],2)
        self.assertEqual(result["proof_observations"],10)
        self.assertEqual(result["eligible_direction_roots"],4)
        self.assertEqual(result["proved_direction_roots"],4)
        self.assertAlmostEqual(
            result["confidence_shrunk_posterior"]
                  ["macro_impression_brier"],
            (0.0625+0.25)/2,
        )
        self.assertAlmostEqual(
            result["raw_strength_without_confidence"]
                  ["macro_impression_brier"],
            (0.0+0.25)/2,
        )
        balanced=result["constant_probability_baselines"][
            "constant_0_5_balanced_pair_target"
        ]
        empirical=result["constant_probability_baselines"][
            "constant_empirical_active_proof_rate"
        ]
        self.assertEqual(balanced["probability"],0.5)
        self.assertAlmostEqual(balanced["macro_impression_brier"],0.25)
        self.assertAlmostEqual(
            balanced["macro_impression_log_loss"],math.log(2.0)
        )
        self.assertEqual(empirical["probability"],0.5)
        self.assertEqual(
            result["evaluated_probability"]["ordinal_ranking_used"],False
        )
        shrunk_minus_balanced=result["paired_deltas"][
            "confidence_shrunk_minus_constant_0_5"
        ]
        self.assertAlmostEqual(
            shrunk_minus_balanced["macro_impression_brier"],-0.09375
        )
        self.assertEqual(
            len(shrunk_minus_balanced["macro_impression_brier_95_ci"]),2
        )
        self.assertEqual(len(result["brier_delta_95_ci"]),2)

    def test_pair_proof_reliability_empirical_constant_is_impression_macro(self):
        prepared=[
            ({"id":"i1","relevant":["a"]},[],0,0),
            ({"id":"i2","relevant":["c"]},[],0,0),
        ]
        planned=[
            ([{"article":{"id":"a"}},{"article":{"id":"b"}}],
             ([(0,1,"ab","ba")],[])),
            ([{"article":{"id":"c"}},{"article":{"id":"d"}}],
             ([(0,1,"cd","dc")],[])),
        ]
        # Active-proof target rates are 1 for i1 and 1/2 for i2. Equal
        # impression weighting therefore gives the empirical constant 3/4,
        # even though i2 has twice as many proof observations.
        proof_map={
            "ab":["(proof (STV 0.8 1.0))"],
            "cd":["(proof (STV 0.4 1.0))"],
            "dc":["(proof (STV 0.4 1.0))"],
        }

        result=Lab._pair_proof_reliability(prepared,planned,proof_map)

        baseline=result["constant_probability_baselines"][
            "constant_empirical_active_proof_rate"
        ]
        self.assertAlmostEqual(baseline["probability"],0.75)
        self.assertAlmostEqual(baseline["macro_impression_brier"],0.1875)
        self.assertAlmostEqual(baseline["equal_impression_weighted_ece"],0.0)
        self.assertAlmostEqual(
            result["confidence_shrunk_posterior"]["macro_impression_brier"],
            0.15,
        )
        delta=result["descriptive_same_cohort_deltas"][
            "confidence_shrunk_minus_empirical_constant"
        ]
        self.assertAlmostEqual(delta["macro_impression_brier"],-0.0375)
        self.assertNotIn("macro_impression_brier_95_ci",delta)
        self.assertNotIn("macro_impression_log_loss_95_ci",delta)
        self.assertIn("no confidence interval",delta["inference"])

    def test_balanced_margin_comparison_reliability_is_pre_tournament(self):
        structured=_rule(
            "pair_mined_cluster_1","pair_long_affinity",strength=0.8
        )
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8",
            strength=0.6,
        )
        lab=_bare_lab("balanced_margin",[structured,semantic])
        prepared=[
            ({"id":"i1","relevant":["a"]},[],0,0),
            ({"id":"i2","relevant":["d"]},[],0,0),
        ]
        planned=[
            ([{"article":{"id":"a"}},{"article":{"id":"b"}}],
             ([(0,1,"ab","ba")],[])),
            ([{"article":{"id":"c"}},{"article":{"id":"d"}}],
             ([(0,1,"cd","dc")],[])),
        ]
        proof_map={
            # i1: structured forward margin 0.6 and semantic reverse margin
            # 0.2 give a family-balanced fused margin (0.6 - 0.2) / 2 = 0.2.
            "ab":["(proof pair_mined_cluster_1 (STV 0.8 1.0))"],
            "ba":["(proof pair_mined_cluster_2 (STV 0.6 1.0))"],
            # i2: one structured dependency gives 0.2 - 0.6 = -0.4.
            "cd":["(proof pair_mined_cluster_1 (STV 0.6 1.0))"],
            "dc":["(proof pair_mined_cluster_1 (STV 0.8 1.0))"],
        }

        result=Lab._pair_proof_reliability(
            prepared,planned,proof_map,
            comparison_probability=lab._balanced_margin_probability_evaluator(),
        )

        # Reliability owns a private cache. The subsequently timed production
        # rank must still consume cold dependency margins itself.
        self.assertEqual(lab._pair_margin_cache,{})
        aggregated=result["aggregated_balanced_margin_pair_preference"]
        self.assertEqual(aggregated["status"],"computed")
        self.assertEqual(aggregated["active_comparisons"],2)
        self.assertEqual(
            aggregated["evaluated_probability"]["ordinal_ranking_used"],False
        )
        q1=1/(1+math.exp(-0.2)); q2=1/(1+math.exp(0.4))
        expected_brier=((q1-1)**2+q2**2)/2
        self.assertAlmostEqual(
            aggregated["confidence_aware_probability"]
                      ["macro_impression_brier"],
            expected_brier,
        )
        balanced=aggregated["constant_probability_baselines"][
            "constant_0_5_neutral_pair_preference"
        ]
        empirical=aggregated["constant_probability_baselines"][
            "constant_empirical_left_click_rate"
        ]
        self.assertAlmostEqual(balanced["macro_impression_brier"],0.25)
        self.assertAlmostEqual(empirical["probability"],0.5)
        delta=aggregated["paired_deltas"][
            "confidence_aware_minus_constant_0_5"
        ]
        self.assertAlmostEqual(
            delta["macro_impression_brier"],expected_brier-0.25
        )
        self.assertEqual(len(delta["macro_impression_brier_95_ci"]),2)
        empirical_delta=aggregated["descriptive_same_cohort_deltas"][
            "confidence_aware_minus_empirical_constant"
        ]
        self.assertNotIn("macro_impression_brier_95_ci",empirical_delta)
        self.assertIn("no confidence interval",empirical_delta["inference"])

        lab._pairwise_rank(_rows()[:2],[],planned[0][1],proof_map)
        self.assertGreater(len(lab._pair_margin_cache),0)

    def test_pair_plan_fails_closed_before_quadratic_budget_is_exceeded(self):
        lab=_bare_lab("flat_margin",[])
        lab.config["max_pair_comparisons"]=9
        lab.article=lambda article_id:{"id":article_id}
        lab._pair_spec=lambda left,right:(
            f'pair_{left[0]["id"]}_{right[0]["id"]}',{}
        )
        rows=[{"article":{"id":f"a{index}"}} for index in range(5)]
        specs=[(row["article"]["id"],"unused",{}, {}) for row in rows]
        with self.assertRaisesRegex(
                ValueError,"10 unordered pairs.*max_pair_comparisons=9"):
            lab._pairwise_plan(rows,specs)

    def test_bounded_opponent_plan_obeys_the_same_comparison_budget(self):
        lab=_bare_lab("flat_margin",[])
        lab.config.update(pairwise_opponents=1,max_pair_comparisons=5)
        lab.article=lambda article_id:{"id":article_id}
        lab._pair_spec=lambda left,right:(
            f'pair_{left[0]["id"]}_{right[0]["id"]}',{}
        )
        rows=[{"article":{"id":f"a{index}"}} for index in range(5)]
        specs=[(row["article"]["id"],"unused",{}, {}) for row in rows]
        comparisons,_pair_specs=lab._pairwise_plan(rows,specs)
        self.assertEqual(len(comparisons),5)
        lab.config["max_pair_comparisons"]=4
        with self.assertRaisesRegex(
                ValueError,"more than 4 unordered pairs"):
            lab._pairwise_plan(rows,specs)

    def test_pair_case_cache_fails_before_partial_publication(self):
        lab=_bare_lab("flat_margin",[])
        lab.config["max_proof_cache_entries"]=1
        lab._pair_case_attrs={}
        with self.assertRaisesRegex(RuntimeError,"pair case cache limit"):
            lab._ensure_pair_specs([
                ("pair_a",{"pair_topic":"left"}),
                ("pair_b",{"pair_topic":"right"}),
            ])
        self.assertEqual(lab._pair_case_attrs,{})

    def test_pair_margin_cache_fails_closed_at_bound(self):
        lab=_bare_lab("flat_margin",[])
        lab.config["max_proof_cache_entries"]=1
        lab._pair_proof_origins={}
        lab._proof_dependency_margins(
            "pair_a",[_proof("pair_mined_cluster_1")]
        )
        with self.assertRaisesRegex(RuntimeError,"pair margin cache limit"):
            lab._proof_dependency_margins(
                "pair_b",[_proof("pair_mined_cluster_2")]
            )
        self.assertEqual(len(lab._pair_margin_cache),1)

    def test_one_family_balanced_rank_is_flat_margin_equivalent(self):
        rules=[_rule("pair_mined_cluster_1","pair_long_affinity")]
        proofs={
            "ab":[_proof("pair_mined_cluster_1")], "ba":[],
            "ac":[_proof("pair_mined_cluster_1")], "ca":[],
            "bc":[_proof("pair_mined_cluster_1")], "cb":[],
        }
        flat=_bare_lab("flat_margin",rules)._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        balanced=_bare_lab("balanced_rank",rules)._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        self.assertEqual(
            [row["article"]["id"] for row in flat],
            [row["article"]["id"] for row in balanced],
        )
        for left,right in zip(flat,balanced):
            self.assertEqual(
                (left["ranking_score"],left["pairwise_score"],
                 left["pairwise_margin_score"],left["pairwise_rank_score"],
                 left["pairwise_proof_coverage"],
                 left["pairwise_directional_coverage"]),
                (right["ranking_score"],right["pairwise_score"],
                 right["pairwise_margin_score"],right["pairwise_rank_score"],
                 right["pairwise_proof_coverage"],
                 right["pairwise_directional_coverage"]),
            )
        self.assertEqual(
            [row["pairwise_rank_score"] for row in flat],
            [1.0,0.5,0.0],
        )

    def test_balanced_rank_weights_two_families_equally(self):
        structured=(
            _rule("pair_mined_cluster_1","pair_long_affinity"),
            _rule("pair_mined_cluster_2","pair_subcategory_affinity"),
        )
        semantic=_rule(
            "pair_mined_cluster_3","pair_text_semantic_attention_t8"
        )
        proofs={
            # Two structured dependencies vote a>b>c, while the single text
            # dependency votes c>b>a.  Family balancing must yield a tie even
            # though the flat dependency sum favours a.
            "ab":[_proof(rule["dependency_id"]) for rule in structured],
            "ba":[_proof(semantic["dependency_id"])],
            "ac":[_proof(rule["dependency_id"]) for rule in structured],
            "ca":[_proof(semantic["dependency_id"])],
            "bc":[_proof(rule["dependency_id"]) for rule in structured],
            "cb":[_proof(semantic["dependency_id"])],
        }
        rules=[*structured,semantic]
        flat=_bare_lab("flat_margin",rules)._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        balanced=_bare_lab("balanced_rank",rules)._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        self.assertGreater(flat[0]["pairwise_rank_score"],
                           flat[-1]["pairwise_rank_score"])
        self.assertEqual(
            {row["pairwise_rank_score"] for row in balanced},{0.5}
        )
        by_id={row["article"]["id"]:row for row in balanced}
        self.assertEqual(
            by_id["a"]["pairwise_family_rank_scores"],
            {"structured_symbolic":1.0,"text_semantic":0.0},
        )
        self.assertEqual(
            by_id["c"]["pairwise_family_rank_scores"],
            {"structured_symbolic":0.0,"text_semantic":1.0},
        )

    def test_balanced_margin_retains_confidence_adjusted_magnitude(self):
        structured=_rule(
            "pair_mined_cluster_1","pair_long_affinity",strength=0.9
        )
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8",
            strength=0.9,
        )
        proofs={
            # Both families induce exactly opposite ordinal rankings.  The
            # structured proofs are confident; the semantic proofs are weakly
            # supported.  balanced_rank intentionally makes this a tie, while
            # balanced_margin must retain PeTTa's confidence shrinkage.
            "ab":[_proof(structured["dependency_id"],0.9,0.95)],
            "ba":[_proof(semantic["dependency_id"],0.9,0.10)],
            "ac":[_proof(structured["dependency_id"],0.9,0.95)],
            "ca":[_proof(semantic["dependency_id"],0.9,0.10)],
            "bc":[_proof(structured["dependency_id"],0.9,0.95)],
            "cb":[_proof(semantic["dependency_id"],0.9,0.10)],
        }
        rules=[structured,semantic]
        ordinal=_bare_lab("balanced_rank",rules)._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        magnitude=_bare_lab("balanced_margin",rules)._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        self.assertEqual(
            {row["pairwise_rank_score"] for row in ordinal},{0.5}
        )
        self.assertEqual(
            [row["article"]["id"] for row in magnitude],["a","b","c"]
        )
        self.assertEqual(
            [row["pairwise_rank_score"] for row in magnitude],[1.0,0.5,0.0]
        )
        by_id={row["article"]["id"]:row for row in magnitude}
        self.assertGreater(
            by_id["a"]["pairwise_family_margin_scores"]["structured_symbolic"],
            abs(by_id["a"]["pairwise_family_margin_scores"]["text_semantic"]),
        )
        # c(2s-1): (0.95*0.8 - 0.10*0.8) / two families = 0.34.
        self.assertAlmostEqual(
            by_id["a"]["pairwise_fused_margin_score"],0.34
        )
        self.assertAlmostEqual(
            by_id["c"]["pairwise_fused_margin_score"],-0.34
        )

    def test_balanced_margin_averages_dependencies_before_families(self):
        structured=(
            _rule("pair_mined_cluster_1","pair_long_affinity"),
            _rule("pair_mined_cluster_2","pair_subcategory_affinity"),
        )
        semantic=_rule(
            "pair_mined_cluster_3","pair_text_semantic_attention_t8"
        )
        proofs={
            # Duplicating identical dependencies in the structured family must
            # not double that family's capacity relative to text semantics.
            "ab":[_proof(rule["dependency_id"]) for rule in structured],
            "ba":[_proof(semantic["dependency_id"])],
            "ac":[_proof(rule["dependency_id"]) for rule in structured],
            "ca":[_proof(semantic["dependency_id"])],
            "bc":[_proof(rule["dependency_id"]) for rule in structured],
            "cb":[_proof(semantic["dependency_id"])],
        }
        ranked=_bare_lab("balanced_margin",[
            *structured,semantic
        ])._pairwise_rank(_rows(),[],PLAN,proofs)
        self.assertEqual(
            {row["pairwise_rank_score"] for row in ranked},{0.5}
        )
        self.assertEqual(
            {row["pairwise_fused_margin_score"] for row in ranked},{0.0}
        )

    def test_balanced_margin_is_invariant_to_compiled_inactive_rules(self):
        active=_rule("pair_mined_cluster_1","pair_long_affinity")
        proofs={
            "ab":[_proof(active["dependency_id"])], "ba":[],
            "ac":[_proof(active["dependency_id"])], "ca":[],
            "bc":[_proof(active["dependency_id"])], "cb":[],
        }
        baseline=_bare_lab("balanced_margin",[active])._pairwise_rank(
            _rows(),[],PLAN,proofs
        )
        # These rules are present in the compiled structured family, but none
        # has a forward or reverse PeTTa proof for any comparison. They must
        # abstain rather than enter a denominator and weaken active evidence.
        inactive=[
            _rule(f"pair_mined_cluster_{index}","pair_subcategory_affinity")
            for index in range(2,102)
        ]
        extended=_bare_lab("balanced_margin",[
            active,*inactive
        ])._pairwise_rank(_rows(),[],PLAN,proofs)
        self.assertEqual(extended,baseline)

    def test_balanced_margin_missing_family_abstains_without_dilution(self):
        structured=_rule("pair_mined_cluster_1","pair_long_affinity")
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8"
        )
        proofs={
            "ab":[_proof(structured["dependency_id"])], "ba":[],
            "ac":[_proof(structured["dependency_id"])], "ca":[],
            "bc":[_proof(structured["dependency_id"])], "cb":[],
        }
        only={row["article"]["id"]:row for row in
              _bare_lab("balanced_margin",[structured])._pairwise_rank(
                  _rows(),[],PLAN,proofs
              )}
        with_absent={row["article"]["id"]:row for row in
                     _bare_lab("balanced_margin",[
                         structured,semantic
                     ])._pairwise_rank(_rows(),[],PLAN,proofs)}
        for article in only:
            self.assertEqual(
                with_absent[article]["pairwise_fused_margin_score"],
                only[article]["pairwise_fused_margin_score"],
            )
            self.assertEqual(
                with_absent[article]["pairwise_score"],
                only[article]["pairwise_score"],
            )

    def test_balanced_margin_active_dependency_replication_has_no_scale_effect(self):
        first=_rule("pair_mined_cluster_1","pair_long_affinity")
        duplicate=_rule("pair_mined_cluster_2","pair_subcategory_affinity")
        semantic=_rule(
            "pair_mined_cluster_3","pair_text_semantic_attention_t8"
        )
        baseline_proofs={
            "ab":[_proof(first["dependency_id"])],
            "ba":[_proof(semantic["dependency_id"])],
            "ac":[_proof(first["dependency_id"])],
            "ca":[_proof(semantic["dependency_id"])],
            "bc":[_proof(first["dependency_id"])],
            "cb":[_proof(semantic["dependency_id"])],
        }
        replicated_proofs={
            case:(proofs+([_proof(duplicate["dependency_id"])]
                          if case in {"ab","ac","bc"} else []))
            for case,proofs in baseline_proofs.items()
        }
        baseline={row["article"]["id"]:row for row in
                  _bare_lab("balanced_margin",[
                      first,semantic
                  ])._pairwise_rank(_rows(),[],PLAN,baseline_proofs)}
        replicated={row["article"]["id"]:row for row in
                    _bare_lab("balanced_margin",[
                        first,duplicate,semantic
                    ])._pairwise_rank(_rows(),[],PLAN,replicated_proofs)}
        for article in baseline:
            self.assertEqual(
                replicated[article]["pairwise_fused_margin_score"],
                baseline[article]["pairwise_fused_margin_score"],
            )

    def test_balanced_margin_is_reversal_symmetric(self):
        structured=_rule("pair_mined_cluster_1","pair_long_affinity")
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8"
        )
        proofs={
            "ab":[_proof(structured["dependency_id"],0.9,0.9)], "ba":[],
            "ac":[_proof(semantic["dependency_id"],0.7,0.4)], "ca":[],
            "bc":[_proof(structured["dependency_id"],0.8,0.6)], "cb":[],
        }
        reversed_proofs={
            forward:copy.deepcopy(proofs[reverse])
            for forward,reverse in (
                ("ab","ba"),("ba","ab"),("ac","ca"),
                ("ca","ac"),("bc","cb"),("cb","bc"),
            )
        }
        first={row["article"]["id"]:row for row in
               _bare_lab("balanced_margin",[
                   structured,semantic
               ])._pairwise_rank(_rows(),[],PLAN,proofs)}
        second={row["article"]["id"]:row for row in
                _bare_lab("balanced_margin",[
                    structured,semantic
                ])._pairwise_rank(_rows(),[],PLAN,reversed_proofs)}
        for article in first:
            self.assertAlmostEqual(
                first[article]["pairwise_fused_margin_score"],
                -second[article]["pairwise_fused_margin_score"],
            )
            self.assertAlmostEqual(
                first[article]["pairwise_score"]
                +second[article]["pairwise_score"],1.0,
            )
            self.assertAlmostEqual(
                first[article]["pairwise_rank_score"]
                +second[article]["pairwise_rank_score"],1.0,
            )

    def test_balanced_margin_equal_opposing_evidence_is_a_neutral_tie(self):
        structured=_rule("pair_mined_cluster_1","pair_long_affinity")
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8"
        )
        proofs={
            "ab":[_proof(structured["dependency_id"])],
            "ba":[_proof(semantic["dependency_id"])],
            "ac":[_proof(structured["dependency_id"])],
            "ca":[_proof(semantic["dependency_id"])],
            "bc":[_proof(structured["dependency_id"])],
            "cb":[_proof(semantic["dependency_id"])],
        }
        ranked=_bare_lab("balanced_margin",[
            structured,semantic
        ])._pairwise_rank(_rows(),[],PLAN,proofs)
        self.assertEqual({row["pairwise_fused_margin_score"] for row in ranked},{0.0})
        self.assertEqual({row["pairwise_score"] for row in ranked},{0.5})
        self.assertEqual({row["pairwise_rank_score"] for row in ranked},{0.5})

    def test_balanced_margin_probability_is_bounded_for_large_log_odds(self):
        rule=_rule("pair_mined_cluster_1","pair_long_affinity",0.999999999)
        proofs={
            "ab":[_proof(rule["dependency_id"],0.999999999,1.0)], "ba":[],
            "ac":[_proof(rule["dependency_id"],0.999999999,1.0)], "ca":[],
            "bc":[_proof(rule["dependency_id"],0.999999999,1.0)], "cb":[],
        }
        lab=_bare_lab("balanced_margin",[rule])
        lab.config["pair_margin_transform"]="log_odds"
        ranked=lab._pairwise_rank(_rows(),[],PLAN,proofs)
        by_id={row["article"]["id"]:row for row in ranked}
        self.assertTrue(all(0.0<=row["pairwise_score"]<=1.0 for row in ranked))
        self.assertGreater(by_id["a"]["pairwise_score"],0.99)
        self.assertLess(by_id["c"]["pairwise_score"],0.01)

    def test_balanced_rank_is_reversal_symmetric(self):
        structured=_rule("pair_mined_cluster_1","pair_long_affinity")
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8"
        )
        proofs={
            "ab":[_proof(structured["dependency_id"]),
                  _proof(semantic["dependency_id"])], "ba":[],
            "ac":[_proof(structured["dependency_id"]),
                  _proof(semantic["dependency_id"])], "ca":[],
            "bc":[_proof(structured["dependency_id"])],
            "cb":[_proof(semantic["dependency_id"])],
        }
        reversed_proofs={
            forward:copy.deepcopy(proofs[reverse])
            for forward,reverse in (
                ("ab","ba"),("ba","ab"),("ac","ca"),
                ("ca","ac"),("bc","cb"),("cb","bc"),
            )
        }
        original=_bare_lab("balanced_rank",[structured,semantic])
        reversed_lab=_bare_lab("balanced_rank",[structured,semantic])
        first={row["article"]["id"]:row for row in original._pairwise_rank(
            _rows(),[],PLAN,proofs
        )}
        second={row["article"]["id"]:row for row in reversed_lab._pairwise_rank(
            _rows(),[],PLAN,reversed_proofs
        )}
        for article in first:
            self.assertAlmostEqual(
                first[article]["pairwise_rank_score"]
                +second[article]["pairwise_rank_score"],1.0
            )
            for family,value in first[article][
                    "pairwise_family_rank_scores"].items():
                self.assertAlmostEqual(
                    value+second[article]["pairwise_family_rank_scores"][family],
                    1.0,
                )

    def test_family_classification_and_config_are_binary_and_non_mining(self):
        self.assertEqual(
            Lab._pair_rule_family({"premises":[
                ("pair_long_affinity","left"),
                ("pair_text_semantic_top1","left"),
            ]}),
            "text_semantic",
        )
        self.assertNotIn("pair_family_fusion",MINING_CONFIG_KEYS)
        lab=_bare_lab("flat_margin",[])
        # Supply the remaining defaults read by configure; no mining or proof
        # cache invalidation is needed because this setting changes fusion only.
        lab.config.update({"conjunctions":2,"pair_conjunctions":2,
                           "pair_numeric_bins":4,
                           "miner_strategy":"fixed_combinations"})
        lab._feed_sessions={"old":{}}
        lab.feed_cache={"old":[]}
        lab._proof_cache={"proof":"kept"}
        lab._pair_proof_cache={"proof":"kept"}
        lab.configure({"pair_family_fusion":"balanced_rank"})
        self.assertEqual(lab.config["pair_family_fusion"],"balanced_rank")
        self.assertFalse(lab._feed_sessions)
        self.assertFalse(lab.feed_cache)
        self.assertEqual(lab._pair_proof_cache,{"proof":"kept"})
        lab.configure({"pair_family_fusion":"balanced_margin"})
        self.assertEqual(lab.config["pair_family_fusion"],"balanced_margin")
        with self.assertRaisesRegex(ValueError,"pair_family_fusion"):
            lab.configure({"pair_family_fusion":"invalid"})


if __name__ == "__main__":
    unittest.main()
