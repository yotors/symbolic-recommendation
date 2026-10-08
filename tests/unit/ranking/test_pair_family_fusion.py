import copy
import math
import unittest

from recommendation.app.server import Lab


def _rule(dependency, predicate, strength=0.8):
    return {
        "id":dependency,
        "dependency_id":dependency,
        "premises":[(predicate,"left")],
        "strength":strength,
        "confidence":1.0,
    }


def _proof(dependency, strength=0.8, confidence=1.0):
    return f"(by {dependency} (STV {strength} {confidence}))"


def _rows():
    return [{
        "article":{"id":article},
        "score":0.5,
        "stv":{"strength":0.5,"confidence":0.0},
        "tie_break":{
            "topic_prior":0.0,
            "format_prior":0.0,
            "subcategory_prior":0.0,
        },
    } for article in ("a","b","c")]


def _bare_lab(rules):
    lab=object.__new__(Lab)
    lab.config={
        "pairwise_weight":1.0,
        "pair_margin_transform":"log_odds",
        "pair_margin_power":1.0,
        "max_pair_comparisons":8192,
        "max_total_pair_comparisons":1_000_000,
        "max_proof_cache_entries":250_000,
    }
    lab.pair_rules=rules
    lab.version=1
    lab._pair_margin_cache={}
    lab._pair_proof_origins={}
    return lab


PLAN=([(0,1,"ab","ba"),(0,2,"ac","ca"),(1,2,"bc","cb")],[])


class PairFamilyFusionTest(unittest.TestCase):
    def test_reliability_uses_equal_total_weight_per_impression(self):
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
            "cd":["(proof (STV 0.5 1.0))"]*4,
            "dc":["(proof (STV 0.5 1.0))"]*4,
        }

        result=Lab._pair_proof_reliability(prepared,planned,proof_map)

        self.assertEqual(result["status"],"computed")
        self.assertEqual(result["impressions"],2)
        self.assertEqual(result["proof_observations"],10)
        self.assertAlmostEqual(
            result["confidence_shrunk_posterior"]["macro_impression_brier"],
            (0.0625+0.25)/2,
        )
        baseline=result["constant_probability_baselines"][
            "constant_0_5_balanced_pair_target"
        ]
        self.assertAlmostEqual(baseline["macro_impression_brier"],0.25)
        self.assertAlmostEqual(baseline["macro_impression_log_loss"],math.log(2))

    def test_pair_plan_fails_closed_before_budget_is_exceeded(self):
        lab=_bare_lab([])
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

    def test_pair_case_cache_fails_before_partial_publication(self):
        lab=_bare_lab([])
        lab.config["max_proof_cache_entries"]=1
        lab._pair_case_attrs={}
        with self.assertRaisesRegex(RuntimeError,"pair case cache limit"):
            lab._ensure_pair_specs([
                ("pair_a",{"pair_long_affinity":"left"}),
                ("pair_b",{"pair_long_affinity":"right"}),
            ])
        self.assertEqual(lab._pair_case_attrs,{})

    def test_pair_margin_cache_fails_closed_at_bound(self):
        lab=_bare_lab([])
        lab.config["max_proof_cache_entries"]=1
        lab._proof_dependency_margins(
            "pair_a",[_proof("pair_mined_cluster_1")]
        )
        with self.assertRaisesRegex(RuntimeError,"pair margin cache limit"):
            lab._proof_dependency_margins(
                "pair_b",[_proof("pair_mined_cluster_2")]
            )
        self.assertEqual(len(lab._pair_margin_cache),1)

    def test_balanced_family_rank_orders_a_complete_preference_chain(self):
        rule=_rule("pair_mined_cluster_1","pair_long_affinity")
        proofs={
            "ab":[_proof(rule["dependency_id"])],"ba":[],
            "ac":[_proof(rule["dependency_id"])],"ca":[],
            "bc":[_proof(rule["dependency_id"])],"cb":[],
        }

        ranked=_bare_lab([rule])._pairwise_rank(_rows(),[],PLAN,proofs)

        self.assertEqual([row["article"]["id"] for row in ranked],["a","b","c"])
        self.assertEqual(
            [row["pairwise_rank_score"] for row in ranked],[1.0,0.5,0.0]
        )

    def test_balanced_family_rank_is_reversal_symmetric(self):
        structured=_rule("pair_mined_cluster_1","pair_long_affinity")
        semantic=_rule(
            "pair_mined_cluster_2","pair_text_semantic_attention_t8"
        )
        proofs={
            "ab":[_proof(structured["dependency_id"]),
                  _proof(semantic["dependency_id"])],"ba":[],
            "ac":[_proof(structured["dependency_id"]),
                  _proof(semantic["dependency_id"])],"ca":[],
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
        first={row["article"]["id"]:row for row in
               _bare_lab([structured,semantic])._pairwise_rank(
                   _rows(),[],PLAN,proofs
               )}
        second={row["article"]["id"]:row for row in
                _bare_lab([structured,semantic])._pairwise_rank(
                    _rows(),[],PLAN,reversed_proofs
                )}
        for article in first:
            self.assertAlmostEqual(
                first[article]["pairwise_rank_score"]
                +second[article]["pairwise_rank_score"],1.0
            )


if __name__ == "__main__":
    unittest.main()
