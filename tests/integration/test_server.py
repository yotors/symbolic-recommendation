import unittest

from recommendation.app.server import Lab, fixture
from recommendation.mining.rule_parser import balanced_forms


class ChampionServerTest(unittest.TestCase):
    def setUp(self):
        self.lab=Lab(data=fixture(),config={
            "min_support":2,
            "pair_min_support":2,
            "max_rules":8,
            "pair_max_rules":8,
        })
        self.addCleanup(self.lab.close)

    def test_real_fpminer_rules_are_compiled_into_petta(self):
        self.assertTrue(self.lab.mined_rules)
        self.assertTrue(balanced_forms(" ".join(self.lab.mined_output)))
        self.assertTrue(all(
            rule["source"].endswith("miner/fpMiner.metta")
            for rule in self.lab.mined_rules
        ))
        self.assertIsNotNone(self.lab.engine.pid)

    def test_architecture_is_fixed_but_bounded_parameters_are_configurable(self):
        configured=self.lab.configure({"min_support":3,"top_k":4})
        self.assertEqual(configured["min_support"],3)
        self.assertEqual(configured["top_k"],4)
        with self.assertRaisesRegex(ValueError,"fixed to"):
            self.lab.configure({"ranking_mode":"pointwise"})
        with self.assertRaisesRegex(ValueError,"fixed to"):
            self.lab.configure({"miner_strategy":"fixed_combinations"})
        configured=self.lab.configure({
            "pair_feature_profile":"llm_conditional_quantile_causal_popularity"
        })
        self.assertEqual(
            configured["pair_feature_profile"],
            "llm_conditional_quantile_causal_popularity",
        )
        with self.assertRaisesRegex(ValueError,"must be one of"):
            self.lab.configure({"pair_feature_profile":"unknown"})

    def test_benchmark_runs_the_champion_path(self):
        result=self.lab.benchmark({
            "remine":False,
            "eval_case_limit":2,
            "max_candidates":0,
        })
        self.assertEqual(result["cases"],2)
        self.assertIsNotNone(result["auc"])
        self.assertEqual(
            result["config"]["miner_strategy"],"petta_conditional_seed_only"
        )
        self.assertEqual(
            result["config"]["pair_feature_profile"],
            "llm_conditional_quantile",
        )

    def test_feedback_updates_and_reranks_the_same_session(self):
        page=self.lab.feed_page("u1",limit=3)
        selected=page["feed"][0]
        result=self.lab.event(
            "u1",selected["article"]["id"],"skip",
            context=selected.get("context"),
            impression=selected.get("impression"),
            feed_session=page["session"],
            queue_revision=page["queue_revision"],
            feed_position=page["position"],
        )
        self.assertTrue(result["profile_changed"])
        self.assertEqual(
            result["queue_revision"]["session"],page["session"]
        )
        self.assertGreaterEqual(
            result["queue_revision"]["reranked_candidates"],1
        )

    def test_external_candidate_slate_is_proof_ranked(self):
        result=self.lab.rerank("u1",["n1","n2","n3"],limit=2)
        self.assertEqual(result["engine"],"PeTTaChainer")
        self.assertEqual(result["candidate_count"],3)
        self.assertEqual(result["result_count"],2)
        self.assertEqual(len(result["recommendations"]),2)
        self.assertTrue(all(row["engine"]=="PeTTaChainer"
                            for row in result["recommendations"]))

    def test_external_candidate_slate_is_bounded_and_unambiguous(self):
        with self.assertRaisesRegex(ValueError,"duplicates"):
            self.lab.rerank("u1",["n1","n1"],limit=1)
        with self.assertRaisesRegex(ValueError,"outside the slate"):
            self.lab.rerank(
                "u1",["n1"],contexts={"n2":{"topic":"technology"}},
                limit=1,
            )
        self.lab.configure({"rerank_candidate_limit":2})
        with self.assertRaisesRegex(ValueError,"rerank_candidate_limit=2"):
            self.lab.rerank("u1",["n1","n2","n3"],limit=1)


if __name__ == "__main__":
    unittest.main()
