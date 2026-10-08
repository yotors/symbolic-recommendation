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

    def test_benchmark_runs_the_champion_path(self):
        result=self.lab.benchmark({
            "remine":False,
            "eval_case_limit":2,
            "max_candidates":0,
        })
        self.assertEqual(result["cases"],2)
        self.assertIsNotNone(result["auc"])
        self.assertEqual(
            result["config"]["miner_strategy"],"conditional_llm_seed_only"
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


if __name__ == "__main__":
    unittest.main()
