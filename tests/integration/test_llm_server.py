"""Champion LLM observations feed mined rules; they never rank directly."""

import copy
import unittest

from recommendation.app.server import Lab
from recommendation.tests.fixtures import conditional_annotation_fixture


class ChampionLLMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab=Lab(conditional_annotation_fixture(),config={
            "min_support":2,
            "pair_min_support":2,
            "max_rules":8,
            "pair_max_rules":8,
        })
        cls.addClassCleanup(cls.lab.close)

    def test_fixed_profile_uses_conditional_llm_seed_mining(self):
        self.assertEqual(
            self.lab.config["pair_feature_profile"],
            "llm_conditional_quantile",
        )
        search=self.lab.last_pair_mining["target_search"]
        self.assertEqual(search["kind"],"conditional_llm_seed_only")
        self.assertTrue(search["requires_real_fpminer_seed"])
        self.assertTrue(all(
            rule["source"].endswith((
                "miner/fpMiner.metta","mining/conditional_llm_mining.py"
            ))
            for rule in self.lab.pair_rules
        ))

    def test_llm_quantile_fit_is_label_independent(self):
        def fit(actions):
            lab=object.__new__(Lab)
            lab.config={
                "pair_feature_profile":"llm_conditional_quantile",
                "pair_numeric_bins":4,
            }
            lab.data={"events":[
                {"impression":"train","score":score,"action":action}
                for score,action in zip((0.0,0.1,0.4,0.9),actions)
            ]}
            lab.event_features=lambda event:{
                "llm_concept_affinity":event["score"]
            }
            lab._fit_pair_numeric_encoders()
            return lab._numeric_pair_encoders[
                "llm_concept_affinity"
            ].to_json()

        self.assertEqual(
            fit(("click","skip","skip","click")),
            fit(("skip","click","click","skip")),
        )

    def test_missing_llm_observation_abstains_symmetrically(self):
        predicate="pair_llm_concept_affinity_quantile"
        present={"llm_concept_affinity":0.8}
        missing={}
        forward=self.lab._pair_features(
            present,missing,{}, {},needed={predicate}
        )
        reverse=self.lab._pair_features(
            missing,present,{}, {},needed={predicate}
        )
        self.assertEqual(forward[predicate],"incomparable")
        self.assertEqual(reverse[predicate],"incomparable")

    def test_annotation_provenance_is_verified_before_mining(self):
        corrupted=copy.deepcopy(conditional_annotation_fixture())
        corrupted["llm_article_annotations"][next(iter(
            corrupted["llm_article_annotations"]
        ))]["concepts"].append("tampered")
        with self.assertRaisesRegex(ValueError,"provenance"):
            Lab(corrupted)


if __name__ == "__main__":
    unittest.main()
