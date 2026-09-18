import copy
import json
import unittest

from recommendation.app.server import Lab, fixture


class ServingModelTest(unittest.TestCase):
    def test_round_trip_skips_mining_and_preserves_ranking(self):
        source=Lab(data=fixture())
        self.addCleanup(source.close)
        model=json.loads(json.dumps(source.serving_model()))

        replica=Lab(data=fixture(),serving_model=model)
        self.addCleanup(replica.close)

        self.assertEqual(replica._startup_mode,"serving_model")
        self.assertEqual(
            replica._serving_model_sha256,model["model_sha256"]
        )
        self.assertEqual(replica.last_mining["execution"],"serving_model_load")
        self.assertEqual(
            json.loads(json.dumps(replica.mined_rules)),model["point_rules"]
        )
        self.assertEqual(
            json.loads(json.dumps(replica.pair_rules)),model["pair_rules"]
        )

        def signature(rows):
            return [
                (row["article"]["id"],Lab._ranking_signature(row))
                for row in rows
            ]

        self.assertEqual(signature(replica.score("u1")),
                         signature(source.score("u1")))

    def test_model_is_deterministic_and_tamper_evident(self):
        source=Lab(data=fixture())
        self.addCleanup(source.close)
        first=source.serving_model()
        self.assertEqual(first,source.serving_model())

        tampered=copy.deepcopy(first)
        tampered["point_rules"][0]["strength"]=0.0
        with self.assertRaisesRegex(ValueError,"digest"):
            Lab(data=fixture(),serving_model=tampered)


if __name__ == "__main__":
    unittest.main()
