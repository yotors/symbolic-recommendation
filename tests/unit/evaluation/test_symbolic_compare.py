import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recommendation.evaluation.symbolic_compare import compare_artifacts


def _artifacts():
    data = {
        "events": [{"user": "u1"}],
        "evaluation": [
            {"id": "i1", "user": "u1", "history": []},
            {"id": "i2", "user": "u1", "history": ["past"]},
            {"id": "i3", "user": "u2", "history": list(range(10))},
        ],
    }
    digest = hashlib.sha256(json.dumps(data).encode()).hexdigest()
    baseline = {"dataset_sha256": digest, "result": {
        "auc_cases": 3, "cases": 3, "candidates": 6,
        "auc_per_impression": [
            {"id": identity, "index": index, "candidates": 2, "positives": 1,
             "negatives": 1, "auc": 0.5, "auc_proof_only": 0.4}
            for index, identity in enumerate(("i1", "i2", "i3"))
        ],
    }}
    challenger = copy.deepcopy(baseline)
    for row in challenger["result"]["auc_per_impression"]:
        row["auc"] += 0.1
        row["auc_proof_only"] += 0.2
    return baseline, challenger, data


class SymbolicComparisonTest(unittest.TestCase):
    def test_exact_effect_constant_delta_and_paired_cluster_interval(self):
        baseline, challenger, data = _artifacts()
        result = compare_artifacts(baseline, challenger, data=data, repetitions=100)
        self.assertAlmostEqual(result["metrics"]["served"]["delta"], 0.1)
        self.assertAlmostEqual(result["metrics"]["proof_only"]["delta"], 0.2)
        for signal, delta in (("served", 0.1), ("proof_only", 0.2)):
            for method in ("paired_impression_95_ci", "paired_user_cluster_95_ci"):
                for bound in result["metrics"][signal][method]:
                    self.assertAlmostEqual(bound, delta)
        self.assertEqual(result["cohort"]["unique_users"], 2)
        self.assertEqual(result["subgroups"]["seen_user"]["impressions"], 2)
        self.assertEqual(result["subgroups"]["new_user"]["impressions"], 1)
        self.assertEqual(result["subgroups"]["history_cold"]["impressions"], 1)

    def test_deterministic_and_record_order_independent(self):
        baseline, challenger, data = _artifacts()
        challenger["result"]["auc_per_impression"][0]["auc"] = 0.1
        expected = compare_artifacts(baseline, challenger, data=data, repetitions=100)
        challenger["result"]["auc_per_impression"].reverse()
        baseline["result"]["auc_per_impression"].reverse()
        self.assertEqual(expected, compare_artifacts(baseline, challenger, data=data, repetitions=100))

    def test_user_cluster_resamples_whole_users_not_individual_impressions(self):
        baseline, challenger, data = _artifacts()
        for case in data["evaluation"]:
            case["user"] = "same_user"
        for row, value in zip(challenger["result"]["auc_per_impression"], (0.2, 0.5, 0.8)):
            row["auc"] = value
        result = compare_artifacts(baseline, challenger, data=data, repetitions=100)
        metric = result["metrics"]["served"]
        self.assertAlmostEqual(metric["delta"], 0.0)
        for bound in metric["paired_user_cluster_95_ci"]:
            self.assertAlmostEqual(bound, 0.0)
        self.assertLess(metric["paired_impression_95_ci"][0], 0.0)
        self.assertGreater(metric["paired_impression_95_ci"][1], 0.0)

    def test_ranking_ablation_selection(self):
        baseline, challenger, _ = _artifacts()
        baseline["ranking_ablations"] = [{"result": challenger["result"]}]
        result = compare_artifacts(baseline, baseline, challenger_ablation=0, repetitions=100)
        self.assertAlmostEqual(result["metrics"]["served"]["delta"], 0.1)
        self.assertEqual(result["challenger_selection"], "ranking_ablations[0]")
        self.assertIsNone(result["metrics"]["served"]["paired_user_cluster_95_ci"])

    def test_mismatched_dataset_cohort_counts_and_duplicate_ids_rejected(self):
        for mutation in ("hash", "cohort", "counts", "duplicate", "empty"):
            baseline, challenger, _ = _artifacts()
            rows = challenger["result"]["auc_per_impression"]
            if mutation == "hash":
                challenger["dataset_sha256"] = "0" * 64
            elif mutation == "cohort":
                rows[0]["id"] = "different"
            elif mutation == "counts":
                rows[0].update(candidates=3, negatives=2)
            elif mutation == "duplicate":
                rows[1]["id"] = rows[0]["id"]
            else:
                rows.clear()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_artifacts(baseline, challenger, repetitions=100)

    def test_malformed_auc_values_are_rejected(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.7", None):
            baseline, challenger, _ = _artifacts()
            challenger["result"]["auc_per_impression"][0]["auc_proof_only"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite AUC"):
                compare_artifacts(baseline, challenger, repetitions=100)

    def test_dataset_aliases_and_gaps_preserve_source_order_thirds(self):
        baseline, challenger, data = _artifacts()
        source = data.pop("evaluation")
        data["impressions"] = [source[0], {"id": "not_auc_eligible", "user": "u9"}, source[1], source[2]]
        result = compare_artifacts(baseline, challenger, data=data, repetitions=100)
        self.assertEqual(result["subgroups"]["source_order_third_1"]["impressions"], 1)
        self.assertEqual(result["subgroups"]["source_order_third_2"]["impressions"], 1)
        self.assertEqual(result["subgroups"]["source_order_third_3"]["impressions"], 1)
        data["impressions"].append(copy.deepcopy(source[0]))
        with self.assertRaisesRegex(ValueError, "duplicate dataset"):
            compare_artifacts(baseline, challenger, data=data, repetitions=100)

    def test_cli_gzip_dataset_hash_is_verified(self):
        baseline, challenger, data = _artifacts()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data.json.gz"
            source.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            for name, artifact in (("baseline", baseline), ("challenger", challenger)):
                artifact["dataset_sha256"] = digest
                (root / f"{name}.json").write_text(json.dumps(artifact))
            output = root / "comparison.json"
            subprocess.run([
                sys.executable, "-m", "recommendation.evaluation.symbolic_compare",
                "--baseline", str(root / "baseline.json"), "--challenger", str(root / "challenger.json"),
                "--data", str(source), "--output", str(output), "--repetitions", "100",
            ], check=True, capture_output=True, text=True, timeout=10)
            result = json.loads(output.read_text())
            self.assertTrue(result["dataset_bytes_verified"])
            self.assertEqual(result["dataset_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
