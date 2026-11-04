"""Parent-conditioned selection rejects redundant conjunctions."""
import multiprocessing as mp
import re
import unittest

try:
    from petta import PeTTa
except (ImportError, OSError):
    PeTTa = None


def _worker(connection):
    try:
        from petta import PeTTa as Runtime

        runtime = Runtime()
        runtime.load_metta_file("recommendation/miner/helpers.metta")
        runtime.load_metta_file("recommendation/miner/fpMiner.metta")
        runtime.load_metta_file("recommendation/miner/parentAwareSelection.metta")
        runtime.load_metta_file("recommendation/miner/predicateRankingPolicy.metta")
        runtime.load_metta_file("recommendation/miner/adaptiveParentPolicy.metta")
        runtime.process_metta_string("!(bind! &parent_test (new-space))")
        for index in range(40):
            outcome = "click" if index < 10 else "skip"
            facts = [f'(engagement c{index} "{outcome}")']
            if index < 20:
                facts.extend((f'(a c{index} "x")', f'(b c{index} "y")'))
            if index < 10 or index >= 20:
                facts.append(f'(c c{index} "yes")')
            for fact in facts:
                runtime.process_metta_string(f"!(add-atom &parent_test {fact})")
        result = runtime.process_metta_string(
            '!(parent-minimum-gain &parent_test '
            '(, (a $x "x") (b $x "y") (engagement $x "click")) '
            '(engagement $x "click") 0.5)'
        )
        useful = runtime.process_metta_string(
            '!(parent-minimum-gain &parent_test '
            '(, (a $x "x") (c $x "yes") (engagement $x "click")) '
            '(engagement $x "click") 0.5)'
        )
        redundant_bound = runtime.process_metta_string(
            '!(parent-minimum-penalty-bound &parent_test '
            '(, (a $x "x") (b $x "y") (engagement $x "click")) '
            '(engagement $x "click"))'
        )
        useful_bound = runtime.process_metta_string(
            '!(parent-minimum-penalty-bound &parent_test '
            '(, (a $x "x") (c $x "yes") (engagement $x "click")) '
            '(engagement $x "click"))'
        )
        original = runtime.process_metta_string(
            '!(target-aware-formatter (, (a $x "x") (engagement $x "click")) '
            '&parent_test 1 "click" 800.0 0.5)'
        )
        cached = runtime.process_metta_string(
            '!(target-aware-formatter-cached (, (a $x "x") (engagement $x "click")) '
            '&parent_test 1 "click" 800.0 0.5 40 10)'
        )
        runtime.process_metta_string("!(bind! &ranking_test (new-space))")
        for fact in (
            "(CandidateCase p1)", "(CandidateCase p2)",
            "(RuleFired p1 a)", "(RuleFired p1 b)",
            "(RuleMeta a 1 0.1 0.7 0.5 True (topic))",
            "(RuleMeta b 2 0.2 0.9 0.5 False (topic affinity))",
        ):
            runtime.process_metta_string(f"!(add-atom &ranking_test {fact})")
        unfiltered = runtime.process_metta_string(
            "!(predicate-rank-scores &ranking_test False none 0.1 shrink)"
        )
        parent_only = runtime.process_metta_string(
            "!(predicate-rank-scores &ranking_test True none 0.1 shrink)"
        )
        excluded = runtime.process_metta_string(
            "!(predicate-rank-scores &ranking_test True topic 0.1 raw)"
        )
        runtime.process_metta_string("!(bind! &adaptive_test (new-space))")
        for fact in (
            "(CandidateCase p1)", "(RuleFired p1 parent)", "(RuleFired p1 child)",
            "(RuleMeta parent 1 0.1 0.7 0.5 1000000000.0 (topic))",
            "(RuleMeta child 2 0.2 0.9 0.5 0.1 (topic affinity))",
        ):
            runtime.process_metta_string(f"!(add-atom &adaptive_test {fact})")
        loose = runtime.process_metta_string(
            "!(adaptive-parent-scores &adaptive_test 0.05 0.1 shrink)"
        )
        strict = runtime.process_metta_string(
            "!(adaptive-parent-scores &adaptive_test 0.5 0.1 shrink)"
        )
        connection.send(("ok", result, useful, unfiltered, parent_only, excluded,
                         redundant_bound, useful_bound, loose, strict, original, cached))
    except BaseException as error:
        connection.send(("error", str(error)))
    finally:
        connection.close()


@unittest.skipIf(PeTTa is None, "PeTTa is unavailable")
class ParentAwareSelectionTest(unittest.TestCase):
    def test_identical_parent_coverage_is_rejected(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(child,))
        process.start()
        self.assertTrue(parent.poll(30), "PeTTa worker timed out")
        payload = parent.recv()
        process.join(5)
        self.assertEqual(payload[0], "ok", payload)
        self.assertEqual(float(str(payload[1][0])), -1.0)
        self.assertGreater(float(str(payload[2][0])), 0.0)
        self.assertIn("(PredicateScore p1 0.5)", str(payload[3]))
        self.assertIn("(PredicateScore p1 0.4)", str(payload[4]))
        self.assertIn("(PredicateScore p1 0.1)", str(payload[5]))
        self.assertEqual(float(str(payload[6][0])), -1.0)
        self.assertGreater(float(str(payload[7][0])), 0.5)
        self.assertIn("(PredicateScore p1 0.5)", str(payload[8]))
        self.assertIn("(PredicateScore p1 0.4)", str(payload[9]))
        original = re.sub(r"\$_\d+", "$x", str(payload[10]))
        cached = re.sub(r"\$_\d+", "$x", str(payload[11]))
        self.assertEqual(original, cached)


if __name__ == "__main__":
    unittest.main()
