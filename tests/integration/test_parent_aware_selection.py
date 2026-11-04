"""Parent-conditioned selection rejects redundant conjunctions."""
import multiprocessing as mp
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
        runtime.load_metta_file("recommendation/miner/parentAwareSelection.metta")
        runtime.load_metta_file("recommendation/miner/predicateRankingPolicy.metta")
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
        connection.send(("ok", result, useful, unfiltered, parent_only, excluded))
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


if __name__ == "__main__":
    unittest.main()
