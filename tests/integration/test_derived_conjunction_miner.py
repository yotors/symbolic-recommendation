"""Integration coverage for derived conjunction mining and PeTTa backoff."""
import multiprocessing as mp
import unittest

try:
    from petta import PeTTa
except (ImportError, OSError):
    PeTTa = None


FACTS = (
    '(a c1 "x")', '(b c1 "y")', '(engagement c1 "click")',
    '(a c2 "x")', '(b c2 "y")', '(engagement c2 "click")',
    '(a c3 "x")', '(b c3 "z")', '(engagement c3 "skip")',
    '(a c4 "q")', '(b c4 "y")', '(engagement c4 "skip")',
)


def _worker(connection):
    try:
        from petta import PeTTa as WorkerPeTTa

        runtime = WorkerPeTTa()
        runtime.load_metta_file("recommendation/miner/helpers.metta")
        runtime.load_metta_file("recommendation/miner/fpMiner.metta")
        runtime.load_metta_file(
            "recommendation/miner/derivedConjunctionPolicy.metta"
        )
        runtime.process_metta_string("!(bind! &full (new-space))")
        runtime.process_metta_string("!(bind! &positive (new-space))")
        for fact in FACTS:
            runtime.process_metta_string(f"!(add-atom &full {fact})")
        for fact in FACTS[:6]:
            runtime.process_metta_string(f"!(add-atom &positive {fact})")
        mined = runtime.process_metta_string(
            "!(connected-frequency-pattern-miner-range-prefiltered "
            '&full &positive 2 2 3 "click" 2.0 0.0 '
            "((a exposure value value) (b exposure value value)) () 100)"
        )
        runtime.process_metta_string("!(bind! &policy (new-space))")
        for fact in (
            "(CandidateCase p1)", "(CandidateCase p2)",
            "(RuleEvidence p1 r1 1 0.2 0.6)",
            "(RuleEvidence p1 r2 2 0.1 0.7)",
            "(RuleEvidence p1 r3 2 0.3 0.8)",
        ):
            runtime.process_metta_string(f"!(add-atom &policy {fact})")
        scores = runtime.process_metta_string(
            "!(derived-case-scores &policy 2 0.04)"
        )
        connection.send(("ok", mined, scores))
    except BaseException as error:
        connection.send(("error", type(error).__name__, str(error)))
    finally:
        connection.close()


@unittest.skipIf(PeTTa is None, "PeTTa Python runtime is unavailable")
class DerivedConjunctionMinerIntegrationTest(unittest.TestCase):
    def test_prefiltered_apriori_and_symbolic_backoff(self):
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_worker, args=(child,))
        process.start()
        self.assertTrue(parent.poll(30), "PeTTa worker timed out")
        payload = parent.recv()
        process.join(5)
        self.assertEqual(payload[0], "ok", payload)
        mined = " ".join(map(str, payload[1]))
        self.assertIn('(a $', mined)
        self.assertIn('"x"', mined)
        self.assertIn('(b $', mined)
        self.assertGreaterEqual(mined.count("supportOf"), 3)
        scores = " ".join(map(str, payload[2]))
        self.assertIn("(DerivedScore p1 0.8)", scores)
        self.assertIn("(DerivedScore p2 0.04)", scores)


if __name__ == "__main__":
    unittest.main()
