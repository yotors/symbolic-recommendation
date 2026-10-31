from __future__ import annotations

import multiprocessing as mp
import re
import traceback
import unittest


try:
    from petta import PeTTa
except (ImportError, OSError):
    PeTTa = None


SCHEMAS = (
    "((Viewer exposure user join) (Candidate exposure article join) "
    "(InterestedIn user topic join) (HasTopic article topic join) "
    "(Premium exposure level value))"
)
CONNECTED_PATTERN = re.compile(
    r"\(Candidate\s+(\$[^\s()]+)\s+(\$[^\s()]+)\)\s+"
    r"\(HasTopic\s+\2\s+(\$[^\s()]+)\)\s+"
    r"\(InterestedIn\s+(\$[^\s()]+)\s+\3\)\s+"
    r"\(Viewer\s+\1\s+\4\)\s+"
    r"\(engagement\s+\1\s+\"click\"\)"
)
FACTS = (
    "(Viewer e1 u1)", "(Candidate e1 a1)", "(Premium e1 yes)",
    '(engagement e1 "click")',
    "(Viewer e2 u1)", "(Candidate e2 a2)", "(Premium e2 yes)",
    '(engagement e2 "click")',
    "(Viewer e3 u2)", "(Candidate e3 a3)", "(Premium e3 no)",
    '(engagement e3 "skip")',
    "(Viewer e4 u2)", "(Candidate e4 a4)", "(Premium e4 no)",
    '(engagement e4 "skip")',
    # Each clicked exposure has two matching topic paths. Support must still
    # be two distinct exposures, not four graph bindings.
    "(InterestedIn u1 crypto)", "(InterestedIn u1 ai)",
    "(InterestedIn u2 sports)",
    "(HasTopic a1 crypto)", "(HasTopic a1 ai)",
    "(HasTopic a2 crypto)", "(HasTopic a2 ai)",
    "(HasTopic a3 finance)", "(HasTopic a4 finance)",
)


def _mine_worker(
    connection, length_cost: float, max_patterns: int, depth_range: bool,
) -> None:
    try:
        from petta import PeTTa as WorkerPeTTa

        petta = WorkerPeTTa()
        petta.load_metta_file("recommendation/miner/helpers.metta")
        petta.load_metta_file("recommendation/miner/fpMiner.metta")
        petta.process_metta_string("!(bind! &connected_test (new-space))")
        for fact in FACTS:
            petta.process_metta_string(
                f"!(add-atom &connected_test {fact})"
            )
        entry = (
            "connected-frequency-pattern-miner-range &connected_test 2 2 5"
            if depth_range
            else "connected-frequency-pattern-miner &connected_test 2 5"
        )
        result = petta.process_metta_string(
            f'!({entry} "click" 2.0 {length_cost} {SCHEMAS} '
            f'{"() " if depth_range else ""}{max_patterns})'
        )
        connection.send(("ok", result))
    except BaseException as exc:
        connection.send(
            ("error", type(exc).__name__, str(exc), traceback.format_exc())
        )
    finally:
        connection.close()


@unittest.skipIf(PeTTa is None, "PeTTa Python runtime is unavailable")
class ConnectedTargetMinerIntegrationTest(unittest.TestCase):
    def mine(
        self, *, length_cost: float, max_patterns: int = 10_000,
        depth_range: bool = False,
    ):
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_mine_worker,
            args=(child, length_cost, max_patterns, depth_range),
            daemon=False,
        )
        process.start()
        child.close()
        try:
            if not parent.poll(15):
                process.terminate()
                self.fail("connected PeTTa miner timed out")
            payload = parent.recv()
        finally:
            parent.close()
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join()
        if payload[0] != "ok":
            self.fail("\n".join(payload[1:]))
        return payload[1]

    def test_discovers_typed_multi_variable_rule_with_distinct_support(self):
        raw = " ".join(map(str, self.mine(length_cost=0.0)))
        self.assertIsNotNone(CONNECTED_PATTERN.search(raw), raw)
        self.assertRegex(raw, r"\(STV 1\.0 0\.5\)")
        for match in CONNECTED_PATTERN.finditer(raw):
            tail = raw[match.end():]
            self.assertRegex(tail, r"\)\s+2\)")

    def test_mdl_cost_is_configurable_and_can_reject_overlong_rules(self):
        accepted = " ".join(map(str, self.mine(length_cost=0.1)))
        rejected = " ".join(map(str, self.mine(length_cost=0.3)))
        self.assertIsNotNone(CONNECTED_PATTERN.search(accepted), accepted)
        self.assertNotIn("supportOf", rejected)

    def test_search_bound_fails_explicitly(self):
        raw = " ".join(map(str, self.mine(length_cost=0.0, max_patterns=1)))
        self.assertRegex(raw, r"MiningLimit (candidates|patterns)")

    def test_range_mdl_prefers_equally_predictive_short_rule(self):
        raw = " ".join(map(str, self.mine(length_cost=0.3, depth_range=True)))
        self.assertRegex(raw, r'\(Premium\s+\$[^\s()]+\s+yes\)')
        self.assertIsNone(CONNECTED_PATTERN.search(raw), raw)


if __name__ == "__main__":
    unittest.main()
