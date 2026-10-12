from __future__ import annotations

import unittest
import uuid

from recommendation.mining.conditional_model import (
    ConditionalMiningConfig,
    FpMinerUnary,
)
from recommendation.mining.petta_conditional_mining import (
    mine_petta_conditional_patterns,
)


try:
    from petta import PeTTa
except (ImportError, OSError):
    PeTTa = None


@unittest.skipIf(PeTTa is None, "PeTTa Python runtime is unavailable")
class ConditionalMinerIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.petta = PeTTa()
        self.petta.load_metta_file("recommendation/miner/helpers.metta")
        self.petta.load_metta_file("recommendation/miner/fpMiner.metta")
        self.petta.load_metta_file("recommendation/miner/conditionalMiner.metta")
        self.space = f"&conditional_test_{uuid.uuid4().hex}"
        self.petta.process_metta_string(f"!(bind! {self.space} (new-space))")

    def tearDown(self):
        self.petta.process_metta_string(
            f"!(let $atom (superpose (collapse (match {self.space} $x $x))) "
            f"(remove-atom {self.space} $atom))"
        )

    def test_weighted_temporally_stable_child_is_mined(self):
        for fold in range(3):
            for suffix, semantic, context, outcome in (
                ("ax", "a", "x", "click"),
                ("ay", "a", "y", "skip"),
                ("bx", "b", "x", "skip"),
                ("by", "b", "y", "click"),
            ):
                case = f"case_{fold}_{suffix}"
                for fact in (
                    f'(semantic {case} "{semantic}")',
                    f'(context {case} "{context}")',
                    f'(engagement {case} "{outcome}")',
                    f'(case-weight {case} 1.0)',
                    f'(case-fold {case} {fold})',
                ):
                    self.petta.process_metta_string(
                        f"!(add-atom {self.space} {fact})"
                    )

        result = mine_petta_conditional_patterns(
            self.petta,
            self.space,
            row_count=12,
            fpminer_unaries=(FpMinerUnary(
                rule_id="seed_a",
                predicate="semantic",
                value="a",
                target="click",
                source="recommendation/miner/fpMiner.metta",
            ),),
            positive_target="click",
            context_predicates=("context",),
            predicate_lineages={"semantic": "semantic", "context": "context"},
            folds=(0, 1, 2),
            config=ConditionalMiningConfig(
                min_support=3.0,
                fold_min_support=1.0,
                max_depth=2,
                top_k=10,
                min_usable_folds=2,
                max_generated_candidates=100,
            ),
        )
        self.assertEqual(len(result.patterns), 1)
        pattern = result.patterns[0]
        self.assertEqual(
            tuple((atom.predicate, atom.value) for atom in pattern.premises),
            (("context", "x"), ("semantic", "a")),
        )
        self.assertEqual(pattern.fpminer_seed_rule_ids, ("seed_a",))
        self.assertEqual(pattern.weighted.as_dict(), {
            "tp": 3.0, "fp": 0.0, "fn": 3.0, "tn": 6.0,
        })
        self.assertEqual(pattern.counts.as_dict(), {
            "tp": 3, "fp": 0, "fn": 3, "tn": 6,
        })
        self.assertEqual(pattern.precision, 1.0)
        self.assertEqual(pattern.parent_precision, 0.5)
        self.assertEqual(pattern.stable_effect, 0.5)
        self.assertEqual(pattern.stable_incremental_effect, 0.5)
        self.assertEqual(pattern.robust_incremental_wracc, 0.125)
        self.assertEqual(pattern.usable_fold_count, 3)
        self.assertEqual(result.audit.executor.split()[0],
                         "recommendation/miner/conditionalMiner.metta")


if __name__ == "__main__":
    unittest.main()
