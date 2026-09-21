from __future__ import annotations

import re
import unittest
import uuid

from recommendation.mining.incremental_fpminer import (
    IncrementalFpMinerCache,
    parse_fpminer_supports,
)


try:
    from petta import PeTTa
except (ImportError, OSError):  # The ordinary Python unit environment has no PeTTa.
    PeTTa = None


STV_RE = re.compile(r"\(STV\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\)")


def closed_case(case_id: str, topic: str, fmt: str, outcome: str):
    return (
        f'(topic {case_id} "{topic}")',
        f'(format {case_id} "{fmt}")',
        f'(engagement {case_id} "{outcome}")',
    )


@unittest.skipIf(PeTTa is None, "PeTTa Python runtime is unavailable")
class IncrementalFpMinerPeTTaIntegrationTest(unittest.TestCase):
    """Compare append-incremental support with a fresh full MeTTa query."""

    def setUp(self):
        self.petta = PeTTa()
        self.petta.load_metta_file("recommendation/miner/helpers.metta")
        self.petta.load_metta_file("recommendation/miner/fpMiner.metta")
        self.space = f"&incremental_test_{uuid.uuid4().hex}"
        self.petta.process_metta_string(f"!(bind! {self.space} (new-space))")
        self.cache = IncrementalFpMinerCache(
            self.petta, namespace="integration", batch_size=4
        )

    def tearDown(self):
        self.cache.prune()
        self.petta.process_metta_string(
            f"!(let $atom (superpose (collapse (match {self.space} $x $x))) "
            f"(remove-atom {self.space} $atom))"
        )

    def replace_full_space(self, cases):
        self.petta.process_metta_string(
            f"!(let $atom (superpose (collapse (match {self.space} $x $x))) "
            f"(remove-atom {self.space} $atom))"
        )
        for case_id in sorted(cases):
            for fact in cases[case_id]:
                self.petta.process_metta_string(
                    f"!(add-atom {self.space} {fact})"
                )

    def append_full_space(self, case_facts):
        for fact in case_facts:
            self.petta.process_metta_string(f"!(add-atom {self.space} {fact})")

    def full_reference(self, min_support):
        return self.petta.process_metta_string(
            f'!(frequency-pattern-miner {self.space} {min_support} '
            '3 "click" 2.0)'
        )

    @staticmethod
    def parsed(raw):
        return parse_fpminer_supports(
            raw, features=("topic", "format"),
            expected_target="click", expected_depth=3,
        )

    def test_delta_output_equals_full_query_across_append_and_rebuild(self):
        cases = {
            "c1": closed_case("c1", "news", "long", "click"),
            "c2": closed_case("c2", "news", "long", "skip"),
            "c3": closed_case("c3", "sports", "short", "skip"),
        }
        self.replace_full_space(cases)
        seeded = self.cache.mine(
            plan="pair/topic-format/3", full_space=self.space, cases=cases,
            features=("topic", "format"), depth=3,
            min_support=2, evidence_k=2.0,
        )
        self.assertEqual(seeded.audit.mode, "full_seed")
        self.assertEqual(seeded.output, ())

        # A pattern retained internally at support one crosses the public
        # threshold using only a one-case MeTTa delta query.
        cases["c4"] = closed_case("c4", "news", "long", "click")
        self.append_full_space(cases["c4"])
        crossed = self.cache.mine(
            plan="pair/topic-format/3", full_space=self.space, cases=cases,
            features=("topic", "format"), depth=3,
            min_support=2, evidence_k=2.0,
        )
        self.assertEqual(crossed.audit.mode, "delta_updated")
        self.assertEqual(crossed.audit.researched_cases, 1)
        self.assertEqual(self.parsed(crossed.output), self.parsed(self.full_reference(2)))
        positive, negative = [
            tuple(map(float, values)) for values in STV_RE.findall(crossed.output[0])
        ]
        self.assertAlmostEqual(positive[0], 2.0 / 3.0)
        self.assertAlmostEqual(positive[1], 3.0 / 5.0)
        self.assertAlmostEqual(negative[0], 0.0)
        self.assertAlmostEqual(negative[1], 1.0 / 3.0)

        # A skip changes nA and therefore the rule CTV, even though its target
        # support stays two. This is why both delta outcomes are mined.
        cases["c5"] = closed_case("c5", "news", "long", "skip")
        self.append_full_space(cases["c5"])
        skipped = self.cache.mine(
            plan="pair/topic-format/3", full_space=self.space, cases=cases,
            features=("topic", "format"), depth=3,
            min_support=2, evidence_k=2.0,
        )
        self.assertEqual(self.parsed(skipped.output), self.parsed(self.full_reference(2)))
        positive, negative = [
            tuple(map(float, values)) for values in STV_RE.findall(skipped.output[0])
        ]
        self.assertAlmostEqual(positive[0], 0.5)
        self.assertAlmostEqual(positive[1], 4.0 / 6.0)
        self.assertAlmostEqual(negative[0], 0.0)
        self.assertAlmostEqual(negative[1], 1.0 / 3.0)

        # Mutating a closed case is not additive. The cache must discard its
        # sufficient statistics and derive them again from the rebuilt space.
        cases["c5"] = closed_case("c5", "culture", "short", "skip")
        self.replace_full_space(cases)
        rebuilt = self.cache.mine(
            plan="pair/topic-format/3", full_space=self.space, cases=cases,
            features=("topic", "format"), depth=3,
            min_support=2, evidence_k=2.0,
        )
        self.assertEqual(rebuilt.audit.mode, "rebuilt_full")
        self.assertEqual(rebuilt.audit.changed_cases, 1)
        self.assertEqual(self.parsed(rebuilt.output), self.parsed(self.full_reference(2)))

    def test_target_metrics_and_auc_gate_are_computed_by_petta(self):
        cases = {
            "c1": closed_case("c1", "news", "long", "click"),
            "c2": closed_case("c2", "news", "long", "click"),
            "c3": closed_case("c3", "news", "short", "skip"),
            "c4": closed_case("c4", "sports", "short", "skip"),
        }
        self.replace_full_space(cases)
        raw = self.petta.process_metta_string(
            f'!(target-aware-frequency-pattern-miner {self.space} '
            '1 2 "click" 2.0 0.3)'
        )
        text = " ".join(map(str, raw))
        self.assertIn('(format ', text)
        self.assertIn('(AUC 1.0)', text)
        self.assertIn('(Contingency 2 0 0 2)', text)
        self.assertIn('(Information-Gain 1.0)', text)
        # Topic has AUC 0.75 (gain 0.25), so the symbolic gate excludes it.
        self.assertNotIn('(topic ', text)

        # Here neither unary parent predicts the target, but their conjunction
        # does. The conditional miner must calculate and gate that incremental
        # precision inside PeTTa.
        conditional_cases = {
            "c1": closed_case("c1", "news", "long", "click"),
            "c2": closed_case("c2", "sports", "long", "skip"),
            "c3": closed_case("c3", "news", "short", "skip"),
            "c4": closed_case("c4", "sports", "short", "click"),
        }
        self.replace_full_space(conditional_cases)
        conditional = self.petta.process_metta_string(
            f'!(conditional-target-aware-frequency-pattern-miner {self.space} '
            '1 (format) (topic) "click" 2.0 0.0 0.0)'
        )
        conditional_text = " ".join(map(str, conditional))
        self.assertIn('(format ', conditional_text)
        self.assertIn('(topic ', conditional_text)
        self.assertIn('(AUC 0.75)', conditional_text)
        self.assertIn('(Incremental-Precision ', conditional_text)
        self.assertIn('(MDL-Gain ', conditional_text)
        self.assertIn(
            '(Hierarchical-Precision 0.6666666666666666)',
            conditional_text,
        )
        rejected = self.petta.process_metta_string(
            f'!(conditional-target-aware-frequency-pattern-miner {self.space} '
            '1 (format) (topic) "click" 2.0 0.0 0.5)'
        )
        self.assertNotIn('(targetScoreOf ', " ".join(map(str, rejected)))


if __name__ == "__main__":
    unittest.main()
