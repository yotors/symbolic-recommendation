"""A shared decision target must not invent evidence in another proof family."""

import re
import unittest

from recommendation.app.server import LAB, IsolatedPeTTaChainer, proof_tv


PREDICATES = (
    "Pair_Text_Semantic_Attention_T8",
    "Pair_Recent_Subcategory_Transition",
    "Pair_Subcategory_Affinity",
    "Pair_Long_Affinity",
    "Pair_Title_Overlap",
    "Pair_Entity_Recent_Top1_Similarity",
)


def _sources():
    """Six independent rule families, without dependency on a saved dataset."""
    result = []
    for index, predicate in enumerate(PREDICATES, 1):
        dependency = f'"pair_mined_cluster_{index}"'
        # Unequal strengths make accidental copying between roots observable.
        strength = 0.56 + 0.02 * index
        result.extend((
            f'(: pair_mined_cluster_{index}_v1 '
            f'(Implication ({predicate} $pair "left") '
            f'(MinedPairPreference $pair {dependency})) '
            f'(CTV (STV {strength} 0.9) (STV {1.0 - strength} 0.9)))',
            f'(: pair_decision_rule_{index} '
            f'(Implication (MinedPairPreference $pair {dependency}) '
            f'(PairSignal $pair {dependency})) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
            f'(: (no_inverse pair_merge_rule_{index}) '
            f'(Implication (PairSignal $pair {dependency}) (PairWin $pair)) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
        ))
    return result


def _facts(case, *, text_matches):
    return [
        f'(: fact_{case}_{index} ({predicate} {case} '
        f'"{"left" if index != 1 or text_matches else "equal"}") (STV 1.0 1.0))'
        for index, predicate in enumerate(PREDICATES, 1)
    ]


def _roots(case):
    return [f'(: $proof (PairSignal {case} "pair_mined_cluster_{index}") $tv)'
            for index in range(1, 7)]


class ProofIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = IsolatedPeTTaChainer()
        cls.addClassCleanup(cls.engine.close)
        # Exercise the application's ordinary worker/compiler boundary. The
        # wrapper is a supported proof term, not a special host-side filter.
        cls.engine.replace(_sources())

    def query(self, roots):
        return self.engine.query_many(roots, steps=1000, timeout_sec=0)

    def assert_same_truth_values(self, before, after):
        self.assertEqual(len(before), len(after))
        for expected, actual in zip(before, after):
            self.assertEqual(bool(expected), bool(actual))
            if expected:
                self.assertEqual(len(expected), len(actual))
                for old, new in zip(sorted(expected), sorted(actual)):
                    for old_value, new_value in zip(proof_tv(old), proof_tv(new)):
                        self.assertAlmostEqual(old_value, new_value, places=8)

    def test_cold_text_stays_unproved_after_warm_evidence_and_shared_batches(self):
        self.engine.add_atoms_no_check(_facts("cold", text_matches=False))
        cold_before = self.query(_roots("cold"))
        self.assertEqual(cold_before[0], [])
        self.assertTrue(all(cold_before[1:]))

        # A real warm text witness supplies the base-rate evidence that made
        # the original inverse adapter invent a cold text signal. Testing only
        # a cold context never exposed that bug.
        self.engine.add_atoms_no_check(_facts("warm", text_matches=True))
        warm_before = self.query(_roots("warm"))
        self.assertTrue(all(warm_before))
        roots = _roots("warm") + _roots("cold")
        together = self.query(roots)
        self.assertEqual(together[6], [])
        self.assert_same_truth_values(warm_before, together[:6])
        self.assert_same_truth_values(cold_before, together[6:])
        self.assert_same_truth_values(cold_before, self.query(_roots("cold")))

        # Query ordering must not change which family can supply evidence.
        reversed_results = self.query(list(reversed(roots)))
        self.assert_same_truth_values(together, list(reversed(reversed_results)))
        self.assertNotIn("inverted pair_merge_rule", repr(together))

        # Disabling inversion preserves the adapter's intended forward role,
        # including the posterior mode that queries the common PairWin target.
        decisions = self.query([
            "(: $proof (PairWin cold) $tv)",
            "(: $proof (PairWin warm) $tv)",
        ])
        self.assertTrue(all(decisions))
        for proofs in decisions:
            self.assertTrue(any("no_inverse pair_merge_rule" in proof for proof in proofs))
            for proof in proofs:
                strength, confidence = proof_tv(proof)
                self.assertGreater(strength, 0.5)
                self.assertGreater(confidence, 0.0)
        # Materializing/searching the decision cannot create a text witness.
        self.assertEqual(self.query(_roots("cold"))[0], [])

    def test_every_generated_lab_merge_adapter_explicitly_disables_inversion(self):
        self.assertIsNotNone(LAB)
        bridges = [source for source in LAB._pair_rule_sources if "(PairWin $pair)" in source]
        self.assertTrue(bridges, "Fixture must exercise actual Lab rule generation")
        clusters = {rule.get("dependency_id", rule["id"]) for rule in LAB.pair_rules}
        self.assertEqual(len(bridges), len(clusters))
        for source in bridges:
            self.assertRegex(source, r'^\(: \(no_inverse pair_merge_rule_\d+\) ')
            self.assertEqual(len(re.findall(r'\(no_inverse pair_merge_rule_\d+\)', source)), 1)
            self.assertIn("(Implication (PairSignal $pair", source)


if __name__ == "__main__":
    unittest.main()
