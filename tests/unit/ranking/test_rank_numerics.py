"""Proof-family ranks must not magnify floating-point addition-order noise."""

import unittest

from recommendation.app.server import Lab


class RankNumericsTest(unittest.TestCase):
    def test_equivalent_sums_share_a_midrank(self):
        self.assertEqual(Lab._midrank_scores([0.3,0.1+0.2,0.0]),[0.75,0.75,0.0])

    def test_observable_differences_are_preserved(self):
        self.assertEqual(Lab._midrank_scores([0.3,0.300001,0.0]),[0.5,1.0,0.0])

    def test_permutations_preserve_the_same_candidate_ranks(self):
        values=[0.3,0.1+0.2,-0.2,0.7]
        expected=Lab._midrank_scores(values)
        order=[2,0,3,1]
        actual=Lab._midrank_scores([values[index] for index in order])
        self.assertEqual(actual,[expected[index] for index in order])


if __name__=='__main__':
    unittest.main()
