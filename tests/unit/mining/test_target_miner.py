from __future__ import annotations

import itertools
import random
import unittest

from recommendation.mining.target_miner import (
    Atom,
    TargetAwareConjunctionMiner,
    TargetMinerConfig,
    mine_target_patterns,
)


def pattern_key(pattern):
    return tuple((atom.predicate, type(atom.value).__name__, atom.value)
                 for atom in pattern.premises)


class TargetAwareConjunctionMinerTest(unittest.TestCase):
    def test_weighted_contingency_wracc_and_antecedent_only_output(self):
        rows = [
            {"color": "red", "shape": "round"},
            {"color": "red", "shape": "square"},
            {"color": "blue", "shape": "round"},
            {"color": "blue", "shape": "square"},
        ]
        result = mine_target_patterns(
            rows,
            [1, 0, 1, 0],
            weights=[2.0, 1.0, 3.0, 4.0],
            config=TargetMinerConfig(
                min_support=1.0,
                max_depth=2,
                top_k=None,
                objective="absolute",
                exhaustive=True,
            ),
        )
        by_key = {pattern_key(pattern): pattern for pattern in result.patterns}
        red = by_key[(("color", "str", "red"),)]
        self.assertEqual(red.weighted.as_dict(), {
            "tp": 2.0, "fp": 1.0, "fn": 3.0, "tn": 4.0,
        })
        self.assertEqual(red.counts.as_dict(), {
            "tp": 1, "fp": 1, "fn": 1, "tn": 1,
        })
        self.assertAlmostEqual(red.coverage, 0.3)
        self.assertAlmostEqual(red.precision, 2.0 / 3.0)
        self.assertAlmostEqual(red.base_rate, 0.5)
        self.assertAlmostEqual(red.wracc, 0.05)
        self.assertAlmostEqual(red.positive_wracc_upper_bound, 0.1)
        self.assertAlmostEqual(red.negative_wracc_upper_bound, 0.05)

        red_round = by_key[(
            ("color", "str", "red"),
            ("shape", "str", "round"),
        )]
        self.assertAlmostEqual(red_round.wracc, 0.1)
        self.assertTrue(all(
            atom.predicate != "target"
            and atom.predicate != "engagement"
            for pattern in result.patterns for atom in pattern.premises
        ))

    def test_exhaustive_mode_matches_manual_canonical_enumeration(self):
        rows = [
            {"a": "x", "b": "m", "c": "u"},
            {"a": "x", "b": "n", "c": "v"},
            {"a": "y", "b": "m", "c": "v"},
            {"a": "y", "b": "n", "c": "u"},
        ]
        targets = [1, 0, 1, 0]
        result = mine_target_patterns(
            rows,
            targets,
            config=TargetMinerConfig(
                min_support=1,
                max_depth=3,
                top_k=None,
                objective="absolute",
                exhaustive=True,
            ),
        )

        values = {
            predicate: sorted({row[predicate] for row in rows})
            for predicate in ("a", "b", "c")
        }
        expected = {}
        for depth in range(1, 4):
            for predicates in itertools.combinations(sorted(values), depth):
                for selected in itertools.product(*(values[p] for p in predicates)):
                    premises = tuple(zip(predicates, selected))
                    covered = [
                        index for index, row in enumerate(rows)
                        if all(row[predicate] == value
                               for predicate, value in premises)
                    ]
                    if len(covered) < 1:
                        continue
                    tp = sum(targets[index] for index in covered)
                    fp = len(covered) - tp
                    # Base rate is 1/2 and all weights are one.
                    wracc = tp / len(rows) - len(covered) / len(rows) * 0.5
                    key = tuple((predicate, "str", value)
                                for predicate, value in premises)
                    expected[key] = (tp, fp, wracc)

        actual = {
            pattern_key(pattern): (
                pattern.counts.tp, pattern.counts.fp, pattern.wracc
            )
            for pattern in result.patterns
        }
        self.assertEqual(set(actual), set(expected))
        for key, (tp, fp, wracc) in expected.items():
            self.assertEqual(actual[key][:2], (tp, fp))
            self.assertAlmostEqual(actual[key][2], wracc)

    def test_bound_pruning_matches_exhaustive_for_all_objectives(self):
        targets = [1, 1, 1, 1, 0, 0, 0, 0]
        rows = []
        for index, target in enumerate(targets):
            rows.append({
                # Sorted first and perfectly discriminative in both directions.
                "a_signal": "positive" if target else "negative",
                "b_sparse": "one" if index in {0, 4} else "other",
                "c_noise": f"bucket_{index % 4}",
                "d_noise": "left" if index % 2 else "right",
            })

        for objective in ("positive", "negative", "absolute"):
            common = dict(
                min_support=1,
                max_depth=3,
                top_k=1,
                objective=objective,
            )
            bounded = mine_target_patterns(
                rows, targets,
                config=TargetMinerConfig(**common, exhaustive=False),
            )
            exhaustive = mine_target_patterns(
                rows, targets,
                config=TargetMinerConfig(**common, exhaustive=True),
            )
            self.assertEqual(
                [pattern.as_dict() for pattern in bounded.patterns],
                [pattern.as_dict() for pattern in exhaustive.patterns],
                objective,
            )
            self.assertGreater(bounded.audit.bound_checks, 0)
            self.assertGreater(bounded.audit.bound_pruned, 0)
            self.assertEqual(exhaustive.audit.bound_checks, 0)

        positive = mine_target_patterns(
            rows, targets,
            config=TargetMinerConfig(
                min_support=1, max_depth=2, top_k=1,
                objective="positive", exhaustive=False,
            ),
        ).patterns[0]
        negative = mine_target_patterns(
            rows, targets,
            config=TargetMinerConfig(
                min_support=1, max_depth=2, top_k=1,
                objective="negative", exhaustive=False,
            ),
        ).patterns[0]
        self.assertGreater(positive.wracc, 0)
        self.assertLess(negative.wracc, 0)

    def test_random_small_workspaces_match_exhaustive_oracle(self):
        local = random.Random(20260901)
        for trial in range(20):
            row_count = local.randint(6, 12)
            rows = [
                {
                    f"p{predicate}": f"v{local.randrange(3)}"
                    for predicate in range(4)
                }
                for _ in range(row_count)
            ]
            targets = [local.randrange(2) for _ in range(row_count)]
            weights = [local.choice((0.25, 0.5, 1.0, 2.0))
                       for _ in range(row_count)]
            for objective in ("positive", "negative", "absolute"):
                common = dict(
                    min_support=0.5,
                    max_depth=3,
                    top_k=5,
                    objective=objective,
                )
                bounded = mine_target_patterns(
                    rows, targets, weights=weights,
                    config=TargetMinerConfig(**common, exhaustive=False),
                )
                exhaustive = mine_target_patterns(
                    rows, targets, weights=weights,
                    config=TargetMinerConfig(**common, exhaustive=True),
                )
                self.assertEqual(
                    [pattern.as_dict() for pattern in bounded.patterns],
                    [pattern.as_dict() for pattern in exhaustive.patterns],
                    (trial, objective),
                )

    def test_one_value_per_predicate_min_support_depth_and_top_k(self):
        rows = [
            {"topic": "a", "format": "short", "region": "x"},
            {"topic": "a", "format": "long", "region": "x"},
            {"topic": "b", "format": "short", "region": "y"},
            {"topic": "b", "format": "long", "region": "y"},
            {"topic": "rare", "format": "short", "region": "z"},
        ]
        result = mine_target_patterns(
            rows,
            [1, 1, 0, 0, 1],
            config=TargetMinerConfig(
                min_support=2,
                max_depth=2,
                top_k=4,
                objective="absolute",
                exhaustive=True,
            ),
        )
        self.assertLessEqual(len(result.patterns), 4)
        self.assertTrue(all(pattern.depth <= 2 for pattern in result.patterns))
        self.assertTrue(all(pattern.weighted.support >= 2 for pattern in result.patterns))
        self.assertTrue(all(
            len({atom.predicate for atom in pattern.premises}) == pattern.depth
            for pattern in result.patterns
        ))
        self.assertNotIn(Atom("topic", "rare"), result.frequent_atoms)
        self.assertEqual(result.audit.atom_support_pruned, 2)  # topic=rare, region=z

    def test_fold_stability_stats_use_weighted_fold_baselines(self):
        rows = [
            {"signal": "hot"}, {"signal": "cold"},
            {"signal": "hot"}, {"signal": "cold"},
            {"signal": "hot"}, {"signal": "cold"},
        ]
        targets = [1, 0, 1, 0, 0, 1]
        weights = [2, 2, 1, 1, 3, 1]
        folds = ["f1", "f1", "f2", "f2", "f3", "f3"]
        result = mine_target_patterns(
            rows,
            targets,
            weights=weights,
            folds=folds,
            config=TargetMinerConfig(
                min_support=1,
                max_depth=1,
                top_k=None,
                objective="absolute",
                exhaustive=True,
                fold_min_support=1,
            ),
        )
        hot = next(
            pattern for pattern in result.patterns
            if pattern.premises == (Atom("signal", "hot"),)
        )
        self.assertEqual(
            [fold.fold for fold in hot.fold_statistics], ["f1", "f2", "f3"]
        )
        effects = [fold.effect for fold in hot.fold_statistics]
        self.assertAlmostEqual(effects[0], 0.5)
        self.assertAlmostEqual(effects[1], 0.5)
        self.assertAlmostEqual(effects[2], -0.25)
        self.assertAlmostEqual(hot.stable_positive_effect, -0.25)
        self.assertAlmostEqual(hot.stable_negative_effect, -0.5)
        self.assertTrue(all(fold.usable for fold in hot.fold_statistics))

    def test_output_is_deterministic_across_row_and_mapping_order(self):
        rows = [
            {"topic": "a", "format": "short", "known": True},
            {"topic": "b", "format": "long", "known": False},
            {"topic": "a", "format": "long", "known": True},
            {"topic": "b", "format": "short", "known": False},
        ]
        targets = [1, 0, 1, 0]
        weights = [1.0, 2.0, 3.0, 4.0]
        folds = [2, 1, 2, 1]
        config = TargetMinerConfig(
            min_support=1,
            max_depth=3,
            top_k=8,
            objective="absolute",
            exhaustive=True,
        )
        first = TargetAwareConjunctionMiner(config).mine(
            rows, targets, weights=weights, folds=folds
        )
        reordered_rows = [dict(reversed(list(row.items()))) for row in reversed(rows)]
        second = TargetAwareConjunctionMiner(config).mine(
            reordered_rows,
            reversed(targets),
            weights=reversed(weights),
            folds=reversed(folds),
        )
        self.assertEqual(
            [pattern.as_dict() for pattern in first.patterns],
            [pattern.as_dict() for pattern in second.patterns],
        )
        self.assertEqual(
            [atom.as_dict() for atom in first.frequent_atoms],
            [atom.as_dict() for atom in second.frequent_atoms],
        )
        # Search topology and cache hit counts are deterministic too.
        self.assertEqual(first.audit.as_dict(), second.audit.as_dict())

    def test_weighted_statistics_are_stable_under_ill_conditioned_row_order(self):
        rows = [
            {"signal": "hot", "bucket": "all"},
            {"signal": "hot", "bucket": "all"},
            {"signal": "hot", "bucket": "all"},
            {"signal": "cold", "bucket": "all"},
        ]
        targets = [1, 1, 1, 0]
        weights = [1e16, 1.0, 1.0, 3.0]
        config = TargetMinerConfig(
            min_support=1,
            max_depth=2,
            top_k=None,
            objective="absolute",
            exhaustive=False,
        )
        first = mine_target_patterns(rows, targets, weights=weights, config=config)
        second = mine_target_patterns(
            reversed(rows),
            reversed(targets),
            weights=reversed(weights),
            config=config,
        )
        self.assertEqual(
            [pattern.as_dict() for pattern in first.patterns],
            [pattern.as_dict() for pattern in second.patterns],
        )
        self.assertEqual(first.audit.as_dict(), second.audit.as_dict())

    def test_validation_and_type_sensitive_values(self):
        result = mine_target_patterns(
            [{"value": True}, {"value": 1}, {"value": 1.0}],
            [1, 0, 1],
            config=TargetMinerConfig(
                min_support=1, max_depth=1, top_k=None,
                exhaustive=True,
            ),
        )
        self.assertEqual(len(result.frequent_atoms), 3)
        self.assertIn(Atom("value", True), result.frequent_atoms)
        self.assertIn(Atom("value", 1), result.frequent_atoms)
        self.assertIn(Atom("value", 1.0), result.frequent_atoms)

        with self.assertRaisesRegex(ValueError, "targets length"):
            mine_target_patterns([{"x": "a"}], [])
        with self.assertRaisesRegex(ValueError, "positive weight"):
            mine_target_patterns([{"x": "a"}], [1], weights=[0])
        with self.assertRaisesRegex(ValueError, "NaN"):
            mine_target_patterns([{"x": float("nan")}], [1])
        with self.assertRaisesRegex(ValueError, "max_depth"):
            TargetMinerConfig(max_depth=0)


if __name__ == "__main__":
    unittest.main()
