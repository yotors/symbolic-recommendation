from __future__ import annotations

import math
import unittest

from recommendation.mining.conditional_llm_mining import (
    ConditionalMiningConfig,
    ConditionalMiningLimitError,
    FpMinerUnary,
    mine_conditional_llm_patterns,
)


SEMANTIC = ("llm_concept", "llm_format", "llm_intent")
CONTEXT = ("history_band", "same_topic")
LINEAGES = {
    "llm_concept": "llm_text",
    "llm_format": "llm_text",
    "llm_intent": "llm_text",
    "history_band": "history_interest",
    "same_topic": "article_taxonomy",
}


def seed(predicate="llm_concept", value="relevant", *, target="click", rule_id="fp_1"):
    return FpMinerUnary(
        rule_id=rule_id,
        predicate=predicate,
        value=value,
        target=target,
        source="recommendation/miner/fpMiner.metta",
    )


def stable_synergy(*, weighted=False):
    rows = []
    targets = []
    weights = []
    folds = []
    # Neither unary is sufficient: each covers one positive and one negative
    # per fold. Their conjunction is the stable conditional signal.
    for fold in range(3):
        examples = (
            ("relevant", "loyal", 1),
            ("relevant", "new", 0),
            ("other", "loyal", 0),
            ("other", "new", 0),
        )
        for concept, history, target in examples:
            rows.append({
                "llm_concept": concept,
                "history_band": history,
                "same_topic": "same" if history == "loyal" else "different",
                "ignored_dataset_column": f"source-{fold}",
            })
            targets.append(target)
            weights.append(2.0 if weighted and target else 1.0)
            folds.append(fold)
    return rows, targets, weights, folds


def run(rows, targets, weights, folds, *, seeds=None, config=None):
    return mine_conditional_llm_patterns(
        rows,
        targets,
        weights=weights,
        folds=folds,
        fpminer_unaries=[seed()] if seeds is None else seeds,
        positive_target="click",
        semantic_predicates=SEMANTIC,
        context_predicates=CONTEXT,
        predicate_lineages=LINEAGES,
        config=config or ConditionalMiningConfig(
            min_support=3,
            fold_min_support=1,
            max_depth=2,
            top_k=20,
            min_usable_folds=3,
        ),
    )


class ConditionalLlmMiningTest(unittest.TestCase):
    def test_discovers_stable_cross_role_increment_over_fpminer_parent(self):
        rows, targets, weights, folds = stable_synergy()
        result = run(rows, targets, weights, folds)
        pattern = next(
            candidate for candidate in result.patterns
            if {(atom.predicate, atom.value) for atom in candidate.premises}
            == {("llm_concept", "relevant"), ("history_band", "loyal")}
        )
        self.assertEqual(pattern.fpminer_seed_rule_ids, ("fp_1",))
        self.assertEqual(pattern.counts.tp, 3)
        self.assertEqual(pattern.counts.fp, 0)
        self.assertEqual(pattern.support, 3.0)
        self.assertAlmostEqual(pattern.base_rate, 0.25)
        self.assertAlmostEqual(pattern.precision, 1.0)
        self.assertAlmostEqual(pattern.parent_precision, 0.5)
        self.assertAlmostEqual(pattern.incremental_effect, 0.5)
        self.assertAlmostEqual(pattern.stable_incremental_effect, 0.5)
        self.assertEqual(pattern.usable_fold_count, 3)
        self.assertEqual(pattern.semantic_lineages, ("llm_text",))
        self.assertEqual(pattern.context_lineages, ("history_interest",))
        self.assertEqual(
            pattern.evidence_lineages, ("history_interest", "llm_text")
        )
        self.assertEqual(len(pattern.lineage_signature), 16)
        self.assertEqual(len(pattern.variant_signature), 16)
        self.assertTrue(all(
            any(atom == seed().atom for atom in candidate.premises)
            for candidate in result.patterns
        ))

    def test_exact_weighted_statistics_and_fold_statistics(self):
        rows, targets, weights, folds = stable_synergy(weighted=True)
        result = run(rows, targets, weights, folds)
        pattern = next(
            candidate for candidate in result.patterns
            if {atom.value for atom in candidate.premises} == {"relevant", "loyal"}
        )
        self.assertEqual(pattern.weighted.as_dict(), {
            "tp": 6.0, "fp": 0.0, "fn": 0.0, "tn": 9.0,
        })
        self.assertEqual(pattern.counts.as_dict(), {
            "tp": 3, "fp": 0, "fn": 0, "tn": 9,
        })
        self.assertAlmostEqual(pattern.coverage, 0.4)
        self.assertAlmostEqual(pattern.base_rate, 0.4)
        self.assertAlmostEqual(pattern.parent_precision, 2.0 / 3.0)
        self.assertAlmostEqual(pattern.incremental_effect, 1.0 / 3.0)
        for fold in pattern.fold_statistics:
            self.assertEqual(fold.support, 2.0)
            self.assertAlmostEqual(fold.coverage, 0.4)
            self.assertAlmostEqual(fold.incremental_effect, 1.0 / 3.0)

    def test_unstable_conditional_signal_is_rejected(self):
        rows, targets, weights, folds = stable_synergy()
        # Reverse the relationship in the middle temporal window while keeping
        # its local base rate non-degenerate.
        targets[4:8] = [0, 0, 0, 1]
        result = run(rows, targets, weights, folds)
        keys = [
            {(atom.predicate, atom.value) for atom in pattern.premises}
            for pattern in result.patterns
        ]
        self.assertNotIn(
            {("llm_concept", "relevant"), ("history_band", "loyal")}, keys
        )
        self.assertGreater(
            result.audit.unstable_effect_rejections
            + result.audit.unstable_incremental_rejections,
            0,
        )

    def test_multivalued_semantic_fact_is_supported_but_context_is_scalar(self):
        rows, targets, weights, folds = stable_synergy()
        for index, row in enumerate(rows):
            row["llm_concept"] = [row["llm_concept"], "policy"]
        result = run(rows, targets, weights, folds)
        self.assertTrue(any(
            atom.predicate == "llm_concept" and atom.value == "relevant"
            for pattern in result.patterns for atom in pattern.premises
        ))
        rows[0]["history_band"] = ["loyal", "new"]
        with self.assertRaisesRegex(ValueError, "context predicate.*scalar"):
            run(rows, targets, weights, folds)

    def test_missing_semantic_evidence_is_omitted_not_zero_filled(self):
        rows, targets, weights, folds = stable_synergy()
        del rows[0]["llm_concept"]
        result = run(
            rows, targets, weights, folds,
            config=ConditionalMiningConfig(
                min_support=2, fold_min_support=1, max_depth=2,
                top_k=20, min_usable_folds=2,
            ),
        )
        self.assertNotIn(None, [atom.value for atom in result.frequent_atoms])

    def test_real_fpminer_provenance_and_positive_target_are_mandatory(self):
        rows, targets, weights, folds = stable_synergy()
        with self.assertRaisesRegex(ValueError, "fpMiner.metta"):
            FpMinerUnary("fake", "llm_concept", "relevant", "click", "host.py")
        with self.assertRaisesRegex(ValueError, "at least one real fpMiner"):
            run(rows, targets, weights, folds, seeds=[])
        with self.assertRaisesRegex(ValueError, "no real fpMiner unary"):
            run(rows, targets, weights, folds, seeds=[seed(target="skip")])

    def test_no_unseeded_or_single_role_pattern_can_escape(self):
        rows, targets, weights, folds = stable_synergy()
        # Only a structured seed is supplied. Every result must therefore use
        # that precise seed and at least one LLM predicate.
        structured_seed = seed(
            "history_band", "loyal", rule_id="fp_context"
        )
        result = run(rows, targets, weights, folds, seeds=[structured_seed])
        self.assertTrue(result.patterns)
        for pattern in result.patterns:
            predicates = {atom.predicate for atom in pattern.premises}
            self.assertIn("history_band", predicates)
            self.assertTrue(predicates.intersection(SEMANTIC))
            self.assertTrue(predicates.intersection(CONTEXT))
            self.assertEqual(pattern.fpminer_seed_rule_ids, ("fp_context",))

    def test_target_like_antecedents_are_rejected_even_when_not_active(self):
        rows, targets, weights, folds = stable_synergy()
        rows[0]["engagement"] = "click"
        with self.assertRaisesRegex(ValueError, "target-like predicates"):
            run(rows, targets, weights, folds)
        with self.assertRaisesRegex(ValueError, "target-like predicates"):
            mine_conditional_llm_patterns(
                rows, targets, weights=weights, folds=folds,
                fpminer_unaries=[seed()], positive_target="click",
                semantic_predicates=("target",),
                context_predicates=CONTEXT,
                predicate_lineages={**LINEAGES, "target": "bad"},
                config=ConditionalMiningConfig(
                    min_support=1, fold_min_support=1, max_depth=2,
                ),
            )

    def test_search_limits_fail_closed_without_partial_output(self):
        rows, targets, weights, folds = stable_synergy()
        with self.assertRaisesRegex(
            ConditionalMiningLimitError, "max_generated_candidates"
        ):
            run(
                rows, targets, weights, folds,
                config=ConditionalMiningConfig(
                    min_support=1, fold_min_support=1, max_depth=3,
                    top_k=20, min_usable_folds=2,
                    max_generated_candidates=1,
                ),
            )
        with self.assertRaisesRegex(
            ConditionalMiningLimitError, "max_values_per_predicate"
        ):
            run(
                rows, targets, weights, folds,
                config=ConditionalMiningConfig(
                    min_support=1, fold_min_support=1, max_depth=2,
                    top_k=20, min_usable_folds=2,
                    max_values_per_predicate=1,
                ),
            )

    def test_output_and_signatures_are_deterministic_under_row_order(self):
        rows, targets, weights, folds = stable_synergy(weighted=True)
        first = run(rows, targets, weights, folds)
        second = run(
            [dict(reversed(tuple(row.items()))) for row in reversed(rows)],
            reversed(targets), reversed(weights), reversed(folds),
        )
        def without_indexes(result):
            return [pattern.as_dict() for pattern in result.patterns]
        self.assertEqual(without_indexes(first), without_indexes(second))
        self.assertTrue(math.isclose(
            first.audit.total_weight, second.audit.total_weight
        ))


if __name__ == "__main__":
    unittest.main()
