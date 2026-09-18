from __future__ import annotations

import re
import unittest

from recommendation.mining.incremental_fpminer import (
    IncrementalFpMinerCache,
    parse_fpminer_supports,
)


def case(case_id: str, topic: str, outcome: str, *, fmt: str = "article"):
    return (
        f'(topic {case_id} "{topic}")',
        f'(format {case_id} "{fmt}")',
        f'(engagement {case_id} "{outcome}")',
    )


def support_form(
    premises: tuple[tuple[str, str], ...],
    target: str,
    support: int,
    *,
    variable: str = "$_123",
) -> str:
    clauses = " ".join(
        f'({predicate} {variable} "{value}")'
        for predicate, value in premises
    )
    return (
        f"(supportOf (({clauses} (engagement {variable} \"{target}\")) "
        "(CTV (STV 0.5 0.1) (STV 0.4 0.2))) "
        f"{support})"
    )


class ScriptedPeTTa:
    """Small command recorder with ordered fpMiner query responses."""

    def __init__(self):
        self.calls: list[str] = []
        self.responses: list[tuple[str, str, object]] = []
        self.fail_next_target: str | None = None

    def queue(self, space_kind: str, target: str, *forms: str) -> None:
        # Real PeTTa returns one collapsed outer list which can contain
        # duplicate supportOf forms, so the fake deliberately does the same.
        value = [f"({' '.join(forms)})"] if forms else []
        self.responses.append((space_kind, target, value))

    def process_metta_string(self, source: str):
        self.calls.append(source)
        if "frequency-pattern-miner" not in source:
            return []
        target = "click" if '"click"' in source else "skip"
        if self.fail_next_target == target:
            self.fail_next_target = None
            raise RuntimeError("injected miner failure")
        if not self.responses:
            raise AssertionError(f"unexpected fpMiner query: {source}")
        expected_space, expected_target, response = self.responses.pop(0)
        if expected_space == "full":
            assert "&full" in source and "&rec_inc_" not in source
        elif expected_space == "delta":
            assert "&rec_inc_" in source
        else:
            raise AssertionError(f"unknown scripted space kind: {expected_space}")
        assert target == expected_target
        return response

    @property
    def miner_calls(self):
        return [call for call in self.calls if "frequency-pattern-miner" in call]


def stvs(form: str):
    return [tuple(map(float, values)) for values in re.findall(
        r"\(STV\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\)", form
    )]


class ParseFpMinerSupportsTest(unittest.TestCase):
    def test_parses_real_shape_alpha_names_and_deduplicates(self):
        first = support_form((("topic", "news"),), "click", 3, variable="$_25214")
        duplicate = support_form((("topic", "news"),), "click", 3, variable="$_999")
        parsed = parse_fpminer_supports(
            [f"({first} {duplicate})"],
            features=("topic",), expected_target="click", expected_depth=2,
        )
        self.assertEqual(list(parsed), [(("topic", "news"),)])
        self.assertEqual(parsed[(("topic", "news"),)].support, 3)

    def test_canonicalizes_deeper_premises_by_feature_plan(self):
        raw = support_form(
            (("format", "long"), ("topic", "news")), "click", 2
        )
        parsed = parse_fpminer_supports(
            [raw], features=("topic", "format"),
            expected_target="click", expected_depth=3,
        )
        self.assertEqual(
            list(parsed), [(("topic", "news"), ("format", "long"))]
        )

    def test_rejects_conflicting_duplicate_or_wrong_target(self):
        one = support_form((("topic", "news"),), "click", 1)
        two = support_form((("topic", "news"),), "click", 2)
        with self.assertRaisesRegex(ValueError, "conflicting support"):
            parse_fpminer_supports(
                [one, two], features=("topic",),
                expected_target="click", expected_depth=2,
            )
        with self.assertRaisesRegex(ValueError, "unexpected target"):
            parse_fpminer_supports(
                [support_form((("topic", "news"),), "skip", 1)],
                features=("topic",), expected_target="click", expected_depth=2,
            )


class IncrementalFpMinerCacheTest(unittest.TestCase):
    def setUp(self):
        self.petta = ScriptedPeTTa()
        self.cache = IncrementalFpMinerCache(
            self.petta, namespace="unit", batch_size=2
        )

    def seed_news_and_sports(self, *, min_support=2, evidence_k=2.0):
        news_click = support_form((("topic", "news"),), "click", 1)
        news_skip = support_form((("topic", "news"),), "skip", 1)
        sports_skip = support_form((("topic", "sports"),), "skip", 1)
        self.petta.queue("full", "click", news_click, news_click)
        self.petta.queue("full", "skip", news_skip, sports_skip)
        cases = {
            "c1": case("c1", "news", "click"),
            "c2": case("c2", "news", "skip"),
            "c3": case("c3", "sports", "skip"),
        }
        result = self.cache.mine(
            plan="point/topic/2", full_space="&full", cases=cases,
            features=("topic",), depth=2, min_support=min_support,
            evidence_k=evidence_k,
        )
        return cases, result

    def test_full_seed_keeps_support_one_patterns_below_output_threshold(self):
        cases, first = self.seed_news_and_sports()
        self.assertEqual(first.output, ())
        self.assertEqual(first.audit.mode, "full_seed")
        self.assertEqual(first.audit.tracked_patterns, 2)
        self.assertEqual(first.audit.pattern_limit, 250_000)
        self.assertEqual(first.audit.cached_plan_limit, 512)
        self.assertEqual(first.audit.cached_plans_after_commit, 1)
        self.assertEqual(first.audit.full_miner_calls, 2)
        self.assertTrue(first.audit.full_structure_research)

        calls_before = len(self.petta.calls)
        lowered = self.cache.mine(
            plan="point/topic/2", full_space="&full", cases=cases,
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
        )
        self.assertEqual(lowered.audit.mode, "reused")
        self.assertEqual(lowered.audit.miner_calls, 0)
        self.assertEqual(len(self.petta.calls), calls_before)
        self.assertEqual(len(lowered.output), 1)
        positive, negative = stvs(lowered.output[0])
        self.assertEqual(positive, (0.5, 0.5))
        self.assertAlmostEqual(negative[0], 0.0)
        self.assertAlmostEqual(negative[1], 1.0 / 3.0)

    def test_append_queries_only_delta_and_crosses_threshold(self):
        cases, _first = self.seed_news_and_sports()
        self.petta.queue(
            "delta", "click",
            support_form((("topic", "news"),), "click", 1),
        )
        self.petta.queue("delta", "skip")
        cases = {**cases, "c4": case("c4", "news", "click")}
        result = self.cache.mine(
            plan="point/topic/2", full_space="&full", cases=cases,
            features=("topic",), depth=2, min_support=2, evidence_k=2.0,
        )

        self.assertEqual(result.audit.mode, "delta_updated")
        self.assertEqual(result.audit.appended_cases, 1)
        self.assertEqual(result.audit.researched_cases, 1)
        self.assertEqual(result.audit.full_miner_calls, 0)
        self.assertEqual(result.audit.delta_miner_calls, 2)
        self.assertFalse(result.audit.full_structure_research)
        self.assertTrue(result.audit.delta_space.startswith("&rec_inc_unit_"))
        parsed = parse_fpminer_supports(
            result.output, features=("topic",),
            expected_target="click", expected_depth=2,
        )
        self.assertEqual(parsed[(("topic", "news"),)].support, 2)
        positive, negative = stvs(result.output[0])
        self.assertAlmostEqual(positive[0], 2.0 / 3.0)
        self.assertAlmostEqual(positive[1], 3.0 / 5.0)
        self.assertAlmostEqual(negative[0], 0.0)
        self.assertAlmostEqual(negative[1], 1.0 / 3.0)

    def test_appended_skip_updates_antecedent_and_both_ctv_branches(self):
        self.petta.queue(
            "full", "click",
            support_form((("topic", "news"),), "click", 1),
        )
        self.petta.queue("full", "skip")
        initial = {"c1": case("c1", "news", "click")}
        first = self.cache.mine(
            plan="skip-ctv", full_space="&full", cases=initial,
            features=("topic",), depth=2, min_support=1, evidence_k=1.0,
        )
        self.assertEqual(stvs(first.output[0])[0], (1.0, 0.5))

        self.petta.queue("delta", "click")
        self.petta.queue(
            "delta", "skip",
            support_form((("topic", "news"),), "skip", 1),
        )
        updated = self.cache.mine(
            plan="skip-ctv", full_space="&full",
            cases={**initial, "c2": case("c2", "news", "skip")},
            features=("topic",), depth=2, min_support=1, evidence_k=1.0,
        )
        positive, negative = stvs(updated.output[0])
        self.assertEqual(positive, (0.5, 2.0 / 3.0))
        self.assertEqual(negative, (0.0, 0.0))

    def test_changed_or_removed_case_forces_transactional_full_rebuild(self):
        cases, _first = self.seed_news_and_sports(min_support=1)
        changed = {
            "c1": case("c1", "science", "skip"),
            "c2": cases["c2"],
        }
        self.petta.queue("full", "click")
        self.petta.queue(
            "full", "skip",
            support_form((("topic", "news"),), "skip", 1),
            support_form((("topic", "science"),), "skip", 1),
        )
        rebuilt = self.cache.mine(
            plan="point/topic/2", full_space="&full", cases=changed,
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
        )
        self.assertEqual(rebuilt.audit.mode, "rebuilt_full")
        self.assertEqual(rebuilt.audit.changed_cases, 1)
        self.assertEqual(rebuilt.audit.removed_cases, 1)
        self.assertEqual(rebuilt.audit.full_miner_calls, 2)
        self.assertEqual(rebuilt.output, ())

    def test_expired_causal_unit_forces_exact_full_rebuild(self):
        click = support_form((('topic', 'news'),), 'click', 1)
        self.petta.queue("full", "click", click)
        self.petta.queue("full", "skip")
        cases = {"c2": case("c2", "news", "click")}
        first_audit = {
            "policy": "newest_complete_unit_suffix_v1",
            "retained_units": 2,
            "retained_cases": 1,
            "expired_units": 0,
            "expired_cases": 0,
        }
        self.cache.mine(
            plan="retained", full_space="&full", cases=cases,
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
            retention_units=("impression-1", "impression-2"),
            retention_audit=first_audit,
        )

        # The selected discovery row happens to be unchanged, but the global
        # causal window lost impression-1. Approximate subtraction or reuse is
        # forbidden: both target passes must recount the complete full space.
        self.petta.queue("full", "click", click)
        self.petta.queue("full", "skip")
        next_audit = {
            **first_audit,
            "expired_units": 1,
        }
        rebuilt = self.cache.mine(
            plan="retained", full_space="&full", cases=cases,
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
            retention_units=("impression-2", "impression-3"),
            retention_audit=next_audit,
        )

        self.assertEqual(rebuilt.audit.mode, "retention_expired_full")
        self.assertEqual(rebuilt.audit.full_miner_calls, 2)
        self.assertEqual(rebuilt.audit.delta_miner_calls, 0)
        self.assertEqual(rebuilt.audit.researched_cases, 1)
        self.assertEqual(rebuilt.audit.expired_retained_units, 1)
        self.assertEqual(rebuilt.audit.appended_retained_units, 1)
        self.assertTrue(rebuilt.audit.expiration_forced_full_rebuild)
        self.assertEqual(
            rebuilt.audit.rebuild_reason,
            "causal_units_expired_exact_full_rebuild",
        )
        self.assertEqual(rebuilt.audit.retention, next_audit)

    def test_case_state_limit_fails_before_miner_query(self):
        cache = IncrementalFpMinerCache(
            self.petta, namespace="bounded_cases", batch_size=2,
            max_cases_per_plan=1,
        )
        with self.assertRaisesRegex(MemoryError, "case limit exceeded"):
            cache.mine(
                plan="bounded-cases", full_space="&full",
                cases={
                    "c1": case("c1", "news", "click"),
                    "c2": case("c2", "sports", "skip"),
                },
                features=("topic",), depth=2, min_support=1,
                evidence_k=2.0,
            )
        self.assertEqual(self.petta.miner_calls, [])
        self.assertEqual(cache.plans(), ())

    def test_shape_change_under_same_plan_reseeds_instead_of_reusing(self):
        cases, _first = self.seed_news_and_sports(min_support=1)
        self.petta.queue(
            "full", "click",
            support_form(
                (("topic", "news"), ("format", "article")), "click", 1
            ),
        )
        self.petta.queue(
            "full", "skip",
            support_form(
                (("topic", "news"), ("format", "article")), "skip", 1
            ),
            support_form(
                (("topic", "sports"), ("format", "article")), "skip", 1
            ),
        )
        result = self.cache.mine(
            plan="point/topic/2", full_space="&full", cases=cases,
            features=("topic", "format"), depth=3,
            min_support=1, evidence_k=2.0,
        )
        self.assertEqual(result.audit.mode, "reconfigured_full")
        self.assertEqual(result.audit.full_miner_calls, 2)

    def test_failed_second_delta_pass_does_not_commit_partial_support(self):
        cases, _first = self.seed_news_and_sports(min_support=1)
        appended = {**cases, "c4": case("c4", "news", "click")}
        click_delta = support_form((("topic", "news"),), "click", 1)
        self.petta.queue("delta", "click", click_delta)
        self.petta.queue("delta", "skip")
        self.petta.fail_next_target = "skip"
        with self.assertRaisesRegex(RuntimeError, "injected miner failure"):
            self.cache.mine(
                plan="point/topic/2", full_space="&full", cases=appended,
                features=("topic",), depth=2, min_support=1, evidence_k=2.0,
            )
        # The failed skip response was never consumed by the fake.
        self.petta.responses.pop(0)
        self.petta.queue("delta", "click", click_delta)
        self.petta.queue("delta", "skip")
        retried = self.cache.mine(
            plan="point/topic/2", full_space="&full", cases=appended,
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
        )
        parsed = parse_fpminer_supports(
            retried.output, features=("topic",),
            expected_target="click", expected_depth=2,
        )
        self.assertEqual(retried.audit.mode, "delta_updated")
        self.assertEqual(parsed[(("topic", "news"),)].support, 2)
        self.assertTrue(any("remove-atom" in call for call in self.petta.calls))

    def test_new_pattern_can_first_arrive_in_delta(self):
        cases, _first = self.seed_news_and_sports(min_support=1)
        self.petta.queue(
            "delta", "click",
            support_form((("topic", "culture"),), "click", 1),
        )
        self.petta.queue("delta", "skip")
        updated = self.cache.mine(
            plan="point/topic/2", full_space="&full",
            cases={**cases, "c4": case("c4", "culture", "click")},
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
        )
        parsed = parse_fpminer_supports(
            updated.output, features=("topic",),
            expected_target="click", expected_depth=2,
        )
        self.assertEqual(parsed[(("topic", "culture"),)].support, 1)
        self.assertEqual(updated.audit.tracked_patterns, 3)

    def test_pattern_limit_fails_before_transactional_commit(self):
        cache = IncrementalFpMinerCache(
            self.petta, namespace="bounded", batch_size=2,
            max_tracked_patterns_per_plan=1,
        )
        self.petta.queue(
            "full", "click",
            support_form((("topic", "news"),), "click", 1),
            support_form((("topic", "sports"),), "click", 1),
        )
        self.petta.queue("full", "skip")
        with self.assertRaisesRegex(MemoryError, "pattern limit exceeded"):
            cache.mine(
                plan="bounded", full_space="&full",
                cases={
                    "c1": case("c1", "news", "click"),
                    "c2": case("c2", "sports", "click"),
                },
                features=("topic",), depth=2, min_support=1,
                evidence_k=2.0,
            )
        self.assertEqual(cache.plans(), ())

    def test_cached_plan_limit_requires_explicit_pruning(self):
        cache = IncrementalFpMinerCache(
            self.petta, namespace="bounded_plans", batch_size=2,
            max_cached_plans=1,
        )
        self.petta.queue(
            "full", "click",
            support_form((("topic", "news"),), "click", 1),
        )
        self.petta.queue("full", "skip")
        cases = {"c1": case("c1", "news", "click")}
        cache.mine(
            plan="first", full_space="&full", cases=cases,
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
        )
        with self.assertRaisesRegex(MemoryError, "cached-plan limit reached"):
            cache.mine(
                plan="second", full_space="&full", cases=cases,
                features=("topic",), depth=2, min_support=1,
                evidence_k=2.0,
            )
        self.assertEqual(cache.plans(), ("first",))

    def test_invalid_closed_case_contract_is_rejected_before_petta(self):
        invalid = {
            "c1": ('(topic c1 "news")',),
        }
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.cache.mine(
                plan="bad", full_space="&full", cases=invalid,
                features=("topic",), depth=2, min_support=1, evidence_k=2.0,
            )
        self.assertEqual(self.petta.calls, [])

    def test_cross_case_selected_feature_is_rejected_before_petta(self):
        invalid = {
            "c1": (
                '(topic c2 "news")',
                '(engagement c1 "click")',
            ),
        }
        with self.assertRaisesRegex(
            ValueError, "selected-feature fact case does not match"
        ):
            self.cache.mine(
                plan="cross-case", full_space="&full", cases=invalid,
                features=("topic",), depth=2, min_support=1,
                evidence_k=2.0,
            )
        self.assertEqual(self.petta.calls, [])

    def test_invalidate_and_prune_control_lifecycle(self):
        cases, _first = self.seed_news_and_sports(min_support=1)
        self.assertEqual(self.cache.plans(), ("point/topic/2",))
        self.assertTrue(self.cache.invalidate("point/topic/2"))
        self.assertEqual(self.cache.plans(), ())

        self.petta.queue("full", "click")
        self.petta.queue(
            "full", "skip",
            support_form((("topic", "news"),), "skip", 1),
            support_form((("topic", "sports"),), "skip", 1),
        )
        reseeded = self.cache.mine(
            plan="point/topic/2", full_space="&full",
            cases={key: (*facts[:-1], f'(engagement {key} "skip")')
                   for key, facts in cases.items()},
            features=("topic",), depth=2, min_support=1, evidence_k=2.0,
        )
        self.assertEqual(reseeded.audit.mode, "full_seed")
        self.assertEqual(self.cache.prune(), ("point/topic/2",))
        self.assertEqual(self.cache.plans(), ())


if __name__ == "__main__":
    unittest.main()
