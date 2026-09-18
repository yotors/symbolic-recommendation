from __future__ import annotations

import unittest

from recommendation.mining.petta_workspace import PeTTaWorkspaceCache


class RecordingPeTTa:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_once_on: str | None = None

    def process_metta_string(self, source: str):
        self.calls.append(source)
        if self.fail_once_on and self.fail_once_on in source:
            self.fail_once_on = None
            raise RuntimeError("injected PeTTa failure")
        return []


def cases(*identifiers: str) -> dict[str, tuple[str, ...]]:
    return {
        identifier: (
            f'(topic {identifier} "news")',
            f'(engagement {identifier} "click")',
        )
        for identifier in identifiers
    }


class PeTTaWorkspaceCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.petta = RecordingPeTTa()
        self.cache = PeTTaWorkspaceCache(
            self.petta, namespace="unit", batch_size=2
        )

    def test_first_sync_binds_a_unique_space_and_records_counts(self):
        first = self.cache.sync("point:topic:2", cases("case_1", "case_2"))
        second = self.cache.sync("pair:topic:2", cases("pair_1"))

        self.assertEqual(first.mode, "created")
        self.assertEqual(first.as_dict()["appended"], 2)
        self.assertEqual(first.reused, 0)
        self.assertEqual(first.appended_facts, 4)
        self.assertNotEqual(first.space, second.space)
        self.assertTrue(first.space.startswith("&rec_mine_unit_"))
        self.assertIn(f"!(bind! {first.space} (new-space))", self.petta.calls)

    def test_identical_sync_is_a_zero_command_reuse(self):
        original = {
            "case_1": ('(topic case_1 "news")', '(engagement case_1 "click")')
        }
        self.cache.sync("point", original)
        calls_before = len(self.petta.calls)

        # Fact ordering and duplicates are deliberately hash-invariant.
        result = self.cache.sync(
            "point",
            {"case_1": (
                '(engagement case_1 "click")',
                '(topic case_1 "news")',
                '(topic case_1 "news")',
            )},
        )

        self.assertEqual(result.mode, "reused")
        self.assertEqual((result.reused, result.appended), (1, 0))
        self.assertEqual(len(self.petta.calls), calls_before)

    def test_unchanged_subset_appends_only_new_cases(self):
        first_cases = cases("case_1", "case_2")
        self.cache.sync("point", first_cases)
        calls_before = len(self.petta.calls)

        result = self.cache.sync("point", {**first_cases, **cases("case_3")})
        delta_calls = self.petta.calls[calls_before:]

        self.assertEqual(result.mode, "appended")
        self.assertEqual((result.reused, result.appended), (2, 1))
        self.assertEqual(result.appended_facts, 2)
        self.assertEqual(len(delta_calls), 1)
        self.assertIn("case_3", delta_calls[0])
        self.assertNotIn("case_1", delta_calls[0])
        self.assertNotIn("case_2", delta_calls[0])

    def test_changed_or_deleted_case_forces_complete_rebuild(self):
        original = cases("case_1", "case_2")
        first = self.cache.sync("point", original)
        calls_before = len(self.petta.calls)
        changed = {
            "case_1": (
                '(topic case_1 "science")',
                '(engagement case_1 "skip")',
            )
        }

        result = self.cache.sync("point", changed)
        delta_calls = self.petta.calls[calls_before:]

        self.assertEqual(result.mode, "rebuilt")
        self.assertEqual((result.reused, result.appended), (0, 1))
        self.assertEqual((result.changed, result.removed), (1, 1))
        self.assertIn(f"match {first.space}", delta_calls[0])
        self.assertTrue(any("science" in call for call in delta_calls[1:]))

    def test_partial_append_failure_marks_space_for_rebuild(self):
        self.cache.sync("point", cases("case_1"))
        self.petta.fail_once_on = "case_2"
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.cache.sync("point", cases("case_1", "case_2"))
        calls_before = len(self.petta.calls)

        recovered = self.cache.sync("point", cases("case_1", "case_2"))
        recovery_calls = self.petta.calls[calls_before:]

        self.assertEqual(recovered.mode, "rebuilt")
        self.assertEqual(recovered.appended, 2)
        self.assertIn("remove-atom", recovery_calls[0])

    def test_failed_initial_population_releases_plan_and_clears_partial_space(self):
        self.petta.fail_once_on = "case_2"

        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.cache.sync("point", cases("case_1", "case_2"))

        self.assertEqual(self.cache.plans(), ())
        self.assertIn("remove-atom", self.petta.calls[-1])

    def test_case_and_fact_limits_fail_before_binding_or_population(self):
        case_bounded = PeTTaWorkspaceCache(
            self.petta, namespace="case_bound", max_cases_per_plan=1,
        )
        with self.assertRaisesRegex(MemoryError, "case limit exceeded"):
            case_bounded.sync("point", cases("case_1", "case_2"))

        fact_bounded = PeTTaWorkspaceCache(
            self.petta, namespace="fact_bound", max_facts_per_plan=1,
        )
        with self.assertRaisesRegex(MemoryError, "fact limit exceeded"):
            fact_bounded.sync("point", cases("case_1"))

        self.assertEqual(self.petta.calls, [])
        self.assertEqual(case_bounded.plans(), ())
        self.assertEqual(fact_bounded.plans(), ())

    def test_cached_plan_limit_fails_before_walking_rejected_cases(self):
        cache = PeTTaWorkspaceCache(
            self.petta, namespace="plan_bound", max_cached_plans=1,
        )
        cache.sync("first", cases("case_1"))
        calls_before = len(self.petta.calls)

        with self.assertRaisesRegex(MemoryError, "cached-plan limit reached"):
            cache.sync("second", {"case_2": None})

        self.assertEqual(len(self.petta.calls), calls_before)
        self.assertEqual(cache.plans(), ("first",))

    def test_rollback_discards_only_the_exact_staged_workspace(self):
        cache = PeTTaWorkspaceCache(
            self.petta, namespace="rollback", max_cached_plans=1,
        )
        staged = cache.sync("point", cases("case_1"))

        self.assertTrue(cache.rollback(staged))
        self.assertEqual(cache.plans(), ())
        replacement = cache.sync("point", cases("case_2"))
        self.assertNotEqual(replacement.space, staged.space)
        self.assertFalse(cache.rollback(staged))
        self.assertEqual(cache.space_for("point"), replacement.space)

    def test_sync_audit_reports_enforced_limits(self):
        cache = PeTTaWorkspaceCache(
            self.petta, namespace="audit", max_cached_plans=3,
            max_cases_per_plan=4, max_facts_per_plan=5,
        )

        result = cache.sync("point", cases("case_1"))

        self.assertEqual(result.total_facts, 2)
        self.assertEqual(result.case_limit, 4)
        self.assertEqual(result.fact_limit, 5)
        self.assertEqual(result.cached_plan_limit, 3)
        self.assertEqual(result.cached_plans_after_commit, 1)

    def test_constructor_rejects_invalid_resource_limits(self):
        for name in (
            "max_cached_plans", "max_cases_per_plan", "max_facts_per_plan"
        ):
            for value in (0, -1, True, 1.5):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        PeTTaWorkspaceCache(self.petta, **{name: value})

    def test_clear_and_prune_manage_lifecycle_without_space_collision(self):
        first = self.cache.sync("point", cases("case_1"))
        self.cache.sync("pair", cases("pair_1"))

        self.assertEqual(self.cache.clear("point"), 1)
        appended = self.cache.sync("point", cases("case_2"))
        self.assertEqual(appended.mode, "appended")
        self.assertEqual(appended.space, first.space)

        removed = self.cache.prune(keep=("pair",))
        self.assertEqual(removed, ("point",))
        self.assertEqual(self.cache.plans(), ("pair",))
        recreated = self.cache.sync("point", cases("case_3"))
        self.assertNotEqual(recreated.space, first.space)

    def test_rejects_commands_and_multiple_expressions(self):
        invalid = (
            "!(remove-all-atoms &self)",
            '(topic case_1 "news") (engagement case_1 "click")',
            '(topic case_1 "unterminated)',
        )
        for source in invalid:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    self.cache.sync("point", {"case_1": (source,)})


if __name__ == "__main__":
    unittest.main()
