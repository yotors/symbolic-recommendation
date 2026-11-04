"""Evaluation cohorts must exclude earlier impressions and cluster users."""
from recommendation.evaluation.adaptive_parent_experiment import (
    disjoint_tests,
    user_interval,
)


def test_disjoint_tests_excludes_overlap_and_ineligible_cases():
    fresh = [
        {"source_impression_id": "a", "candidates": [1, 2], "relevant": [1]},
        {"source_impression_id": "b", "candidates": [1, 2], "relevant": [1]},
        {"source_impression_id": "c", "candidates": [1], "relevant": [1]},
    ]
    tests, audit = disjoint_tests(fresh, [{"source_impression_id": "a"}])
    assert [test["source_impression_id"] for test in tests] == ["b"]
    assert audit == {"overlap_excluded": 1, "ineligible_excluded": 1}


def test_user_bootstrap_is_deterministic_and_preserves_zero_delta():
    rows = [
        {"id": "a", "user": "u1", "auc": 0.4},
        {"id": "b", "user": "u1", "auc": 0.8},
        {"id": "c", "user": "u2", "auc": 0.6},
    ]
    assert user_interval(rows, rows, repetitions=100) == [0.0, 0.0]
