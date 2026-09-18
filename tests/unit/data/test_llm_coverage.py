import copy
import gzip
import hashlib
import json
import math
import random

import pytest

from recommendation.cli.llm_coverage import build_llm_coverage, main
from recommendation.features.llm_workspace import LLM_NUMERIC_FEATURES


def _context(value=None, *, second=None, history=0.5):
    return {LLM_NUMERIC_FEATURES[0]: value, LLM_NUMERIC_FEATURES[1]: second,
            "llm_history_coverage": history}


def _data(contexts):
    return {
        "llm_article_annotations": {"a": {}, "b": {}},
        "events": [{"user": "u", "impression": "i", "article": identifier, **context}
                   for identifier, context in contexts.items()],
        "eval_impressions": [{"id": "e", "user": "u", "candidates": list(contexts),
                              "candidate_context": copy.deepcopy(contexts)}],
    }


def test_zeros_missingness_direction_and_all_slate_denominators():
    data = _data({"a": _context(0.0), "b": _context(0.0), "c": _context(1.0),
                  "d": _context(), "e": _context(second=0.0)})
    data["eval_impressions"].extend([
        {"id": "unknown", "user": "v", "candidates": ["f", "g"],
         "candidate_context": {"f": _context(history=0.0), "g": _context(history=0.0)}},
        {"id": "empty", "user": "v", "candidates": [], "candidate_context": {}},
    ])
    before = copy.deepcopy(data)
    report = build_llm_coverage(data)
    assert data == before
    training, evaluation = report["training"], report["evaluation"]
    assert training["candidate_context_count"] == 5
    assert training["fully_missing_llm_numeric_count"] == 1
    assert training["annotated_candidate_fraction"] == 2 / 5
    assert training["all_unordered_pairs"] == 10
    assert training["comparable_unordered_pairs"] == 3
    assert training["directional_unordered_pairs"] == 2
    assert training["history_coverage_mean"] == 0.5
    assert evaluation["impression_count"] == 3
    assert evaluation["candidate_context_count"] == 7
    assert evaluation["fully_missing_llm_numeric_count"] == 3
    assert evaluation["all_unordered_pairs"] == 11
    assert evaluation["comparable_unordered_pairs"] == 3
    assert evaluation["directional_unordered_pairs"] == 2
    assert evaluation["comparable_pair_fraction"] == 3 / 11
    assert evaluation["directional_pair_fraction"] == 2 / 11
    assert evaluation["history_coverage_mean"] == 2.5 / 7
    assert evaluation["slates"][1]["annotated_candidate_fraction"] == 0.0
    assert evaluation["slates"][2]["comparable_pair_fraction"] is None


def test_nonfinite_boolean_string_values_are_not_known():
    data = _data({str(index): _context(value, history=None)
                  for index, value in enumerate([None, float("nan"), float("inf"), True, "0", 0, 0.0])})
    report = build_llm_coverage(data)["evaluation"]
    assert report["candidate_context_count"] == 7
    assert report["fully_missing_llm_numeric_count"] == 5
    assert report["comparable_unordered_pairs"] == 1
    assert report["directional_unordered_pairs"] == 0
    assert report["history_coverage_known_count"] == 0
    assert report["history_coverage_missing_count"] == 7
    assert report["history_coverage_mean"] is None


def test_training_groups_are_user_and_impression_scoped_and_labels_unused():
    data = _data({"a": _context(0), "b": _context(1)})
    data["events"] += [dict(data["events"][0], user="v"),
                       dict(data["events"][1], impression="other")]
    original = build_llm_coverage(data)
    assert original["training"]["impression_count"] == 3
    assert original["training"]["all_unordered_pairs"] == 1
    for event in data["events"]:
        event["action"] = "click"
    data["eval_impressions"][0]["labels"] = [1, 0]
    data["eval_impressions"][0]["relevant"] = ["a"]
    assert build_llm_coverage(data) == original


def test_grouped_pair_counts_match_bruteforce_all_eight_features():
    rng = random.Random(19)
    contexts = {str(index): {feature: rng.choice([None, None, 0, 0, 0.5, 1])
                             for feature in LLM_NUMERIC_FEATURES} for index in range(45)}
    # Duplicate signatures exercise multiplicity, rather than unique-row counts.
    contexts["duplicate"] = copy.deepcopy(contexts["0"])
    expected_comparable = expected_directional = 0
    rows = list(contexts.values())
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            common = [feature for feature in LLM_NUMERIC_FEATURES
                      if left[feature] is not None and right[feature] is not None]
            expected_comparable += bool(common)
            expected_directional += any(left[feature] != right[feature] for feature in common)
    report = build_llm_coverage(_data(contexts))["evaluation"]
    assert report["all_unordered_pairs"] == math.comb(len(rows), 2)
    assert report["comparable_unordered_pairs"] == expected_comparable
    assert report["directional_unordered_pairs"] == expected_directional


def test_missing_recorded_candidate_context_rejected_not_silently_dropped():
    data = _data({"a": _context(0)})
    data["eval_impressions"][0]["candidates"].append("unknown")
    with pytest.raises(ValueError, match="recorded context"):
        build_llm_coverage(data)


def test_empty_splits_have_explicit_zero_counts_and_undefined_means():
    report = build_llm_coverage({"events": [], "eval_impressions": [], "llm_article_annotations": {}})
    for split in ("training", "evaluation"):
        assert report[split]["impression_count"] == 0
        assert report[split]["all_unordered_pairs"] == 0
        assert report[split]["history_coverage_mean"] is None
        assert report[split]["annotated_candidate_fraction"] is None


def test_cli_reads_gzip_and_publishes_exclusively(tmp_path):
    source, output = tmp_path / "source.json.gz", tmp_path / "coverage.json"
    source.write_bytes(gzip.compress(json.dumps(_data({"a": _context(0), "b": _context(1)})).encode()))
    before = source.read_bytes()
    main(["--data", str(source), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["source"]["dataset_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["evaluation"]["directional_unordered_pairs"] == 1
    preserved = output.read_bytes()
    with pytest.raises(SystemExit):
        main(["--data", str(source), "--output", str(output)])
    assert output.read_bytes() == preserved
    assert source.read_bytes() == before
    assert not list(tmp_path.glob(".llm-coverage-*"))


def test_cli_never_overwrites_a_concurrently_created_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source.json", tmp_path / "coverage.json"
    source.write_text(json.dumps(_data({"a": _context(0)})))
    import recommendation.cli.llm_coverage as module
    original_link = module.os.link

    def race(source_path, destination):
        destination.write_text("another writer")
        return original_link(source_path, destination)

    monkeypatch.setattr(module.os, "link", race)
    with pytest.raises(FileExistsError):
        main(["--data", str(source), "--output", str(output)])
    assert output.read_text() == "another writer"
    assert not list(tmp_path.glob(".llm-coverage-*"))
