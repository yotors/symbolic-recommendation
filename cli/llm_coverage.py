"""Read-only annotation evidence coverage over every recorded replay slate.

This module never extracts facts, samples candidates, mines rules, or scores
recommendations. A comparable pair is an evidence opportunity, not a proof or
a prediction. Training totals describe the exposures retained in the snapshot,
which need not be the complete source-dataset impression slates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import gzip
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile

from ..features.llm_workspace import LLM_NUMERIC_FEATURES


LLM_COVERAGE_SCHEMA = "mindplex-llm-replay-coverage-v1"
_EVALUATION_KEYS = ("eval_impressions", "evaluation", "tests", "impressions")
_SUM_FIELDS = (
    "candidate_context_count", "fully_missing_llm_numeric_count",
    "annotated_candidate_count", "history_coverage_known_count",
    "history_coverage_missing_count", "all_unordered_pairs",
    "comparable_unordered_pairs", "directional_unordered_pairs",
)


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(source):
    if isinstance(source, Mapping):
        return source, {"kind": "in-memory-mapping"}
    path = Path(source)
    digest = _file_sha256(path)
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as stream:
        data = json.load(stream)
    if _file_sha256(path) != digest:
        raise ValueError("dataset changed while reading coverage")
    if not isinstance(data, Mapping):
        raise ValueError("coverage dataset must be a JSON mapping")
    return data, {"kind": "file-bytes", "dataset_sha256": digest}


def _ratios(counts):
    candidates, pairs = counts["candidate_context_count"], counts["all_unordered_pairs"]
    known = counts["history_coverage_known_count"]
    counts["annotated_candidate_fraction"] = (
        counts["annotated_candidate_count"] / candidates if candidates else None
    )
    counts["history_coverage_mean"] = counts["history_coverage_sum"] / known if known else None
    for name in ("comparable", "directional"):
        counts[f"{name}_pair_fraction"] = counts[f"{name}_unordered_pairs"] / pairs if pairs else None
    return counts


def _slate_counts(rows, annotations):
    signatures = Counter()
    history_values = []
    counts = dict.fromkeys(_SUM_FIELDS, 0)
    counts["candidate_context_count"] = len(rows)
    for candidate, context in rows:
        if not isinstance(context, Mapping):
            raise ValueError("every candidate must have its recorded context mapping")
        signature = tuple(_finite(context.get(feature)) for feature in LLM_NUMERIC_FEATURES)
        signatures[signature] += 1
        counts["fully_missing_llm_numeric_count"] += int(all(value is None for value in signature))
        counts["annotated_candidate_count"] += int(isinstance(annotations.get(str(candidate)), Mapping))
        coverage = _finite(context.get("llm_history_coverage"))
        if coverage is not None:
            history_values.append(coverage)
    counts["history_coverage_known_count"] = len(history_values)
    counts["history_coverage_missing_count"] = len(rows) - len(history_values)
    counts["history_coverage_sum"] = math.fsum(history_values)
    counts["all_unordered_pairs"] = len(rows) * (len(rows) - 1) // 2
    groups = [(signature, size, sum(1 << index for index, value in enumerate(signature)
                                    if value is not None))
              for signature, size in signatures.items()]
    for index, (left, left_size, left_mask) in enumerate(groups):
        # Equal finite zeros are genuinely comparable, but never directional.
        if left_mask:
            counts["comparable_unordered_pairs"] += left_size * (left_size - 1) // 2
        for right, right_size, right_mask in groups[index + 1:]:
            common = left_mask & right_mask
            if not common:
                continue
            multiplicity = left_size * right_size
            counts["comparable_unordered_pairs"] += multiplicity
            if any(common & (1 << feature) and a != b
                   for feature, (a, b) in enumerate(zip(left, right))):
                counts["directional_unordered_pairs"] += multiplicity
    return _ratios(counts)


def _split_report(slates, annotations):
    records = []
    for identifier, user, rows in slates:
        records.append({"id": identifier, "user": user, **_slate_counts(rows, annotations)})
    totals = {field: sum(record[field] for record in records) for field in _SUM_FIELDS}
    totals["history_coverage_sum"] = math.fsum(record["history_coverage_sum"] for record in records)
    return {
        "impression_count": len(records),
        **_ratios(totals),
        "impressions_with_comparable_pairs": sum(record["comparable_unordered_pairs"] > 0 for record in records),
        "impressions_with_directional_pairs": sum(record["directional_unordered_pairs"] > 0 for record in records),
        "slates": records,
    }


def build_llm_coverage(source):
    """Audit a prepared projection mapping or JSON/JSON.GZ path without edits.

    All candidate occurrences count, including duplicates and unannotated
    articles. History coverage is the mean of finite recorded coverage values
    over candidate contexts, with a separate missing-value denominator. It is
    not recomputed from a later profile. Pair fractions always divide by all
    unordered pairs within each retained impression, never only known pairs.
    """
    data, provenance = _read(source)
    annotations, events = data.get("llm_article_annotations"), data.get("events")
    if not isinstance(annotations, Mapping) or not isinstance(events, list):
        raise ValueError("prepared coverage data requires article annotations and an events list")
    training = {}
    for event in events:
        if not isinstance(event, Mapping) or any(key not in event for key in ("user", "impression", "article")):
            raise ValueError("every training exposure requires user, impression and article IDs")
        identity = (event["user"], event["impression"])
        try:
            training.setdefault(identity, []).append((event["article"], event))
        except TypeError as exc:
            raise ValueError("training user and impression IDs must be scalar") from exc
    keys = [key for key in _EVALUATION_KEYS if isinstance(data.get(key), list)]
    if len(keys) != 1:
        raise ValueError("coverage requires exactly one evaluation slate list")
    evaluation = []
    for case in data[keys[0]]:
        if (not isinstance(case, Mapping) or "id" not in case or "user" not in case
                or not isinstance(case.get("candidates"), list)
                or not isinstance(case.get("candidate_context"), Mapping)):
            raise ValueError("evaluation slates require IDs, ordered candidates and recorded contexts")
        rows = [(candidate, case["candidate_context"].get(str(candidate))) for candidate in case["candidates"]]
        evaluation.append((case["id"], case["user"], rows))
    return {
        "schema": LLM_COVERAGE_SCHEMA,
        "source": provenance,
        "llm_numeric_features": list(LLM_NUMERIC_FEATURES),
        "definitions": {
            "annotated_candidate": "article has an annotation record, including explicit no-facts records",
            "known_numeric": "finite real number; booleans and numeric strings are missing",
            "comparable_pair": "at least one identical numeric feature is finite for both candidates",
            "directional_pair": "at least one comparable feature has unequal numeric values",
            "pair_denominator": "all unordered candidate-occurrence pairs within every retained impression",
            "history_coverage_mean": "candidate-context weighted mean of finite recorded values; missing count reported",
            "training_scope": "retained snapshot exposure groups, not necessarily complete source impression slates",
            "evaluation_scope": "all recorded slates and candidates; no label, coverage or positive-only filtering",
        },
        "training": _split_report(((impression, user, rows) for (user, impression), rows in training.items()), annotations),
        "evaluation": _split_report(evaluation, annotations),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("output must be a new path; existing files are never overwritten")
    report = build_llm_coverage(args.data)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=".llm-coverage-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Exclusive atomic publication; never replace.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "training_impressions": report["training"]["impression_count"],
                      "evaluation_impressions": report["evaluation"]["impression_count"]}))


if __name__ == "__main__":
    main()
