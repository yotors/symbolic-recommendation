"""Compare frozen symbolic evaluation artifacts without mining or inference.

The paired statistic is the macro mean of per-impression challenger minus
baseline AUC. Bootstrap replicates resample the same impressions (or complete
user clusters) for both methods. They describe uncertainty on this cohort, not
generalization guarantees or correction for trying many configurations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any


_EVALUATION_KEYS = ("eval_impressions", "evaluation", "tests", "impressions")
_AUC_FIELDS = {"served": "auc", "proof_only": "auc_proof_only"}


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _result(artifact: Mapping[str, Any], ablation: int | None) -> Mapping[str, Any]:
    selected = artifact
    if ablation is not None:
        _integer(ablation, "ablation index")
        variants = artifact.get("ranking_ablations")
        if not isinstance(variants, list) or ablation >= len(variants):
            raise ValueError(f"ranking_ablations index {ablation} is unavailable")
        selected = variants[ablation]
    if not isinstance(selected, Mapping) or not isinstance(selected.get("result"), Mapping):
        raise ValueError("artifact selection must contain a result mapping")
    return selected["result"]


def _records(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = result.get("auc_per_impression")
    if not isinstance(rows, list) or not rows:
        raise ValueError("result must contain nonempty auc_per_impression records")
    found: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("per-impression record must be a mapping")
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("per-impression id must be a nonempty string")
        if identity in found:
            raise ValueError(f"duplicate impression id: {identity}")
        candidates = _integer(row.get("candidates"), "candidates", minimum=2)
        positives = _integer(row.get("positives"), "positives", minimum=1)
        negatives = _integer(row.get("negatives"), "negatives", minimum=1)
        if candidates != positives + negatives:
            raise ValueError(f"candidate counts do not sum for impression {identity}")
        if "index" in row:
            _integer(row["index"], "impression index")
        for field in _AUC_FIELDS.values():
            value = row.get(field)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{field} must be a finite AUC in [0, 1] for {identity}")
        found[identity] = row
    if "auc_cases" in result and _integer(result["auc_cases"], "auc_cases", minimum=1) != len(found):
        raise ValueError("auc_cases disagrees with the per-impression records")
    if "cases" in result and _integer(result["cases"], "cases", minimum=1) < len(found):
        raise ValueError("cases cannot be less than AUC-eligible impression count")
    if "candidates" in result and _integer(result["candidates"], "result candidates", minimum=2) < sum(
        row["candidates"] for row in found.values()
    ):
        raise ValueError("result candidates cannot be less than AUC-eligible candidate count")
    return found


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(fraction * (len(ordered) - 1))] for fraction in (0.025, 0.975)]


def _dataset_cases(data: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    cases = next((data[key] for key in _EVALUATION_KEYS if isinstance(data.get(key), list)), None)
    if cases is None:
        raise ValueError("dataset has no supported evaluation list")
    by_id = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("dataset evaluation record must be a mapping")
        raw = case.get("source_impression_id") or case.get("id") or case.get("impression_id")
        if raw is None or not str(raw).strip():
            raise ValueError("dataset evaluation record lacks an impression id")
        identity = str(raw)
        if identity in by_id:
            raise ValueError(f"duplicate dataset impression id: {identity}")
        by_id[identity] = case
    return cases, by_id


def compare_artifacts(
    baseline: Mapping[str, Any], challenger: Mapping[str, Any], *,
    baseline_ablation: int | None = None, challenger_ablation: int | None = None,
    data: Mapping[str, Any] | None = None, data_sha256: str | None = None,
    seed: int = 37, repetitions: int = 1000,
) -> dict[str, Any]:
    """Return exact paired AUC effects and deterministic bootstrap intervals.

    ``data_sha256`` is the hash of the optional dataset file's original bytes,
    including gzip bytes when compressed. The CLI always verifies this hash.
    An in-memory ``data`` mapping may be supplied without a file hash, but the
    report then explicitly records that its bytes were not verified.
    """

    _integer(seed, "seed")
    _integer(repetitions, "repetitions", minimum=100)
    digest = baseline.get("dataset_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("baseline dataset_sha256 must be a lowercase SHA256 digest")
    if challenger.get("dataset_sha256") != digest:
        raise ValueError("baseline and challenger dataset_sha256 differ")
    if data_sha256 is not None and data_sha256 != digest:
        raise ValueError("provided dataset bytes do not match artifact dataset_sha256")
    if data_sha256 is not None and data is None:
        raise ValueError("data_sha256 requires its dataset mapping")
    left_result = _result(baseline, baseline_ablation)
    right_result = _result(challenger, challenger_ablation)
    left, right = _records(left_result), _records(right_result)
    if left.keys() != right.keys():
        raise ValueError("baseline and challenger impression cohorts differ")
    ids = sorted(left)
    for identity in ids:
        for field in ("candidates", "positives", "negatives"):
            if left[identity][field] != right[identity][field]:
                raise ValueError(f"{field} differs for impression {identity}")
    for field in ("cases", "candidates"):
        if field in left_result or field in right_result:
            if left_result.get(field) != right_result.get(field):
                raise ValueError(f"result {field} differs between evaluation cohorts")

    changes = {
        label: [float(right[identity][field]) - float(left[identity][field]) for identity in ids]
        for label, field in _AUC_FIELDS.items()
    }
    stats = {
        label: {
            "baseline_auc": _mean([float(left[identity][field]) for identity in ids]),
            "challenger_auc": _mean([float(right[identity][field]) for identity in ids]),
            "delta": _mean(changes[label]),
        }
        for label, field in _AUC_FIELDS.items()
    }
    rng = random.Random(seed)
    bootstrap = {label: [] for label in changes}
    for _ in range(repetitions):
        sample = [rng.randrange(len(ids)) for _ in ids]
        for label, values in changes.items():
            bootstrap[label].append(_mean([values[index] for index in sample]))
    for label in stats:
        stats[label]["paired_impression_95_ci"] = _interval(bootstrap[label])
        stats[label]["paired_user_cluster_95_ci"] = None

    groups: dict[str, list[int]] = defaultdict(list)
    slices: dict[str, list[int]] = defaultdict(list)
    if data is not None:
        cases, by_id = _dataset_cases(data)
        if set(ids).difference(by_id):
            raise ValueError("compared impressions are missing from the supplied dataset")
        order = {str(case.get("source_impression_id") or case.get("id") or case.get("impression_id")): index
                 for index, case in enumerate(cases)}
        train_users = {
            str(event.get("user", event.get("user_id")))
            for event in data.get("events", [])
        }
        for index, identity in enumerate(ids):
            case = by_id[identity]
            raw_user = case.get("user", case.get("user_id"))
            if raw_user is None or not str(raw_user).strip():
                raise ValueError(f"dataset impression {identity} has no user")
            user = str(raw_user)
            groups[user].append(index)
            history = case.get("history")
            if not isinstance(history, (list, tuple)):
                bucket = "unknown"
            else:
                count = len(history)
                bucket = "cold" if count == 0 else "short" if count <= 5 else "medium" if count <= 20 else "long"
            slices[f"history_{bucket}"].append(index)
            slices["seen_user" if user in train_users else "new_user"].append(index)
            third = min(2, order[identity] * 3 // max(1, len(cases)))
            slices[f"source_order_third_{third + 1}"].append(index)

        ordered_groups = [groups[user] for user in sorted(groups)]
        group_sums = {
            label: [math.fsum(values[index] for index in group) for group in ordered_groups]
            for label, values in changes.items()
        }
        group_counts = [len(group) for group in ordered_groups]
        cluster_rng = random.Random(seed ^ 0xC1A57E)
        clustered = {label: [] for label in changes}
        for _ in range(repetitions):
            sample = [cluster_rng.randrange(len(ordered_groups)) for _ in ordered_groups]
            count = sum(group_counts[index] for index in sample)
            for label, values in group_sums.items():
                clustered[label].append(math.fsum(values[index] for index in sample) / count)
        for label in stats:
            stats[label]["paired_user_cluster_95_ci"] = _interval(clustered[label])

    subgroup_stats = {}
    for name, indices in sorted(slices.items()):
        subgroup_stats[name] = {
            "impressions": len(indices),
            **{
                label: {
                    "baseline_auc": _mean([float(left[ids[index]][field]) for index in indices]),
                    "challenger_auc": _mean([float(right[ids[index]][field]) for index in indices]),
                    "delta": _mean([changes[label][index] for index in indices]),
                }
                for label, field in _AUC_FIELDS.items()
            },
        }
    return {
        "dataset_sha256": digest,
        "dataset_bytes_verified": data is not None and data_sha256 is not None,
        "baseline_selection": "result" if baseline_ablation is None else f"ranking_ablations[{baseline_ablation}]",
        "challenger_selection": "result" if challenger_ablation is None else f"ranking_ablations[{challenger_ablation}]",
        "cohort": {
            "auc_impressions": len(ids),
            "candidates": sum(left[identity]["candidates"] for identity in ids),
            "positives": sum(left[identity]["positives"] for identity in ids),
            "negatives": sum(left[identity]["negatives"] for identity in ids),
            "impression_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "unique_users": len(groups) if data is not None else None,
        },
        "metrics": stats,
        "bootstrap": {
            "seed": seed, "repetitions": repetitions,
            "paired": True,
            "user_cluster_statistic": "resample whole users; retain macro-per-impression weighting",
            "scope": "cohort sampling uncertainty; no multiple-search correction or cross-dataset guarantee",
        },
        "subgroups": subgroup_stats,
        "subgroup_order_scope": "thirds of the supplied dataset evaluation row order; not wall-clock periods",
        "per_impression": [
            {"id": identity, "served_delta": changes["served"][index],
             "proof_only_delta": changes["proof_only"][index]}
            for index, identity in enumerate(ids)
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--baseline-ablation", type=int)
    parser.add_argument("--challenger-ablation", type=int)
    parser.add_argument("--data", type=Path, help="Original JSON/gzip dataset, required for user/subgroup diagnostics")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    result = compare_artifacts(
        _load(args.baseline), _load(args.challenger),
        baseline_ablation=args.baseline_ablation, challenger_ablation=args.challenger_ablation,
        data=_load(args.data) if args.data else None,
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data else None,
        seed=args.seed, repetitions=args.repetitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cohort": result["cohort"], "metrics": result["metrics"]}))


if __name__ == "__main__":
    main()
