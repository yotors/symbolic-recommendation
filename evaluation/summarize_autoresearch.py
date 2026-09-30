"""Summarize the symbolic AUC study with paired, selection-aware inference."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _load_runs(pattern: str) -> list[tuple[Path, dict]]:
    runs = []
    for name in sorted(glob.glob(pattern)):
        path = Path(name)
        value = json.loads(path.read_text())
        if value.get("schema") == "recommendation-autoresearch-training-tail-v1":
            runs.append((path, value))
    if not runs:
        raise ValueError(f"no autoresearch run artifacts matched {pattern!r}")
    return runs


def _records(run: dict) -> list[dict]:
    records = run.get("result", {}).get("auc_per_impression", [])
    if not records:
        raise ValueError("run has no per-impression AUC records")
    return records


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="results/autoresearch_*.json")
    parser.add_argument("--baseline", default="results/autoresearch_baseline.json")
    parser.add_argument("--output", default="results/autoresearch_summary.json")
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_928)
    args = parser.parse_args()
    if args.resamples < 1_000:
        parser.error("--resamples must be at least 1000")

    runs = _load_runs(args.pattern)
    baseline_path = Path(args.baseline)
    baseline_matches = [(path, run) for path, run in runs if path == baseline_path]
    if len(baseline_matches) != 1:
        raise ValueError("the baseline must match exactly one run artifact")
    baseline = baseline_matches[0][1]
    challengers = [(path, run) for path, run in runs if path != baseline_path]

    baseline_records = _records(baseline)
    identity = [(row["id"], row["slate_digest"]) for row in baseline_records]
    baseline_auc = np.asarray([row["auc"] for row in baseline_records], dtype=np.float64)
    differences = []
    run_rows = []
    for path, run in challengers:
        records = _records(run)
        observed_identity = [(row["id"], row["slate_digest"]) for row in records]
        if observed_identity != identity:
            raise ValueError(f"cohort mismatch in {path}")
        auc = np.asarray([row["auc"] for row in records], dtype=np.float64)
        differences.append(auc - baseline_auc)
        run_rows.append({
            "artifact": str(path),
            "overrides": run["overrides"],
            "auc": float(auc.mean()),
            "auc_delta": float((auc - baseline_auc).mean()),
            "mrr": float(run["result"]["mrr"]),
            "ndcg_at_5": float(run["result"]["ndcg_at_5"]),
            "ndcg_at_10": float(run["result"]["ndcg_at_10"]),
            "proof_coverage": float(run["result"]["proof_coverage"]),
        })
    difference_matrix = np.stack(differences)
    observed = difference_matrix.mean(axis=1)
    winner_index = int(np.argmax(observed))

    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty((len(challengers), args.resamples), dtype=np.float64)
    null = np.empty_like(bootstrap)
    batch_size = 1_000
    for start in range(0, args.resamples, batch_size):
        stop = min(start + batch_size, args.resamples)
        width = stop - start
        indices = rng.integers(0, len(identity), size=(width, len(identity)))
        bootstrap[:, start:stop] = difference_matrix[:, indices].mean(axis=2)
        signs = rng.choice((-1.0, 1.0), size=(width, len(identity)))
        null[:, start:stop] = difference_matrix @ signs.T / len(identity)

    max_abs_deviation = np.max(
        np.abs(bootstrap - observed[:, np.newaxis]), axis=0,
    )
    simultaneous_radius = float(np.quantile(max_abs_deviation, 0.95))
    max_null = np.max(null, axis=0)
    for index, row in enumerate(run_rows):
        row["paired_bootstrap_delta_ci_95"] = [
            float(np.quantile(bootstrap[index], 0.025)),
            float(np.quantile(bootstrap[index], 0.975)),
        ]
        row["simultaneous_delta_ci_95"] = [
            float(observed[index] - simultaneous_radius),
            float(observed[index] + simultaneous_radius),
        ]
        row["randomization_p_one_sided"] = float(
            (1 + np.count_nonzero(null[index] >= observed[index]))
            / (args.resamples + 1)
        )
        row["max_t_adjusted_p_one_sided"] = float(
            (1 + np.count_nonzero(max_null >= observed[index]))
            / (args.resamples + 1)
        )

    baseline_bootstrap = np.empty(args.resamples, dtype=np.float64)
    for start in range(0, args.resamples, batch_size):
        stop = min(start + batch_size, args.resamples)
        indices = rng.integers(
            0, len(identity), size=(stop - start, len(identity)),
        )
        baseline_bootstrap[start:stop] = baseline_auc[indices].mean(axis=1)

    winner = run_rows[winner_index]
    output = {
        "schema": "recommendation-autoresearch-summary-v1",
        "unit_of_analysis": "one complete MIND training impression",
        "primary_metric": "macro mean within-impression AUC",
        "selection_scope": (
            f"{len(challengers)} challengers on one fixed chronological training tail"
        ),
        "inference": {
            "resamples": args.resamples,
            "seed": args.seed,
            "paired_bootstrap": "resample complete impressions with replacement",
            "randomization": "common impression-level Rademacher sign flips",
            "multiplicity": "max-T across all challenger deltas",
            "interval_scope": "simultaneous 95% intervals across all challengers",
        },
        "cohort": {
            "impressions": len(identity),
            "fingerprint": baseline["split"]["cohort_fingerprint"],
            "dataset_sha256": baseline["dataset_sha256"],
            "all_run_identities_equal": True,
        },
        "baseline": {
            "artifact": str(baseline_path),
            "auc": float(baseline_auc.mean()),
            "bootstrap_auc_ci_95": [
                float(np.quantile(baseline_bootstrap, 0.025)),
                float(np.quantile(baseline_bootstrap, 0.975)),
            ],
        },
        "winner": winner,
        "challengers": sorted(run_rows, key=lambda row: row["auc"], reverse=True),
        "interpretation": (
            "The winner is a training-tail development result. Promotion requires "
            "an untouched confirmation cohort; neither an unadjusted interval nor this "
            "selection-aware analysis makes it an official MIND test result."
        ),
    }
    _atomic_json(Path(args.output), output)
    print(json.dumps({"output": args.output, "winner": winner}, indent=2))


if __name__ == "__main__":
    main()
