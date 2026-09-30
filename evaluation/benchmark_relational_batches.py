"""Benchmark exact-root PeTTa batching on real relational plans."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from ..features.relational_workspace import build_relational_plans
from ..pipelines.relational_data import (
    _article_mapping,
    _contexts,
    _evaluation,
    _read,
    _run_isolated_reasoner_shard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--roots", type=int, default=100)
    parser.add_argument("--batch-sizes", default="5,10,20,50")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must name a new file")
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    if args.roots < 1 or any(value < 1 for value in batch_sizes):
        parser.error("roots and batch sizes must be positive")

    original, source_sha256, _source_kind = _read(args.data)
    data = copy.deepcopy(original)
    evaluation_key, _ = _evaluation(data)
    articles = _article_mapping(data)
    annotations = data.get("llm_article_annotations", {})
    entity_cache = {}
    concept_cache = {}
    planned = {}
    for _context, candidate, history, user, _split in _contexts(data, evaluation_key):
        key = (user, candidate, tuple(history))
        if key not in planned:
            planned[key] = build_relational_plans(
                candidate, history, articles, annotations, user_id=user,
                entity_observation_cache=entity_cache,
                concept_observation_cache=concept_cache,
            )

    by_family = {"entity": [], "concept": []}
    for entity_plan, concept_plan in planned.values():
        for family, plan in (("entity", entity_plan), ("concept", concept_plan)):
            by_family[family].extend(
                (plan, root.query, "exact", f"{root.origin_id}/{root.matched_value_id}")
                for root in plan.proof_roots
            )
    # Interleave families so the benchmark includes both graph depths.
    selected = []
    for index in range(max(map(len, by_family.values()))):
        for family in ("entity", "concept"):
            if index < len(by_family[family]):
                selected.append(by_family[family][index])
                if len(selected) == args.roots:
                    break
        if len(selected) == args.roots:
            break
    indexed = tuple(
        (index, plan, query, kind, diagnostic)
        for index, (plan, query, kind, diagnostic) in enumerate(selected)
    )

    results = []
    for batch_size in batch_sizes:
        started = time.perf_counter()
        status = "ok"
        error = None
        metrics = None
        try:
            metrics = _run_isolated_reasoner_shard(
                indexed, add_batch_size=5000,
                query_batch_size=batch_size, query_steps=args.steps,
            )
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - started
        results.append({
            "batch_size": batch_size,
            "status": status,
            "error": error,
            "seconds": seconds,
            "roots_per_second": len(indexed) / seconds,
            "projected_exact_root_hours": (
                sum(map(len, by_family.values())) / (len(indexed) / seconds) / 3600
            ),
            "metrics": metrics,
        })
        print(json.dumps(results[-1]), flush=True)

    output = {
        "schema": "recommendation-relational-batch-benchmark-v1",
        "source_sha256": source_sha256,
        "total_entity_exact_roots": len(by_family["entity"]),
        "total_concept_exact_roots": len(by_family["concept"]),
        "benchmark_roots": len(indexed),
        "query_steps_per_root": args.steps,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
