"""Materialize label-free graph observations for affordable symbolic mining.

Unlike ``relational_data``, this pipeline computes the finite closure of the
typed continuity hypergraph directly and does not claim an offline PeTTa proof
ledger. The output is recomputed and hash-checked when loaded. fpMiner still
discovers recommendation rules and PeTTaChainer still proves those rules; live
present-user graph facts continue through the chained PeTTa path.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import time

from ..features.relational_workspace import (
    MATERIALIZED_RELATIONAL_WORKSPACE_SCHEMA,
    RELATIONAL_WORKSPACE_FEATURES,
    build_relational_plans,
    materialized_relational_features,
    relational_context_observations_sha256,
    validate_materialized_relational_projection,
)
from .relational_data import _article_mapping, _contexts
from .recency_data import _evaluation, _read


def build_materialized_relational_projection(source: Path) -> dict:
    started = time.perf_counter()
    original, source_sha256, source_kind = _read(source)
    data = copy.deepcopy(original)
    metadata = data.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    if metadata.get("relational_workspace") is not None:
        raise ValueError("source already contains a relational workspace")
    evaluation_key, _ = _evaluation(data)
    articles = _article_mapping(data)
    annotations = data.get("llm_article_annotations") or {}
    if not isinstance(annotations, dict):
        raise ValueError("llm_article_annotations must be a mapping")

    entity_cache = {}
    concept_cache = {}
    planned = {}
    counts = Counter()
    contexts = 0
    for context, candidate, history, user, split in _contexts(data, evaluation_key):
        if any(field in context for field in RELATIONAL_WORKSPACE_FEATURES):
            raise ValueError("source already contains relational observations")
        key = (user, candidate, tuple(history))
        if key not in planned:
            planned[key] = build_relational_plans(
                candidate, history, articles, annotations, user_id=user,
                entity_observation_cache=entity_cache,
                concept_observation_cache=concept_cache,
            )
        features = materialized_relational_features(*planned[key])
        context.update(features)
        counts.update(f"{name}:{value}" for name, value in features.items())
        counts[f"split:{split}"] += 1
        contexts += 1

    metadata["relational_workspace"] = {
        "schema": MATERIALIZED_RELATIONAL_WORKSPACE_SCHEMA,
        "source_sha256": source_sha256,
        "source_kind": source_kind,
        "method": "deterministic_typed_hypergraph_closure",
        "ranking_role": "fpMiner observation; never a direct score",
        "offline_petta_proof_ledger": False,
        "live_petta_chaining_preserved": True,
        "contexts": contexts,
        "unique_plans": len(planned),
        "feature_counts": dict(sorted(
            (key, value) for key, value in counts.items()
            if not key.startswith("split:")
        )),
        "split_counts": dict(sorted(
            (key.removeprefix("split:"), value)
            for key, value in counts.items() if key.startswith("split:")
        )),
        "context_observations_sha256": relational_context_observations_sha256(data),
        "seconds": time.perf_counter() - started,
    }
    validate_materialized_relational_projection(data)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        parser.error("output must be a new path")
    data = build_materialized_relational_projection(args.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", dir=output.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as binary:
            if output.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                        json.dump(data, text, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            else:
                binary.write(json.dumps(
                    data, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8"))
            binary.flush()
            os.fsync(binary.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(output), "bytes": output.stat().st_size,
        "relational_workspace": data["metadata"]["relational_workspace"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
