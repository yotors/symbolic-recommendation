"""Enrich a preserved replay with causal recency observations, without replacing evidence.

The companion snapshot supplies only explicitly restored training histories.
It must contain exactly the same ordered exposures and validation slates. The
original article corpus must match the complete frozen text-vector sidecar.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

from ..features.recency_workspace import (
    RECENCY_WORKSPACE_FEATURES, RECENCY_WORKSPACE_SCHEMA,
    build_recency_workspace_facts,
)
from ..features.text_embeddings import (
    TextEmbeddingError, _canonical_corpus, load_text_embedding_sidecar,
)


RECENCY_PROJECTION_SCHEMA = "mindplex-preserved-recency-projection-v1"
_EVALUATION_KEYS = ("eval_impressions", "evaluation", "tests", "impressions")


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(value):
    if isinstance(value, Mapping):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        return value, hashlib.sha256(encoded).hexdigest(), "canonical-json-mapping"
    path = Path(value)
    before = _file_hash(path)
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as stream:
        data = json.load(stream)
    if before != _file_hash(path):
        raise ValueError("dataset changed while loading")
    if not isinstance(data, Mapping):
        raise ValueError("dataset must be a JSON mapping")
    return data, before, "file-bytes"


def _evaluation(data):
    keys = [key for key in _EVALUATION_KEYS if isinstance(data.get(key), list)]
    if len(keys) != 1:
        raise ValueError("dataset must contain exactly one evaluation list")
    return keys[0], data[keys[0]]


def _history(row, label):
    history = row.get("history")
    if not isinstance(history, list) or any(not isinstance(x, str) or not x for x in history):
        raise ValueError(f"{label} must have an explicit ordered history of article IDs")
    return history


def _corpus(data):
    return _canonical_corpus(
        ((article["id"], article.get("source_id", article["id"]),
          article.get("title"), article.get("abstract"))
         for article in data["articles"]),
        source={"kind": "preserved-replay"},
    )


def build_recency_projection(source, histories, *, sidecar_path=None):
    """Add histories and two observations while preserving every original field.

History rows are joined by exact rowwise identities, never a latest user
profile. Validation candidates, labels, and their preceding histories must
match exactly. A missing vector remains unavailable, not a zero similarity.
"""
    original, source_hash, source_kind = _read(source)
    companion, history_hash, history_kind = _read(histories)
    if original.get("metadata", {}).get("recency_workspace"):
        raise ValueError("source already contains a recency workspace")
    source_events = original.get("events")
    history_events = companion.get("events")
    if (not isinstance(source_events, list) or not isinstance(history_events, list)
            or len(source_events) != len(history_events)):
        raise ValueError("training exposure counts must match")
    impression_histories = {}
    for index, (event, restored) in enumerate(zip(source_events, history_events)):
        for key in ("user", "article", "action", "impression"):
            if key not in event or event[key] != restored.get(key):
                raise ValueError(f"training row identity mismatch at {index}: {key}")
        for key in ("source_impression_id", "timestamp", "position"):
            if event.get(key) != restored.get(key):
                raise ValueError(f"training row identity mismatch at {index}: {key}")
        history = _history(restored, f"training row {index}")
        if "history" in event and event["history"] != history:
            raise ValueError(f"original training history mismatch at {index}")
        impression = (event["user"], event["impression"])
        if impression in impression_histories and impression_histories[impression] != history:
            raise ValueError("candidates in one training impression have different histories")
        impression_histories[impression] = history
    evaluation_key, source_tests = _evaluation(original)
    _, history_tests = _evaluation(companion)
    if len(source_tests) != len(history_tests):
        raise ValueError("evaluation impression counts must match")
    for index, (case, restored) in enumerate(zip(source_tests, history_tests)):
        if any(key not in case for key in ("id", "user", "candidates", "history")):
            raise ValueError("evaluation impressions need IDs, users, candidates and histories")
        for key in ("id", "source_impression_id", "user", "candidates", "history", "labels", "relevant"):
            if case.get(key) != restored.get(key):
                raise ValueError(f"evaluation slate mismatch at {index}: {key}")
        _history(case, f"evaluation impression {index}")
        if not isinstance(case.get("candidates"), list):
            raise ValueError("evaluation candidates must be an ordered list")
        contexts = case.get("candidate_context")
        if not isinstance(contexts, Mapping) or any(
            not isinstance(contexts.get(str(candidate)), Mapping) for candidate in case["candidates"]
        ):
            raise ValueError("each evaluation candidate needs its original context")
    corpus = _corpus(original)
    companion_corpus = _corpus(companion)
    companion_articles = {item.article_id: item for item in companion_corpus.articles}
    for item in corpus.articles:
        if companion_articles.get(item.article_id) != item:
            raise TextEmbeddingError(f"history companion article content mismatch: {item.article_id}")
    embedding_path = sidecar_path or original.get("metadata", {}).get("text_embedding_sidecar")
    if not embedding_path:
        raise ValueError("a frozen text embedding sidecar is required")
    embedding_path = Path(embedding_path).resolve()
    embedding_hash = _file_hash(embedding_path)
    sidecar = load_text_embedding_sidecar(
        embedding_path, expected_content_sha256=corpus.content_sha256,
    )
    if embedding_hash != _file_hash(embedding_path):
        raise TextEmbeddingError("embedding sidecar changed while loading")
    corpus_pairs = {(item.article_id, item.source_id) for item in corpus.articles}
    if set(zip(sidecar.article_ids, sidecar.source_ids)) != corpus_pairs:
        raise TextEmbeddingError("sidecar IDs do not match the complete original corpus")
    recorded_content = original.get("metadata", {}).get("text_embedding_content_sha256")
    if recorded_content and recorded_content != corpus.content_sha256:
        raise TextEmbeddingError("source embedding content fingerprint is stale")
    vectors = sidecar.as_mapping()
    data = copy.deepcopy(original)
    cache = {}

    def enrich(candidate, history, context):
        if any(feature in context for feature in RECENCY_WORKSPACE_FEATURES):
            raise ValueError("source already contains recency observation fields")
        key = (candidate, tuple(history))
        if key not in cache:
            cache[key] = build_recency_workspace_facts(candidate, history, vectors)
        context.update(cache[key])

    for event, restored in zip(data["events"], history_events):
        event["history"] = list(restored["history"])
        enrich(event["article"], event["history"], event)
    for case in data[evaluation_key]:
        for candidate in case["candidates"]:
            enrich(candidate, case["history"], case["candidate_context"][str(candidate)])
    metadata = data.setdefault("metadata", {})
    metadata["text_embedding_sidecar"] = str(embedding_path)
    metadata["recency_workspace"] = {
        "schema": RECENCY_PROJECTION_SCHEMA,
        "observation_schema": RECENCY_WORKSPACE_SCHEMA,
        "source_dataset_sha256": source_hash,
        "source_dataset_sha256_kind": source_kind,
        "histories_dataset_sha256": history_hash,
        "histories_dataset_sha256_kind": history_kind,
        "embedding_file_sha256": embedding_hash,
        "embedding_vector_sha256": sidecar.metadata["vector_sha256"],
        "embedding_content_sha256": corpus.content_sha256,
        "embedding_model": dict(sidecar.metadata["model"]),
        "formula": "sum softmax(8*cosine-log(2)*history_lag/half_life) * (cosine+1)/2; half_life in [8,16]",
        "history_policy": "explicit causal companion histories; rowwise training and complete evaluation slate identity verified",
        "training_exposures": len(source_events),
        "evaluation_impressions": len(source_tests),
        "evaluation_candidates": sum(len(case["candidates"]) for case in source_tests),
        "covered_article_texts_verified": len(corpus_pairs),
        "cached_candidate_history_pairs": len(cache),
        "preserved_original_evidence": True,
        "neural_role": "frozen content observations only; actual miner discovers rules and PeTTaChainer proofs rank",
        "nl2pln_used": False,
    }
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--histories", required=True, type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        parser.error("output must be a new path; existing files are never overwritten")
    data = build_recency_projection(args.data, args.histories, sidecar_path=args.embeddings)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as binary:
            if output.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                        json.dump(data, text, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            else:
                encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                binary.write(encoded.encode("utf-8"))
            binary.flush()
            os.fsync(binary.fileno())
        # Atomic exclusive publication also refuses an output created mid-build.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "recency_workspace": data["metadata"]["recency_workspace"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
