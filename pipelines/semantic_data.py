"""Build a content-verified, training-frozen semantic observation workspace.

Embeddings describe immutable article text; they never predict engagement here.
The same history projector supplies training and evaluation observations, which
remain inputs to the actual miner and PeTTaChainer rather than a neural ranker.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..features.semantic_workspace import build_semantic_workspace_facts, fit_semantic_workspace_model
from .symbolic_data import build_symbolic_projection
from ..features.text_embeddings import TextEmbeddingError, article_text, load_text_embedding_sidecar, read_article_corpus


SEMANTIC_PROJECTION_SCHEMA = "mindplex-semantic-projection-v1"
_EVALUATION_KEYS = ("eval_impressions", "evaluation", "tests", "impressions")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def build_semantic_projection(
    source: str | Path | Mapping[str, Any],
    sidecar_path: str | Path,
    *,
    provenance_corpus: str | Path,
) -> dict[str, Any]:
    """Return an independent projection without fitting on evaluation items.

``provenance_corpus`` must be the complete text corpus used to build the
sidecar, not merely a same-ID dataset. Its canonical content fingerprint and
every covered dataset article's normalized title/abstract must agree. Unknown
article vectors stay missing. Prior histories must already be explicit.
    """
    if isinstance(source, Mapping):
        original = source
        source_hash = _json_sha256(original)
        source_hash_kind = "canonical-json-mapping"
    else:
        input_path = Path(source)
        source_hash = _file_sha256(input_path)
        source_hash_kind = "file-bytes"
        opener = gzip.open if input_path.suffix == ".gz" else open
        with opener(input_path, "rt", encoding="utf-8") as handle:
            original = json.load(handle)
        if source_hash != _file_sha256(input_path):
            raise ValueError("source dataset changed while loading")
    if not isinstance(original, Mapping):
        raise TypeError("dataset must be a JSON mapping")
    seen_ids: set[str] = set()
    for article in original.get("articles", []):
        identifier = str(article.get("id", ""))
        if not identifier or identifier in seen_ids:
            raise ValueError("dataset articles must have unique non-empty IDs")
        seen_ids.add(identifier)
    # Discard old derived semantics, entity annotations and vector payloads
    # before adding this explicitly versioned evidence. Never mutate input.
    data = build_symbolic_projection(original, None, fresh_eval_impressions=0)
    embedding_path = Path(sidecar_path).resolve()
    corpus_path = Path(provenance_corpus).resolve()
    corpus = read_article_corpus(corpus_path)
    embedding_hash = _file_sha256(embedding_path)
    sidecar = load_text_embedding_sidecar(
        embedding_path, expected_content_sha256=corpus.content_sha256,
    )
    if embedding_hash != _file_sha256(embedding_path):
        raise TextEmbeddingError("embedding sidecar changed while loading")
    corpus_pairs = {(item.article_id, item.source_id) for item in corpus.articles}
    if set(zip(sidecar.article_ids, sidecar.source_ids)) != corpus_pairs:
        raise TextEmbeddingError("sidecar IDs do not match the provenance corpus")
    corpus_by_source = {item.source_id: item for item in corpus.articles}
    source_vectors = sidecar.as_mapping("source_id")
    articles = {str(article["id"]): article for article in data["articles"]}
    vectors: dict[str, Any] = {}
    source_ids: set[str] = set()
    for identifier, article in articles.items():
        source_id = str(article.get("source_id", identifier))
        if not source_id or source_id in source_ids:
            raise ValueError("dataset articles must have unique non-empty source IDs")
        source_ids.add(source_id)
        if source_id not in source_vectors:
            continue
        expected = corpus_by_source[source_id]
        if article_text(article.get("title"), article.get("abstract")) != expected.text:
            raise TextEmbeddingError(f"article content mismatch for source ID {source_id!r}")
        vectors[identifier] = source_vectors[source_id]

    training_ids: set[str] = set()
    for event in data.get("events", []):
        training_ids.add(str(event["article"]))
        training_ids.update(map(str, event["history"]))
    fitted_ids = sorted(training_ids.intersection(vectors))
    model = fit_semantic_workspace_model({identifier: vectors[identifier] for identifier in fitted_ids})
    data["semantic_workspace_model"] = model
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def evidence(candidate: str, history: tuple[str, ...]) -> dict[str, Any]:
        key = (candidate, history)
        if key not in cache:
            cache[key] = build_semantic_workspace_facts(candidate, history, vectors, model)
        return cache[key]

    def project(rows):
        candidates: set[str] = set()
        histories: set[str] = set()
        counts = {
            "contexts": 0, "unknown_candidate_contexts": 0,
            "no_usable_history_contexts": 0, "empty_history_contexts": 0,
            "unknown_contexts": 0, "history_occurrences": 0,
            "unknown_history_occurrences": 0,
        }
        for candidate, history, target in rows:
            history = tuple(map(str, history))
            candidate = str(candidate)
            target.update(evidence(candidate, history))
            candidate_known = candidate in vectors
            usable_history = sum(identifier in vectors for identifier in history)
            candidates.add(candidate)
            histories.update(history)
            counts["contexts"] += 1
            counts["unknown_candidate_contexts"] += int(not candidate_known)
            counts["no_usable_history_contexts"] += int(not usable_history)
            counts["empty_history_contexts"] += int(not history)
            counts["unknown_contexts"] += int(not candidate_known or not usable_history)
            counts["history_occurrences"] += len(history)
            counts["unknown_history_occurrences"] += len(history) - usable_history
        return {
            **counts,
            "unique_candidates": len(candidates),
            "covered_unique_candidates": len(candidates.intersection(vectors)),
            "unique_history_items": len(histories),
            "covered_unique_history_items": len(histories.intersection(vectors)),
        }

    training_coverage = project(
        (event["article"], event["history"], event) for event in data.get("events", [])
    )
    evaluation_key = next((key for key in _EVALUATION_KEYS if key in data), "tests")
    evaluation_coverage = project(
        (candidate, case["history"], case["candidate_context"][str(candidate)])
        for case in data.get(evaluation_key, []) for candidate in case["candidates"]
    )
    metadata = data.setdefault("metadata", {})
    metadata["text_embedding_sidecar"] = str(embedding_path)
    metadata["semantic_workspace"] = {
        "schema": SEMANTIC_PROJECTION_SCHEMA,
        "source_dataset": str(Path(source).resolve()) if not isinstance(source, Mapping) else "in-memory JSON mapping",
        "source_dataset_sha256": source_hash,
        "source_dataset_sha256_kind": source_hash_kind,
        # Fresh-holdout selection occurred upstream. The symbolic sanitizing
        # pass retains those slates but has its own fresh=False audit; keep
        # the original selection/exclusion evidence intact and independent.
        "source_projection": copy.deepcopy(original.get("metadata", {}).get("symbolic_projection")),
        "embedding_file_sha256": embedding_hash,
        "embedding_vector_sha256": sidecar.metadata["vector_sha256"],
        "embedding_model": dict(sidecar.metadata["model"]),
        "embedding_schema": sidecar.metadata["schema"],
        "embedding_text_recipe": sidecar.metadata["text_recipe"],
        "provenance_corpus": str(corpus_path),
        "provenance_corpus_content_sha256": corpus.content_sha256,
        "provenance_corpus_source": dict(corpus.source),
        "model_sha256": _json_sha256(model),
        "fit_scope": "unique retained training candidate IDs and their explicit preceding history IDs; no evaluation items or labels",
        "training_item_ids": len(training_ids),
        "training_item_ids_sha256": _json_sha256(sorted(training_ids)),
        "fitted_item_ids": len(fitted_ids),
        "fitted_item_ids_sha256": _json_sha256(fitted_ids),
        "cached_candidate_history_pairs": len(cache),
        "source_checks": {
            "full_corpus_fingerprint_verified": True,
            "sidecar_id_pairs_verified": len(corpus_pairs),
            "dataset_articles": len(articles),
            "covered_article_texts_verified": len(vectors),
            "articles_without_vectors": len(articles) - len(vectors),
        },
        "coverage": {"training": training_coverage, "evaluation": evaluation_coverage},
        "neural_role": "frozen content encoder only; no click prediction, rule generation or ranking",
        "nl2pln_used": False,
    }
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--provenance-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output in {args.data.resolve(), args.embeddings.resolve(), args.provenance_corpus.resolve()}:
        parser.error("output must not overwrite the dataset, embeddings or provenance corpus")
    started = time.perf_counter()
    data = build_semantic_projection(
        args.data, args.embeddings, provenance_corpus=args.provenance_corpus,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        if output.suffix == ".gz":
            with temporary.open("wb") as binary:
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                        json.dump(data, text, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                binary.flush()
                os.fsync(binary.fileno())
        else:
            with temporary.open("w", encoding="utf-8") as text:
                json.dump(data, text, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                text.flush()
                os.fsync(text.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(output), "bytes": output.stat().st_size,
        "seconds": round(time.perf_counter() - started, 3),
        "semantic_workspace": data["metadata"]["semantic_workspace"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
