"""Freeze train-derived canonical IDs for an immutable article-facts export.

Only article content/entity metadata and the identities in causal training
histories are inspected. Outcomes, users, validation slates and vectors never
enter the registry or transformation. The resulting envelope is accepted
directly by :mod:`recommendation.pipelines.llm_data` so registry provenance is retained.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..features.concept_canonicalization import (
    build_canonical_registry, canonicalize_annotations,
)


def _read(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _records(payload):
    if not isinstance(payload, dict):
        raise ValueError("annotation export must be a JSON object")
    records = payload.get("extraction_records", payload)
    if not isinstance(records, dict):
        raise ValueError("annotation export records must be a mapping")
    return records


def _training_article_ids(history_snapshot):
    events = history_snapshot.get("events") if isinstance(history_snapshot, dict) else None
    if not isinstance(events, list):
        raise ValueError("history snapshot requires a training events list")
    identifiers = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("article"), str):
            raise ValueError(f"training event {index} lacks an article ID")
        history = event.get("history")
        if not isinstance(history, list) or any(not isinstance(item, str) for item in history):
            raise ValueError(f"training event {index} lacks an explicit ordered history")
        identifiers.add(event["article"])
        identifiers.update(history)
    return identifiers


def build_frozen_canonical_snapshot(source, histories, annotations, *,
                                    include_entity_concepts=False,
                                    max_entity_concepts=8):
    articles = source.get("articles") if isinstance(source, dict) else None
    if not isinstance(articles, list):
        raise ValueError("source dataset requires an articles list")
    by_id = {}
    for article in articles:
        identifier = article.get("id") if isinstance(article, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in by_id:
            raise ValueError("source articles require unique nonempty string IDs")
        by_id[identifier] = article
    records = _records(annotations)
    training_ids = _training_article_ids(histories)
    unknown = training_ids.difference(by_id)
    if unknown:
        raise ValueError("training histories reference articles absent from the source corpus")
    registry_ids = sorted(training_ids.intersection(records))
    registry_records = {identifier: records[identifier] for identifier in registry_ids}
    registry_articles = [by_id[identifier] for identifier in registry_ids]
    registry = build_canonical_registry(registry_records, registry_articles)
    result = canonicalize_annotations(
        annotations, source, registry=registry,
        include_entity_concepts=include_entity_concepts,
        max_entity_concepts=max_entity_concepts,
    )
    result["provenance"]["registry_training_scope"] = {
        "policy": "training candidates plus explicit preceding training histories; no evaluation slate membership or outcomes",
        "training_article_ids": len(training_ids),
        "annotated_training_article_ids": len(registry_ids),
        "unannotated_training_article_ids": len(training_ids) - len(registry_ids),
        "evaluation_inputs_used": False,
        "behavioral_fields_used": [],
    }
    return result


def _write_exclusive(path: Path, value):
    target = path.resolve()
    if target.exists() or path.is_symlink():
        raise ValueError("output must be a new path; existing files are never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.", dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        with temporary.open("wb") as binary:
            if target.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as stream:
                    stream.write(payload)
            else:
                binary.write(payload)
            binary.flush()
            os.fsync(binary.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--histories", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-entity-concepts", action="store_true")
    parser.add_argument("--max-entity-concepts", type=int, default=8)
    args = parser.parse_args(argv)
    source, histories, annotations = (
        _read(args.data), _read(args.histories), _read(args.annotations)
    )
    result = build_frozen_canonical_snapshot(
        source, histories, annotations,
        include_entity_concepts=args.include_entity_concepts,
        max_entity_concepts=args.max_entity_concepts,
    )
    result["provenance"]["input_files"] = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in (
            ("data", args.data), ("histories", args.histories),
            ("annotations", args.annotations),
        )
    }
    # The output hash covers records only and remains valid after adding the
    # outer file provenance above.
    _write_exclusive(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "records": len(result["annotations"]),
        "registry_training_scope":
            result["provenance"]["registry_training_scope"],
        "statistics": result["provenance"]["statistics"],
        "include_entity_concepts": args.include_entity_concepts,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
