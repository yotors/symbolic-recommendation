"""Attach content-verified article annotations to an unchanged causal replay.

This projector performs no language-model calls or preference fitting. The
companion provides preceding training histories, and the shared observation
builder supplies facts to the miner and PeTTaChainer.
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
from collections import Counter
from collections.abc import Mapping

from ..features.llm_workspace import LLM_WORKSPACE_FEATURES, LLM_WORKSPACE_SCHEMA, build_llm_workspace_facts
from .recency_data import _corpus, _evaluation, _history, _read
from ..features.text_embeddings import article_text


LLM_PROJECTION_SCHEMA = "mindplex-preserved-llm-projection-v1"
ANNOTATION_SNAPSHOT_SCHEMA = "mindplex-llm-article-facts-v1"
CANONICAL_ANNOTATION_SCHEMA = "mindplex-canonical-article-annotations-v1"


def _annotation_records(payload):
    """Read the immutable extractor export, never its mutable cache format."""
    if not isinstance(payload, Mapping):
        raise ValueError("annotations must be a mapping or immutable snapshot")
    if payload.get("schema") == CANONICAL_ANNOTATION_SCHEMA:
        if set(payload) != {"schema", "annotations", "provenance"}:
            raise ValueError("canonical annotations do not match their frozen envelope schema")
        records, provenance = payload.get("annotations"), payload.get("provenance")
        if not isinstance(records, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("canonical annotations require records and provenance mappings")
        expected = provenance.get("output_annotations_sha256")
        actual = hashlib.sha256(json.dumps(
            records, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        if expected != actual:
            raise ValueError("canonical annotation records disagree with their provenance hash")
        return records, {
            "kind": "canonical-annotation-snapshot",
            "schema": CANONICAL_ANNOTATION_SCHEMA,
            "provenance": copy.deepcopy(provenance),
        }
    if "extraction_records" not in payload and not isinstance(payload.get("schema"), str):
        return payload, {"kind": "direct-article-mapping"}
    allowed = {"schema", "model", "prompt_sha256", "progress", "extraction_records"}
    if payload.get("schema") != ANNOTATION_SNAPSHOT_SCHEMA or set(payload) - allowed:
        raise ValueError("annotations must be an article-facts snapshot, not a mutable cache or unknown envelope")
    records = payload.get("extraction_records")
    model, prompt_hash = payload.get("model"), payload.get("prompt_sha256")
    if (not isinstance(records, Mapping) or not isinstance(model, str) or not model.strip()
            or not isinstance(prompt_hash, str) or len(prompt_hash) != 64
            or any(character not in "0123456789abcdef" for character in prompt_hash)
            or not isinstance(payload.get("progress"), Mapping)):
        raise ValueError("invalid article-facts snapshot provenance or records")
    for record in records.values():
        provenance = record.get("provenance", {}) if isinstance(record, Mapping) else {}
        for field in ("model", "prompt_sha256"):
            if field in provenance and provenance[field] != payload[field]:
                raise ValueError(f"annotation record disagrees with snapshot {field}")
    return records, {"kind": "extractor-snapshot", **copy.deepcopy({
        key: value for key, value in payload.items() if key != "extraction_records"
    })}


def _validate_histories(original, companion):
    events, restored_events = original.get("events"), companion.get("events")
    if not isinstance(events, list) or not isinstance(restored_events, list) or len(events) != len(restored_events):
        raise ValueError("training exposure counts must match")
    histories_by_impression = {}
    for index, (event, restored) in enumerate(zip(events, restored_events)):
        for field in ("user", "article", "action", "impression"):
            if field not in event or event[field] != restored.get(field):
                raise ValueError(f"training row identity mismatch at {index}: {field}")
        for field in ("source_impression_id", "timestamp", "position"):
            if event.get(field) != restored.get(field):
                raise ValueError(f"training row identity mismatch at {index}: {field}")
        history = _history(restored, f"training row {index}")
        if "history" in event and event["history"] != history:
            raise ValueError(f"original training history mismatch at {index}")
        identity = (event["user"], event["impression"])
        if identity in histories_by_impression and histories_by_impression[identity] != history:
            raise ValueError("one training impression cannot contain different preceding histories")
        histories_by_impression[identity] = history
    evaluation_key, cases = _evaluation(original)
    _, restored_cases = _evaluation(companion)
    if len(cases) != len(restored_cases):
        raise ValueError("evaluation impression counts must match")
    for index, (case, restored) in enumerate(zip(cases, restored_cases)):
        if any(field not in case for field in ("id", "user", "candidates", "history")):
            raise ValueError("evaluation impressions require identity, candidates and explicit history")
        for field in ("id", "source_impression_id", "user", "candidates", "history", "labels", "relevant"):
            if case.get(field) != restored.get(field):
                raise ValueError(f"evaluation slate mismatch at {index}: {field}")
        _history(case, f"evaluation impression {index}")
        contexts = case.get("candidate_context")
        if not isinstance(case["candidates"], list) or not isinstance(contexts, Mapping) or any(
            not isinstance(contexts.get(str(candidate)), Mapping) for candidate in case["candidates"]
        ):
            raise ValueError("each evaluation candidate requires its original recorded context")
    corpus = _corpus(original)
    companion_articles = {item.article_id: item for item in _corpus(companion).articles}
    for item in corpus.articles:
        if companion_articles.get(item.article_id) != item:
            raise ValueError(f"history companion article content mismatch: {item.article_id}")
    return evaluation_key, corpus


def _verify_annotations(records, articles):
    allowed = {"concepts", "format", "event_types", "intents", "audiences", "provenance"}
    for identifier, record in records.items():
        if identifier not in articles:
            raise ValueError(f"annotation article ID absent from source: {identifier}")
        if not isinstance(record, Mapping) or set(record) - allowed:
            raise ValueError(f"invalid article annotation fields: {identifier}")
        for field in ("concepts", "event_types", "intents", "audiences"):
            values = record.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"annotation {field} must be a list of nonempty strings: {identifier}")
        if record.get("format") is not None and (not isinstance(record["format"], str) or not record["format"].strip()):
            raise ValueError(f"annotation format must be a string or null: {identifier}")
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"annotation lacks content provenance: {identifier}")
        article = articles[identifier]
        expected = hashlib.sha256(article_text(article.get("title"), article.get("abstract")).encode("utf-8")).hexdigest()
        if provenance.get("article_content_sha256") != expected:
            raise ValueError(f"stale annotation content hash: {identifier}")
        if "source_id" in provenance and provenance["source_id"] != article.get("source_id", identifier):
            raise ValueError(f"annotation source ID mismatch: {identifier}")


def build_llm_projection(source, histories, annotations):
    """Preserve source evidence and add facts from explicit preceding histories.

``annotations`` is an article-ID mapping, the extractor's article-facts export,
or a JSON/JSON.GZ file containing either. Mutable extraction caches are rejected.
Incomplete coverage is allowed; it is recorded and never represented
as a known negative preference. Each record must bind to its article text hash.
"""
    original, source_hash, source_kind = _read(source)
    companion, history_hash, history_kind = _read(histories)
    annotation_payload, annotations_hash, annotations_kind = _read(annotations)
    records, annotation_source = _annotation_records(annotation_payload)
    if "llm_article_annotations" in original or original.get("metadata", {}).get("llm_workspace"):
        raise ValueError("source already contains an LLM workspace")
    evaluation_key, corpus = _validate_histories(original, companion)
    articles = {article["id"]: article for article in original["articles"]}
    _verify_annotations(records, articles)
    data = copy.deepcopy(original)
    data["llm_article_annotations"] = copy.deepcopy(records)
    cache = {}
    coverage = {"training": Counter(), "evaluation": Counter()}

    def enrich(candidate, history, context, split):
        if any(feature in context for feature in LLM_WORKSPACE_FEATURES):
            raise ValueError("source already contains LLM observation fields")
        key = (candidate, tuple(history))
        if key not in cache:
            cache[key] = build_llm_workspace_facts(candidate, history, records)
            if set(cache[key]) != set(LLM_WORKSPACE_FEATURES):
                raise ValueError("LLM observation builder returned an unexpected feature schema")
        context.update(cache[key])
        counts = coverage[split]
        counts["contexts"] += 1
        counts["annotated_candidate_contexts"] += int(candidate in records)
        counts["unannotated_candidate_contexts"] += int(candidate not in records)
        counts["history_occurrences"] += len(history)
        counts["annotated_history_occurrences"] += sum(identifier in records for identifier in history)
        counts["empty_history_contexts"] += int(not history)
        counts["contexts_without_annotated_history"] += int(not any(identifier in records for identifier in history))
        counts["observations_available"] += sum(value is not None for value in cache[key].values())
        counts["observations_missing"] += sum(value is None for value in cache[key].values())

    for event, restored in zip(data["events"], companion["events"]):
        event["history"] = list(restored["history"])
        enrich(event["article"], event["history"], event, "training")
    for case in data[evaluation_key]:
        for candidate in case["candidates"]:
            enrich(candidate, case["history"], case["candidate_context"][str(candidate)], "evaluation")
    data.setdefault("metadata", {})["llm_workspace"] = {
        "schema": LLM_PROJECTION_SCHEMA,
        "observation_schema": LLM_WORKSPACE_SCHEMA,
        "source_dataset_sha256": source_hash,
        "source_dataset_sha256_kind": source_kind,
        "histories_dataset_sha256": history_hash,
        "histories_dataset_sha256_kind": history_kind,
        "annotations_sha256": annotations_hash,
        "annotations_sha256_kind": annotations_kind,
        "annotation_source": annotation_source,
        "annotation_records_sha256": hashlib.sha256(json.dumps(
            records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "article_corpus_content_sha256": corpus.content_sha256,
        "article_content_hash_recipe": "SHA256 of NFKC whitespace-folded title plus optional double-newline abstract, UTF-8",
        "history_policy": "exact rowwise exposure and evaluation-slate match; explicit preceding history only",
        "articles": len(articles),
        "annotated_articles": len(records),
        "unannotated_articles": len(articles) - len(records),
        "cached_candidate_history_pairs": len(cache),
        "coverage": {split: dict(values) for split, values in coverage.items()},
        "preserved_original_evidence": True,
        "annotation_role": "content observations only; actual miner discovers preferences and PeTTaChainer proofs rank",
    }
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "histories", "annotations", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        parser.error("output must be a new path; existing files are never overwritten")
    data = build_llm_projection(args.data, args.histories, args.annotations)
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
                binary.write(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            binary.flush()
            os.fsync(binary.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "llm_workspace": data["metadata"]["llm_workspace"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
