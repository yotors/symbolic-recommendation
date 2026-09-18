"""Prepare auditable non-neural recommendation evidence and fresh holdout slates.

The portable operations consume the lab's ordinary JSON schema.  Only the
``_reczoo_*`` helpers know Microsoft/RecZoo field names.  Recorded training
contexts are retained because recalculating a causal transition fact with the
end-of-training model would leak later outcomes.  Explicit historical item IDs
are recovered from the source when an older cache omitted them.

Fresh validation selection uses impression IDs alone, never labels, and excludes
the previously evaluated impressions.  It is a new MIND holdout, not evidence of
cross-dataset generalization.  All model statistics remain training-frozen.
"""

from __future__ import annotations

import csv
import copy
import argparse
import gzip
import hashlib
import heapq
import io
import json
import os
import tempfile
import time
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..features.lexical_workspace import build_lexical_workspace_facts, fit_lexical_idf_model
from ..adapters.mind import _format_bucket, history_feature_context, normalize_label, safe_metta_symbol


SCHEMA_VERSION = "strict-symbolic-history-lexical-v1"
_EVALUATION_KEYS = ("eval_impressions", "evaluation", "tests", "impressions")


def _excluded_field(name: str) -> bool:
    key = name.casefold()
    return (
        "embedding" in key
        or "entity" in key
        or "entities" in key
        or "vector" in key
        or "neural" in key
        or "nl2pln" in key
        or key.startswith(("text_semantic", "mi_semantic", "semantic_", "llm_"))
        or key in {"transformer_model", "encoder_model", "sentence_model", "recency_workspace"}
    )


def strip_neural_evidence(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep JSON-like copy without embeddings or entity annotations.

    Derived cached scalar features are removed too, not only vector matrices.
    Titles, abstracts, editorial taxonomy, causal counts, and the ordinary
    statistical IDF/transition models are retained.  No source object is mutated.
    """
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")

    def clean(value: Any, *, dynamic_keys: bool = False) -> Any:
        if isinstance(value, Mapping):
            return {
                key: (
                    # These statistics are non-neural. Their vocabularies may
                    # legitimately contain words such as "entity" or taxonomy
                    # labels such as "neural_science"; preserve them exactly.
                    copy.deepcopy(item)
                    if key in {"title_idf_model", "lexical_idf_model", "subcategory_transition_model"}
                    else clean(item, dynamic_keys=key in {
                        "users", "articles", "candidate_context", "labels", "idf",
                        "document_frequency", "doc_frequency", "topic_counts",
                        "subcategory_counts", "transition", "candidate",
                    })
                )
                for key, item in value.items()
                if dynamic_keys or not isinstance(key, str) or not _excluded_field(key)
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    return clean(data)


def _read_dataset(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return strip_neural_evidence(source)
    path = Path(source)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return strip_neural_evidence(json.load(handle))


def _evaluation_key(data: Mapping[str, Any]) -> str:
    return next((key for key in _EVALUATION_KEYS if key in data), "tests")


def _prepare_evaluation_ids(data: dict[str, Any]) -> None:
    """Give every portable impression a stable, unambiguous evaluation ID.

    IDs are metadata, never feature inputs.  Generated names depend only on
    source order and reserved names, not outcomes.  Both explicit IDs and the
    effective source-ID-first identity used by the benchmark must be unique.
    """
    cases = data.get(_evaluation_key(data), [])
    reserved: set[str] = set()
    for case in cases:
        for field in ("id", "source_impression_id"):
            value = case.get(field)
            if value is not None and value != "":
                case[field] = str(value)
                reserved.add(str(value))
    ids: set[str] = set()
    identities: set[str] = set()
    for index, case in enumerate(cases):
        if not case.get("id"):
            base = f"symbolic_eval_{index:08d}"
            generated = base
            suffix = 0
            while generated in reserved:
                suffix += 1
                generated = f"{base}_{suffix}"
            case["id"] = generated
            reserved.add(generated)
        identifier = case["id"]
        if identifier in ids:
            raise ValueError(f"duplicate evaluation id: {identifier}")
        ids.add(identifier)
        identity = str(case.get("source_impression_id") or identifier)
        if identity in identities:
            raise ValueError(f"duplicate evaluation impression identity: {identity}")
        identities.add(identity)


def _validate_training_impressions(data: dict[str, Any]) -> None:
    """Pair cases must compare candidates from one user's shared prior state."""
    identities: dict[str, tuple[str, tuple[str, ...]]] = {}
    for event in data.get("events", []):
        impression = event.get("impression")
        # Point-only source rows have no pair group.  Do not accidentally join
        # all such rows into one artificial impression.
        if not impression:
            continue
        group = str(impression)
        identity = (str(event["user"]), tuple(map(str, event["history"])))
        previous = identities.setdefault(group, identity)
        if previous != identity:
            raise ValueError(
                f"training impression {group} must contain one user and "
                "identical ordered pre-impression history for every candidate"
            )


def _source_impression(case: Mapping[str, Any]) -> str:
    source = case.get("source_impression_id")
    if source is not None:
        return str(source)
    value = str(case.get("id", case.get("impression", "")))
    for prefix in ("impression_train_", "impression_valid_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _digest(values: list[str]) -> str:
    encoded = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reczoo_rows(archive: zipfile.ZipFile, filename: str):
    """Parse CSV cheaply; do not build metadata dictionaries for millions of rows."""
    with archive.open(filename) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            rows = csv.reader(text)
            header = next(rows, None)
            required = {"imp_id", "click", "hour", "user_id", "news_id", "news_his"}
            if header is None or not required.issubset(header) or len(header) != len(set(header)):
                raise ValueError(f"{filename}: unsupported RecZoo schema")
            columns = {name: header.index(name) for name in required}
            for row_number, row in enumerate(rows, 2):
                if len(row) != len(header):
                    raise ValueError(f"{filename}:{row_number}: invalid column count")
                yield row_number, row, columns


def _reczoo_news(archive: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    """Read only text/editorial metadata; ignore entity annotation columns."""
    news: dict[str, dict[str, Any]] = {}
    with archive.open("news_corpus.tsv") as binary:
        with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text, delimiter="\t")
            if not {"news_id", "cat", "sub_cat", "title", "abstract"}.issubset(reader.fieldnames or ()):
                raise ValueError("news_corpus.tsv: unsupported RecZoo schema")
            for row in reader:
                source = str(row["news_id"] or "").strip()
                if not source or source in news:
                    raise ValueError("news_corpus.tsv: missing or duplicate news ID")
                title = unicodedata.normalize("NFKC", row["title"] or "").strip()
                topic = normalize_label(row["cat"])
                news[source] = {
                    "id": safe_metta_symbol(source, "article"),
                    "source_id": source,
                    "title": title or source,
                    "abstract": unicodedata.normalize("NFKC", row["abstract"] or "").strip(),
                    "topic": topic,
                    "category": topic,
                    "subcategory": normalize_label(row["sub_cat"], "news"),
                    "format": _format_bucket(title),
                    "url": "",
                }
    return news


def _reczoo_training_histories(
    data: dict[str, Any], archive: zipfile.ZipFile,
) -> dict[str, int]:
    events = data.get("events", [])
    wanted: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if "history" not in event:
            wanted.setdefault(_source_impression(event), []).append(event)
    if not wanted:
        return {"source_rows_scanned": 0, "recovered_impressions": 0, "recovered_events": 0}
    found: dict[str, tuple[str, str]] = {}
    scanned = 0
    for _number, row, columns in _reczoo_rows(archive, "train.csv"):
        scanned += 1
        impression = row[columns["imp_id"]]
        if impression not in wanted:
            continue
        identity = (row[columns["user_id"]], row[columns["news_his"]])
        if impression in found and found[impression] != identity:
            raise ValueError(f"train.csv: inconsistent user/history in impression {impression}")
        found[impression] = identity
    missing = set(wanted).difference(found)
    if missing:
        raise ValueError(f"train.csv: {len(missing)} retained training impressions were not found")
    for impression, group in wanted.items():
        source_user, raw_history = found[impression]
        history = [safe_metta_symbol(source, "article") for source in raw_history.split("^") if source]
        for event in group:
            if str(event["user"]) != safe_metta_symbol(source_user, "user"):
                raise ValueError(f"train.csv: user mismatch for retained impression {impression}")
            event["history"] = list(history)
    return {
        "source_rows_scanned": scanned,
        "recovered_impressions": len(wanted),
        "recovered_events": sum(map(len, wanted.values())),
    }


def _reczoo_fresh_groups(
    archive: zipfile.ZipFile, *, count: int, seed: int, excluded: set[str],
) -> tuple[list[tuple[int, str, list[dict[str, str]]]], dict[str, Any]]:
    """Keep smallest ID hashes, preserving whole slates and original row order."""
    retained: list[tuple[int, str, int, list[dict[str, str]]]] = []
    seen: set[str] = set()
    current: str | None = None
    current_start = current_priority = 0
    current_rows: list[dict[str, str]] = []
    eligible_current = False
    source_rows = 0

    def finish() -> None:
        if current is None or not eligible_current:
            return
        entry = (-current_priority, current, current_start, current_rows)
        if len(retained) < count:
            heapq.heappush(retained, entry)
        elif current_priority < -retained[0][0]:
            heapq.heapreplace(retained, entry)

    for row_number, row, columns in _reczoo_rows(archive, "valid.csv"):
        source_rows += 1
        impression = row[columns["imp_id"]]
        if impression != current:
            finish()
            if impression in seen:
                raise ValueError(f"valid.csv: impression {impression} is not contiguous")
            seen.add(impression)
            current = impression
            current_start = row_number
            current_priority = int.from_bytes(hashlib.sha256(
                f"{seed}\0strict-symbolic-holdout\0{impression}".encode("utf-8")
            ).digest(), "big")
            eligible_current = (
                impression not in excluded
                and (len(retained) < count or current_priority < -retained[0][0])
            )
            current_rows = []
        if eligible_current:
            current_rows.append({name: row[index] for name, index in columns.items()})
    finish()
    if len(retained) != count:
        raise ValueError(f"valid.csv: requested {count} fresh impressions; only {len(retained)} available")
    selected = sorted((start, impression, group) for _priority, impression, start, group in retained)
    return selected, {
        "source_rows_scanned": source_rows,
        "source_impressions_scanned": len(seen),
        "eligible_impressions": len(seen.difference(excluded)),
        "selected_impressions": len(selected),
        "selection": "smallest SHA256(seed, namespace, impression_id); no outcome filtering",
    }


def _fresh_cases(
    data: dict[str, Any], groups: list[tuple[int, str, list[dict[str, str]]]],
    articles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    initialized_users: set[str] = set()
    for start, impression, rows in groups:
        first = rows[0]
        source_user = first["user_id"]
        user = safe_metta_symbol(source_user, "user")
        raw_history = first["news_his"]
        history = [safe_metta_symbol(source, "article") for source in raw_history.split("^") if source]
        candidates: list[str] = []
        labels: dict[str, int] = {}
        contexts: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["user_id"] != source_user or row["news_his"] != raw_history:
                raise ValueError(f"valid.csv: inconsistent user/history in impression {impression}")
            candidate = safe_metta_symbol(row["news_id"], "article")
            if candidate not in articles:
                raise ValueError(f"valid.csv: candidate {candidate} has no source metadata")
            if row["click"] not in {"0", "1"}:
                raise ValueError(f"valid.csv: invalid label in impression {impression}")
            label = int(row["click"])
            if candidate in labels:
                if labels[candidate] != label:
                    raise ValueError(f"valid.csv: conflicting duplicate candidate in impression {impression}")
                continue
            candidates.append(candidate)
            labels[candidate] = label
            context = history_feature_context(
                articles[candidate], history, articles,
                entity_vectors={}, text_semantic_vectors={},
                title_idf_model=data.get("title_idf_model"),
                transition_model=data.get("subcategory_transition_model"),
                hour=row["hour"],
            )
            # This helper has history but not the adapter's full exposure-count
            # snapshots.  Omit unavailable priors instead of fabricating zeros.
            for unavailable in ("ctr_bucket", "freshness_bucket", "position_bucket"):
                context.pop(unavailable, None)
            contexts[candidate] = strip_neural_evidence(context)
        topics = Counter(articles[item]["topic"] for item in history if item in articles)
        ordered_topics = [topic for topic, _ in sorted(topics.items(), key=lambda item: (-item[1], item[0]))]
        if user not in initialized_users:
            data.setdefault("users", {})[user] = {
                "history": list(history), "topics": ordered_topics,
                "recent_subcategories": [articles[item]["subcategory"] for item in history[-5:] if item in articles],
            }
            initialized_users.add(user)
        cases.append({
            "id": safe_metta_symbol(f"valid_{impression}", "impression"),
            "source_impression_id": impression,
            "user": user, "source_user_id": source_user,
            "timestamp": f"valid-row-{start:09d}", "hour": first["hour"],
            "history": history, "history_topics": ordered_topics,
            "candidates": candidates,
            "relevant": [candidate for candidate in candidates if labels[candidate]],
            "labels": labels, "candidate_context": contexts,
        })
    return cases


def _enrich_lexical(data: dict[str, Any]) -> dict[str, Any]:
    articles = {str(article["id"]): article for article in data["articles"]}
    training_ids: set[str] = set()
    for event in data.get("events", []):
        training_ids.add(str(event["article"]))
        training_ids.update(map(str, event["history"]))
    model = fit_lexical_idf_model(articles[item] for item in sorted(training_ids) if item in articles)
    data["lexical_idf_model"] = model
    cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def ensure_context(
        target: dict[str, Any], candidate: str, history: list[str],
        *, training: bool, hour: Any = None,
    ) -> None:
        if "history_size_bucket" in target:
            return
        if candidate not in articles:
            raise ValueError(f"candidate {candidate} has no metadata")
        context = history_feature_context(
            articles[candidate], history, articles,
            entity_vectors={}, text_semantic_vectors={},
            title_idf_model=data.get("title_idf_model"),
            # A training event with no saved causal transition cannot consult
            # the final label-derived table.  Its existing captured transition,
            # if any, survives below; validation uses the training-frozen table.
            transition_model=None if training else data.get("subcategory_transition_model"),
            hour=hour,
        )
        for unavailable in ("ctr_bucket", "freshness_bucket", "position_bucket"):
            context.pop(unavailable, None)
        for name, value in strip_neural_evidence(context).items():
            target.setdefault(name, value)

    def evidence(candidate: str, history: list[str]) -> dict[str, Any]:
        key = (candidate, tuple(history))
        if key not in cache:
            if candidate not in articles:
                raise ValueError(f"candidate {candidate} has no metadata")
            cache[key] = build_lexical_workspace_facts(
                articles[candidate], [articles.get(item, {}) for item in history], model,
            )
        return cache[key]

    for event in data.get("events", []):
        candidate = str(event["article"])
        history = list(map(str, event["history"]))
        ensure_context(event, candidate, history, training=True, hour=event.get("hour"))
        event.update(evidence(candidate, history))
    for case in data.get(_evaluation_key(data), []):
        if "history" not in case:
            raise ValueError("every evaluation impression needs explicit pre-impression history")
        if not isinstance(case["history"], (list, tuple)):
            raise ValueError("evaluation history must be an ordered list of preceding item IDs")
        history = list(map(str, case["history"]))
        contexts = case.setdefault("candidate_context", {})
        for candidate in case["candidates"]:
            candidate = str(candidate)
            context = contexts.setdefault(candidate, {})
            ensure_context(context, candidate, history, training=False, hour=case.get("hour"))
            context.update(evidence(candidate, history))
    return {
        "training_document_ids": len(training_ids),
        "training_document_ids_sha256": _digest(list(training_ids)),
        "fitted_documents": model["document_count"],
        "fit_scope": "retained training candidates and their preceding histories only",
        "cached_candidate_history_pairs": len(cache),
        "model_sha256": hashlib.sha256(json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def build_symbolic_projection(
    cache_path: str | Path | Mapping[str, Any], archive_path: str | Path | None,
    *, fresh_eval_impressions: int = 500, seed: int = 37,
    exclude_evaluation_sources: Iterable[str | Path | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return sanitized/enriched data; never save or modify the input cache.

    Set ``fresh_eval_impressions=0`` to retain the previous development cohort.
    For a portable JSON dataset with explicit training and evaluation histories,
    use that mode and ``archive_path=None``; no MIND source is then required.
    Additional previously evaluated datasets can be supplied through
    ``exclude_evaluation_sources``. Only their impression IDs participate in
    exclusion; their training rows, labels, metadata and models never enter the
    projection's evidence or model fitting.
    """
    if isinstance(fresh_eval_impressions, bool) or not isinstance(fresh_eval_impressions, int) or fresh_eval_impressions < 0:
        raise ValueError("fresh_eval_impressions must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    data = _read_dataset(cache_path)
    _prepare_evaluation_ids(data)
    evaluation_key = _evaluation_key(data)
    previous = data.get(evaluation_key, [])
    excluded = {_source_impression(case) for case in previous}
    old_users = {str(case["user"]) for case in previous}
    if isinstance(exclude_evaluation_sources, (str, Path, Mapping)):
        raise TypeError("exclude_evaluation_sources must be an iterable of dataset sources")
    additional_exclusions = []
    for source in exclude_evaluation_sources:
        exclusion_data = _read_dataset(source)
        _prepare_evaluation_ids(exclusion_data)
        exclusion_cases = exclusion_data.get(_evaluation_key(exclusion_data), [])
        identities = {_source_impression(case) for case in exclusion_cases}
        excluded.update(identities)
        old_users.update(str(case["user"]) for case in exclusion_cases if "user" in case)
        additional_exclusions.append({
            "source": str(source) if not isinstance(source, Mapping) else "in-memory JSON mapping",
            "impressions": len(identities),
            "impression_ids_sha256": _digest(list(identities)),
        })
    train_users = {str(event["user"]) for event in data.get("events", [])}
    old_train_contexts = _digest([json.dumps(event, sort_keys=True) for event in data.get("events", [])])
    recovery: dict[str, Any] = {"recovered_impressions": 0, "recovered_events": 0}
    selection: dict[str, Any] = {"selection": "unchanged development impressions"}
    archive_manifest: dict[str, Any] = {}
    articles = {str(article["id"]): article for article in data.get("articles", [])}
    if archive_path is not None:
        with zipfile.ZipFile(archive_path) as archive:
            archive_manifest = {
                name: {"crc32": f"{archive.getinfo(name).CRC:08x}", "uncompressed_bytes": archive.getinfo(name).file_size}
                for name in ("news_corpus.tsv", "train.csv", "valid.csv")
            }
            recovery = _reczoo_training_histories(data, archive)
            news = _reczoo_news(archive)
            source_articles = {str(article["id"]): article for article in news.values()}
            # Retain cached metadata for known articles and add missing history
            # and candidate metadata from source; labels never enter this map.
            source_articles.update(articles)
            articles = source_articles
            if fresh_eval_impressions:
                groups, selection = _reczoo_fresh_groups(
                    archive, count=fresh_eval_impressions, seed=seed, excluded=excluded,
                )
                fresh = _fresh_cases(data, groups, articles)
                for key in _EVALUATION_KEYS:
                    data.pop(key, None)
                data[evaluation_key] = fresh
    elif fresh_eval_impressions:
        raise ValueError("fresh RecZoo validation selection requires archive_path")
    missing_histories = sum("history" not in event for event in data.get("events", []))
    if missing_histories:
        raise ValueError(f"{missing_histories} training events lack explicit histories; supply their source archive")
    for event in data.get("events", []):
        if not isinstance(event["history"], (list, tuple)):
            raise ValueError("training history must be an ordered list of preceding item IDs")
    _validate_training_impressions(data)
    required = set(str(article["id"]) for article in data.get("articles", []))
    for event in data.get("events", []):
        required.add(str(event["article"]))
        required.update(map(str, event["history"]))
    cases = data.get(evaluation_key, [])
    for case in cases:
        required.update(map(str, case.get("history", [])))
        required.update(map(str, case["candidates"]))
    data["articles"] = [articles[item] for item in sorted(required) if item in articles]
    lexical = _enrich_lexical(data)
    new_ids = {_source_impression(case) for case in cases}
    new_users = {str(case["user"]) for case in cases}
    if fresh_eval_impressions and excluded.intersection(new_ids):
        raise AssertionError("fresh validation overlaps the excluded development cohort")
    data.setdefault("metadata", {})["symbolic_projection"] = {
        "schema": SCHEMA_VERSION, "seed": seed,
        "source_cache": str(cache_path) if not isinstance(cache_path, Mapping) else "in-memory JSON mapping",
        "source_archive": str(archive_path) if archive_path is not None else None,
        "source_archive_members": archive_manifest,
        "removed_evidence": ["sentence/model embeddings", "pretrained entity vectors", "entity annotations", "cached derived semantic and entity predicates"],
        "training_history_recovery": recovery,
        "training_contexts_before_enrichment_sha256": old_train_contexts,
        "training_context_policy": "retain all original non-neural causal values; add explicit history and training-frozen lexical evidence",
        "lexical": lexical,
        "fresh_holdout": bool(fresh_eval_impressions),
        "validation_selection": selection,
        "excluded_previous_impression_ids": sorted(excluded),
        "excluded_previous_impression_ids_sha256": _digest(list(excluded)),
        "additional_evaluation_exclusions": additional_exclusions,
        "selected_impression_ids_sha256": _digest(list(new_ids)),
        "previous_impression_overlap": len(excluded.intersection(new_ids)),
        "previous_validation_user_overlap": len(old_users.intersection(new_users)),
        "training_user_overlap": len(train_users.intersection(new_users)),
        "validation_users": len(new_users),
        "validation_impressions": len(cases),
        "auc_eligible_impressions": sum(0 < len(case["relevant"]) < len(case["candidates"]) for case in cases),
        "unresolved_history_ids": len(required.difference(articles)),
        "ordering_scope": "RecZoo source-row order; wall-clock temporal ordering is not established",
        "transfer_scope": "new MIND impressions; cross-dataset generalization remains unmeasured",
        "unavailable_fresh_prior_features": ["ctr_bucket", "freshness_bucket", "position_bucket"] if fresh_eval_impressions else [],
    }
    data["metadata"].update(
        train_cases_loaded=len(data.get("events", [])),
        eval_impressions_loaded=len(cases), articles_loaded=len(data["articles"]),
        users_loaded=len(data.get("users", {})),
    )
    return data


def main(argv: list[str] | None = None) -> int:
    """Build a reproducible JSON/gzip artifact without overwriting its sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--fresh-eval-impressions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--exclude-evaluation", type=Path, action="append", default=[],
        help="Previously evaluated JSON/gzip dataset to exclude; repeat for multiple cohorts",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    sources = {args.cache.resolve()}
    if args.archive is not None:
        sources.add(args.archive.resolve())
    sources.update(source.resolve() for source in args.exclude_evaluation)
    if output in sources:
        parser.error("output must not overwrite the cache, source archive or exclusion datasets")
    started = time.perf_counter()
    data = build_symbolic_projection(
        args.cache, args.archive,
        fresh_eval_impressions=args.fresh_eval_impressions, seed=args.seed,
        exclude_evaluation_sources=args.exclude_evaluation,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        if output.suffix == ".gz":
            # No filename/mtime in the gzip header: identical projections can
            # produce byte-identical artifacts across repeated invocations.
            with temporary.open("wb") as binary:
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                        json.dump(data, text, ensure_ascii=False, separators=(",", ":"))
        else:
            with temporary.open("w", encoding="utf-8") as text:
                json.dump(data, text, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    audit = data["metadata"]["symbolic_projection"]
    print(json.dumps({
        "output": str(output), "bytes": output.stat().st_size,
        "seconds": round(time.perf_counter() - started, 3),
        "schema": SCHEMA_VERSION,
        "training_events": len(data.get("events", [])),
        "validation_impressions": audit["validation_impressions"],
        "auc_eligible_impressions": audit["auc_eligible_impressions"],
        "fresh_holdout": audit["fresh_holdout"],
        "previous_impression_overlap": audit["previous_impression_overlap"],
        "excluded_previous_impressions": len(audit["excluded_previous_impression_ids"]),
        "training_history_recovery": audit["training_history_recovery"],
        "lexical_model_sha256": audit["lexical"]["model_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
