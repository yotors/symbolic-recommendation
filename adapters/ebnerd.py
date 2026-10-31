"""Causal adapter for the Ekstra Bladet News Recommendation Dataset.

The adapter translates EB-NeRD Parquet files into the same portable event
contract used by the recommendation engine.  Dataset-specific code stops at
that boundary: mining, CTV calibration, PeTTaChainer inference, and metrics are
shared with every other source.

Sampling is deterministic.  Among complete impressions that contain a clicked
and a nonclicked candidate, selection order depends only on a hash of the
source impression ID.  Every earlier click in the source stream is still
allowed to update the user's causal history.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, deque
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from .mind import (
    _format_bucket,
    fit_title_idf_model,
    history_feature_context,
    normalize_label,
    safe_metta_symbol,
)
from ..features.text_embeddings import article_text
from ..pipelines.llm_data import build_llm_projection


DEFAULT_MAX_TRAIN_CASES = 20_000
DEFAULT_MAX_EVAL_IMPRESSIONS = 2_000
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_RECENT_CTR_HOURS = 6
DEFAULT_RECENT_CTR_PRIOR_WEIGHT = 5.0


class EbnerdDataError(ValueError):
    """Raised when an EB-NeRD directory violates the expected schema."""


def _arrow():
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("EB-NeRD loading requires pyarrow") from exc
    return parquet


def _positive_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _priority(seed: int, split: str, impression: object) -> bytes:
    return hashlib.sha256(f"{seed}:{split}:{impression}".encode()).digest()


def _selected_impressions(
    rows: Iterable[dict[str, Any]], *, split: str, seed: int,
    max_cases: int | None = None, max_impressions: int | None = None,
) -> set[int]:
    eligible = []
    for row in rows:
        candidates = row.get("article_ids_inview") or ()
        clicked = set(row.get("article_ids_clicked") or ())
        if clicked and any(candidate not in clicked for candidate in candidates):
            eligible.append(row)
    eligible.sort(key=lambda row: _priority(seed, split, row["impression_id"]))
    selected: set[int] = set()
    cases = 0
    for row in eligible:
        size = len(row["article_ids_inview"])
        if max_impressions is not None and len(selected) >= max_impressions:
            break
        if max_cases is not None and cases + size > max_cases:
            continue
        selected.add(int(row["impression_id"]))
        cases += size
        if max_cases is not None and cases >= max_cases:
            break
    if not selected:
        raise EbnerdDataError(f"{split} has no eligible complete impressions")
    return selected


def _article_records(root: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    parquet = _arrow()
    columns = (
        "article_id", "title", "subtitle", "url", "article_type",
        "topics", "category_str", "subcategory", "ner_clusters",
        "entity_groups", "sentiment_label", "premium",
    )
    rows = parquet.read_table(root / "articles.parquet", columns=list(columns)).to_pylist()
    articles: dict[int, dict[str, Any]] = {}
    annotations: dict[str, Any] = {}
    for row in rows:
        source = int(row["article_id"])
        identifier = safe_metta_symbol(source, "article")
        topics = [normalize_label(value, "") for value in row.get("topics") or []]
        topics = list(dict.fromkeys(value for value in topics if value))
        category = normalize_label(row.get("category_str"))
        subcategories = [str(value) for value in row.get("subcategory") or []]
        title = str(row.get("title") or source).strip()
        article = {
            "id": identifier,
            "source_id": str(source),
            "title": title,
            "abstract": str(row.get("subtitle") or "").strip(),
            "url": str(row.get("url") or ""),
            "topic": topics[0] if topics else category,
            "category": category,
            "subcategory": topics[1] if len(topics) > 1 else (
                f"subcategory_{subcategories[0]}" if subcategories else category
            ),
            "format": normalize_label(row.get("article_type"), _format_bucket(title)),
            "title_entities": list(row.get("ner_clusters") or []),
            "abstract_entities": list(row.get("entity_groups") or []),
            "topics": topics,
            "sentiment": normalize_label(row.get("sentiment_label"), "unknown"),
            "premium": bool(row.get("premium")),
        }
        articles[source] = article
        concepts = list(dict.fromkeys(
            topics
            + [normalize_label(value, "") for value in row.get("ner_clusters") or []]
            + [normalize_label(value, "") for value in row.get("entity_groups") or []]
        ))
        annotations[identifier] = {
            "concepts": [value for value in concepts if value],
            "format": article["format"],
            "event_types": [],
            "intents": [],
            "audiences": [],
            "provenance": {
                "source_id": str(source),
                "article_content_sha256": hashlib.sha256(
                    article_text(article["title"], article["abstract"]).encode()
                ).hexdigest(),
            },
        }
    if not articles:
        raise EbnerdDataError("articles.parquet contains no articles")
    return articles, annotations


def _histories(path: Path, limit: int) -> dict[int, list[int]]:
    parquet = _arrow()
    rows = parquet.read_table(
        path, columns=["user_id", "article_id_fixed"]
    ).to_pylist()
    return {
        int(row["user_id"]): [int(value) for value in (row["article_id_fixed"] or [])][-limit:]
        for row in rows
    }


def _behaviors(path: Path) -> list[dict[str, Any]]:
    parquet = _arrow()
    columns = (
        "impression_id", "impression_time", "user_id", "session_id",
        "device_type", "article_ids_inview", "article_ids_clicked",
        "next_read_time", "next_scroll_percentage",
    )
    rows = parquet.read_table(path, columns=list(columns)).to_pylist()
    rows.sort(key=lambda row: (row["impression_time"], row["impression_id"]))
    return rows


def _safe_history(
    history: Iterable[int], articles: dict[int, dict[str, Any]], limit: int,
) -> list[str]:
    return [articles[value]["id"] for value in list(history)[-limit:] if value in articles]


def _context(
    source: int, history: list[int], articles: dict[int, dict[str, Any]],
    by_id: dict[str, dict[str, Any]], idf: dict[str, Any], hour: int,
) -> dict[str, Any]:
    article = articles[source]
    return history_feature_context(
        article, _safe_history(history, articles, len(history)), by_id,
        title_idf_model=idf, hour=hour,
    )


def _action(clicked: bool, read_time: object, scroll: object) -> str:
    if not clicked:
        return "skip"
    read = float(read_time or 0.0)
    depth = float(scroll or 0.0)
    return "complete" if read >= 60.0 or depth >= 80.0 else "click"


def load_ebnerd(
    root: str | Path, *, max_train_cases: int | None = DEFAULT_MAX_TRAIN_CASES,
    max_eval_impressions: int | None = DEFAULT_MAX_EVAL_IMPRESSIONS,
    history_limit: int = DEFAULT_HISTORY_LIMIT, seed: int = 7,
    recent_ctr_hours: int = DEFAULT_RECENT_CTR_HOURS,
    recent_ctr_prior_weight: float = DEFAULT_RECENT_CTR_PRIOR_WEIGHT,
    content_workspace: bool = True,
) -> dict[str, Any]:
    """Load a bounded, causal EB-NeRD replay using complete impressions."""
    max_train_cases = _positive_limit("max_train_cases", max_train_cases)
    max_eval_impressions = _positive_limit(
        "max_eval_impressions", max_eval_impressions
    )
    history_limit = _positive_limit("history_limit", history_limit)
    if history_limit is None:
        raise ValueError("history_limit cannot be None")
    recent_ctr_hours = _positive_limit("recent_ctr_hours", recent_ctr_hours)
    if recent_ctr_hours is None:
        raise ValueError("recent_ctr_hours cannot be None")
    if recent_ctr_prior_weight <= 0.0:
        raise ValueError("recent_ctr_prior_weight must be positive")
    root = Path(root).expanduser().resolve()
    required = (
        "articles.parquet", "train/history.parquet", "train/behaviors.parquet",
        "validation/history.parquet", "validation/behaviors.parquet",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"EB-NeRD root is missing: {', '.join(missing)}")

    articles, annotations = _article_records(root)
    by_id = {article["id"]: article for article in articles.values()}
    train_rows = _behaviors(root / "train/behaviors.parquet")
    valid_rows = _behaviors(root / "validation/behaviors.parquet")
    selected_train = _selected_impressions(
        train_rows, split="train", seed=seed, max_cases=max_train_cases,
    )
    selected_valid = _selected_impressions(
        valid_rows, split="validation", seed=seed,
        max_impressions=max_eval_impressions,
    )
    exposed = (
        source for row in train_rows
        if int(row["impression_id"]) in selected_train
        for source in row["article_ids_inview"]
    )
    idf = fit_title_idf_model(
        {str(source): article for source, article in articles.items()}, exposed
    )
    train_history = _histories(root / "train/history.parquet", history_limit)
    valid_history = _histories(root / "validation/history.parquet", history_limit)
    events: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    selected_users: set[str] = set()
    referenced: set[str] = set()
    exposures: Counter[int] = Counter()
    clicks: Counter[int] = Counter()
    recent_exposures: Counter[int] = Counter()
    recent_clicks: Counter[int] = Counter()
    recent_rows: deque[tuple[Any, tuple[int, ...], tuple[int, ...]]] = deque()

    def expire_recent(now: Any) -> None:
        cutoff = now - timedelta(hours=recent_ctr_hours)
        while recent_rows and recent_rows[0][0] < cutoff:
            _time, old_candidates, old_clicked = recent_rows.popleft()
            recent_exposures.subtract(old_candidates)
            recent_clicks.subtract(old_clicked)

    for row in train_rows:
        expire_recent(row["impression_time"])
        user_source = int(row["user_id"])
        history = train_history.setdefault(user_source, [])
        candidates = [int(value) for value in row["article_ids_inview"] if int(value) in articles]
        clicked = {int(value) for value in row["article_ids_clicked"] or []}
        if int(row["impression_id"]) in selected_train and candidates:
            user = safe_metta_symbol(user_source, "user")
            impression = safe_metta_symbol(f"train_{row['impression_id']}", "impression")
            safe_history = _safe_history(history, articles, history_limit)
            selected_users.add(user)
            referenced.update(safe_history)
            for position, source in enumerate(candidates):
                context = _context(
                    source, history[-history_limit:], articles, by_id, idf,
                    row["impression_time"].hour,
                )
                article_id = articles[source]["id"]
                context["empirical_ctr"] = round(
                    (clicks[source] + 1.0) / (exposures[source] + 10.0), 8
                )
                context["recent_empirical_ctr"] = round((
                    recent_clicks[source]
                    + recent_ctr_prior_weight * context["empirical_ctr"]
                ) / (
                    recent_exposures[source] + recent_ctr_prior_weight
                ), 8)
                referenced.add(article_id)
                events.append({
                    "user": user, "article": article_id,
                    "action": _action(
                        source in clicked, row.get("next_read_time"),
                        row.get("next_scroll_percentage"),
                    ),
                    "impression": impression,
                    "source_impression_id": str(row["impression_id"]),
                    "timestamp": row["impression_time"].isoformat(),
                    "position": position, "history": list(safe_history),
                    "read_time": row.get("next_read_time"),
                    "scroll_percentage": row.get("next_scroll_percentage"),
                    **context,
                })
        exposures.update(candidates)
        clicks.update(source for source in clicked if source in articles)
        recent_exposures.update(candidates)
        recent_clicks.update(source for source in clicked if source in articles)
        recent_rows.append((
            row["impression_time"], tuple(candidates),
            tuple(source for source in clicked if source in articles),
        ))
        history.extend(source for source in row["article_ids_clicked"] or [] if int(source) in articles)
        del history[:-history_limit]

    for row in valid_rows:
        user_source = int(row["user_id"])
        history = valid_history.setdefault(user_source, [])
        candidates = [int(value) for value in row["article_ids_inview"] if int(value) in articles]
        clicked = {int(value) for value in row["article_ids_clicked"] or []}
        if int(row["impression_id"]) in selected_valid and candidates and clicked:
            user = safe_metta_symbol(user_source, "user")
            safe_history = _safe_history(history, articles, history_limit)
            contexts = {
                articles[source]["id"]: _context(
                    source, history[-history_limit:], articles, by_id, idf,
                    row["impression_time"].hour,
                )
                for source in candidates
            }
            for source in candidates:
                contexts[articles[source]["id"]]["empirical_ctr"] = round(
                    (clicks[source] + 1.0) / (exposures[source] + 10.0), 8
                )
                contexts[articles[source]["id"]]["recent_empirical_ctr"] = round((
                    recent_clicks[source]
                    + recent_ctr_prior_weight
                    * contexts[articles[source]["id"]]["empirical_ctr"]
                ) / (
                    recent_exposures[source] + recent_ctr_prior_weight
                ), 8)
            candidate_ids = [articles[source]["id"] for source in candidates]
            relevant = [articles[source]["id"] for source in candidates if source in clicked]
            identifier = safe_metta_symbol(
                f"validation_{row['impression_id']}", "impression"
            )
            tests.append({
                "id": identifier, "source_impression_id": str(row["impression_id"]),
                "user": user, "candidates": candidate_ids, "relevant": relevant,
                "labels": {candidate: int(candidate in relevant) for candidate in candidate_ids},
                "history": safe_history, "candidate_context": contexts,
                "timestamp": row["impression_time"].isoformat(),
            })
            selected_users.add(user)
            referenced.update(safe_history)
            referenced.update(candidate_ids)
        history.extend(source for source in row["article_ids_clicked"] or [] if int(source) in articles)
        del history[:-history_limit]

    profile_histories = {
        safe_metta_symbol(user, "user"): _safe_history(history, articles, history_limit)
        for user, history in valid_history.items()
        if safe_metta_symbol(user, "user") in selected_users
    }
    users = {
        user: {
            "history": profile_histories.get(user, []),
            "topics": [topic for topic, _count in Counter(
                by_id[item]["topic"] for item in profile_histories.get(user, []) if item in by_id
            ).most_common()],
        }
        for user in sorted(selected_users)
    }
    data = {
        "users": users,
        "articles": sorted(
            (article for article in by_id.values() if article["id"] in referenced),
            key=lambda article: article["id"],
        ),
        "events": events, "tests": tests, "title_idf_model": idf,
        "metadata": {
            "dataset": "EB-NeRD-small", "root": str(root), "seed": seed,
            "sampling": "seeded hash-priority complete impressions",
            "history_policy": (
                "split-initial history plus every chronologically earlier source click"
            ),
            "history_limit": history_limit,
            "source_train_impressions": len(train_rows),
            "source_eval_impressions": len(valid_rows),
            "train_impressions_loaded": len(selected_train),
            "train_cases_loaded": len(events),
            "eval_impressions_loaded": len(tests),
            "users_loaded": len(users), "articles_loaded": len(referenced),
            "target": "clicked article; complete is a positive click subtype",
            "empirical_ctr": (
                "Beta(1,9) posterior from chronologically earlier train exposures; "
                "frozen for validation"
            ),
            "recent_empirical_ctr": (
                f"{recent_ctr_hours}-hour train window shrunk toward empirical_ctr "
                f"with prior weight {recent_ctr_prior_weight:g}; frozen for validation"
            ),
            "complete_definition": "next read >=60 seconds or next scroll >=80 percent",
        },
    }
    if content_workspace:
        retained_annotations = {
            article["id"]: annotations[article["id"]] for article in data["articles"]
        }
        data = build_llm_projection(data, data, retained_annotations)
        data["metadata"]["content_observation_source"] = (
            "publisher topics, named-entity clusters, entity groups, and article type"
        )
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                    json.dump(data, text, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-train-cases", type=int, default=DEFAULT_MAX_TRAIN_CASES)
    parser.add_argument("--max-eval-impressions", type=int, default=DEFAULT_MAX_EVAL_IMPRESSIONS)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--recent-ctr-hours", type=int, default=DEFAULT_RECENT_CTR_HOURS)
    parser.add_argument(
        "--recent-ctr-prior-weight", type=float,
        default=DEFAULT_RECENT_CTR_PRIOR_WEIGHT,
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-content-workspace", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new path")
    data = load_ebnerd(
        args.root, max_train_cases=args.max_train_cases,
        max_eval_impressions=args.max_eval_impressions,
        history_limit=args.history_limit, seed=args.seed,
        recent_ctr_hours=args.recent_ctr_hours,
        recent_ctr_prior_weight=args.recent_ctr_prior_weight,
        content_workspace=not args.no_content_workspace,
    )
    _write(args.output.resolve(), data)
    print(json.dumps({"output": str(args.output), **data["metadata"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
