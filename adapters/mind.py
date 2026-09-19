"""Bounded, deterministic adapter for the Microsoft MIND dataset.

``load_mind`` expects an extracted directory containing a training split and
either a ``valid``/``validation`` or ``dev`` split.  Split directories may be
named simply ``train`` and ``dev`` or use archive-style names such as
``MINDsmall_train`` and ``MINDsmall_dev``.

The returned mapping intentionally follows :func:`recommendation.app.server.fixture`:

``users``
    ``{safe_user_id: [history_topic, ...]}``.
``articles``
    Records containing at least ``id``, ``title``, ``topic`` and ``format``.
    MIND has no format field, so ``format`` is a bounded title-length tier;
    the original subcategory is retained separately.
``events``
    Closed training exposures containing ``user``, ``article`` and a
    ``click``/``skip`` action.  Impression, timestamp and history-affinity
    fields are retained as additional context.
``tests``
    Chronologically ordered development impressions.  Every test keeps its
    original candidate order and clicked candidates in ``relevant``.  Its
    ``candidate_context`` is derived only from the history supplied for that
    impression, preventing outcome leakage.

Raw split sampling uses independent seeded reservoirs. The streamable RecZoo
archive uses deterministic seed-controlled hash-priority sampling of complete
impressions after scanning each population, while retaining chronological
output and causal feature snapshots.
"""

from __future__ import annotations

import csv
import functools
import gzip
import hashlib
import heapq
import io
import json
import math
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..core.multi_interest import (
    PreparedMultiInterestHistory, PreparedSemanticHistory,
    build_multi_interest_facts, build_semantic_match_facts,
    prepare_multi_interest_history, prepare_semantic_history,
)
from ..features.text_embeddings import load_text_embedding_sidecar


NEWS_COLUMNS = 8
BEHAVIOR_COLUMNS = 5
DEFAULT_MAX_TRAIN_CASES = 5_000
DEFAULT_MAX_EVAL_IMPRESSIONS = 500
_RECZOO_CACHE_VERSION = "whole-impression-text-semantic-attention-v13"
_TIME_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",  # MIND-small/full canonical format
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)
_SAFE_SYMBOL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TITLE_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_TITLE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "who", "with",
}


@dataclass(frozen=True)
class HistoryFeatureWorkspace:
    """Candidate-independent causal history state for one scoring slate."""

    history_ids: tuple[str, ...]
    normalized_articles: dict[str, dict[str, Any]]
    topics: Counter[str]
    known: int
    recent_ids: tuple[str, ...]
    recent_topics: Counter[str]
    recent_known: int
    subcategories: Counter[str]
    subcategory_known: int
    recent_subcategories: tuple[str, ...]
    multi_interest: PreparedMultiInterestHistory
    text_semantic: PreparedSemanticHistory

# Deliberately fixed, low-cardinality thresholds.  Keeping them here makes the
# experiment reproducible and prevents the miner vocabulary growing with raw
# numeric values.
FEATURE_THRESHOLDS = {
    "affinity_level": "none=count 0; low=share <0.10; medium=share <0.30; high=share >=0.30",
    "history_size_bucket": "cold=0; light=1-5; regular=6-20; heavy=21+",
    "entity_overlap": "none=0 shared entities; low=1; high=2+",
    "ctr_bucket": "cold=0 prior exposures; low=smoothed CTR <0.05; medium=<0.20; high=>=0.20; Beta(1,9)",
    "freshness_bucket": "new=0 prior exposures; recent=first seen within 1000 prior exposure positions; established=older",
    "time_bucket": "night=00-05; morning=06-11; afternoon=12-17; evening=18-23",
    "recent_affinity": "same affinity thresholds over the last up-to-5 preceding history articles",
    "long_affinity": "same affinity thresholds over the full preceding history",
    "topic_affinity": "exact candidate-topic share of the full preceding history",
    "recent_topic_affinity": "exact candidate-topic share of the last up-to-5 history articles",
    "history_topic_count_bucket": "zero=0; one=1; two=2; three_plus=3+ articles of candidate topic",
    "recent_topic_count_bucket": "zero=0; one=1; two=2; three_plus=3+ recent topic articles",
    "topic_rank_bucket": "none=0; top=highest historical topic count; secondary=other present topic",
    "subcategory_affinity": "none=count 0; low=share <0.10; medium=<0.30; high>=0.30 in history",
    "subcategory_affinity_score": "exact candidate-subcategory share of the full preceding history",
    "recent_subcategory_affinity_score": "exact candidate-subcategory share of the last up-to-5 preceding history articles",
    "topic_recency_bucket": "none=no preceding match; immediate=last item; recent=2nd-5th item; older=earlier history",
    "subcategory_recency_bucket": "none=no preceding match; immediate=last item; recent=2nd-5th item; older=earlier history",
    "topic_recency_score": "1/(1+d), where d is the candidate-topic distance from the end of the preceding history; 0 when absent",
    "subcategory_recency_score": "1/(1+d), where d is the candidate-subcategory distance from the end of the preceding history; 0 when absent",
    "entity_overlap_detail": "none=0; one=1; two=2; three_plus=3+ shared entities",
    "position_bucket": "top=positions 0-1; early=2-4; middle=5-9; late=10+ in source impression",
    "title_overlap_detail": "none=0; one=1; two=2; three_plus=3+ shared non-stop title tokens",
    "title_history_idf_jaccard": "maximum train-corpus-IDF-weighted title-token Jaccard against any preceding history item; unavailable=None",
    "entity_recent_top1_similarity": "maximum cosine similarity to the last 5 history article entity vectors; unavailable=None",
    "entity_long_mean_similarity": "mean cosine similarity to all known preceding history article entity vectors; unavailable=None",
    "recent_subcategory_transition_score": "Bayesian P(click|candidate subcategory,recent history subcategory), max over the last 5; alpha=10 with candidate/global backoff",
    "text_semantic_top3_mean_similarity": "mean of the three strongest cosine similarities between a frozen candidate text vector and causally preceding history vectors; cosine is mapped to [0,1]",
    "text_semantic_attention_t8_similarity": "candidate-aware sum softmax(8 * raw cosine) * mapped cosine over every compatible causally preceding frozen text vector",
    "text_semantic_attention_t12_similarity": "candidate-aware sum softmax(12 * raw cosine) * mapped cosine over every compatible causally preceding frozen text vector",
    "text_semantic_sidecar": "offline title+abstract vectors with pinned model/content/checksum provenance; no outcomes are encoder inputs",
}


class MindDataError(ValueError):
    """Raised when an extracted MIND split violates its documented schema."""


def safe_metta_symbol(value: object, prefix: str = "id") -> str:
    """Return a stable unquoted MeTTa symbol without collision-prone cleanup.

    Already-safe identifiers retain a readable form.  If characters must be
    replaced (or the identifier is very long), a digest of the original value
    is appended so distinct source identifiers cannot collapse to one symbol.
    """

    raw = unicodedata.normalize("NFKC", str(value)).strip()
    if not raw:
        raise MindDataError("MeTTa identifiers cannot be empty")
    clean_prefix = re.sub(r"[^A-Za-z0-9_]", "_", prefix).strip("_") or "id"
    if not re.match(r"^[A-Za-z_]", clean_prefix):
        clean_prefix = "id_" + clean_prefix
    clean = re.sub(r"[^A-Za-z0-9_]", "_", raw).strip("_") or "value"
    changed = clean != raw or len(clean) > 64
    clean = clean[:64]
    if changed:
        clean += "_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    symbol = f"{clean_prefix}_{clean}"
    if not _SAFE_SYMBOL_RE.fullmatch(symbol):  # Defensive invariant.
        raise MindDataError(f"Failed to normalize MeTTa symbol: {raw!r}")
    return symbol


def normalize_label(value: object, default: str = "unknown") -> str:
    """Normalize a categorical MIND field while keeping it human-readable."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = " ".join(text.split()).casefold()
    return text or default


def _format_bucket(title: str) -> str:
    """Project MIND text into a low-cardinality editorial format feature."""

    words=len(title.split())
    if words<=7: return "short"
    if words<=14: return "medium"
    return "long"


def _affinity_level(count: int, size: int) -> str:
    if not size or not count:
        return "none"
    share = count / size
    if share < 0.10:
        return "low"
    if share < 0.30:
        return "medium"
    return "high"


def _history_size_bucket(size: int) -> str:
    if size == 0:
        return "cold"
    if size <= 5:
        return "light"
    if size <= 20:
        return "regular"
    return "heavy"


def _history_topic_count_bucket(count: int) -> str:
    if count <= 0:
        return "zero"
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    return "three_plus"


def _topic_rank_bucket(count: int, topics: Counter[str]) -> str:
    if count <= 0:
        return "none"
    return "top" if count == max(topics.values(), default=0) else "secondary"


def _history_recency_evidence(
    article: dict[str, Any],
    history: Iterable[str],
    news: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return candidate-aware sequence facts from preceding history only.

    Distances count original history positions, including unknown article IDs,
    so missing metadata cannot pull an older click into a more recent bucket.
    The values describe observed recency; they do not prescribe whether recency
    predicts engagement.  That implication remains a mining decision.
    """

    history_ids = [str(value) for value in history]
    topic = str(article.get("topic", article.get("category", "unknown")))
    subcategory = str(article.get("subcategory", "unknown"))
    topic_known = topic not in {"", "unknown"}
    subcategory_known = subcategory not in {"", "unknown"}
    topic_distance: int | None = None
    subcategory_distance: int | None = None
    for distance, history_id in enumerate(reversed(history_ids)):
        previous = news.get(history_id)
        if previous is None:
            continue
        previous_topic = str(
            previous.get("topic", previous.get("category", "unknown"))
        )
        previous_subcategory = str(previous.get("subcategory", "unknown"))
        if (topic_known and previous_topic not in {"", "unknown"}
                and topic_distance is None and previous_topic == topic):
            topic_distance = distance
        if (subcategory_known and previous_subcategory not in {"", "unknown"}
                and subcategory_distance is None
                and previous_subcategory == subcategory):
            subcategory_distance = distance
        if topic_distance is not None and subcategory_distance is not None:
            break

    def projection(distance: int | None) -> tuple[float, str]:
        if distance is None:
            return 0.0, "none"
        if distance == 0:
            bucket = "immediate"
        elif distance < 5:
            bucket = "recent"
        else:
            bucket = "older"
        return round(1.0 / (1.0 + distance), 8), bucket

    recent_window = history_ids[-5:]
    recent_subcategory_count = sum(
        history_id in news
        and str(news[history_id].get("subcategory", "unknown"))
            not in {"", "unknown"}
        and str(news[history_id].get("subcategory", "unknown")) == subcategory
        for history_id in recent_window
    )
    topic_score, topic_bucket = projection(topic_distance)
    subcategory_score, subcategory_bucket = projection(subcategory_distance)
    return {
        "recent_subcategory_affinity_score": (
            round(recent_subcategory_count / len(recent_window), 8)
            if subcategory_known and recent_window else
            0.0 if subcategory_known else None
        ),
        "topic_recency_score": topic_score if topic_known else None,
        "subcategory_recency_score": (
            subcategory_score if subcategory_known else None
        ),
        "topic_recency_bucket": topic_bucket if topic_known else "unknown",
        "subcategory_recency_bucket": (
            subcategory_bucket if subcategory_known else "unknown"
        ),
    }


def _entity_keys(value: Any) -> set[str]:
    """Extract stable entity IDs from either raw MIND JSON or RecZoo IDs."""

    if isinstance(value, str):
        normalized = normalize_label(value, "")
        return {normalized} if normalized else set()
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_entity_keys(item))
        return result
    if isinstance(value, dict):
        # WdId is stable across title/abstract mentions; Label is the safe
        # fallback used by a few older MIND exports.
        for key in ("WikidataId", "WikidataID", "WdId", "EntityId", "Label"):
            if value.get(key):
                return _entity_keys(str(value[key]))
    return set()


def _article_entities(article: dict[str, Any]) -> set[str]:
    return _entity_keys(article.get("title_entities", [])) | _entity_keys(
        article.get("abstract_entities", [])
    )


@functools.lru_cache(maxsize=131_072)
def _cached_title_tokens(title: str) -> frozenset[str]:
    """Tokenize immutable title text once across replay candidates.

    The cache has a fixed upper bound, so a production corpus cannot create an
    unbounded process-global vocabulary.  These are lexical evidence tokens,
    not labels or dataset-specific categories.
    """

    normalized = normalize_label(title, "")
    return frozenset(
        token for token in _TITLE_WORD_RE.findall(normalized)
        if len(token) > 2 and token not in _TITLE_STOPWORDS
    )


def _title_tokens(article: dict[str, Any]) -> frozenset[str]:
    return _cached_title_tokens(str(article.get("title", "")))


def fit_title_idf_model(
    articles: dict[str, dict[str, Any]],
    exposed_article_ids: Iterable[str],
) -> dict[str, Any]:
    """Fit a JSON-safe smoothed IDF model on unique training exposures only.

    ``articles`` and ``exposed_article_ids`` must use the same ID namespace.
    Each exposed article contributes one document regardless of how often the
    logging policy exposed it.  This prevents popular items from redefining
    lexical rarity merely because they were shown more often.
    """

    document_ids = sorted({str(value) for value in exposed_article_ids})
    documents = [articles[item_id] for item_id in document_ids if item_id in articles]
    document_frequency: Counter[str] = Counter()
    for article in documents:
        document_frequency.update(_title_tokens(article))
    document_count = len(documents)
    if not document_count:
        return {
            "version": "smooth-log-v1",
            "document_count": 0,
            "idf": {},
            "default_idf": None,
        }
    idf = {
        token: round(math.log((document_count + 1.0) / (frequency + 1.0)) + 1.0, 12)
        for token, frequency in sorted(document_frequency.items())
    }
    return {
        "version": "smooth-log-v1",
        "document_count": document_count,
        "idf": idf,
        # OOV tokens are still comparable at serving time, but their weight is
        # determined solely by the training document count.
        "default_idf": round(math.log(document_count + 1.0) + 1.0, 12),
    }


def title_history_idf_jaccard(
    article: dict[str, Any],
    history: Iterable[str],
    articles: dict[str, dict[str, Any]],
    model: dict[str, Any] | None,
) -> float | None:
    """Return maximum IDF-weighted title Jaccard against preceding history.

    ``None`` means the model, candidate tokens, or usable history text is
    missing.  ``0.0`` is retained as real evidence when comparable titles are
    present but lexically disjoint.
    """

    if not model or int(model.get("document_count", 0) or 0) <= 0:
        return None
    candidate_tokens = _title_tokens(article)
    if not candidate_tokens:
        return None
    weights = model.get("idf")
    default_weight = model.get("default_idf")
    if not isinstance(weights, dict) or not isinstance(default_weight, (int, float)):
        return None
    default_weight = float(default_weight)
    if not math.isfinite(default_weight) or default_weight <= 0.0:
        return None

    best: float | None = None
    for history_id in history:
        previous = articles.get(str(history_id))
        if previous is None:
            continue
        previous_tokens = _title_tokens(previous)
        if not previous_tokens:
            continue
        union = candidate_tokens | previous_tokens
        denominator = math.fsum(
            float(weights.get(token, default_weight)) for token in union
        )
        if not math.isfinite(denominator) or denominator <= 0.0:
            continue
        numerator = math.fsum(
            float(weights.get(token, default_weight))
            for token in candidate_tokens & previous_tokens
        )
        score = max(0.0, min(1.0, numerator / denominator))
        best = score if best is None else max(best, score)
    return None if best is None else round(best, 8)


def _title_overlap_detail(
    article: dict[str, Any], history: Iterable[str], news: dict[str, dict[str, Any]]
) -> str:
    history_tokens: set[str] = set()
    for source_id in history:
        previous = news.get(source_id)
        if previous is not None:
            history_tokens.update(_title_tokens(previous))
    overlap = len(_title_tokens(article) & history_tokens)
    if overlap <= 0:
        return "none"
    if overlap == 1:
        return "one"
    if overlap == 2:
        return "two"
    return "three_plus"


def _entity_overlap_detail(
    article: dict[str, Any], history: Iterable[str], news: dict[str, dict[str, Any]]
) -> str:
    history_entities: set[str] = set()
    for source_id in history:
        previous = news.get(source_id)
        if previous is not None:
            history_entities.update(_article_entities(previous))
    overlap = len(_article_entities(article) & history_entities)
    if overlap <= 0:
        return "none"
    if overlap == 1:
        return "one"
    if overlap == 2:
        return "two"
    return "three_plus"


def _hour_value(raw: object) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw if 0 <= raw <= 23 else None
    text = str(raw or "").strip().upper().replace(" ", "")
    match = re.fullmatch(r"(\d{1,2})(?::\d{2})?(AM|PM)?", text)
    if not match:
        return None
    hour = int(match.group(1))
    suffix = match.group(2)
    if suffix:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if suffix == "PM" else 0)
    return hour if 0 <= hour <= 23 else None


def _time_bucket(raw: object) -> str:
    hour = _hour_value(raw)
    if hour is None:
        return "unknown"
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _ctr_bucket(exposures: int, clicks: int) -> str:
    if exposures == 0:
        return "cold"
    smoothed = (clicks + 1) / (exposures + 10)  # Beta(1, 9) prior.
    if smoothed < 0.05:
        return "low"
    if smoothed < 0.20:
        return "medium"
    return "high"


def _freshness_bucket(source_id: str, first_seen: dict[str, int], sequence: int) -> str:
    first = first_seen.get(source_id)
    if first is None:
        return "new"
    return "recent" if sequence - first <= 1_000 else "established"


def _position_bucket(position: int | None) -> str:
    """Project source-impression order into a small, pre-click feature.

    This is useful for replay evaluation because MIND candidates retain the
    position at which the original system exposed them.  Live candidates have
    no source position and therefore receive ``unknown``; the feature is kept
    in an opt-in accuracy profile rather than silently affecting live ranking.
    """

    if position is None or position < 0:
        return "unknown"
    if position < 2:
        return "top"
    if position < 5:
        return "early"
    if position < 10:
        return "middle"
    return "late"


def _positive_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None")
    return value


def _parse_time(raw: str, path: Path, line_number: int) -> datetime:
    value = raw.strip()
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise MindDataError(f"{path}:{line_number}: unsupported MIND timestamp {raw!r}")


def _split_kind(name: str) -> str | None:
    tokens = set(filter(None, re.split(r"[^a-z0-9]+", name.casefold())))
    if "train" in tokens or name.casefold() == "train":
        return "train"
    if tokens.intersection({"valid", "validation", "dev"}):
        return "eval"
    return None


def _resolve_splits(root: Path) -> tuple[Path, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"MIND root is not a directory: {root}")

    candidates: list[tuple[str, Path]] = []
    for behaviors in root.rglob("behaviors.tsv"):
        directory = behaviors.parent
        if (directory / "news.tsv").is_file():
            kind = _split_kind(directory.name)
            if kind:
                candidates.append((kind, directory))

    def choose(kind: str) -> Path:
        matches = [path for candidate_kind, path in candidates if candidate_kind == kind]
        if not matches:
            expected = "train" if kind == "train" else "valid/dev"
            raise FileNotFoundError(
                f"No {expected} MIND split with news.tsv and behaviors.tsv under {root}"
            )
        # Prefer conventional exact names, then the shallowest deterministic path.
        preferred = {
            "train": {"train": 0},
            "eval": {"valid": 0, "validation": 1, "dev": 2},
        }[kind]
        return min(
            matches,
            key=lambda path: (
                preferred.get(path.name.casefold(), 3),
                len(path.relative_to(root).parts),
                str(path),
            ),
        )

    return choose("train"), choose("eval")


def _parse_entities(raw: str, path: Path, line_number: int, field: str) -> list[Any]:
    if not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MindDataError(f"{path}:{line_number}: invalid {field} JSON") from exc
    if not isinstance(value, list):
        raise MindDataError(f"{path}:{line_number}: {field} must be a JSON list")
    return value


def _read_news(path: Path, split_name: str) -> dict[str, dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, 1):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) != NEWS_COLUMNS:
                raise MindDataError(
                    f"{path}:{line_number}: expected {NEWS_COLUMNS} news columns, got {len(row)}"
                )
            source_id, category, subcategory, title, abstract, url, title_json, abstract_json = row
            source_id = source_id.strip()
            if not source_id:
                raise MindDataError(f"{path}:{line_number}: empty news ID")
            topic = normalize_label(category)
            normalized_subcategory = normalize_label(subcategory, "news")
            article = {
                "id": safe_metta_symbol(source_id, "article"),
                "source_id": source_id,
                "title": unicodedata.normalize("NFKC", title).strip() or source_id,
                "abstract": unicodedata.normalize("NFKC", abstract).strip(),
                "url": url.strip(),
                "topic": topic,
                "category": topic,
                "subcategory": normalized_subcategory,
                # MIND has no editorial-format column. A title-length tier is
                # both observable before the click and bounded enough for the
                # symbolic miner; the original subcategory remains available.
                "format": _format_bucket(title),
                "title_entities": _parse_entities(title_json, path, line_number, "title entities"),
                "abstract_entities": _parse_entities(abstract_json, path, line_number, "abstract entities"),
                "source_splits": [split_name],
            }
            previous = articles.get(source_id)
            if previous and any(
                previous[key] != article[key] for key in ("topic", "subcategory", "title")
            ):
                raise MindDataError(f"{path}:{line_number}: conflicting duplicate news ID {source_id}")
            articles[source_id] = previous or article
    if not articles:
        raise MindDataError(f"No news records found in {path}")
    return articles


def _merge_news(*sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for source_id, article in source.items():
            previous = merged.get(source_id)
            if previous is None:
                merged[source_id] = article
                continue
            if any(previous[key] != article[key] for key in ("topic", "subcategory", "title")):
                raise MindDataError(f"Conflicting news metadata across splits for {source_id}")
            previous["source_splits"] = sorted(
                set(previous["source_splits"] + article["source_splits"])
            )
            for key in ("abstract", "url", "title_entities", "abstract_entities"):
                if not previous[key] and article[key]:
                    previous[key] = article[key]
    return merged


def _parse_impressions(raw: str, path: Path, line_number: int) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for token in raw.split():
        if "-" not in token:
            raise MindDataError(f"{path}:{line_number}: unlabeled impression token {token!r}")
        source_id, label = token.rsplit("-", 1)
        if not source_id or label not in {"0", "1"}:
            raise MindDataError(f"{path}:{line_number}: invalid impression token {token!r}")
        results.append((source_id, int(label)))
    if not results:
        raise MindDataError(f"{path}:{line_number}: impression has no candidates")
    return results


def _history_context(
    history: Iterable[str], news: dict[str, dict[str, Any]]
) -> tuple[Counter[str], int]:
    topics: Counter[str] = Counter()
    known = 0
    for source_id in history:
        article = news.get(source_id)
        if article is not None:
            topic = str(article.get("topic", article.get("category", "unknown")))
            if topic not in {"", "unknown"}:
                topics[topic] += 1
                known += 1
    return topics, known


def _subcategory_context(
    history: Iterable[str], news: dict[str, dict[str, Any]]
) -> tuple[Counter[str], int]:
    subcategories: Counter[str] = Counter()
    known = 0
    for source_id in history:
        article = news.get(source_id)
        if article is not None:
            subcategory = str(article.get("subcategory", "unknown"))
            if subcategory not in {"", "unknown"}:
                subcategories[subcategory] += 1
                known += 1
    return subcategories, known


def _affinity(article: dict[str, Any], topics: Counter[str], known: int) -> dict[str, Any]:
    count = topics[article["topic"]]
    return {
        "affinity": "high" if count else "low",
        "topic_affinity": round(count / known, 8) if known else 0.0,
        "history_topic_count": count,
        "history_size": known,
    }


def _feature_context(
    article: dict[str, Any],
    *,
    history: Iterable[str],
    news: dict[str, dict[str, Any]],
    topics: Counter[str],
    known: int,
    recent_topics: Counter[str] | None,
    recent_known: int | None,
    hour: object,
    exposures: Counter[str],
    clicks: Counter[str],
    first_seen: dict[str, int],
    sequence: int,
    subcategories: Counter[str] | None = None,
    subcategory_known: int | None = None,
    position: int | None = None,
    title_history_idf_jaccard: float | None = None,
    entity_recent_top1_similarity: float | None = None,
    entity_long_mean_similarity: float | None = None,
    recent_subcategory_transition_score: float | None = None,
) -> dict[str, Any]:
    """Build only pre-outcome categorical features for one candidate."""

    history = [str(value) for value in history]
    source_id = article["source_id"]
    count = topics[article["topic"]]
    recent_topics = recent_topics if recent_topics is not None else topics
    recent_known = recent_known if recent_known is not None else known
    recent_count = recent_topics[article["topic"]]
    subcategories = subcategories if subcategories is not None else Counter()
    subcategory_known = subcategory_known if subcategory_known is not None else 0
    subcategory_count = subcategories[article["subcategory"]]
    overlap_detail = _entity_overlap_detail(article, history, news)
    overlap_bucket = {"none": "none", "one": "low", "two": "high", "three_plus": "high"}[overlap_detail]
    title_overlap_detail = _title_overlap_detail(article, history, news)
    sequence_evidence = _history_recency_evidence(article, history, news)
    return {
        **_affinity(article, topics, known),
        "subcategory": article["subcategory"],
        "affinity_level": _affinity_level(count, known),
        "recent_affinity": _affinity_level(recent_count, recent_known),
        "recent_topic_affinity": round(recent_count/recent_known,8)
                                 if recent_known else 0.0,
        "long_affinity": _affinity_level(count, known),
        "history_size_bucket": _history_size_bucket(known),
        "entity_overlap": overlap_bucket,
        "entity_overlap_detail": overlap_detail,
        "history_topic_count_bucket": _history_topic_count_bucket(count),
        "recent_topic_count_bucket": _history_topic_count_bucket(recent_count),
        "topic_rank_bucket": _topic_rank_bucket(count, topics),
        "subcategory_affinity": _affinity_level(subcategory_count, subcategory_known),
        "subcategory_affinity_score": round(
            subcategory_count/subcategory_known,8
        ) if subcategory_known else 0.0,
        "time_bucket": _time_bucket(hour),
        "ctr_bucket": _ctr_bucket(exposures[source_id], clicks[source_id]),
        "freshness_bucket": _freshness_bucket(source_id, first_seen, sequence),
        "position_bucket": _position_bucket(position),
        "title_overlap_detail": title_overlap_detail,
        "title_history_idf_jaccard": title_history_idf_jaccard,
        "entity_recent_top1_similarity": entity_recent_top1_similarity,
        "entity_long_mean_similarity": entity_long_mean_similarity,
        "recent_subcategory_transition_score": recent_subcategory_transition_score,
        **sequence_evidence,
    }


def _reservoir_add(
    reservoir: list[Any], item: Any, seen: int, limit: int | None, rng: random.Random
) -> int:
    seen += 1
    if limit is None or len(reservoir) < limit:
        reservoir.append(item)
    else:
        replacement = rng.randrange(seen)
        if replacement < limit:
            reservoir[replacement] = item
    return seen


def _behavior_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, 1):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) != BEHAVIOR_COLUMNS:
                raise MindDataError(
                    f"{path}:{line_number}: expected {BEHAVIOR_COLUMNS} behavior columns, got {len(row)}"
                )
            impression_id, source_user, raw_time, raw_history, raw_impressions = row
            if not impression_id.strip() or not source_user.strip():
                raise MindDataError(f"{path}:{line_number}: empty impression or user ID")
            yield (
                line_number,
                impression_id.strip(),
                source_user.strip(),
                _parse_time(raw_time, path, line_number),
                raw_history.split(),
                _parse_impressions(raw_impressions, path, line_number),
            )


def _topic_list(article_ids: Iterable[str], news: dict[str, dict[str, Any]]) -> list[str]:
    counts = Counter(
        news[source_id]["topic"] for source_id in set(article_ids) if source_id in news
    )
    return [topic for topic, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _caret_values(raw: str) -> list[str]:
    return [value for value in raw.split("^") if value]


def _reczoo_news(archive: zipfile.ZipFile) -> dict[str, dict[str, Any]]:
    """Read the public RecZoo projection of MIND-small from its archive."""

    articles: dict[str, dict[str, Any]] = {}
    with archive.open("news_corpus.tsv") as binary:
        reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"), delimiter="\t")
        expected = {
            "news_id", "cat", "sub_cat", "title_entities",
            "abstract_entities", "title", "abstract",
        }
        if set(reader.fieldnames or ()) != expected:
            raise MindDataError(
                "news_corpus.tsv has an unsupported RecZoo MIND schema: "
                f"{reader.fieldnames}"
            )
        for line_number, row in enumerate(reader, 2):
            source_id = (row["news_id"] or "").strip()
            if not source_id:
                raise MindDataError(f"news_corpus.tsv:{line_number}: empty news ID")
            topic = normalize_label(row["cat"])
            subcategory = normalize_label(row["sub_cat"], "news")
            articles[source_id] = {
                "id": safe_metta_symbol(source_id, "article"),
                "source_id": source_id,
                "title": unicodedata.normalize("NFKC", row["title"] or "").strip() or source_id,
                "abstract": unicodedata.normalize("NFKC", row["abstract"] or "").strip(),
                "url": "",
                "topic": topic,
                "category": topic,
                "subcategory": subcategory,
                "format": _format_bucket(row["title"] or ""),
                "title_entities": _caret_values(row["title_entities"] or ""),
                "abstract_entities": _caret_values(row["abstract_entities"] or ""),
                "source_splits": ["train", "valid"],
            }
    if not articles:
        raise MindDataError("news_corpus.tsv contains no news records")
    return articles


def _reczoo_article_entity_vectors(
    archive: zipfile.ZipFile,
    news: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build normalized article vectors from the archive's entity embedding HDF5.

    The HDF5 payload is intentionally opened from memory so the large public
    archive remains streamable and never needs extraction. Articles without a
    known title/abstract entity, archives without the optional embedding file,
    and environments without h5py simply have no vector and therefore expose
    an unavailable similarity rather than a fabricated value.
    """

    embedding_name = "entity_emb_dim100.h5"
    if embedding_name not in archive.namelist():
        return {}
    try:
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return {}

    try:
        payload = io.BytesIO(archive.read(embedding_name))
        with h5py.File(payload, "r") as embeddings:
            if "key" not in embeddings or "value" not in embeddings:
                raise MindDataError(
                    f"{embedding_name} must contain key and value datasets"
                )
            raw_keys = embeddings["key"][:]
            values = np.asarray(embeddings["value"][:], dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise MindDataError(f"Invalid {embedding_name} in MIND archive") from exc

    if values.ndim != 2 or len(raw_keys) != values.shape[0]:
        raise MindDataError(
            f"{embedding_name} has incompatible key/value dimensions"
        )
    entity_rows: dict[str, int] = {}
    for index, raw_key in enumerate(raw_keys):
        if isinstance(raw_key, bytes):
            key = raw_key.decode("utf-8")
        else:
            key = str(raw_key)
        entity_rows[normalize_label(key, "")] = index

    article_vectors: dict[str, Any] = {}
    for source_id, article in news.items():
        rows = [
            entity_rows[entity_id]
            for entity_id in _article_entities(article)
            if entity_id in entity_rows
        ]
        if not rows:
            continue
        mean_vector = values[rows].mean(axis=0)
        norm = float(np.linalg.norm(mean_vector))
        if not math.isfinite(norm) or norm <= 0.0:
            continue
        article_vectors[source_id] = mean_vector / norm
    return article_vectors


def _prepared_vector(
    values: Iterable[float],
) -> tuple[tuple[float, ...], float] | None:
    try:
        normalized = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not normalized or not all(math.isfinite(value) for value in normalized):
        return None
    norm = math.sqrt(math.fsum(value * value for value in normalized))
    return (normalized, norm) if norm > 0.0 else None


def _prepared_cosine_similarity(
    left: tuple[tuple[float, ...], float],
    right: tuple[tuple[float, ...], float],
) -> float | None:
    left_values, left_norm = left
    right_values, right_norm = right
    if len(left_values) != len(right_values):
        return None
    cosine = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    ) / (left_norm * right_norm)
    return max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else None


def _entity_similarity_evidence(
    source_article: str,
    history: Iterable[str],
    article_vectors: dict[str, Any],
) -> tuple[float | None, float | None]:
    """Compute recent-maximum and long-mean cosine in one history pass."""

    history_ids = [str(value) for value in history]
    candidate = article_vectors.get(str(source_article))
    prepared_candidate = (
        None if candidate is None else _prepared_vector(candidate)
    )
    if prepared_candidate is None:
        return None, None
    recent_start = max(0, len(history_ids) - 5)
    all_similarities: list[float] = []
    recent_similarities: list[float] = []
    for index, history_id in enumerate(history_ids):
        previous = article_vectors.get(history_id)
        prepared_previous = (
            None if previous is None else _prepared_vector(previous)
        )
        if prepared_previous is None:
            continue
        similarity = _prepared_cosine_similarity(
            prepared_candidate, prepared_previous
        )
        if similarity is None:
            continue
        all_similarities.append(similarity)
        if index >= recent_start:
            recent_similarities.append(similarity)
    recent = (
        None if not recent_similarities else round(max(recent_similarities), 8)
    )
    long_mean = (
        None
        if not all_similarities
        else round(math.fsum(all_similarities) / len(all_similarities), 8)
    )
    return recent, long_mean


def entity_long_mean_similarity(
    source_article: str,
    history: Iterable[str],
    article_vectors: dict[str, Any],
) -> float | None:
    """Return mean cosine against every known preceding history vector.

    Article embeddings are fixed metadata evidence.  Missing or malformed
    candidate/history vectors produce no fact instead of a fabricated zero.
    """

    _recent, long_mean = _entity_similarity_evidence(
        str(source_article), history, article_vectors
    )
    return long_mean


def subcategory_transition_score(
    candidate_subcategory: object,
    recent_history_subcategories: Iterable[object],
    model: dict[str, Any] | None,
) -> float | None:
    """Return the learned recent-subcategory transition posterior.

    The model is fitted exclusively from training rows.  A candidate-specific
    Bayesian posterior is the backoff; each unique subcategory among the last
    five clicked history items can refine it, and the strongest learned
    transition is retained.  This is a numeric premise—not a recommendation
    rule.  fpMiner still has to discover its directional implication and
    PeTTaChainer still has to prove that implication for a grounded pair.
    """

    if not model:
        return None
    candidate = normalize_label(candidate_subcategory)
    candidate_scores = model.get("candidate", {})
    backoff = candidate_scores.get(candidate, model.get("global"))
    if backoff is None:
        return None
    transition_scores = model.get("transition", {})
    scores = [float(backoff)]
    for history_subcategory in dict.fromkeys(
        normalize_label(value) for value in list(recent_history_subcategories)[-5:]
    ):
        value = transition_scores.get(history_subcategory, {}).get(candidate)
        if value is not None:
            scores.append(float(value))
    return round(max(scores), 8)


def prepare_history_feature_workspace(
    history: Iterable[str],
    articles: dict[str, dict[str, Any]],
    *,
    entity_vectors: dict[str, Iterable[float]] | None = None,
    text_semantic_vectors: dict[str, Iterable[float]] | None = None,
) -> HistoryFeatureWorkspace:
    """Prepare history-only evidence once for all candidates in a slate."""
    history_ids=tuple(str(value) for value in history)
    normalized={}
    for item_id in dict.fromkeys(history_ids):
        value=articles.get(item_id)
        if value is None:
            continue
        normalized[item_id]={
            **value,"source_id":item_id,
            "topic":value.get("topic",value.get("category","unknown")),
            "subcategory":value.get("subcategory","unknown"),
        }
    topics,known=_history_context(history_ids,normalized)
    recent_ids=history_ids[-5:]
    recent_topics,recent_known=_history_context(recent_ids,normalized)
    subcategories,subcategory_known=_subcategory_context(
        history_ids,normalized
    )
    recent_subcategories=tuple(
        normalized[item_id].get("subcategory","unknown")
        for item_id in recent_ids if item_id in normalized
    )
    return HistoryFeatureWorkspace(
        history_ids=history_ids,normalized_articles=normalized,
        topics=topics,known=known,recent_ids=recent_ids,
        recent_topics=recent_topics,recent_known=recent_known,
        subcategories=subcategories,subcategory_known=subcategory_known,
        recent_subcategories=recent_subcategories,
        multi_interest=prepare_multi_interest_history(
            history_ids,normalized,semantic_vectors=entity_vectors or {}
        ),
        text_semantic=prepare_semantic_history(
            history_ids,text_semantic_vectors or {}
        ),
    )


def history_feature_context(
    article: dict[str, Any],
    history: Iterable[str],
    articles: dict[str, dict[str, Any]],
    *,
    entity_vectors: dict[str, Iterable[float]] | None = None,
    text_semantic_vectors: dict[str, Iterable[float]] | None = None,
    title_idf_model: dict[str, Any] | None = None,
    transition_model: dict[str, Any] | None = None,
    hour: object = None,
    workspace: HistoryFeatureWorkspace | None = None,
) -> dict[str, Any]:
    """Build the same causal history facts for replay and live serving.

    The function consumes only an item's metadata, the user's preceding item
    IDs and models fitted on training data.  Dataset adapters can therefore
    use arbitrary item IDs and taxonomies; no MIND label is consulted here.
    """

    # Preserve every raw position before resolving metadata. Unknown IDs must
    # occupy their original slots in recent-window predicates; removing them
    # first would incorrectly pull older known items into the last five.
    history_ids=[str(value) for value in history]
    prepared=(workspace or prepare_history_feature_workspace(
        history_ids,articles,entity_vectors=entity_vectors,
        text_semantic_vectors=text_semantic_vectors,
    ))
    if tuple(history_ids)!=prepared.history_ids:
        raise ValueError("history feature workspace does not match history")
    candidate={**article}
    candidate_id=str(candidate.get("id",candidate.get("source_id","candidate")))
    candidate.setdefault("id",candidate_id)
    candidate["source_id"]=candidate_id
    candidate.setdefault("topic",normalize_label(candidate.get("category")))
    candidate.setdefault("subcategory",normalize_label(candidate.get("subcategory")))
    # Only the candidate and its history can participate in these predicates.
    # Normalizing the complete corpus for every candidate turns a linear feed
    # pass into O(corpus²) work on real datasets.
    normalized_articles=dict(prepared.normalized_articles)
    normalized_articles[candidate_id]={
        **candidate,"source_id":candidate_id,
        "topic":candidate.get("topic",candidate.get("category","unknown")),
        "subcategory":candidate.get("subcategory","unknown"),
    }
    topics,known=prepared.topics,prepared.known
    recent_ids=prepared.recent_ids
    recent_topics,recent_known=(prepared.recent_topics,
                                prepared.recent_known)
    subcategories,subcategory_known=(prepared.subcategories,
                                     prepared.subcategory_known)
    recent_subcategories=list(prepared.recent_subcategories)

    vectors=entity_vectors or {}
    recent_entity_similarity,long_entity_similarity=(
        _entity_similarity_evidence(candidate_id,history_ids,vectors)
    )

    context=_feature_context(
        candidate,history=history_ids,news=normalized_articles,
        topics=topics,known=known,recent_topics=recent_topics,
        recent_known=recent_known,hour=hour,exposures=Counter(),clicks=Counter(),
        first_seen={},sequence=0,subcategories=subcategories,
        subcategory_known=subcategory_known,
        title_history_idf_jaccard=title_history_idf_jaccard(
            candidate, history_ids, normalized_articles, title_idf_model
        ),
        entity_recent_top1_similarity=recent_entity_similarity,
        entity_long_mean_similarity=long_entity_similarity,
        recent_subcategory_transition_score=subcategory_transition_score(
            candidate.get("subcategory"),recent_subcategories,transition_model
        ),
    )
    multi_interest = build_multi_interest_facts(
        candidate,
        history_ids,
        normalized_articles,
        semantic_vectors=vectors,
        prepared_history=prepared.multi_interest,
    )
    text_semantic = build_semantic_match_facts(
        candidate_id,
        history_ids,
        text_semantic_vectors or {},
        prefix="text_semantic",
        prepared_history=prepared.text_semantic,
    )
    return {"topic":candidate["topic"],
            "format":candidate.get("format",_format_bucket(str(candidate.get("title","")))),
            "recent_history_subcategories":recent_subcategories,
            **context,
            **multi_interest,
            **text_semantic}


def _reczoo_context(
    row: dict[str, str],
    article: dict[str, Any],
    news: dict[str, dict[str, Any]],
    exposures: Counter[str],
    clicks: Counter[str],
    first_seen: dict[str, int],
    sequence: int,
    position: int | None = None,
    article_entity_vectors: dict[str, Any] | None = None,
    article_text_vectors: dict[str, Any] | None = None,
    subcategory_transition_model: dict[str, Any] | None = None,
    title_idf_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = _caret_values(row.get("news_his", ""))
    history_topics = [
        news[source_id]["topic"] for source_id in history if source_id in news
    ]
    counts = Counter(history_topics)
    history_subcategories = [
        news[source_id]["subcategory"] for source_id in history if source_id in news
    ]
    recent_ids = history[-5:]
    recent = [
        news[source_id]["topic"] for source_id in recent_ids if source_id in news
    ]
    recent_subcategories = [
        news[source_id]["subcategory"]
        for source_id in recent_ids if source_id in news
    ]
    recent_entity_similarity,long_entity_similarity=_entity_similarity_evidence(
        article["source_id"],history,article_entity_vectors or {}
    )
    context = _feature_context(
        article,
        history=history,
        news=news,
        topics=counts,
        known=len(history_topics),
        recent_topics=Counter(recent),
        recent_known=len(recent),
        hour=row.get("hour", ""),
        exposures=exposures,
        clicks=clicks,
        first_seen=first_seen,
        sequence=sequence,
        subcategories=Counter(history_subcategories),
        subcategory_known=len(history_subcategories),
        position=position,
        title_history_idf_jaccard=title_history_idf_jaccard(
            article, history, news, title_idf_model
        ),
        entity_recent_top1_similarity=recent_entity_similarity,
        entity_long_mean_similarity=long_entity_similarity,
        recent_subcategory_transition_score=subcategory_transition_score(
            article["subcategory"], recent_subcategories,
            subcategory_transition_model,
        ),
    )
    multi_interest = build_multi_interest_facts(
        {**article, "id": article["source_id"]},
        history,
        news,
        semantic_vectors=article_entity_vectors or {},
    )
    text_semantic = build_semantic_match_facts(
        article["source_id"],
        history,
        article_text_vectors or {},
        prefix="text_semantic",
    )
    return {
        "topic": article["topic"], "format": article["format"],
        "recent_history_subcategories": recent_subcategories,
        **context,
        **multi_interest,
        **text_semantic,
    }


def _reczoo_reader(archive: zipfile.ZipFile, name: str):
    binary = archive.open(name)
    text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
    reader = csv.DictReader(text)
    required = {
        "imp_id", "click", "hour", "user_id", "news_id", "cat", "sub_cat",
        "title_entities", "abstract_entities", "news_his", "cat_his", "subcat_his",
    }
    if set(reader.fieldnames or ()) != required:
        text.close()
        raise MindDataError(f"{name} has an unsupported RecZoo MIND schema: {reader.fieldnames}")
    return binary, text, reader


def _load_reczoo_archive(
    path: Path,
    *,
    max_train_cases: int | None,
    max_eval_impressions: int | None,
    seed: int,
    text_embedding_path: Path | None = None,
) -> dict[str, Any]:
    """Load the 8.58M-row public MIND-small projection without extracting 6.3 GB.

    RecZoo stores one candidate per CSV row, grouped by impression. Bounded runs
    scan the complete split and retain a deterministic seed-controlled
    hash-priority sample of whole impressions. Training contexts are captured
    against the complete preceding exposure stream, and validation priors use
    the complete scanned training population. Passing ``None`` retains the
    complete respective split.
    """

    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise MindDataError(f"Invalid MIND zip archive: {path}") from exc
    with archive:
        required_files = {"news_corpus.tsv", "train.csv", "valid.csv"}
        missing = required_files.difference(archive.namelist())
        if missing:
            raise MindDataError(
                f"Unsupported MIND archive {path}; missing {', '.join(sorted(missing))}"
            )
        news = _reczoo_news(archive)
        article_entity_vectors = _reczoo_article_entity_vectors(archive, news)
        text_embedding_sidecar = None
        article_text_vectors: dict[str, Any] = {}
        if text_embedding_path is not None:
            text_embedding_sidecar = load_text_embedding_sidecar(
                text_embedding_path
            )
            article_text_vectors = text_embedding_sidecar.as_mapping("source_id")
        title_idf_model: dict[str, Any] = {}
        train_exposed_articles: set[str] = set()
        pending_semantic_histories: dict[str, list[str]] = {}
        selected_events: list[dict[str, Any]] = []
        selected_tests: list[dict[str, Any]] = []
        profile_topics: dict[str, Counter[str]] = {}
        profile_histories: dict[str, list[str]] = {}
        referenced_articles: set[str] = set()
        train_exposures: Counter[str] = Counter()
        train_clicks: Counter[str] = Counter()
        transition_candidate_exposures: Counter[str] = Counter()
        transition_candidate_clicks: Counter[str] = Counter()
        transition_exposures: Counter[tuple[str, str]] = Counter()
        transition_clicks: Counter[tuple[str, str]] = Counter()
        transition_global_exposures = 0
        transition_global_clicks = 0
        subcategory_transition_model: dict[str, Any] = {}
        first_seen: dict[str, int] = {}
        exposure_sequence = 0
        train_interactions_scanned = 0
        train_impressions_scanned = 0

        def projected_article_id(source_id: str) -> str:
            article = news.get(source_id)
            return (
                article["id"] if article is not None
                else safe_metta_symbol(source_id, "article")
            )

        def sample_priority(split: str, impression_id: str) -> int:
            payload = f"{seed}\0{split}\0{impression_id}".encode("utf-8")
            return int.from_bytes(hashlib.sha256(payload).digest(), "big")

        binary, text, rows = _reczoo_reader(archive, "train.csv")
        try:
            # Heap entries retain the smallest hash priorities. Negated values
            # make heapq remove the worst (largest) priority first. A group
            # larger than the complete case budget can never be selected.
            sampled_train: list[tuple[int, int, int, list[dict[str, Any]]]] = []
            sampled_train_cases = 0
            group: list[dict[str, Any]] = []
            group_id: str | None = None
            group_start = 0

            def append_training_group(selected_group: list[dict[str, Any]]) -> None:
                first = selected_group[0]["row"]
                source_user = first["user_id"]
                source_impression = first["imp_id"]
                safe_user = safe_metta_symbol(source_user, "user")
                safe_impression = safe_metta_symbol(
                    f"train_{source_impression}", "impression"
                )
                source_history_ids = _caret_values(first.get("news_his", ""))
                history_ids=[projected_article_id(source_id)
                             for source_id in source_history_ids]
                profile_histories[safe_user]=history_ids
                referenced_articles.update(history_ids)
                if not title_idf_model:
                    pending_semantic_histories[safe_impression] = source_history_ids
                for item in selected_group:
                    row = item["row"]
                    source_article = item["source_article"]
                    article = news[source_article]
                    prior_exposures = Counter(
                        {source_article: item["prior_exposures"]}
                    )
                    prior_clicks = Counter({source_article: item["prior_clicks"]})
                    prior_first_seen = (
                        {}
                        if item["prior_first_seen"] is None
                        else {source_article: item["prior_first_seen"]}
                    )
                    context = _reczoo_context(
                        row,
                        article,
                        news,
                        prior_exposures,
                        prior_clicks,
                        prior_first_seen,
                        item["prior_sequence"],
                        item["position"],
                        article_entity_vectors,
                        article_text_vectors,
                        subcategory_transition_model,
                        title_idf_model,
                    )
                    # This value was captured from counters frozen before the
                    # complete impression.  Never replace it with the later
                    # full-training model used by validation/live serving.
                    context["recent_subcategory_transition_score"]=(
                        item["prior_transition_score"]
                    )
                    selected_events.append({
                        "user": safe_user,
                        "source_user_id": source_user,
                        "article": article["id"],
                        "source_article_id": source_article,
                        "action": "click" if item["label"] else "skip",
                        "impression": safe_impression,
                        "source_impression_id": source_impression,
                        "timestamp": f"train-row-{item['row_number']:09d}",
                        "hour": row["hour"],
                        "position": item["position"],
                        **context,
                    })
                    referenced_articles.add(article["id"])

            def flush_training_group() -> None:
                nonlocal exposure_sequence, sampled_train_cases
                nonlocal train_impressions_scanned, group
                nonlocal transition_global_exposures, transition_global_clicks
                if not group:
                    return
                train_impressions_scanned += 1
                first = group[0]["row"]
                source_user = first["user_id"]
                if any(item["row"]["user_id"] != source_user for item in group):
                    raise MindDataError(
                        f"train.csv:{group_start}: impression {group_id} contains multiple users"
                    )
                safe_user = safe_metta_symbol(source_user, "user")
                incoming = Counter(
                    news[source_id]["topic"]
                    for source_id in _caret_values(first.get("news_his", ""))
                    if source_id in news
                )
                if sum(incoming.values()) > sum(profile_topics.get(safe_user, Counter()).values()):
                    profile_topics[safe_user] = incoming

                # RecZoo repeats the same history on each candidate row. Parse
                # each distinct history once per impression.
                recent_by_raw_history={}
                for item in group:
                    raw_history=item["row"].get("news_his", "")
                    if raw_history not in recent_by_raw_history:
                        recent_by_raw_history[raw_history]=tuple(dict.fromkeys(
                            news[source_id]["subcategory"]
                            for source_id in _caret_values(raw_history)[-5:]
                            if source_id in news
                        ))
                    item["recent_subcategories"]=(
                        recent_by_raw_history[raw_history]
                    )

                size = len(group)
                retained=max_train_cases is None
                if max_train_cases is None:
                    pass
                elif size <= max_train_cases:
                    priority = sample_priority("train", str(group_id))
                    heapq.heappush(
                        sampled_train,
                        (-priority, -group_start, group_start, group),
                    )
                    retained=True
                    sampled_train_cases += size
                    while sampled_train_cases > max_train_cases:
                        removed = heapq.heappop(sampled_train)
                        sampled_train_cases -= len(removed[3])
                        if removed[3] is group:
                            retained=False

                if retained:
                    # Prequential target encoding: all retained rows in this
                    # impression use the identical model fitted only on
                    # preceding impressions. Population counters below still
                    # consume every row; only materialization is sample-bound.
                    prior_global_rate=(
                        transition_global_clicks/transition_global_exposures
                        if transition_global_exposures else 0.0
                    )
                    for item in group:
                        candidate=item["candidate_subcategory"]
                        candidate_exposures=(
                            transition_candidate_exposures[candidate]
                        )
                        candidate_score=(
                            transition_candidate_clicks[candidate]
                            +20.0*prior_global_rate
                        )/(candidate_exposures+20.0)
                        scores=[candidate_score]
                        for history_subcategory in item["recent_subcategories"]:
                            key=(history_subcategory,candidate)
                            exposures=transition_exposures[key]
                            scores.append((
                                transition_clicks[key]+10.0*candidate_score
                            )/(exposures+10.0))
                        item["prior_transition_score"]=round(max(scores),8)
                    if max_train_cases is None:
                        append_training_group(group)

                # Outcomes become prior evidence only after every row in the
                # impression has captured the same pre-impression snapshot.
                for item in group:
                    source_id = item["source_article"]
                    label = item["label"]
                    candidate_subcategory=item["candidate_subcategory"]
                    recent_subcategories=item["recent_subcategories"]
                    first_seen.setdefault(source_id, exposure_sequence)
                    train_exposures[source_id] += 1
                    train_clicks[source_id] += label
                    transition_global_exposures += 1
                    transition_global_clicks += label
                    transition_candidate_exposures[candidate_subcategory] += 1
                    transition_candidate_clicks[candidate_subcategory] += label
                    for history_subcategory in recent_subcategories:
                        key=(history_subcategory,candidate_subcategory)
                        transition_exposures[key] += 1
                        transition_clicks[key] += label
                    exposure_sequence += 1
                group = []

            for row_number, row in enumerate(rows, 2):
                source_article = row["news_id"]
                article = news.get(source_article)
                if article is None:
                    raise MindDataError(
                        f"train.csv:{row_number}: unknown news ID {source_article!r}"
                    )
                label = row["click"]
                if label not in {"0", "1"}:
                    raise MindDataError(f"train.csv:{row_number}: invalid click label {label!r}")
                label_value = int(label)
                source_impression = row["imp_id"]
                if group_id is None:
                    group_id = source_impression
                    group_start = row_number
                elif source_impression != group_id:
                    flush_training_group()
                    group_id = source_impression
                    group_start = row_number
                train_interactions_scanned += 1
                train_exposed_articles.add(source_article)
                group.append({
                    "row_number": row_number,
                    "row": row,
                    "source_article": source_article,
                    "label": label_value,
                    "candidate_subcategory":article["subcategory"],
                    "position": len(group),
                    # Only these candidate-specific prior values are needed to
                    # recreate the context if this group survives sampling.
                    "prior_exposures": train_exposures[source_article],
                    "prior_clicks": train_clicks[source_article],
                    "prior_first_seen": first_seen.get(source_article),
                    "prior_sequence": exposure_sequence,
                })
            flush_training_group()

            # Only candidate articles actually exposed by the training logger
            # define lexical rarity. Validation candidates and the full news
            # catalog cannot affect this model.
            title_idf_model.update(
                fit_title_idf_model(news, train_exposed_articles)
            )
            # With an explicitly unbounded training projection, groups are
            # materialized as they stream. Recompute just this model-dependent
            # fact now; the shared history is retained once per impression.
            if pending_semantic_histories:
                for event in selected_events:
                    event["title_history_idf_jaccard"] = title_history_idf_jaccard(
                        news[event["source_article_id"]],
                        pending_semantic_histories.get(event["impression"], ()),
                        news,
                        title_idf_model,
                    )
                pending_semantic_histories.clear()

            # Fit the compact transition table from the complete training
            # population before projecting either the representative training
            # sample or validation candidates.  Beta-style hierarchical
            # smoothing was selected on a later, disjoint training slice:
            # global -> candidate subcategory (alpha=20) -> recent transition
            # (alpha=10).
            global_rate=(transition_global_clicks/transition_global_exposures
                         if transition_global_exposures else 0.0)
            candidate_scores={
                candidate:(
                    transition_candidate_clicks[candidate]+20.0*global_rate
                )/(exposures+20.0)
                for candidate,exposures in transition_candidate_exposures.items()
            }
            transition_scores: dict[str, dict[str, float]] = defaultdict(dict)
            for (history_subcategory,candidate),exposures in transition_exposures.items():
                backoff=candidate_scores.get(candidate,global_rate)
                transition_scores[history_subcategory][candidate]=round(
                    (transition_clicks[(history_subcategory,candidate)]+10.0*backoff)
                    /(exposures+10.0),8
                )
            subcategory_transition_model.update({
                "global":round(global_rate,8),
                "candidate":{key:round(value,8)
                             for key,value in candidate_scores.items()},
                "transition":dict(transition_scores),
            })

            if max_train_cases is not None:
                chosen_train = [(entry[2], entry[3]) for entry in sampled_train]
                for _start, selected_group in sorted(
                    chosen_train, key=lambda item: item[0]
                ):
                    append_training_group(selected_group)

        finally:
            text.close()
            binary.close()

        eval_interactions_scanned = 0
        eval_impressions_scanned = 0
        binary, text, rows = _reczoo_reader(archive, "valid.csv")
        try:
            sampled_eval: list[
                tuple[int, int, int, list[tuple[int, dict[str, str]]]]
            ] = []
            group: list[tuple[int, dict[str, str]]] = []
            group_id: str | None = None
            group_start = 0

            def append_validation_group(
                selected_group: list[tuple[int, dict[str, str]]]
            ) -> None:
                first = selected_group[0][1]
                source_user = first["user_id"]
                source_impression = first["imp_id"]
                safe_user = safe_metta_symbol(source_user, "user")
                history_ids=[
                    projected_article_id(source_id)
                    for source_id in _caret_values(first.get("news_his",""))
                ]
                profile_histories[safe_user]=history_ids
                referenced_articles.update(history_ids)
                candidate_ids: list[str] = []
                relevant: list[str] = []
                labels: dict[str, int] = {}
                contexts: dict[str, dict[str, Any]] = {}
                for _row_number, row in selected_group:
                    source_article = row["news_id"]
                    article = news[source_article]
                    position = len(candidate_ids)
                    article_id = article["id"]
                    if article_id in labels:
                        continue
                    labels[article_id] = int(row["click"])
                    candidate_ids.append(article_id)
                    contexts[article_id] = _reczoo_context(
                        row, article, news, train_exposures, train_clicks,
                        first_seen, exposure_sequence, position,
                        article_entity_vectors,
                        article_text_vectors,
                        subcategory_transition_model,
                        title_idf_model,
                    )
                    referenced_articles.add(article_id)
                    if row["click"] == "1":
                        relevant.append(article_id)
                if not relevant:
                    return
                incoming = Counter(
                    news[source_id]["topic"]
                    for source_id in _caret_values(first.get("news_his", ""))
                    if source_id in news
                )
                if sum(incoming.values()) > sum(profile_topics.get(safe_user, Counter()).values()):
                    profile_topics[safe_user] = incoming
                selected_tests.append({
                    "id": safe_metta_symbol(f"valid_{source_impression}", "impression"),
                    "source_impression_id": source_impression,
                    "user": safe_user,
                    "source_user_id": source_user,
                    "timestamp": f"valid-row-{selected_group[0][0]:09d}",
                    "hour": first["hour"],
                    "history": [
                        projected_article_id(source_id)
                        for source_id in _caret_values(first["news_his"])
                    ],
                    "history_topics": [
                        topic for topic, _ in sorted(
                            incoming.items(), key=lambda item: (-item[1], item[0])
                        )
                    ],
                    "candidates": candidate_ids,
                    "relevant": relevant,
                    "labels": labels,
                    "candidate_context": contexts,
                })

            def flush_validation_group() -> None:
                nonlocal eval_impressions_scanned, group
                if not group:
                    return
                eval_impressions_scanned += 1
                first = group[0][1]
                source_user = first["user_id"]
                for row_number, row in group:
                    if row["user_id"] != source_user:
                        raise MindDataError(
                            f"valid.csv:{row_number}: impression {group_id} contains multiple users"
                        )
                    source_article = row["news_id"]
                    article = news.get(source_article)
                    if article is None:
                        raise MindDataError(
                            f"valid.csv:{row_number}: unknown news ID {source_article!r}"
                        )
                    label = row["click"]
                    if label not in {"0", "1"}:
                        raise MindDataError(
                            f"valid.csv:{row_number}: invalid click label {label!r}"
                        )
                if not any(row["click"] == "1" for _row_number, row in group):
                    group = []
                    return
                if max_eval_impressions is None:
                    append_validation_group(group)
                else:
                    priority = sample_priority("valid", str(group_id))
                    heapq.heappush(
                        sampled_eval,
                        (-priority, -group_start, group_start, group),
                    )
                    if len(sampled_eval) > max_eval_impressions:
                        heapq.heappop(sampled_eval)
                group = []

            for row_number, row in enumerate(rows, 2):
                incoming_id = row["imp_id"]
                if group_id is None:
                    group_id = incoming_id
                    group_start = row_number
                elif incoming_id != group_id:
                    flush_validation_group()
                    group_id = incoming_id
                    group_start = row_number
                eval_interactions_scanned += 1
                group.append((row_number, row))
            flush_validation_group()

            if max_eval_impressions is not None:
                chosen_eval = [(entry[2], entry[3]) for entry in sampled_eval]
                for _start, selected_group in sorted(
                    chosen_eval, key=lambda item: item[0]
                ):
                    append_validation_group(selected_group)
        finally:
            text.close()
            binary.close()

    if not selected_events:
        raise MindDataError("train.csv contains no selected training cases")
    if not selected_tests:
        raise MindDataError("valid.csv contains no selected labeled impressions")
    selected_users = {event["user"] for event in selected_events}
    selected_users.update(test["user"] for test in selected_tests)
    recent_by_user: dict[str, list[str]] = {}
    article_by_safe_id={article["id"]:article for article in news.values()}
    for event in selected_events:
        recent_by_user[event["user"]] = list(
            event.get("recent_history_subcategories", [])
        )[-5:]
    for test in selected_tests:
        recent_by_user[test["user"]] = [
            article_by_safe_id[article_id]["subcategory"]
            for article_id in test.get("history", [])
            if article_id in article_by_safe_id
        ][-5:]
    users = {
        user: {
            "topics":[topic for topic, _ in sorted(
                profile_topics.get(user, Counter()).items(),
                key=lambda item: (-item[1], item[0]),
            )],
            "recent_subcategories":recent_by_user.get(user,[]),
            "history":profile_histories.get(user,[]),
        }
        for user in sorted(selected_users)
    }
    articles = sorted(
        (article for article in news.values() if article["id"] in referenced_articles),
        key=lambda article: article["id"],
    )
    serialized_vectors={
        article["id"]:[float(value) for value in article_entity_vectors[source_id]]
        for source_id,article in news.items()
        if article["id"] in referenced_articles and source_id in article_entity_vectors
    }
    return {
        "users": users,
        "articles": articles,
        "events": selected_events,
        "tests": selected_tests,
        "article_entity_vectors":serialized_vectors,
        "title_idf_model": title_idf_model,
        "subcategory_transition_model": subcategory_transition_model,
        "metadata": {
            "dataset": "MIND-small",
            "projection": "RecZoo MIND_small_x1",
            "root": str(path),
            "seed": seed,
            "sampling": "seeded hash-priority whole impressions",
            "source_train_interactions": 5_843_444,
            "source_eval_interactions": 2_740_998,
            "train_interactions_scanned": train_interactions_scanned,
            "train_impressions_scanned": train_impressions_scanned,
            "train_impressions_loaded": len({event["impression"] for event in selected_events}),
            "train_cases_loaded": len(selected_events),
            "eval_interactions_scanned": eval_interactions_scanned,
            "eval_impressions_scanned": eval_impressions_scanned,
            "eval_impressions_loaded": len(selected_tests),
            "articles_loaded": len(articles),
            "users_loaded": len(users),
            "entity_vector_articles": len(article_entity_vectors),
            "text_embedding_articles": len(article_text_vectors),
            "text_embedding_sidecar": (
                str(text_embedding_path) if text_embedding_path is not None else None
            ),
            "text_embedding_model": (
                dict(text_embedding_sidecar.metadata.get("model", {}))
                if text_embedding_sidecar is not None else None
            ),
            "text_embedding_content_sha256": (
                text_embedding_sidecar.metadata.get("content_sha256")
                if text_embedding_sidecar is not None else None
            ),
            "title_idf_documents": title_idf_model.get("document_count", 0),
            "subcategory_transition_cells": len(transition_exposures),
            "transition_training_encoding":"prequential-impression-v1",
            "format_source": "title length tier (short/medium/long)",
            "affinity_definition": "candidate category frequency in the impression's preceding history",
            "feature_thresholds": FEATURE_THRESHOLDS,
        },
    }


def load_mind(
    root: str | Path,
    *,
    max_train_cases: int | None = DEFAULT_MAX_TRAIN_CASES,
    max_eval_impressions: int | None = DEFAULT_MAX_EVAL_IMPRESSIONS,
    seed: int = 7,
    text_embedding_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load bounded MIND train/dev projections for the live recommendation lab.

    ``max_train_cases`` limits individual candidate exposures.  Evaluation is
    sampled at whole-impression granularity, so candidate sets are never
    truncated.  ``None`` disables the respective bound.
    """

    max_train_cases = _positive_limit("max_train_cases", max_train_cases)
    max_eval_impressions = _positive_limit("max_eval_impressions", max_eval_impressions)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    dataset_root = Path(root).expanduser().resolve()
    if dataset_root.is_dir() and (dataset_root / "MIND_small_x1.zip").is_file():
        dataset_root = dataset_root / "MIND_small_x1.zip"
    if dataset_root.is_file():
        if text_embedding_path is None:
            candidate = dataset_root.with_name(
                f"{dataset_root.stem}.text-embeddings.npz"
            )
            resolved_text_embedding = candidate if candidate.is_file() else None
        else:
            resolved_text_embedding = Path(text_embedding_path).expanduser().resolve()
            if not resolved_text_embedding.is_file():
                raise MindDataError(
                    f"Text embedding sidecar does not exist: {resolved_text_embedding}"
                )
        stat=dataset_root.stat()
        text_embedding_stat=(
            resolved_text_embedding.stat()
            if resolved_text_embedding is not None else None
        )
        cache_signature=json.dumps({
            "version":_RECZOO_CACHE_VERSION,"path":str(dataset_root),
            "size":stat.st_size,"mtime_ns":stat.st_mtime_ns,
            "max_train_cases":max_train_cases,
            "max_eval_impressions":max_eval_impressions,"seed":seed,
            "text_embedding_path":(
                str(resolved_text_embedding)
                if resolved_text_embedding is not None else None
            ),
            "text_embedding_size":(
                text_embedding_stat.st_size if text_embedding_stat is not None else None
            ),
            "text_embedding_mtime_ns":(
                text_embedding_stat.st_mtime_ns
                if text_embedding_stat is not None else None
            ),
        },sort_keys=True,separators=(",",":"))
        cache_key=hashlib.sha256(cache_signature.encode("utf-8")).hexdigest()[:20]
        cache_path=dataset_root.parent/f".mind-replay-{cache_key}.json.gz"
        if cache_path.is_file():
            try:
                with gzip.open(cache_path,"rt",encoding="utf-8") as handle:
                    cached=json.load(handle)
                cached["metadata"]["cache"]="available"
                cached["metadata"]["cache_path"]=str(cache_path)
                return cached
            except (OSError,json.JSONDecodeError,KeyError,TypeError):
                cache_path.unlink(missing_ok=True)
        loaded=_load_reczoo_archive(
            dataset_root,
            max_train_cases=max_train_cases,
            max_eval_impressions=max_eval_impressions,
            seed=seed,
            text_embedding_path=resolved_text_embedding,
        )
        loaded["metadata"]["cache"]="available"
        loaded["metadata"]["cache_path"]=str(cache_path)
        temporary=cache_path.with_suffix(cache_path.suffix+".tmp")
        try:
            with gzip.open(temporary,"wt",encoding="utf-8") as handle:
                json.dump(loaded,handle,ensure_ascii=False,separators=(",",":"))
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return loaded
    train_dir, eval_dir = _resolve_splits(dataset_root)
    train_news = _read_news(train_dir / "news.tsv", train_dir.name)
    eval_news = _read_news(eval_dir / "news.tsv", eval_dir.name)
    news = _merge_news(train_news, eval_news)

    train_rng = random.Random(seed ^ 0x4D494E44)
    eval_rng = random.Random(seed ^ 0x4556414C)
    selected_events: list[dict[str, Any]] = []
    selected_tests: list[dict[str, Any]] = []
    train_seen = 0
    eval_seen = 0
    skipped_eval_without_click = 0
    unknown_history_articles = 0
    train_exposures: Counter[str] = Counter()
    train_clicks: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    exposure_sequence = 0

    # Profiles used by the fixture-compatible users mapping contain only
    # training information.  Training clicks are valid pre-evaluation history.
    train_profile_articles: dict[str, set[str]] = defaultdict(set)
    train_profile_histories: dict[str, list[str]] = {}
    earliest_eval_history: dict[str, tuple[datetime, list[str]]] = {}

    train_path = train_dir / "behaviors.tsv"
    # Raw archives do not promise behavior-file order. Sorting impressions by
    # timestamp guarantees that CTR/freshness never observe a later outcome.
    train_rows = sorted(_behavior_rows(train_path), key=lambda row: (row[3], row[0]))
    title_idf_model = fit_title_idf_model(
        news,
        (
            source_article
            for _line, _impression, _user, _time, _history, candidates in train_rows
            for source_article, _label in candidates
        ),
    )
    for line_number, impression_id, source_user, timestamp, history, candidates in train_rows:
        safe_user = safe_metta_symbol(source_user, "user")
        safe_impression = safe_metta_symbol(f"train_{impression_id}", "impression")
        topics, known_history = _history_context(history, news)
        recent_topics, recent_known = _history_context(history[-5:], news)
        subcategories, subcategory_known = _subcategory_context(history, news)
        unknown_history_articles += len(history) - known_history
        train_profile_articles[safe_user].update(source_id for source_id in history if source_id in news)
        latest_profile_history = [
            news[source_id]["id"] for source_id in history if source_id in news
        ]
        for position, (source_article, label) in enumerate(candidates):
            article = news.get(source_article)
            if article is None:
                raise MindDataError(
                    f"{train_path}:{line_number}: candidate news ID {source_article!r} is absent from news.tsv"
                )
            if label:
                train_profile_articles[safe_user].add(source_article)
                latest_profile_history.append(article["id"])
            context = _feature_context(
                article,
                history=history,
                news=news,
                topics=topics,
                known=known_history,
                recent_topics=recent_topics,
                recent_known=recent_known,
                hour=timestamp.hour,
                exposures=train_exposures,
                clicks=train_clicks,
                first_seen=first_seen,
                sequence=exposure_sequence,
                subcategories=subcategories,
                subcategory_known=subcategory_known,
                position=position,
                title_history_idf_jaccard=title_history_idf_jaccard(
                    article, history, news, title_idf_model
                ),
            )
            event = {
                "user": safe_user,
                "source_user_id": source_user,
                "article": article["id"],
                "source_article_id": source_article,
                "action": "click" if label else "skip",
                "impression": safe_impression,
                "source_impression_id": impression_id,
                "timestamp": timestamp.isoformat(),
                "hour": timestamp.hour,
                "position": position,
                **context,
                "_sort": (timestamp, line_number, position),
            }
            train_seen = _reservoir_add(
                selected_events, event, train_seen, max_train_cases, train_rng
            )
        # Outcomes become prior evidence only after every candidate in the
        # impression has received its context.
        for position, (source_article, label) in enumerate(candidates):
            first_seen.setdefault(source_article, exposure_sequence)
            train_exposures[source_article] += 1
            train_clicks[source_article] += label
            exposure_sequence += 1
        train_profile_histories[safe_user] = latest_profile_history[-200:]

    eval_path = eval_dir / "behaviors.tsv"
    for line_number, impression_id, source_user, timestamp, history, candidates in _behavior_rows(eval_path):
        safe_user = safe_metta_symbol(source_user, "user")
        previous = earliest_eval_history.get(safe_user)
        known_history_ids = [source_id for source_id in history if source_id in news]
        if previous is None or timestamp < previous[0]:
            earliest_eval_history[safe_user] = (timestamp, known_history_ids)
        topics, known_history = _history_context(history, news)
        recent_topics, recent_known = _history_context(history[-5:], news)
        subcategories, subcategory_known = _subcategory_context(history, news)
        unknown_history_articles += len(history) - known_history

        candidate_ids: list[str] = []
        relevant: list[str] = []
        labels: dict[str, int] = {}
        contexts: dict[str, dict[str, Any]] = {}
        for position, (source_article, label) in enumerate(candidates):
            article = news.get(source_article)
            if article is None:
                raise MindDataError(
                    f"{eval_path}:{line_number}: candidate news ID {source_article!r} is absent from news.tsv"
                )
            article_id = article["id"]
            if article_id in labels:
                if labels[article_id] != label:
                    raise MindDataError(
                        f"{eval_path}:{line_number}: candidate {source_article!r} has conflicting labels"
                    )
                continue
            labels[article_id] = label
            candidate_ids.append(article_id)
            contexts[article_id] = _feature_context(
                article,
                history=history,
                news=news,
                topics=topics,
                known=known_history,
                recent_topics=recent_topics,
                recent_known=recent_known,
                hour=timestamp.hour,
                exposures=train_exposures,
                clicks=train_clicks,
                first_seen=first_seen,
                sequence=exposure_sequence,
                subcategories=subcategories,
                subcategory_known=subcategory_known,
                position=position,
                title_history_idf_jaccard=title_history_idf_jaccard(
                    article, history, news, title_idf_model
                ),
            )
            if label:
                relevant.append(article_id)

        # Ranking metrics such as MRR/nDCG are undefined without a positive.
        if not relevant:
            skipped_eval_without_click += 1
            continue
        test = {
            "id": safe_metta_symbol(f"{eval_dir.name}_{impression_id}", "impression"),
            "source_impression_id": impression_id,
            "user": safe_user,
            "source_user_id": source_user,
            "timestamp": timestamp.isoformat(),
            "hour": timestamp.hour,
            "history": [
                news[source_id]["id"]
                if source_id in news else safe_metta_symbol(source_id, "article")
                for source_id in history
            ],
            "history_topics": [
                topic for topic, _ in sorted(topics.items(), key=lambda item: (-item[1], item[0]))
            ],
            "candidates": candidate_ids,
            "relevant": relevant,
            "labels": labels,
            "candidate_context": contexts,
            "_sort": (timestamp, line_number),
        }
        eval_seen = _reservoir_add(
            selected_tests, test, eval_seen, max_eval_impressions, eval_rng
        )

    if not selected_events:
        raise MindDataError(f"No labeled training candidate cases found in {train_path}")
    if not selected_tests:
        raise MindDataError(f"No evaluable impressions with a clicked candidate found in {eval_path}")

    selected_events.sort(key=lambda event: event["_sort"])
    selected_tests.sort(key=lambda test: test["_sort"])
    for event in selected_events:
        event.pop("_sort")
    for test in selected_tests:
        test.pop("_sort")

    selected_users = {event["user"] for event in selected_events}
    selected_users.update(test["user"] for test in selected_tests)
    evaluation_histories: dict[str, list[str]] = {}
    for test in selected_tests:
        evaluation_histories[test["user"]] = list(test.get("history", []))[-200:]
    article_by_id = {article["id"]: article for article in news.values()}
    users: dict[str, dict[str, list[str]]] = {}
    for user in sorted(selected_users):
        profile = train_profile_articles.get(user)
        if not profile and user in earliest_eval_history:
            profile = earliest_eval_history[user][1]
        ordered_history = evaluation_histories.get(user)
        if ordered_history is None:
            ordered_history = train_profile_histories.get(user)
        if ordered_history is None and user in earliest_eval_history:
            ordered_history = [
                news[source_id]["id"]
                for source_id in earliest_eval_history[user][1]
                if source_id in news
            ][-200:]
        ordered_history = list(ordered_history or ())
        users[user] = {
            "topics": _topic_list(profile or (), news),
            "history": ordered_history,
            "recent_subcategories": [
                article_by_id[article_id]["subcategory"]
                for article_id in ordered_history[-5:]
                if article_id in article_by_id
            ],
        }

    referenced_articles = {event["article"] for event in selected_events}
    referenced_articles.update(
        article_id for test in selected_tests for article_id in test["candidates"]
    )
    referenced_articles.update(
        article_id for test in selected_tests for article_id in test.get("history", [])
    )
    for user in selected_users:
        referenced_articles.update(
            news[source_id]["id"]
            for source_id in train_profile_articles.get(user, ())
            if source_id in news
        )
    articles = sorted(
        (article for article in news.values() if article["id"] in referenced_articles),
        key=lambda article: article["id"],
    )

    return {
        "users": users,
        "articles": articles,
        "events": selected_events,
        "tests": selected_tests,
        "article_entity_vectors": {},
        "title_idf_model": title_idf_model,
        "subcategory_transition_model": {},
        "metadata": {
            "dataset": "MIND",
            "root": str(dataset_root),
            "train_split": train_dir.name,
            "eval_split": eval_dir.name,
            "seed": seed,
            "max_train_cases": max_train_cases,
            "max_eval_impressions": max_eval_impressions,
            "train_cases_seen": train_seen,
            "train_cases_loaded": len(selected_events),
            "eval_impressions_seen": eval_seen,
            "eval_impressions_loaded": len(selected_tests),
            "eval_impressions_without_click": skipped_eval_without_click,
            "unknown_history_articles": unknown_history_articles,
            "articles_loaded": len(articles),
            "title_idf_documents": title_idf_model.get("document_count", 0),
            "format_source": "title length tier (short/medium/long)",
            "affinity_definition": "candidate category frequency in the impression's preceding history",
            "feature_thresholds": FEATURE_THRESHOLDS,
        },
    }


__all__ = [
    "DEFAULT_MAX_EVAL_IMPRESSIONS",
    "DEFAULT_MAX_TRAIN_CASES",
    "HistoryFeatureWorkspace",
    "MindDataError",
    "entity_long_mean_similarity",
    "fit_title_idf_model",
    "history_feature_context",
    "load_mind",
    "normalize_label",
    "prepare_history_feature_workspace",
    "safe_metta_symbol",
    "subcategory_transition_score",
    "title_history_idf_jaccard",
]
