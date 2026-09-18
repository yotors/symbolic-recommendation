"""Dataset-agnostic, candidate-aware symbolic multi-interest features.

The builder consumes only article metadata and an ordered *prior* history.  It
never accepts outcomes, and its output contains neither article IDs nor raw
topic/entity names.  The resulting fixed-cardinality mapping can therefore be
loaded directly as ``(predicate case value)`` mining facts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

try:  # Optional fast path for dense sidecar vectors.
    import numpy as _np
except ImportError:  # The metadata-only builder remains stdlib-only.
    _np = None


SEMANTIC_ATTENTION_TEMPERATURES = (8.0, 12.0)


@dataclass(frozen=True)
class MultiInterestConfig:
    """Portable bounds for the symbolic workspace."""

    max_topic_prototypes: int = 3
    max_subcategory_prototypes: int = 3
    max_entity_prototypes: int = 4
    max_semantic_prototypes: int = 3
    recent_window: int = 20
    recency_decay: float = 0.85
    semantic_cluster_threshold: float = 0.72

    def __post_init__(self) -> None:
        for name in (
            "max_topic_prototypes", "max_subcategory_prototypes",
            "max_entity_prototypes", "max_semantic_prototypes", "recent_window",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0.0 < self.recency_decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        if not -1.0 <= self.semantic_cluster_threshold <= 1.0:
            raise ValueError("semantic_cluster_threshold must be between -1 and 1")


def _label(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.split()).casefold()


def _count_bucket(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "one"
    if count <= 3:
        return "few"
    return "many"


def _support_bucket(value: float) -> str:
    if value <= 0.0:
        return "none"
    if value < 0.10:
        return "low"
    if value < 0.30:
        return "medium"
    return "high"


def _recency_bucket(lag: int | None) -> str:
    if lag is None:
        return "none"
    if lag == 0:
        return "latest"
    if lag <= 4:
        return "recent"
    return "old"


def _similarity_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.40:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _entity_keys(value: Any) -> set[str]:
    if isinstance(value, str):
        normalized = _label(value)
        return {normalized} if normalized else set()
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(_entity_keys(item))
        return result
    if isinstance(value, Mapping):
        for key in (
            "WikidataId", "WikidataID", "WdId", "EntityId", "entity_id",
            "id", "Label", "label",
        ):
            if value.get(key):
                return _entity_keys(value[key])
    return set()


def _article_entities(article: Mapping[str, Any]) -> set[str]:
    result = _entity_keys(article.get("entities", ()))
    result.update(_entity_keys(article.get("title_entities", ())))
    result.update(_entity_keys(article.get("abstract_entities", ())))
    return result


def _resolve_history(
    history: Iterable[object], articles: Mapping[str, Mapping[str, Any]]
) -> list[tuple[int, Mapping[str, Any], str | None]]:
    raw = list(history)
    resolved: list[tuple[int, Mapping[str, Any], str | None]] = []
    size = len(raw)
    for position, item in enumerate(raw):
        article_id: str | None = None
        if isinstance(item, Mapping):
            article = item
            if item.get("id") is not None:
                article_id = str(item["id"])
        else:
            article_id = str(item)
            article = articles.get(article_id)
            if article is None:
                continue
        # Unknown IDs still occupy their original position, so recency does not
        # silently change when an adapter lacks metadata for one history item.
        resolved.append((size - position - 1, article, article_id))
    return resolved


def _metadata_prototypes(
    resolved: Sequence[tuple[int, Mapping[str, Any], str | None]],
    extractor,
    decay: float,
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "weight": 0.0, "latest_lag": None}
    )
    for lag, article, _article_id in resolved:
        values = extractor(article)
        if isinstance(values, str):
            values = (values,)
        for value in set(values or ()):
            normalized = _label(value)
            if not normalized:
                continue
            row = stats[normalized]
            row["count"] += 1
            row["weight"] += decay**lag
            row["latest_lag"] = (
                lag if row["latest_lag"] is None else min(row["latest_lag"], lag)
            )
    return [
        {"key": key, **value}
        for key, value in sorted(
            stats.items(),
            key=lambda item: (
                -item[1]["weight"], -item[1]["count"],
                item[1]["latest_lag"], item[0],
            ),
        )
    ]


def _add_discrete_slots(
    facts: dict[str, str | float],
    prefix: str,
    candidate_values: set[str],
    prototypes: Sequence[Mapping[str, Any]],
    limit: int,
    history_size: int,
    decay: float,
) -> None:
    facts[f"{prefix}_prototype_count"] = _count_bucket(len(prototypes))
    matched_rank: int | None = None
    total_weight = math.fsum(float(row["weight"]) for row in prototypes)
    matched = [row for row in prototypes if str(row["key"]) in candidate_values]
    matched_weight = math.fsum(float(row["weight"]) for row in matched)
    matched_count = sum(int(row["count"]) for row in matched)
    matched_recency = max(
        (decay ** int(row["latest_lag"]) for row in matched), default=0.0
    )
    # These candidate-conditioned summaries preserve a multi-modal history:
    # the prototype slots remain available to the miner, while the aggregate
    # scores expose how much of the complete profile supports this candidate.
    # They use no raw taxonomy/entity name and are therefore portable across
    # adapters.
    facts[f"{prefix}_candidate_affinity_score"] = round(
        matched_weight / total_weight if total_weight else 0.0, 8
    )
    facts[f"{prefix}_candidate_support_score"] = round(
        min(1.0, matched_count / max(1, history_size)), 8
    )
    facts[f"{prefix}_candidate_recency_score"] = round(matched_recency, 8)
    for index in range(limit):
        stem = f"{prefix}_slot{index + 1}"
        if index >= len(prototypes):
            facts[f"{stem}_present"] = "no"
            facts[f"{stem}_candidate_match"] = "no"
            facts[f"{stem}_support"] = "none"
            facts[f"{stem}_recency"] = "none"
            facts[f"{stem}_support_score"] = 0.0
            facts[f"{stem}_recency_score"] = 0.0
            continue
        prototype = prototypes[index]
        support = float(prototype["count"]) / max(1, history_size)
        lag = int(prototype["latest_lag"])
        matched = str(prototype["key"]) in candidate_values
        if matched and matched_rank is None:
            matched_rank = index
        facts[f"{stem}_present"] = "yes"
        facts[f"{stem}_candidate_match"] = "yes" if matched else "no"
        facts[f"{stem}_support"] = _support_bucket(support)
        facts[f"{stem}_recency"] = _recency_bucket(lag)
        facts[f"{stem}_support_score"] = round(min(1.0, support), 8)
        facts[f"{stem}_recency_score"] = round(decay**lag, 8)
    facts[f"{prefix}_candidate_match_rank"] = (
        "none" if matched_rank is None else "top" if matched_rank == 0 else "secondary"
    )


def _normalize_vector(value: object) -> tuple[float, ...] | None:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        raw = tuple(value)  # Accept NumPy/HDF5 arrays without importing them.
        vector = tuple(float(item) for item in raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not vector:
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(math.fsum(item * item for item in vector))
    if norm <= 0.0:
        return None
    return tuple(item / norm for item in vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        return None
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right))))


def _article_vector(
    article: Mapping[str, Any], article_id: str | None,
    vectors: Mapping[str, Sequence[float]] | None,
) -> tuple[float, ...] | None:
    raw = vectors.get(article_id) if vectors is not None and article_id is not None else None
    if raw is None:
        raw = article.get("semantic_vector", article.get("embedding"))
    return _normalize_vector(raw)


def _mapped_semantic_vector(
    item: object,
    vectors: Mapping[object, Sequence[float]],
) -> tuple[float, ...] | None:
    """Resolve an opaque item through a vector map without exposing its key."""
    key = item.get("id") if isinstance(item, Mapping) else item
    if key is None:
        return None
    try:
        raw = vectors.get(key)
    except (TypeError, ValueError):
        # Unhashable adapter objects are simply unavailable evidence.
        return None
    return _normalize_vector(raw)


def _unit_cosine(
    left: Sequence[float] | None,
    right: Sequence[float] | None,
) -> float | None:
    if left is None or right is None:
        return None
    cosine = _cosine(left,right)
    return None if cosine is None else max(0.0,min(1.0,(cosine+1.0)/2.0))


def _vector_centroid(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...] | None:
    if not vectors:
        return None
    width=len(vectors[0])
    if width<1 or any(len(vector)!=width for vector in vectors):
        return None
    return _normalize_vector(tuple(
        math.fsum(vector[index] for vector in vectors)/len(vectors)
        for index in range(width)
    ))


def build_semantic_match_facts(
    candidate: object,
    history: Iterable[object],
    semantic_vectors: Mapping[object, Sequence[float]],
    *,
    prefix: str = "semantic_match",
    recency_decay: float = 0.85,
) -> dict[str, str | float]:
    """Build generic candidate-relative vector facts from prior history.

    ``history`` must be ordered oldest-to-newest and must contain only events
    preceding the candidate exposure. Items can be opaque vector-map keys or
    mappings with an ``id`` key. The result never contains those keys, labels,
    outcomes, vector dimensions, or raw vectors.

    Similarities are cosine values mapped from ``[-1, 1]`` into ``[0, 1]``.
    Top-k means average the available compatible matches (up to k); missing
    vectors are represented separately by ``coverage`` rather than treated as
    zero similarity. Candidate-aware attention is calculated over every
    compatible prior vector as ``softmax(T * raw_cosine)`` and returns the
    weighted mapped similarity ``sum(alpha * (cosine + 1) / 2)`` for fixed
    inverse temperatures 8 and 12. Recent windows are sliced by interaction
    position before unavailable vectors are removed, preserving causal recency
    semantics.
    """
    if not isinstance(prefix,str) or not prefix or any(character.isspace()
                                                        for character in prefix):
        raise ValueError("prefix must be a non-empty string without whitespace")
    if not 0.0<recency_decay<=1.0:
        raise ValueError("recency_decay must be in (0, 1]")

    prior=list(history)
    if _np is not None:
        def dense_vector(item: object):
            key=item.get("id") if isinstance(item,Mapping) else item
            if key is None:
                return None
            try:
                raw=semantic_vectors.get(key)
                vector=_np.asarray(raw,dtype=_np.float64)
            except (TypeError,ValueError,OverflowError):
                return None
            if vector.ndim!=1 or vector.size<1 or not _np.isfinite(vector).all():
                return None
            norm=float(_np.linalg.norm(vector))
            return vector/norm if math.isfinite(norm) and norm>0.0 else None

        candidate_vector=dense_vector(candidate)
        resolved=[dense_vector(item) for item in prior]
        history_available=any(vector is not None for vector in resolved)
        compatible=[
            vector for vector in resolved
            if (vector is not None and candidate_vector is not None
                and vector.shape==candidate_vector.shape)
        ]
        similarities=(
            sorted(_np.clip(
                (_np.vstack(compatible)@candidate_vector+1.0)/2.0,0.0,1.0
            ).tolist(),reverse=True)
            if compatible else []
        )
    else:
        candidate_vector=_mapped_semantic_vector(candidate,semantic_vectors)
        resolved=[_mapped_semantic_vector(item,semantic_vectors) for item in prior]
        history_available=any(vector is not None for vector in resolved)
        compatible=[
            vector for vector in resolved
            if (vector is not None and candidate_vector is not None
                and len(vector)==len(candidate_vector))
        ]
        similarities=sorted(
            (similarity for similarity in (
                _unit_cosine(candidate_vector,vector) for vector in compatible
            ) if similarity is not None),
            reverse=True,
        )

    def top_mean(limit: int) -> float:
        selected=similarities[:limit]
        return math.fsum(selected)/len(selected) if selected else 0.0

    def softmax_attention(temperature: float) -> float:
        if not similarities:
            return 0.0
        # ``similarities`` are mapped to [0,1]; recover the native cosine for
        # the attention logits. Subtracting the maximum makes exp stable while
        # preserving the exact softmax ratio. Only the supplied prior history
        # participates, so this remains a causal candidate-relative sensor.
        raw_cosines=[2.0*similarity-1.0 for similarity in similarities]
        largest=max(raw_cosines)
        weights=[math.exp(temperature*(cosine-largest))
                 for cosine in raw_cosines]
        total=math.fsum(weights)
        return (
            math.fsum(weight*similarity
                      for weight,similarity in zip(weights,similarities))/total
            if total else 0.0
        )

    if _np is not None and compatible:
        centroid=_np.vstack(compatible).mean(axis=0)
        centroid_norm=float(_np.linalg.norm(centroid))
        centroid=(centroid/centroid_norm if centroid_norm>0.0 else None)
        centroid_similarity=(
            max(0.0,min(1.0,(float(candidate_vector@centroid)+1.0)/2.0))
            if centroid is not None else None
        )
    else:
        centroid=_vector_centroid(compatible)
        centroid_similarity=_unit_cosine(candidate_vector,centroid)

    recent_resolved=resolved[-5:]
    recent=[
        vector for vector in recent_resolved
        if (vector is not None and candidate_vector is not None
            and len(vector)==len(candidate_vector))
    ]
    if _np is not None and recent:
        recent_matrix=_np.vstack(recent)
        recent_similarities=_np.clip(
            (recent_matrix@candidate_vector+1.0)/2.0,0.0,1.0
        ).tolist()
        recent_centroid=recent_matrix.mean(axis=0)
        recent_norm=float(_np.linalg.norm(recent_centroid))
        recent_centroid_similarity=(
            max(0.0,min(1.0,(
                float(candidate_vector@(recent_centroid/recent_norm))+1.0
            )/2.0)) if recent_norm>0.0 else None
        )
    else:
        recent_similarities=[
            similarity for similarity in (
                _unit_cosine(candidate_vector,vector) for vector in recent
            ) if similarity is not None
        ]
        recent_centroid_similarity=_unit_cosine(
            candidate_vector,_vector_centroid(recent)
        )

    last20_resolved=resolved[-20:]
    weighted=[]
    window_size=len(last20_resolved)
    for position,vector in enumerate(last20_resolved):
        if (_np is not None and vector is not None and candidate_vector is not None
                and vector.shape==candidate_vector.shape):
            similarity=max(0.0,min(1.0,(
                float(candidate_vector@vector)+1.0
            )/2.0))
        else:
            similarity=_unit_cosine(candidate_vector,vector)
        if similarity is None:
            continue
        lag=window_size-position-1
        weighted.append((recency_decay**lag,similarity))
    weight_total=math.fsum(weight for weight,_similarity in weighted)
    recency_decayed=(
        math.fsum(weight*similarity for weight,similarity in weighted)/weight_total
        if weight_total else 0.0
    )

    def bounded(value: float | None) -> float:
        numeric=0.0 if value is None or not math.isfinite(value) else value
        return round(max(0.0,min(1.0,numeric)),8)

    facts: dict[str, str | float]={
        f"{prefix}_candidate_available":"yes" if candidate_vector is not None else "no",
        f"{prefix}_history_available":"yes" if history_available else "no",
        f"{prefix}_match_available":"yes" if similarities else "no",
        f"{prefix}_centroid_available":"yes" if centroid_similarity is not None else "no",
        f"{prefix}_recent5_available":"yes" if recent_similarities else "no",
        f"{prefix}_last20_available":"yes" if weighted else "no",
        f"{prefix}_coverage":bounded(
            len(compatible)/len(prior) if prior else 0.0
        ),
        f"{prefix}_top1_similarity":bounded(top_mean(1)),
        f"{prefix}_top3_mean_similarity":bounded(top_mean(3)),
        f"{prefix}_top5_mean_similarity":bounded(top_mean(5)),
        f"{prefix}_attention_t8_similarity":bounded(softmax_attention(8.0)),
        f"{prefix}_attention_t12_similarity":bounded(softmax_attention(12.0)),
        f"{prefix}_centroid_similarity":bounded(centroid_similarity),
        f"{prefix}_recent5_centroid_similarity":bounded(recent_centroid_similarity),
        f"{prefix}_recent5_max_similarity":bounded(
            max(recent_similarities,default=0.0)
        ),
        f"{prefix}_last20_recency_decayed_similarity":bounded(recency_decayed),
    }
    return facts


def _semantic_prototypes(
    resolved: Sequence[tuple[int, Mapping[str, Any], str | None]],
    vectors: Mapping[str, Sequence[float]] | None,
    decay: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], int]:
    clusters: list[dict[str, Any]] = []
    valid = 0
    # Oldest-to-newest makes the online clustering stable under appending a new
    # prior interaction; final ranking still rewards recent/supporting evidence.
    for lag, article, article_id in resolved:
        vector = _article_vector(article, article_id, vectors)
        if vector is None:
            continue
        valid += 1
        best_index = None
        best_similarity = -2.0
        for index, cluster in enumerate(clusters):
            similarity = _cosine(vector, cluster["centroid"])
            if similarity is not None and similarity > best_similarity:
                best_index, best_similarity = index, similarity
        weight = decay**lag
        if best_index is None or best_similarity < threshold:
            clusters.append({
                "centroid": vector, "weighted_sum": [weight * x for x in vector],
                "weight": weight, "count": 1, "latest_lag": lag,
            })
            continue
        cluster = clusters[best_index]
        if len(cluster["centroid"]) != len(vector):
            continue
        cluster["weight"] += weight
        cluster["count"] += 1
        cluster["latest_lag"] = min(cluster["latest_lag"], lag)
        cluster["weighted_sum"] = [
            current + weight * incoming
            for current, incoming in zip(cluster["weighted_sum"], vector)
        ]
        centroid = _normalize_vector(cluster["weighted_sum"])
        if centroid is not None:
            cluster["centroid"] = centroid
    clusters.sort(key=lambda row: (-row["weight"], -row["count"], row["latest_lag"]))
    return clusters, valid


def build_multi_interest_facts(
    candidate: Mapping[str, Any],
    history: Iterable[object],
    articles: Mapping[str, Mapping[str, Any]],
    *,
    semantic_vectors: Mapping[str, Sequence[float]] | None = None,
    config: MultiInterestConfig | None = None,
) -> dict[str, str | float]:
    """Return bounded candidate-relative facts derived from prior history only.

    ``history`` is ordered oldest-to-newest. Items may be article mappings or
    IDs resolved through ``articles``. Outcome/action fields are deliberately
    ignored. Numeric outputs are finite values in ``[0, 1]``; categorical
    outputs come from fixed vocabularies independent of dataset labels.
    """

    cfg = config or MultiInterestConfig()
    # Apply the window to interaction positions before metadata resolution;
    # otherwise many unknown IDs could pull stale known articles into a recent
    # profile and silently distort both support and recency.
    history_window = list(history)[-cfg.recent_window:]
    resolved = _resolve_history(history_window, articles)
    facts: dict[str, str | float] = {
        "mi_history_size": _count_bucket(len(resolved)),
    }

    topic_prototypes = _metadata_prototypes(
        resolved, lambda article: _label(article.get("topic", article.get("category"))),
        cfg.recency_decay,
    )
    subcategory_prototypes = _metadata_prototypes(
        resolved, lambda article: _label(article.get("subcategory")), cfg.recency_decay,
    )
    entity_prototypes = _metadata_prototypes(
        resolved, _article_entities, cfg.recency_decay,
    )
    _add_discrete_slots(
        facts, "mi_topic", {_label(candidate.get("topic", candidate.get("category")))},
        topic_prototypes, cfg.max_topic_prototypes, len(resolved), cfg.recency_decay,
    )
    _add_discrete_slots(
        facts, "mi_subcategory", {_label(candidate.get("subcategory"))},
        subcategory_prototypes, cfg.max_subcategory_prototypes, len(resolved),
        cfg.recency_decay,
    )
    _add_discrete_slots(
        facts, "mi_entity", _article_entities(candidate), entity_prototypes,
        cfg.max_entity_prototypes, len(resolved), cfg.recency_decay,
    )

    candidate_id = str(candidate["id"]) if candidate.get("id") is not None else None
    candidate_vector = _article_vector(candidate, candidate_id, semantic_vectors)
    semantic, valid_vectors = _semantic_prototypes(
        resolved, semantic_vectors, cfg.recency_decay, cfg.semantic_cluster_threshold
    )
    facts["mi_semantic_available"] = "yes" if candidate_vector is not None else "no"
    facts["mi_semantic_prototype_count"] = _count_bucket(len(semantic))
    # Aggregate candidate affinity over every bounded-history prototype, not
    # merely the few prototypes exported as inspectable slots.  Otherwise a
    # candidate that exactly matches a smaller secondary interest can receive
    # a misleadingly low ``top1`` score just because that interest fell beyond
    # ``max_semantic_prototypes``.
    similarities = ([] if candidate_vector is None else [
        (cosine + 1.0) / 2.0
        for prototype in semantic
        if (cosine := _cosine(candidate_vector, prototype["centroid"])) is not None
    ])
    for index in range(cfg.max_semantic_prototypes):
        stem = f"mi_semantic_slot{index + 1}"
        if index >= len(semantic):
            similarity = None
            support = 0.0
            lag = None
            present = "no"
        else:
            prototype = semantic[index]
            cosine = (_cosine(candidate_vector, prototype["centroid"])
                      if candidate_vector is not None else None)
            similarity = None if cosine is None else (cosine + 1.0) / 2.0
            support = float(prototype["count"]) / max(1, valid_vectors)
            lag = int(prototype["latest_lag"])
            present = "yes"
        facts[f"{stem}_present"] = present
        facts[f"{stem}_similarity"] = _similarity_bucket(similarity)
        facts[f"{stem}_support"] = _support_bucket(support)
        facts[f"{stem}_recency"] = _recency_bucket(lag)
        facts[f"{stem}_similarity_score"] = round(similarity or 0.0, 8)
        facts[f"{stem}_support_score"] = round(min(1.0, support), 8)
        facts[f"{stem}_recency_score"] = round(
            cfg.recency_decay**lag if lag is not None else 0.0, 8
        )
    ordered = sorted(similarities, reverse=True)
    facts["mi_semantic_top1_similarity"] = round(ordered[0] if ordered else 0.0, 8)
    facts["mi_semantic_topk_mean_similarity"] = round(
        math.fsum(ordered[:3]) / min(3, len(ordered)) if ordered else 0.0, 8
    )
    semantic_weight = math.fsum(float(row["weight"]) for row in semantic)
    weighted_similarity = []
    attention_scores = []
    for prototype in semantic:
        cosine = (_cosine(candidate_vector, prototype["centroid"])
                  if candidate_vector is not None else None)
        if cosine is None:
            continue
        similarity = (cosine + 1.0) / 2.0
        normalized_support = (
            float(prototype["weight"]) / semantic_weight if semantic_weight else 0.0
        )
        recency = cfg.recency_decay ** int(prototype["latest_lag"])
        weighted_similarity.append(normalized_support * similarity)
        # Candidate-aware attention remains an observable fact rather than a
        # prescribed click rule. PatternMiner must learn whether high, medium,
        # or low values predict the target and PeTTa must prove that rule.
        attention_scores.append(
            similarity * math.sqrt(max(0.0, normalized_support * recency))
        )
    facts["mi_semantic_weighted_similarity"] = round(
        math.fsum(weighted_similarity), 8
    )
    facts["mi_semantic_attention_score"] = round(
        max(attention_scores, default=0.0), 8
    )
    return facts


__all__ = [
    "MultiInterestConfig", "SEMANTIC_ATTENTION_TEMPERATURES",
    "build_multi_interest_facts",
    "build_semantic_match_facts",
]
