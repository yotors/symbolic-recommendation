"""Frozen-vector observations for the miner and PeTTaChainer workspace.

No model is trained to predict clicks, no labels enter this module, and no
ranking is produced here. ``fit_semantic_workspace_model`` estimates only a
background vector from *training* article vectors. Callers must freeze and
persist that model before evaluating subsequent impressions.

For unit training vectors u_j, the background is mu = sum(u_j) / n. A centered
article vector is z(x) = (unit(x) - mu) / ||unit(x) - mu||. Centered observations
use mapped cosine s = (dot(z(candidate), z(history)) + 1) / 2. They report the
mean of the strongest three available matches, the strongest last-five match,
and sum(p_i * s_i), where p_i = softmax(8 * (2*s_i - 1)). The background itself
is deliberately NOT normalized: doing so would change mean-centering.

Effective support uses the same centered cosine attention weights:
    effective_support_ratio = 1 / (n * sum(p_i**2)).
It distinguishes diffuse support (1) from one dominant neighbor (near 1/n),
without asserting which kind of support predicts a click. Only the miner can
learn that implication. Missing evidence is None, not an artificial zero.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..core.multi_interest import build_semantic_match_facts


SEMANTIC_WORKSPACE_SCHEMA = "mindplex-semantic-workspace-v1"
SEMANTIC_WORKSPACE_FEATURES = (
    "text_semantic_centered_top3_similarity",
    "text_semantic_centered_attention_t8_similarity",
    "text_semantic_centered_recent5_max_similarity",
    "text_semantic_effective_support_ratio",
)


def _unit_vector(value: object, dimensions: int | None = None) -> np.ndarray | None:
    """Resolve one finite nonzero vector; malformed observations are missing."""

    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
            return None
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if dimensions is not None and vector.size != dimensions:
        return None
    if not np.isfinite(vector).all():
        return None
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return None
    # Scaling first avoids overflow for finite, very large input coordinates.
    scaled = vector / scale
    unit = scaled / float(np.linalg.norm(scaled))
    unit[unit == 0.0] = 0.0  # Canonicalize signed zero for provenance hashing.
    return unit


def fit_semantic_workspace_model(
    training_vectors: Mapping[object, Sequence[float]] | Iterable[Sequence[float]],
) -> dict[str, Any]:
    """Fit an ID/order-independent background from training vectors only.

    Mapping keys are opaque article IDs and never enter the model or hash.
    The caller should supply one vector per training article, not one copy per
    exposure. Identical vectors belonging to distinct articles retain their
    multiplicity. Invalid/empty training input fails explicitly. Evaluation
    vectors and click labels are not arguments to this function.
    """

    values = training_vectors.values() if isinstance(training_vectors, Mapping) else training_vectors
    vectors: list[np.ndarray] = []
    dimensions: int | None = None
    for index, raw in enumerate(values):
        vector = _unit_vector(raw, dimensions)
        if vector is None:
            raise ValueError(f"training vector {index} must be finite, nonzero and dimensionally consistent")
        dimensions = int(vector.size)
        vectors.append(vector)
    if not vectors:
        raise ValueError("semantic workspace requires at least one training vector")

    # A canonical vector order makes both the floating-point reduction and
    # provenance independent of article IDs and source iteration order.
    vectors.sort(key=lambda vector: vector.astype("<f8", copy=False).tobytes())
    matrix = np.vstack(vectors)
    mean = matrix.mean(axis=0)
    digest = hashlib.sha256()
    digest.update(f"{SEMANTIC_WORKSPACE_SCHEMA}:{dimensions}:{len(vectors)}:".encode("ascii"))
    digest.update(matrix.astype("<f8", copy=False).tobytes())
    return {
        "schema": SEMANTIC_WORKSPACE_SCHEMA,
        "normalization": "unit-vector-mean-then-renormalize-centered",
        "dimensions": dimensions,
        "training_vector_count": len(vectors),
        "training_vector_sha256": digest.hexdigest(),
        "mean_vector": mean.tolist(),
        "attention_temperature": 8.0,
    }


def _model_mean(model: Mapping[str, Any]) -> np.ndarray:
    if not isinstance(model, Mapping) or model.get("schema") != SEMANTIC_WORKSPACE_SCHEMA:
        raise ValueError("unsupported semantic workspace model")
    dimensions = model.get("dimensions")
    count = model.get("training_vector_count")
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("semantic workspace dimensions must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("semantic workspace training_vector_count must be positive")
    if (model.get("normalization") != "unit-vector-mean-then-renormalize-centered"
            or model.get("attention_temperature") != 8.0):
        raise ValueError("semantic workspace formula does not match this implementation")
    try:
        raw = np.asarray(model.get("mean_vector"))
        if raw.dtype.kind not in "iuf":
            raise ValueError("non-numeric mean")
        mean = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("semantic workspace mean_vector is invalid") from exc
    if (mean.shape != (dimensions,) or not np.isfinite(mean).all()
            or float(np.linalg.norm(mean)) > 1.0 + 1e-8):
        raise ValueError("semantic workspace mean_vector is invalid")
    return mean


def _identifier(item: object) -> object:
    return item.get("id") if isinstance(item, Mapping) else item


def _bounded(value: float | None) -> float | None:
    return None if value is None else round(max(0.0, min(1.0, float(value))), 8)


def _attention_weights(raw_cosines: np.ndarray) -> np.ndarray:
    weights = np.exp(8.0 * (raw_cosines - float(raw_cosines.max())))
    return weights / float(weights.sum())


def build_semantic_workspace_facts(
    candidate_id: object,
    ordered_history_ids: Iterable[object],
    vectors_mapping: Mapping[object, Sequence[float]],
    model: Mapping[str, Any],
) -> dict[str, str | float | None]:
    """Project frozen vectors and an oldest-to-newest *prior* history to facts.

    This function never fits/refits a background, reads outcomes, or mutates
    its inputs. Missing/malformed/dimension-mismatched observations are absent
    evidence; an invalid fitted model raises ValueError. Unknown history IDs
    retain their original positions when computing recent-five evidence.
    Metadata objects are accepted only as ``{'id': opaque_id}``; all remaining
    fields, including labels, are ignored.
    """

    mean = _model_mean(model)
    dimensions = int(mean.size)
    history = [_identifier(item) for item in ordered_history_ids]
    candidate = _identifier(candidate_id)

    def resolve(key: object) -> np.ndarray | None:
        try:
            raw = vectors_mapping.get(key)
        except (TypeError, ValueError):
            return None
        return _unit_vector(raw, dimensions)

    candidate_vector = resolve(candidate)
    resolved = [resolve(item) for item in history]
    # Positional integer keys decouple the helper from the caller's ID type,
    # preserve repeated events, and avoid collisions with the candidate key.
    raw_vectors = {position: vector for position, vector in enumerate(resolved) if vector is not None}
    if candidate_vector is not None:
        raw_vectors[-1] = candidate_vector
    facts: dict[str, str | float | None] = build_semantic_match_facts(
        -1, range(len(history)), raw_vectors, prefix="text_semantic"
    )

    scalar_availability = {
        "top1_similarity": "match_available",
        "top3_mean_similarity": "match_available",
        "top5_mean_similarity": "match_available",
        "attention_t8_similarity": "match_available",
        "attention_t12_similarity": "match_available",
        "centroid_similarity": "centroid_available",
        "recent5_max_similarity": "recent5_available",
        "recent5_centroid_similarity": "recent5_available",
        "last20_recency_decayed_similarity": "last20_available",
    }
    for scalar, availability in scalar_availability.items():
        if facts[f"text_semantic_{availability}"] != "yes":
            facts[f"text_semantic_{scalar}"] = None
    recent_raw = [vector for vector in resolved[-5:] if vector is not None]
    recent_centroid = _unit_vector(np.vstack(recent_raw).mean(axis=0)) if recent_raw else None
    facts["text_semantic_recent5_centroid_available"] = (
        "yes" if candidate_vector is not None and recent_centroid is not None else "no"
    )
    if facts["text_semantic_recent5_centroid_available"] == "no":
        facts["text_semantic_recent5_centroid_similarity"] = None

    def center(vector: np.ndarray | None) -> np.ndarray | None:
        if vector is None:
            return None
        residual = vector - mean
        # Do not turn round-off in a degenerate all-identical background into
        # a spurious unit direction and apparently strong semantic evidence.
        return None if float(np.linalg.norm(residual)) <= 1e-12 else _unit_vector(residual)

    centered_candidate = center(candidate_vector)
    centered_history = [center(vector) for vector in resolved]
    centered_valid = [vector for vector in centered_history if vector is not None]
    centered_recent = [vector for vector in centered_history[-5:] if vector is not None]
    has_matches = centered_candidate is not None and bool(centered_valid)
    facts.update({
        "text_semantic_centered_candidate_available": "yes" if centered_candidate is not None else "no",
        "text_semantic_centered_history_available": "yes" if centered_valid else "no",
        "text_semantic_centered_match_available": "yes" if has_matches else "no",
        "text_semantic_centered_recent5_available": "yes" if centered_candidate is not None and centered_recent else "no",
        "text_semantic_centered_coverage": _bounded(len(centered_valid) / len(history) if history else 0.0),
        "text_semantic_centered_top3_similarity": None,
        "text_semantic_centered_attention_t8_similarity": None,
        "text_semantic_centered_recent5_max_similarity": None,
        "text_semantic_effective_support_ratio": None,
        "text_semantic_effective_support_available": "yes" if has_matches else "no",
    })
    if has_matches:
        raw_cosines = np.clip(np.vstack(centered_valid) @ centered_candidate, -1.0, 1.0)
        similarities = (raw_cosines + 1.0) / 2.0
        facts["text_semantic_centered_top3_similarity"] = _bounded(np.sort(similarities)[-3:].mean())
        attention = _attention_weights(raw_cosines)
        facts["text_semantic_centered_attention_t8_similarity"] = _bounded(float(attention @ similarities))
        facts["text_semantic_effective_support_ratio"] = _bounded(
            1.0 / (len(centered_valid) * float(attention @ attention))
        )
    if centered_candidate is not None and centered_recent:
        recent_cosines = np.clip(np.vstack(centered_recent) @ centered_candidate, -1.0, 1.0)
        facts["text_semantic_centered_recent5_max_similarity"] = _bounded((float(recent_cosines.max()) + 1.0) / 2.0)
    return facts


__all__ = [
    "SEMANTIC_WORKSPACE_FEATURES", "SEMANTIC_WORKSPACE_SCHEMA",
    "fit_semantic_workspace_model", "build_semantic_workspace_facts",
]
