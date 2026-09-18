"""Content observations combining semantic relevance and prior-history recency.

For a candidate unit vector x and preceding history unit vector v_j, let
c_j = clip(dot(x, v_j), -1, 1), and let lag_j count interaction positions from
the newest history entry. For each fixed half-life h in (8, 16), this module
reports sum(alpha_j * (c_j + 1) / 2), where

    alpha_j = softmax(8 * c_j - log(2) * lag_j / h).

These are observations for rule discovery, not recommendation scores. This
module accepts no outcomes, fitted preference model, or rule weights. Callers
must provide an oldest-to-newest history preceding the candidate exposure.
Unknown entries retain their positions; repeated interactions retain their
multiplicity. No compatible observation yields None, rather than zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


RECENCY_WORKSPACE_SCHEMA = "semantic-recency-attention-v1"
RECENCY_ATTENTION_TEMPERATURE = 8.0
RECENCY_HALF_LIVES = (8, 16)
RECENCY_WORKSPACE_FEATURES = tuple(
    f"text_semantic_recency_attention_h{half_life}_similarity"
    for half_life in RECENCY_HALF_LIVES
)


def _unit_vector(value: object, dimensions: int | None = None) -> np.ndarray | None:
    """Treat malformed evidence as missing; normalize finite extremes safely."""

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
    scaled = vector / scale
    return scaled / float(np.linalg.norm(scaled))


def build_recency_workspace_facts(
    candidate_id: object,
    ordered_history_ids: Iterable[object],
    vectors_mapping: Mapping[object, Sequence[float]],
) -> dict[str, float | None]:
    """Return two bounded observations without fitting or mutating anything.

    Candidate/history entries may be opaque IDs or mappings containing ``id``.
    All other metadata is ignored. The candidate establishes vector dimension;
    nonfinite, zero, malformed, or incompatible history vectors are unavailable.
    A missing history position still ages every preceding known interaction.
    """

    def resolve(item: object, dimensions: int | None = None) -> np.ndarray | None:
        identifier = item.get("id") if isinstance(item, Mapping) else item
        if identifier is None:
            return None
        try:
            raw = vectors_mapping.get(identifier)
        except (TypeError, ValueError):
            return None
        return _unit_vector(raw, dimensions)

    facts = dict.fromkeys(RECENCY_WORKSPACE_FEATURES)
    candidate = resolve(candidate_id)
    if candidate is None:
        return facts
    history = list(ordered_history_ids)
    raw_cosines: list[float] = []
    lags: list[int] = []
    for position, item in enumerate(history):
        vector = resolve(item, int(candidate.size))
        if vector is None:
            continue
        raw_cosines.append(max(-1.0, min(1.0, float(candidate @ vector))))
        lags.append(len(history) - position - 1)
    if not raw_cosines:
        return facts

    cosines = np.asarray(raw_cosines, dtype=np.float64)
    positions = np.asarray(lags, dtype=np.float64)
    mapped_cosines = (cosines + 1.0) / 2.0
    for half_life, feature in zip(RECENCY_HALF_LIVES, RECENCY_WORKSPACE_FEATURES):
        logits = RECENCY_ATTENTION_TEMPERATURE * cosines - math.log(2.0) * positions / half_life
        weights = np.exp(logits - float(logits.max()))
        total = float(weights.sum())
        value = float(weights @ mapped_cosines) / total
        facts[feature] = round(max(0.0, min(1.0, value)), 8)
    return facts


__all__ = [
    "RECENCY_WORKSPACE_SCHEMA",
    "RECENCY_ATTENTION_TEMPERATURE",
    "RECENCY_HALF_LIVES",
    "RECENCY_WORKSPACE_FEATURES",
    "build_recency_workspace_facts",
]
