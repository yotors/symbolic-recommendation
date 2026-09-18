"""Leakage-safe chronological confirmation for recommendation challengers.

The public development impressions are deliberately absent from this module.
It turns the chronological tail of the *training* log into complete, immutable
confirmation slates.  Champion and challenger are then mined from the same
earlier prefix and compared with paired impression-level AUC.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


_EVALUATION_KEYS = ("eval_impressions", "evaluation", "tests", "impressions")
_FORBIDDEN_CONTEXT_KEYS = frozenset({
    "action", "click", "clicked", "engagement", "is_click", "label",
    "outcome", "positive", "relevant", "target",
})
_ROW_TIMESTAMP = re.compile(r"^train-row-(\d+)$")

PUBLIC_BUILD_FRACTION = 2.0 / 3.0
PUBLIC_BOOTSTRAP_SEED = 37
PUBLIC_BOOTSTRAP_REPETITIONS = 5_000
PUBLIC_MIN_CONFIRMATION_IMPRESSIONS = 100
PUBLIC_MIN_CONFIRMATION_USERS = 30
_PUBLIC_FIXED_REQUEST_KEYS = frozenset({
    "bootstrap_repetitions", "bootstrap_seed", "build_fraction",
    "minimum_confirmation_impressions", "minimum_confirmation_users",
})
_ATTEMPT_REGISTRY: dict[str, str] = {}
_ATTEMPT_REGISTRY_LOCK = threading.Lock()


_RULE_SUPPORT_FIELDS = (
    "support", "mined_support", "antecedent_support", "joint_support",
    "target_weighted_support", "target_weighted_contingency",
    "target_count_contingency", "calibrated_support",
    "calibrated_support_unit", "required_calibrated_support",
)
_RULE_CALIBRATION_FIELDS = (
    "discovery_ctv", "ctv_calibration", "confidence_basis",
    "calibration_base_rate", "count_confidence", "activation_coverage",
    "lift", "effect", "stable_effect", "vote_weight", "quality",
    "target_wracc", "conditional_incremental_effect",
    "conditional_stable_incremental_effect", "conditional_incremental_wracc",
    "conditional_robust_incremental_wracc", "residual_delta",
    "residual_parent_logit", "residual_parent_count",
)
_RULE_TEMPORAL_FIELDS = (
    "temporal_fold_effects", "temporal_fold_weighted_supports",
    "temporal_fold_kish_effective_impressions", "required_temporal_folds",
    "target_fold_statistics",
)
_RULE_PROVENANCE_FIELDS = (
    "source", "id", "rule_id", "dependency_id", "dependency_owner",
    "variant_id", "proof_channel_id", "parent_id", "parent_rule_id",
    "specificity", "categorical_fact_family", "scoped_categorical_prior",
    "evidence_relationship", "conditional_fpminer_seed_rule_ids",
    "conditional_evidence_lineages", "conditional_lineage_signature",
    "conditional_variant_signature", "proof_factorization",
    "point_variant_id", "point_decision_id", "point_proof_channel_id",
    "point_proof_factorization",
)


@dataclass(frozen=True)
class TrainingGateProtocol:
    """Internal protocol dependency.

    The HTTP/Lab entry point never constructs this from request data.  Small
    unit tests may inject a cheaper protocol explicitly without weakening the
    public confirmation contract.
    """

    build_fraction: float = PUBLIC_BUILD_FRACTION
    bootstrap_seed: int = PUBLIC_BOOTSTRAP_SEED
    bootstrap_repetitions: int = PUBLIC_BOOTSTRAP_REPETITIONS
    minimum_confirmation_impressions: int = PUBLIC_MIN_CONFIRMATION_IMPRESSIONS
    minimum_confirmation_users: int = PUBLIC_MIN_CONFIRMATION_USERS
    enforce_single_attempt: bool = True


def _stable_value(value: Any) -> Any:
    """Convert supported model/data values to an unambiguous JSON form."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprinted values must contain only finite numbers")
        return {"__float__": value.hex()}
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("fingerprinted values must contain only finite numbers")
        return {"__decimal__": str(value.normalize())}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _stable_value(asdict(value)),
        }
    if isinstance(value, Mapping):
        pairs = [(_stable_value(key), _stable_value(item))
                 for key, item in value.items()]
        pairs.sort(key=lambda pair: json.dumps(
            pair[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ))
        return {"__mapping__": pairs}
    if isinstance(value, (set, frozenset)):
        items = [_stable_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ))
        return {"__set__": items}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            "__object__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _stable_value(vars(value)),
        }
    raise ValueError(f"unsupported fingerprint value type: {type(value).__name__}")


def _stable_bytes(value: Any) -> bytes:
    return json.dumps(
        _stable_value(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_stable_bytes(value)).hexdigest()


def _component_digest(value: Any) -> str:
    """Hash large top-level containers without copying the full container."""

    digest = hashlib.sha256()
    if isinstance(value, Mapping):
        ordered = sorted(value, key=lambda key: _stable_bytes(key))
        digest.update(b"mapping\x00")
        for key in ordered:
            digest.update(_stable_bytes(key))
            digest.update(b"\x00")
            digest.update(_stable_bytes(value[key]))
            digest.update(b"\x00")
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\x00")
        for item in value:
            digest.update(_stable_bytes(item))
            digest.update(b"\x00")
    else:
        digest.update(_stable_bytes(value))
    return digest.hexdigest()


_REPRESENTATION_DATA_KEYS = (
    "articles", "article_entity_vectors", "article_text_vectors",
    "title_idf_model", "lexical_idf_model", "subcategory_transition_model",
    "semantic_workspace_model", "llm_article_annotations",
)
def _representation_fingerprint(data: Mapping[str, Any]) -> str:
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("dataset metadata must be an object")
    components = {
        key: _component_digest(data[key])
        for key in _REPRESENTATION_DATA_KEYS if key in data
    }
    # Keep all declared provenance/formula metadata.  The filesystem root is
    # only a locator, and an earlier gate audit is recursive operational state,
    # so neither defines the representation itself.
    components["metadata"] = _stable_digest({
        key: value for key, value in metadata.items()
        if key not in {"root", "training_confirmation"}
    })
    return _stable_digest({
        "schema": "recommendation-representation-fingerprint-v1",
        "components": components,
    })


def confirmation_slate_digest(
    user: object,
    candidates: Sequence[object],
    relevant: Sequence[object],
    candidate_context: Mapping[object, Any],
) -> str:
    """Bind a score record to one ordered slate and its causal snapshots."""

    if not isinstance(candidate_context, Mapping):
        raise ValueError("confirmation candidate_context must be an object")
    candidate_ids = [str(candidate) for candidate in candidates]
    relevant_ids = [str(candidate) for candidate in relevant]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("confirmation slate contains duplicate candidates")
    if not set(relevant_ids).issubset(candidate_ids):
        raise ValueError("confirmation relevant IDs must belong to the slate")
    normalized_context = {}
    for candidate in candidate_ids:
        if candidate not in candidate_context:
            raise ValueError(
                f"confirmation slate lacks causal context for candidate {candidate!r}"
            )
        context = candidate_context[candidate]
        if not isinstance(context, Mapping):
            raise ValueError("confirmation candidate contexts must be objects")
        normalized_context[candidate] = dict(context)
    extra = {str(key) for key in candidate_context}.difference(candidate_ids)
    if extra:
        raise ValueError("confirmation candidate_context contains candidates outside the slate")
    return _stable_digest({
        "schema": "training-confirmation-slate-v1",
        "user": user,
        "ordered_candidates": candidate_ids,
        "relevant_ids": relevant_ids,
        "causal_contexts": [
            [candidate, normalized_context[candidate]] for candidate in candidate_ids
        ],
    })


@dataclass(frozen=True)
class _ChronologyPoint:
    domain: str
    value: tuple[Any, ...]
    display: str


@dataclass(frozen=True)
class TrainingConfirmationSplit:
    """A derived prefix-training dataset and its expected confirmation cohort."""

    data: dict[str, Any]
    audit: dict[str, Any]
    expected: tuple[dict[str, Any], ...]


def _chronology_point(raw: object) -> _ChronologyPoint:
    if isinstance(raw, bool) or raw is None:
        raise ValueError("training timestamps must be row numbers, numbers, or ISO datetimes")
    if isinstance(raw, (int, float, Decimal)):
        try:
            value = Decimal(str(raw))
        except InvalidOperation as exc:
            raise ValueError("training timestamp is not a finite number") from exc
        if not value.is_finite():
            raise ValueError("training timestamp is not a finite number")
        return _ChronologyPoint("numeric", (value,), str(raw))

    if isinstance(raw, datetime):
        parsed = raw
        display = raw.isoformat()
    elif isinstance(raw, str):
        text = raw.strip()
        row = _ROW_TIMESTAMP.fullmatch(text)
        if row:
            return _ChronologyPoint("train-row", (int(row.group(1)),), text)
        try:
            numeric = Decimal(text)
        except InvalidOperation:
            numeric = None
        if numeric is not None and numeric.is_finite():
            return _ChronologyPoint("numeric", (numeric,), text)
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise ValueError(f"unsupported training timestamp {raw!r}") from exc
        display = text
    else:
        raise ValueError("training timestamps must be row numbers, numbers, or ISO datetimes")

    aware = parsed.utcoffset() is not None
    if aware:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return _ChronologyPoint(
        "iso-aware" if aware else "iso-naive",
        (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute,
         parsed.second, parsed.microsecond),
        display,
    )


def _as_fraction(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("build_fraction must be a finite number between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("build_fraction must be a finite number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("build_fraction must be a finite number between 0 and 1")
    return result


def _is_positive(action: object, positive_actions: frozenset[object]) -> bool:
    return action in positive_actions


def prepare_training_confirmation(
    data: Mapping[str, Any],
    *,
    context_features: Iterable[str],
    positive_actions: Iterable[object],
    build_fraction: float = 2.0 / 3.0,
    authoritative_context_markers: Sequence[str] = ("history_size_bucket",),
) -> TrainingConfirmationSplit:
    """Split causal training events by whole chronological impressions.

    Every source impression participates in the cutoff.  Impressions without
    both a positive and a negative candidate are excluded only afterwards, so
    an unusable tail impression can never move a later impression into train.
    """

    if not isinstance(data, Mapping):
        raise ValueError("training confirmation requires a dataset object")
    events = data.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("training confirmation requires a non-empty events list")
    selected_fraction = _as_fraction(build_fraction)
    context = tuple(dict.fromkeys(context_features))
    if not context or any(not isinstance(item, str) or not item for item in context):
        raise ValueError("context_features must contain non-empty strings")
    contaminated = sorted(set(context).intersection(_FORBIDDEN_CONTEXT_KEYS))
    if contaminated:
        raise ValueError("context feature whitelist contains targets: " + ", ".join(contaminated))
    markers = tuple(authoritative_context_markers)
    if not markers or any(marker not in context for marker in markers):
        raise ValueError("authoritative context markers must belong to context_features")
    positives = frozenset(positive_actions)
    if not positives:
        raise ValueError("positive_actions must not be empty")

    grouped: OrderedDict[str, list[tuple[int, Mapping[str, Any]]]] = OrderedDict()
    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, Mapping):
            raise ValueError(f"training event {index} must be an object")
        impression = raw_event.get("impression")
        if impression is None or str(impression) == "":
            raise ValueError(f"training event {index} lacks an impression")
        grouped.setdefault(str(impression), []).append((index, raw_event))
    if len(grouped) < 2:
        raise ValueError("training confirmation requires at least two impressions")

    article_ids = {
        str(article.get("id")) for article in data.get("articles", [])
        if isinstance(article, Mapping) and article.get("id") is not None
    }
    users = data.get("users", {})
    validated = []
    domains = set()
    for impression, indexed_events in grouped.items():
        group_events = [event for _index, event in indexed_events]
        group_users = {event.get("user") for event in group_events}
        if None in group_users or len(group_users) != 1:
            raise ValueError(f"training impression {impression} must contain exactly one user")
        user = next(iter(group_users))
        if user not in users:
            raise ValueError(f"training impression {impression} has unknown user {user!r}")
        source_ids = {
            str(event["source_impression_id"])
            for event in group_events
            if event.get("source_impression_id") is not None
        }
        if len(source_ids) > 1:
            raise ValueError(f"training impression {impression} has conflicting source IDs")

        seen_articles = set()
        points = []
        candidates = []
        relevant = []
        candidate_context = {}
        outcomes = []
        marker_snapshots = {marker: [] for marker in markers}
        for index, event in indexed_events:
            if "action" not in event:
                raise ValueError(f"training event {index} lacks an action")
            if not all(marker in event for marker in markers):
                raise ValueError(
                    f"training event {index} lacks an authoritative causal context snapshot"
                )
            for marker in markers:
                marker_snapshots[marker].append(_stable_bytes(event[marker]))
            if event.get("article") is None:
                raise ValueError(f"training event {index} lacks an article")
            article = str(event["article"])
            if article in seen_articles:
                raise ValueError(
                    f"training impression {impression} repeats candidate {article!r}"
                )
            if article_ids and article not in article_ids:
                raise ValueError(
                    f"training impression {impression} references unknown article {article!r}"
                )
            seen_articles.add(article)
            if "timestamp" not in event:
                raise ValueError(f"training event {index} lacks a timestamp")
            point = _chronology_point(event["timestamp"])
            points.append(point)
            domains.add(point.domain)
            is_positive = _is_positive(event["action"], positives)
            candidates.append(article)
            outcomes.append(is_positive)
            if is_positive:
                relevant.append(article)
            # Presence of the marker is preserved even when its value is
            # ``unknown``. Lab.contextual_features uses that presence to avoid
            # falling back to a later, outcome-contaminated user profile.
            candidate_context[article] = {
                feature: event[feature] for feature in context if feature in event
            }

        for marker, snapshots in marker_snapshots.items():
            if len(set(snapshots)) != 1:
                raise ValueError(
                    f"training impression {impression} has inconsistent authoritative "
                    f"causal context marker {marker!r}"
                )
        if len({point.domain for point in points}) != 1:
            raise ValueError(f"training impression {impression} mixes timestamp domains")
        validated.append({
            "impression": impression,
            "source_impression_id": next(iter(source_ids), impression),
            "user": user,
            "indexed_events": indexed_events,
            "candidates": candidates,
            "relevant": relevant,
            "candidate_context": candidate_context,
            "outcomes": outcomes,
            "timestamps": [
                {"domain": point.domain, "display": point.display}
                for point in points
            ],
            "minimum": min(points, key=lambda item: item.value),
            "maximum": max(points, key=lambda item: item.value),
            "first_index": indexed_events[0][0],
        })
    if len(domains) != 1:
        raise ValueError("training events mix incomparable timestamp domains")

    validated.sort(key=lambda group: (group["minimum"].value, group["first_index"]))
    source_id_owners = {}
    for group in validated:
        identity = str(group["source_impression_id"])
        previous_owner = source_id_owners.get(identity)
        if previous_owner is not None:
            raise ValueError(
                "training confirmation contains duplicate effective impression "
                f"identity {identity!r} for {previous_owner!r} and "
                f"{group['impression']!r}"
            )
        source_id_owners[identity] = group["impression"]
    for previous, current in zip(validated, validated[1:]):
        # Equal cross-impression timestamps provide no defensible ordering.
        # A source-specific sequence key must be supplied instead of resolving
        # such ties from container order.
        if previous["maximum"].value >= current["minimum"].value:
            raise ValueError(
                "training impression chronology overlaps or ties: "
                f"{previous['impression']} and {current['impression']}"
            )

    cutoff = math.floor(len(validated) * selected_fraction)
    if cutoff < 1 or cutoff >= len(validated):
        raise ValueError("build_fraction leaves an empty build or confirmation partition")
    build_groups = validated[:cutoff]
    confirmation_groups = validated[cutoff:]
    build_usable = sum(any(group["outcomes"]) and not all(group["outcomes"])
                       for group in build_groups)
    if not build_usable:
        raise ValueError("training build prefix has no complete positive/negative impression")

    confirmation_cases = []
    expected = []
    effective_ids = set()
    excluded = []
    for group in confirmation_groups:
        positives_count = sum(group["outcomes"])
        negatives_count = len(group["outcomes"]) - positives_count
        if not positives_count or not negatives_count:
            excluded.append(group["impression"])
            continue
        case_id = f"training_confirmation_{group['impression']}"
        # Lab.benchmark deliberately reports source_impression_id when one is
        # present.  Mirror that public identity here; comparing against the
        # synthetic case id would reject a valid real-Lab run even though the
        # two scorers evaluated exactly the same slate.
        effective_id = str(group["source_impression_id"] or case_id)
        if effective_id in effective_ids:
            raise ValueError(
                "training confirmation contains duplicate effective impression "
                f"identity {effective_id!r}"
            )
        effective_ids.add(effective_id)
        case = {
            "id": case_id,
            "source_impression_id": group["source_impression_id"],
            "user": group["user"],
            "candidates": list(group["candidates"]),
            "relevant": list(group["relevant"]),
            "candidate_context": group["candidate_context"],
        }
        slate_digest = confirmation_slate_digest(
            case["user"], case["candidates"], case["relevant"],
            case["candidate_context"],
        )
        case["training_confirmation_slate_digest"] = slate_digest
        confirmation_cases.append(case)
        expected.append({
            "index": len(expected),
            "id": effective_id,
            "user": group["user"],
            "candidates": len(group["candidates"]),
            "positives": positives_count,
            "negatives": negatives_count,
            "candidate_ids": tuple(group["candidates"]),
            "relevant_ids": tuple(group["relevant"]),
            "slate_digest": slate_digest,
        })
    if not confirmation_cases:
        raise ValueError("training confirmation tail has no complete positive/negative impression")

    build_events = [
        event for group in build_groups for _index, event in group["indexed_events"]
    ]
    representation_fingerprint = _representation_fingerprint(data)
    participant_user_fingerprint = _stable_digest({
        group["user"]: users[group["user"]] for group in validated
    })
    cohort_payload = {
        "schema": "training-confirmation-cohort-v2",
        "representation_fingerprint": representation_fingerprint,
        "participant_user_fingerprint": participant_user_fingerprint,
        "build": [{
            "impression": group["impression"],
            "source_impression_id": group["source_impression_id"],
            "user": group["user"],
            "timestamps": group["timestamps"],
            "candidates": group["candidates"],
            "actions": [
                event["action"] for _index, event in group["indexed_events"]
            ],
            "outcomes": group["outcomes"],
            "candidate_context": group["candidate_context"],
        } for group in build_groups],
        "confirmation": [{
            "impression": group["impression"],
            "source_impression_id": group["source_impression_id"],
            "user": group["user"],
            "timestamps": group["timestamps"],
            "candidates": group["candidates"],
            "actions": [
                event["action"] for _index, event in group["indexed_events"]
            ],
            "outcomes": group["outcomes"],
            "candidate_context": group["candidate_context"],
        } for group in confirmation_groups],
    }
    cohort_fingerprint = _stable_digest(cohort_payload)

    derived = dict(data)
    for key in _EVALUATION_KEYS:
        derived.pop(key, None)
    derived["events"] = build_events
    derived["tests"] = confirmation_cases
    metadata = dict(data.get("metadata") or {})
    metadata["training_confirmation"] = {
        "policy": "chronological whole-impression prefix/tail; filter after cutoff",
        "model_fit_scope": "chronological build prefix only",
        "confirmation_context_policy": (
            "prequential causal snapshots: an earlier confirmation outcome may "
            "affect only a later confirmation impression's recorded context"
        ),
        "chronology_domain": next(iter(domains)),
        "cohort_fingerprint": cohort_fingerprint,
        "representation_fingerprint": representation_fingerprint,
        "participant_user_fingerprint": participant_user_fingerprint,
    }
    derived["metadata"] = metadata
    audit = {
        "policy": "chronological_whole_impression_training_confirmation",
        "build_fraction": selected_fraction,
        "chronology_domain": next(iter(domains)),
        "source_events": len(events),
        "source_impressions": len(validated),
        "build_events": len(build_events),
        "build_impressions": len(build_groups),
        "build_auc_usable_impressions": build_usable,
        "confirmation_source_impressions": len(confirmation_groups),
        "confirmation_auc_impressions": len(confirmation_cases),
        "confirmation_users": len({
            _stable_digest(row["user"]) for row in expected
        }),
        "confirmation_excluded_after_cutoff": excluded,
        "cutoff": {
            "last_build_impression": build_groups[-1]["impression"],
            "last_build_timestamp": build_groups[-1]["maximum"].display,
            "first_confirmation_impression": confirmation_groups[0]["impression"],
            "first_confirmation_timestamp": confirmation_groups[0]["minimum"].display,
        },
        "causal_context_markers": list(markers),
        "confirmation_context_policy": (
            "prequential causal snapshots; no confirmation outcome is used to "
            "fit either compared model"
        ),
        "representation_fingerprint": representation_fingerprint,
        "participant_user_fingerprint": participant_user_fingerprint,
        "cohort_fingerprint": cohort_fingerprint,
    }
    return TrainingConfirmationSplit(derived, audit, tuple(expected))


def _finite_auc(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite AUC between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite AUC between 0 and 1")
    return result


def _validated_records(
    run: Mapping[str, Any], expected: Sequence[Mapping[str, Any]], label: str,
) -> list[Mapping[str, Any]]:
    records = run.get("auc_per_impression")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError(f"{label} did not score the complete confirmation cohort")
    if (run.get("cases") != len(expected) or run.get("auc_cases") != len(expected)
            or run.get("sampled_without_positive", 0) != 0):
        raise ValueError(f"{label} did not produce one full-slate AUC per impression")
    seen = set()
    for wanted, actual in zip(expected, records):
        identity = (
            actual.get("index"), str(actual.get("id")),
            actual.get("candidates"), actual.get("positives"), actual.get("negatives"),
        )
        wanted_identity = tuple(wanted[key] for key in (
            "index", "id", "candidates", "positives", "negatives"
        ))
        if identity != wanted_identity:
            raise ValueError(f"{label} scored a different confirmation impression cohort")
        if identity[1] in seen:
            raise ValueError(f"{label} contains duplicate confirmation IDs")
        seen.add(identity[1])
        if ("user" not in wanted or "user" not in actual
                or actual["user"] != wanted["user"]):
            raise ValueError(f"{label} scored a different confirmation user cohort")
        wanted_digest = wanted.get("slate_digest")
        if (not isinstance(wanted_digest, str) or len(wanted_digest) != 64
                or actual.get("slate_digest") != wanted_digest):
            raise ValueError(
                f"{label} scored a different ordered slate or causal context snapshot"
            )
        _finite_auc(actual.get("auc"), f"{label} serving AUC")
        _finite_auc(actual.get("auc_proof_only"), f"{label} proof-only AUC")
    return records


def _bootstrap(values: Sequence[float], seed: int, repetitions: int) -> tuple[float, float]:
    local = random.Random(seed)
    count = len(values)
    estimates = [
        math.fsum(values[local.randrange(count)] for _index in range(count)) / count
        for _repeat in range(repetitions)
    ]
    estimates.sort()
    return (
        estimates[int(0.025 * (repetitions - 1))],
        estimates[int(0.975 * (repetitions - 1))],
    )


def _user_cluster_bootstrap(
    values: Sequence[float], users: Sequence[object], seed: int, repetitions: int,
) -> tuple[tuple[float, float], int]:
    """Resample whole users while retaining the macro-impression estimand."""

    if len(values) != len(users) or not values:
        raise ValueError("user-cluster bootstrap inputs must be non-empty and aligned")
    clusters: OrderedDict[str, list[float]] = OrderedDict()
    for value, user in zip(values, users):
        clusters.setdefault(_stable_digest(user), []).append(value)
    populations = list(clusters.values())
    cluster_count = len(populations)
    local = random.Random(seed)
    estimates = []
    for _repeat in range(repetitions):
        selected = [
            populations[local.randrange(cluster_count)]
            for _draw in range(cluster_count)
        ]
        denominator = sum(len(cluster) for cluster in selected)
        estimates.append(
            math.fsum(value for cluster in selected for value in cluster)
            / denominator
        )
    estimates.sort()
    return (
        estimates[int(0.025 * (repetitions - 1))],
        estimates[int(0.975 * (repetitions - 1))],
    ), cluster_count


def compare_confirmation_runs(
    champion: Mapping[str, Any],
    challenger: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    *,
    min_delta: float,
    bootstrap_seed: int,
    bootstrap_repetitions: int,
    require_proof_noninferiority: bool = True,
) -> dict[str, Any]:
    """Compare raw, paired per-impression AUC values from identical slates."""

    champion_rows = _validated_records(champion, expected, "champion")
    challenger_rows = _validated_records(challenger, expected, "challenger")
    if require_proof_noninferiority is not True:
        raise ValueError("proof noninferiority is mandatory for confirmation comparison")
    deltas = []
    proof_deltas = []
    users = []
    paired = []
    for wanted, base, candidate in zip(expected, champion_rows, challenger_rows):
        delta = float(candidate["auc"]) - float(base["auc"])
        proof_delta = (float(candidate["auc_proof_only"])
                       - float(base["auc_proof_only"]))
        deltas.append(delta)
        proof_deltas.append(proof_delta)
        users.append(wanted["user"])
        paired.append({
            "index": base["index"], "id": str(base["id"]),
            "champion_auc": round(float(base["auc"]), 10),
            "challenger_auc": round(float(candidate["auc"]), 10),
            "delta": round(delta, 10),
            "champion_proof_only_auc": round(float(base["auc_proof_only"]), 10),
            "challenger_proof_only_auc": round(float(candidate["auc_proof_only"]), 10),
            "proof_only_delta": round(proof_delta, 10),
        })
    served_interval = _bootstrap(
        deltas, bootstrap_seed ^ 0x4348414D, bootstrap_repetitions
    )
    proof_interval = _bootstrap(
        proof_deltas, bootstrap_seed ^ 0x50524F4D, bootstrap_repetitions
    )
    user_served_interval, user_count = _user_cluster_bootstrap(
        deltas, users, bootstrap_seed ^ 0x55534552, bootstrap_repetitions
    )
    user_proof_interval, proof_user_count = _user_cluster_bootstrap(
        proof_deltas, users, bootstrap_seed ^ 0x55505246, bootstrap_repetitions
    )
    if proof_user_count != user_count:
        raise RuntimeError("served and proof user-cluster cohorts diverged")
    impression_served_pass = served_interval[0] > min_delta
    user_served_pass = user_served_interval[0] > min_delta
    impression_proof_pass = proof_interval[0] >= 0.0
    user_proof_pass = user_proof_interval[0] >= 0.0
    served_pass = impression_served_pass and user_served_pass
    proof_pass = impression_proof_pass and user_proof_pass
    passed = served_pass and proof_pass
    return {
        "status": "pass" if passed else "hold",
        "criterion": (
            "paired training-confirmation serving AUC lower 95% bounds > min_delta "
            "and proof-only lower 95% bounds >= 0 under both impression and "
            "whole-user cluster bootstraps"
        ),
        "metric": "serving_ranking_auc",
        "paired_impressions": len(deltas),
        "min_delta": min_delta,
        "mean_auc_delta": round(math.fsum(deltas) / len(deltas), 10),
        "delta_95_ci": [round(value, 10) for value in served_interval],
        "user_cluster_delta_95_ci": [
            round(value, 10) for value in user_served_interval
        ],
        "mean_proof_only_auc_delta": round(
            math.fsum(proof_deltas) / len(proof_deltas), 10
        ),
        "proof_only_delta_95_ci": [round(value, 10) for value in proof_interval],
        "user_cluster_proof_only_delta_95_ci": [
            round(value, 10) for value in user_proof_interval
        ],
        "require_proof_noninferiority": require_proof_noninferiority,
        "served_criterion_passed": served_pass,
        "proof_noninferiority_passed": proof_pass,
        "impression_served_criterion_passed": impression_served_pass,
        "user_cluster_served_criterion_passed": user_served_pass,
        "impression_proof_noninferiority_passed": impression_proof_pass,
        "user_cluster_proof_noninferiority_passed": user_proof_pass,
        "bootstrap": {
            "seed": bootstrap_seed, "repetitions": bootstrap_repetitions,
            "unit": "whole training-tail impression",
        },
        "user_cluster_bootstrap": {
            "seed": bootstrap_seed,
            "repetitions": bootstrap_repetitions,
            "clusters": user_count,
            "unit": "whole confirmation user with all of that user's impressions",
            "estimand": "macro-impression AUC delta",
        },
        "per_impression": paired,
    }


def _audit_value(value: Any) -> Any:
    """Return a readable, deterministic JSON value for learned aggregates.

    Unlike :func:`_stable_value`, this representation is intended for people
    inspecting an artifact.  Callers pass only explicitly allow-listed model
    fields; in particular this helper is never given coverage sets, row
    indexes, event records, or confirmation benchmark rows.
    """

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("mining audit values must contain only finite numbers")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("mining audit values must contain only finite numbers")
        return float(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _audit_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _audit_value(value[key])
            for key in sorted(value, key=lambda item: _stable_bytes(item))
        }
    if isinstance(value, (set, frozenset)):
        values = [_audit_value(item) for item in value]
        return sorted(values, key=_stable_bytes)
    if isinstance(value, (list, tuple)):
        return [_audit_value(item) for item in value]
    raise ValueError(f"unsupported mining audit value type: {type(value).__name__}")


def _selected_rule_record(rule: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Project one fitted rule to compact, non-row-level audit evidence."""

    if not isinstance(rule, Mapping):
        raise ValueError("learned rule records must be objects")
    premises = []
    for premise in rule.get("premises", ()):
        if (not isinstance(premise, (list, tuple)) or len(premise) != 2):
            raise ValueError("learned rule premises must be predicate/value pairs")
        premises.append({
            "predicate": str(premise[0]),
            "value": _audit_value(premise[1]),
        })
    if kind == "point":
        conclusion = {
            "predicate": "engagement",
            "action": _audit_value(rule.get("target", "click")),
        }
    elif kind == "pair":
        conclusion = {"predicate": "pair_win", "action": "prefer_left"}
    else:  # Internal programming error, not model input.
        raise ValueError(f"unsupported learned rule kind: {kind}")

    positive = {
        key: _audit_value(rule[key])
        for key in ("strength", "confidence", "proof_strength", "proof_confidence")
        if key in rule
    }
    negative = {
        key.removeprefix("negative_"): _audit_value(rule[key])
        for key in ("negative_strength", "negative_confidence") if key in rule
    }
    record: dict[str, Any] = {
        "kind": kind,
        "premises": premises,
        "conclusion": conclusion,
    }
    if positive or negative:
        record["stv"] = {"positive": positive, "negative": negative}
    support = {
        key: _audit_value(rule[key])
        for key in _RULE_SUPPORT_FIELDS if key in rule
    }
    if support:
        record["support"] = support
    calibration = {
        key: _audit_value(rule[key])
        for key in _RULE_CALIBRATION_FIELDS if key in rule
    }
    if calibration:
        record["calibration"] = calibration
    temporal = {
        key: _audit_value(rule[key])
        for key in _RULE_TEMPORAL_FIELDS if key in rule
    }
    if temporal:
        record["temporal_validation"] = temporal
    provenance = {
        key: _audit_value(rule[key])
        for key in _RULE_PROVENANCE_FIELDS if key in rule
    }
    if provenance:
        record["provenance"] = provenance
    return record


def _compact_target_search(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    fields = (
        "kind", "requires_real_fpminer_seed", "semantic_predicates",
        "context_predicates", "fpminer_semantic_seeds", "candidate_patterns",
        "deeper_candidate_patterns", "discovery_only_seed_count",
        "candidate_conditional_children", "compiled_conditional_children",
        "compiled_conditional_premises", "semantic_seed_policy",
        "backoff_policy", "reason", "config",
    )
    return {key: _audit_value(raw[key]) for key in fields if key in raw}


def _compact_categorical_search(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    fields = (
        "actual_miner", "plans", "plan_population", "miner_depth",
        "emitted_with_side_predicate", "eligible_scoped_rules",
        "rejected_side_rules", "calibrated_scoped_rules",
        "selected_scoped_rules", "selected_scoped_premises", "scope_policy",
        "vocabulary", "host_generated_rules", "calibration", "temporal_gate",
        "support_gate", "redundancy", "dependency_policy",
    )
    return {key: _audit_value(raw[key]) for key in fields if key in raw}


def _learned_mining_audit(lab: Any) -> dict[str, Any]:
    """Describe only learned build-prefix state needed to audit mining.

    The projection is intentionally allow-list based.  Rule match/coverage
    indexes, source cases, user events, labels, and per-impression benchmark
    values cannot enter a future confirmation artifact through this path.
    """

    point_rules = [
        _selected_rule_record(rule, kind="point")
        for rule in getattr(lab, "mined_rules", ())
    ]
    pair_rules = [
        _selected_rule_record(rule, kind="pair")
        for rule in getattr(lab, "pair_rules", ())
    ]
    last_point = getattr(lab, "last_mining", None)
    if not isinstance(last_point, Mapping):
        last_point = {}
    last_pair = getattr(lab, "last_pair_mining", None)
    if not isinstance(last_pair, Mapping):
        last_pair = {}
    categorical = _compact_categorical_search(last_pair.get("categorical_search"))
    actual_miner = categorical.get("actual_miner") if categorical else None
    if actual_miner is None:
        actual_miner = next((
            rule.get("source") for rule in (*getattr(lab, "mined_rules", ()),
                                             *getattr(lab, "pair_rules", ()))
            if isinstance(rule, Mapping)
            and str(rule.get("source", "")).endswith("fpMiner.metta")
        ), None)

    point_fields = (
        "miner_strategy", "miner_calls", "source_cases", "cases", "positives",
        "negatives", "source_positives", "source_negatives", "base_rate",
        "sample_base_rate", "fpminer_min_support", "fpminer_support_unit",
        "target_min_support", "target_support_unit", "ctv_evidence_k",
        "rules_by_premises", "point_proof_channels",
        "point_proof_factorization",
    )
    pair_fields = (
        "miner_strategy", "miner_calls", "source_cases", "cases", "wins",
        "losses", "base_rate", "fpminer_min_support", "fpminer_support_unit",
        "target_min_support", "target_support_unit", "calibration_min_support",
        "ctv_evidence_k", "pair_ctv_mode", "petta_ctv_evidence_k",
        "pair_rule_selection_k", "pair_ctv_evidence_k",
        "pair_ctv_evidence_k_status", "confidence_contract",
        "pair_dependency_mode", "feature_profile", "rules_by_premises",
        "evidence_clusters", "proof_channels", "proof_channel_mode",
        "proof_factorization", "conditional_effective_backoff",
        "residual_hypergraph", "pair_target_semantics", "discovery_sample",
        "ctv_estimation_population", "search_weight_unit",
        "search_weight_total",
    )
    point_output = getattr(lab, "mined_output", ())
    point_sources = getattr(lab, "_point_rule_sources", ())
    point_channel_sources = getattr(lab, "_point_channel_sources", ())
    pair_sources = getattr(lab, "_pair_rule_sources", ())
    point_audit = {
        "selected_rule_count": len(point_rules),
        "selected_rules": point_rules,
        "miner_output_count": len(point_output),
        "miner_output_sha256": _component_digest(point_output),
        "compiled_rule_source_count": len(point_sources),
        "compiled_rule_sources_sha256": _component_digest(point_sources),
        "compiled_channel_source_count": len(point_channel_sources),
        "compiled_channel_sources_sha256": _component_digest(
            point_channel_sources
        ),
        "provenance": {
            key: _audit_value(last_point[key])
            for key in point_fields if key in last_point
        },
    }
    pair_audit = {
        "selected_rule_count": len(pair_rules),
        "selected_rules": pair_rules,
        "compiled_rule_source_count": len(pair_sources),
        "compiled_rule_sources_sha256": _component_digest(pair_sources),
        "provenance": {
            key: _audit_value(last_pair[key])
            for key in pair_fields if key in last_pair
        },
    }
    target_search = _compact_target_search(last_pair.get("target_search"))
    if target_search is not None:
        pair_audit["target_search"] = target_search
    if categorical is not None:
        pair_audit["categorical_search"] = categorical
    return {
        "schema": "recommendation-build-mining-audit-v1",
        "fit_scope": "chronological training build prefix only",
        "actual_pattern_miner": _audit_value(actual_miner),
        "host_generated_categorical_rules": (
            categorical.get("host_generated_rules") if categorical else None
        ),
        "point": point_audit,
        "pair": pair_audit,
    }


def _model_fingerprint(lab: Any) -> str:
    mining_audit = _learned_mining_audit(lab)
    payload = {
        "schema": "recommendation-model-fingerprint-v5",
        "symbolic_only": bool(lab.symbolic_only),
        "config": lab.config,
        "point_rules": lab.mined_rules,
        "point_miner_output": lab.mined_output,
        "point_rule_sources": getattr(lab, "_point_rule_sources", ()),
        "point_channel_sources": getattr(lab, "_point_channel_sources", ()),
        "pair_rules": lab.pair_rules,
        "pair_rule_sources": lab._pair_rule_sources,
        "feature_vocabulary": lab._feature_vocabulary,
        "pair_feature_vocabulary": lab._pair_feature_vocabulary,
        "pair_categorical_labels": lab._pair_categorical_labels,
        "numeric_pair_encoders": lab._numeric_pair_encoders,
        "click_base_rate": lab._click_base_rate,
        "tie_break_priors": lab._tie_break_stats,
        "popularity_prior": lab._popularity,
        "representation_fingerprint": _representation_fingerprint(lab.data),
        "representation_metadata": {
            "semantic_workspace_model": getattr(
                lab, "_semantic_workspace_model", None
            ),
            "recency_workspace": getattr(lab, "_recency_workspace", None),
            "llm_workspace": getattr(lab, "_llm_workspace", None),
            "text_embedding_metadata": getattr(
                lab, "_text_embedding_metadata", None
            ),
        },
        # Bind the deterministic, human-auditable mining projection as well as
        # the complete internal state above. This catches changes in search or
        # calibration policy even when they happen to select identical rules.
        "learned_mining_audit": mining_audit,
    }
    return _stable_digest(payload)


def _run_ephemeral(
    factory: Callable[..., Any], data: dict[str, Any], symbolic_only: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    lab = None
    try:
        lab = factory(data=data, symbolic_only=symbolic_only, config=config)
        result = lab.benchmark({
            "remine": False, "max_candidates": 0, "eval_case_limit": 0,
        })
        return {
            "result": result,
            "model_fingerprint": _model_fingerprint(lab),
            "mining_audit": _learned_mining_audit(lab),
        }
    finally:
        if lab is not None and getattr(lab, "engine", None) is not None:
            lab.engine.close()


def _run_summary(
    run: Mapping[str, Any], model_fingerprint: str,
    mining_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run.get("id"),
        "auc": run.get("auc"),
        "auc_proof_only": run.get("auc_proof_only"),
        "cases": run.get("cases"),
        "candidates": run.get("candidates"),
        "proof_coverage": run.get("proof_coverage"),
        "pairwise_proof_coverage": run.get("pairwise_proof_coverage"),
        "rules": run.get("rules"),
        "pair_rules": run.get("pair_rules"),
        "seconds": run.get("seconds"),
        "config": dict(run.get("config") or {}),
        "model_fingerprint": model_fingerprint,
        "build_mining_audit": dict(mining_audit),
    }


def _validated_protocol(protocol: TrainingGateProtocol) -> TrainingGateProtocol:
    if not isinstance(protocol, TrainingGateProtocol):
        raise TypeError("_protocol must be a TrainingGateProtocol")
    _as_fraction(protocol.build_fraction)
    if (isinstance(protocol.bootstrap_seed, bool)
            or not isinstance(protocol.bootstrap_seed, int)):
        raise ValueError("internal bootstrap_seed must be an integer")
    if (isinstance(protocol.bootstrap_repetitions, bool)
            or not isinstance(protocol.bootstrap_repetitions, int)
            or not 1 <= protocol.bootstrap_repetitions <= 100_000):
        raise ValueError("internal bootstrap_repetitions must be between 1 and 100000")
    if (isinstance(protocol.minimum_confirmation_impressions, bool)
            or not isinstance(protocol.minimum_confirmation_impressions, int)
            or protocol.minimum_confirmation_impressions < 1):
        raise ValueError("internal minimum_confirmation_impressions must be positive")
    if (isinstance(protocol.minimum_confirmation_users, bool)
            or not isinstance(protocol.minimum_confirmation_users, int)
            or protocol.minimum_confirmation_users < 2):
        raise ValueError("internal minimum_confirmation_users must be at least 2")
    if not isinstance(protocol.enforce_single_attempt, bool):
        raise ValueError("internal enforce_single_attempt must be boolean")
    return protocol


def _claim_confirmation_attempt(
    cohort_fingerprint: str,
    registry: MutableMapping[str, str],
    registry_lock: Any,
) -> None:
    with registry_lock:
        if cohort_fingerprint in registry:
            raise ValueError(
                "this immutable training confirmation cohort has already been used; "
                "load a genuinely new host dataset/cohort before another decision"
            )
        registry[cohort_fingerprint] = "running"


def _finish_confirmation_attempt(
    cohort_fingerprint: str,
    registry: MutableMapping[str, str],
    registry_lock: Any,
    status: str,
) -> None:
    with registry_lock:
        # Deliberately retain failed and held attempts: retrying another
        # architecture on the same tail would turn it into a tuning set.
        registry[cohort_fingerprint] = status


def _adopt_staged_model(host: Any, staged: Any) -> dict[str, Any]:
    """Swap a completely mined worker/model into ``host`` without partial state."""

    if staged.engine is None or staged.engine.pid is None:
        raise RuntimeError("staged challenger has no live PeTTaChainer worker")
    # Validate and bind every fingerprinted component before touching the
    # serving host.  A serialization/provenance error cannot leave a promoted
    # model live while reporting the confirmation request as failed.
    staged_fingerprint = _model_fingerprint(staged)
    model_fields = (
        "config", "_feature_vocabulary", "_pair_feature_vocabulary",
        "_pair_categorical_labels", "_numeric_pair_encoders", "mined_rules",
        "mined_output", "_point_rule_sources", "_point_channel_sources",
        "pair_rules",
        "_pair_rule_sources", "_active_rule_ids", "_active_pair_rule_ids",
        "last_pair_mining", "_click_base_rate", "_tie_break_stats",
        "last_mining", "last_mined_at", "_popularity",
    )
    def model_field(owner: Any, name: str) -> Any:
        if name in {"_point_rule_sources", "_point_channel_sources"}:
            return getattr(owner,name,[])
        return getattr(owner,name)

    old_state = {
        name:model_field(host,name)
        for name in model_fields
    }
    old_state.update({
        "engine": host.engine, "version": host.version,
        "pending_events": host.pending_events,
        "feed_cache": host.feed_cache, "_feed_sessions": host._feed_sessions,
        "_proof_cache": host._proof_cache,
        "_loaded_point_channels": getattr(host,"_loaded_point_channels",set()),
        "_point_channel_proof_cache": getattr(
            host,"_point_channel_proof_cache",{}
        ),
        "_point_channel_templates": getattr(host,"_point_channel_templates",{}),
        "_point_query_calls": getattr(host,"_point_query_calls",0),
        "_point_query_roots": getattr(host,"_point_query_roots",0),
        "_point_pruned_query_roots": getattr(
            host,"_point_pruned_query_roots",0
        ),
        "_point_channel_activations": getattr(
            host,"_point_channel_activations",0
        ),
        "_point_reused_channel_activations": getattr(
            host,"_point_reused_channel_activations",0
        ),
        "_last_point_completeness": getattr(
            host,"_last_point_completeness",{}
        ),
        "_last_point_cache_stats": getattr(host,"_last_point_cache_stats",{}),
        "_loaded_candidates": host._loaded_candidates,
        "_loaded_pairs": host._loaded_pairs,
        "_loaded_pair_channels": host._loaded_pair_channels,
        "_pair_proof_cache": host._pair_proof_cache,
        "_pair_channel_proof_cache": host._pair_channel_proof_cache,
        "_pair_margin_cache": host._pair_margin_cache,
        "_pair_case_attrs": host._pair_case_attrs,
        "_pair_channel_templates": host._pair_channel_templates,
        "_pair_proof_origins": host._pair_proof_origins,
        "_pair_query_calls": host._pair_query_calls,
        "_pair_query_roots": host._pair_query_roots,
        "_pair_pruned_query_roots": host._pair_pruned_query_roots,
        "_pair_channel_activations": host._pair_channel_activations,
        "_pair_reused_channel_activations": host._pair_reused_channel_activations,
    })
    new_version = host.version + 1
    new_state = {
        name:model_field(staged,name)
        for name in model_fields
    }
    new_state["config"] = staged.config.copy()
    new_state["_active_rule_ids"] = set(staged._active_rule_ids)
    new_state["_active_pair_rule_ids"] = set(staged._active_pair_rule_ids)
    new_state["engine"] = staged.engine
    new_state["version"] = new_version
    new_state["pending_events"] = 0
    new_state["feed_cache"] = OrderedDict()
    new_state["_feed_sessions"] = {}
    for name in (
        "_proof_cache", "_point_channel_proof_cache",
        "_point_channel_templates", "_last_point_completeness",
        "_last_point_cache_stats",
        "_pair_proof_cache", "_pair_channel_proof_cache",
        "_pair_margin_cache", "_pair_case_attrs", "_pair_channel_templates",
        "_pair_proof_origins",
    ):
        new_state[name] = {}
    for name in (
        "_loaded_candidates", "_loaded_point_channels", "_loaded_pairs",
        "_loaded_pair_channels",
    ):
        new_state[name] = set()
    for name in (
        "_point_query_calls", "_point_query_roots",
        "_point_pruned_query_roots", "_point_channel_activations",
        "_point_reused_channel_activations",
        "_pair_query_calls", "_pair_query_roots", "_pair_pruned_query_roots",
        "_pair_channel_activations", "_pair_reused_channel_activations",
    ):
        new_state[name] = 0
    if isinstance(staged.last_mining, Mapping):
        new_state["last_mining"] = dict(staged.last_mining)
        new_state["last_mining"]["version"] = new_version

    try:
        for name, value in new_state.items():
            setattr(host, name, value)
    except BaseException:
        for name, value in old_state.items():
            setattr(host, name, value)
        raise
    staged.engine = None
    retirement_warning = None
    try:
        old_state["engine"].close()
    except BaseException as exc:  # The new model is already coherent and live.
        retirement_warning = f"previous worker retirement failed: {type(exc).__name__}"
    return {
        "version": new_version,
        "worker_pid": host.engine.pid,
        "model_fingerprint": staged_fingerprint,
        "mining": host.last_mining,
        "retirement_warning": retirement_warning,
    }


def run_training_confirmation(
    host: Any,
    request: Mapping[str, Any],
    *,
    context_features: Iterable[str],
    positive_actions: Iterable[object],
    challenger_config_keys: Iterable[str],
    lab_factory: Callable[..., Any] | None = None,
    _protocol: TrainingGateProtocol | None = None,
    _attempt_registry: MutableMapping[str, str] | None = None,
    _attempt_registry_lock: Any | None = None,
) -> dict[str, Any]:
    """Mine, confirm, and optionally promote one declared architecture."""

    if not isinstance(request, Mapping):
        raise ValueError("training confirmation config must be an object")
    protocol = _validated_protocol(_protocol or TrainingGateProtocol())
    forbidden_protocol = sorted(set(request).intersection(_PUBLIC_FIXED_REQUEST_KEYS))
    if forbidden_protocol:
        raise ValueError(
            "the public confirmation protocol is fixed; remove request fields: "
            + ", ".join(forbidden_protocol)
        )
    allowed_request = {
        "challenger_config", "min_delta", "promote",
        "require_proof_noninferiority",
    }
    unknown_request = sorted(set(request) - allowed_request)
    if unknown_request:
        raise ValueError(
            "training confirmation contains unsupported request fields: "
            + ", ".join(unknown_request)
        )
    lock = getattr(host, "lock", None)
    if lock is None:
        lock = threading.RLock()
    with lock:
        if getattr(host, "_online_events", None):
            raise ValueError(
                "reload the dataset before training confirmation: live feedback "
                "is isolated from the immutable offline protocol"
            )
        raw_challenger = request.get("challenger_config")
        if not isinstance(raw_challenger, Mapping) or not raw_challenger:
            raise ValueError("challenger_config must be a non-empty object")
        allowed = frozenset(challenger_config_keys)
        unknown = sorted(set(raw_challenger) - allowed)
        if unknown:
            raise ValueError(
                "challenger_config contains unsupported keys: " + ", ".join(unknown)
            )
        champion_config = host.config.copy()
        challenger_config = {**champion_config, **dict(raw_challenger)}
        changed = {
            key: value for key, value in raw_challenger.items()
            if champion_config.get(key) != value
        }
        if not changed:
            raise ValueError("challenger_config does not differ from the active champion")

        try:
            min_delta = float(request.get("min_delta", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("min_delta must be a finite number between 0 and 1") from exc
        if not math.isfinite(min_delta) or not 0.0 <= min_delta <= 1.0:
            raise ValueError("min_delta must be a finite number between 0 and 1")
        promote = request.get("promote", False)
        proof_guard = request.get("require_proof_noninferiority", True)
        if not isinstance(promote, bool) or not isinstance(proof_guard, bool):
            raise ValueError("promote and require_proof_noninferiority must be booleans")
        if not proof_guard:
            raise ValueError(
                "require_proof_noninferiority is mandatory for training confirmation"
            )

        split = prepare_training_confirmation(
            host.data,
            context_features=context_features,
            positive_actions=positive_actions,
            build_fraction=protocol.build_fraction,
        )
        eligible = len(split.expected)
        if eligible < protocol.minimum_confirmation_impressions:
            raise ValueError(
                "training confirmation needs at least "
                f"{protocol.minimum_confirmation_impressions} eligible tail impressions; "
                f"found {eligible}"
            )
        if split.audit["confirmation_users"] < protocol.minimum_confirmation_users:
            raise ValueError(
                "training confirmation needs at least "
                f"{protocol.minimum_confirmation_users} distinct tail users "
                "for user-cluster confidence"
            )

        registry = _attempt_registry if _attempt_registry is not None else _ATTEMPT_REGISTRY
        registry_lock = (_attempt_registry_lock if _attempt_registry_lock is not None
                         else _ATTEMPT_REGISTRY_LOCK)
        attempt_claimed = False
        cohort_fingerprint = split.audit["cohort_fingerprint"]
        if protocol.enforce_single_attempt:
            _claim_confirmation_attempt(cohort_fingerprint, registry, registry_lock)
            attempt_claimed = True

        factory = lab_factory or type(host)
        try:
            champion = _run_ephemeral(
                factory, split.data, bool(host.symbolic_only), champion_config
            )
            challenger = _run_ephemeral(
                factory, split.data, bool(host.symbolic_only), challenger_config
            )
            comparison = compare_confirmation_runs(
                champion["result"], challenger["result"], split.expected,
                min_delta=min_delta,
                bootstrap_seed=protocol.bootstrap_seed,
                bootstrap_repetitions=protocol.bootstrap_repetitions,
                require_proof_noninferiority=True,
            )
            # Individual tail outcomes are intentionally not returned by the
            # public gate: they would support manual challenger tuning.
            comparison.pop("per_impression", None)
            passed = comparison["status"] == "pass"
            promotion = None
            if passed and promote:
                staged = None
                try:
                    # This model sees all training impressions only after the
                    # architecture decision. It never runs the public dev replay.
                    staged = factory(
                        data=host.data, symbolic_only=bool(host.symbolic_only),
                        config=challenger_config,
                    )
                    promotion = _adopt_staged_model(host, staged)
                finally:
                    if (staged is not None
                            and getattr(staged, "engine", None) is not None):
                        staged.engine.close()

            decision = "promote_challenger" if passed else "retain_champion"
            result = {
                "kind": "chronological_training_confirmation_gate",
                **comparison,
                "decision": decision,
                "promotion_requested": promote,
                "promoted": promotion is not None,
                "promotion": promotion,
                "split": split.audit,
                "cohort_fingerprint": cohort_fingerprint,
                "challenger_overrides": changed,
                "champion": _run_summary(
                    champion["result"], champion["model_fingerprint"],
                    champion["mining_audit"],
                ),
                "challenger": _run_summary(
                    challenger["result"], challenger["model_fingerprint"],
                    challenger["mining_audit"],
                ),
                "protocol": {
                    "build_fraction": protocol.build_fraction,
                    "bootstrap_seed": protocol.bootstrap_seed,
                    "bootstrap_repetitions": protocol.bootstrap_repetitions,
                    "minimum_confirmation_impressions": (
                        protocol.minimum_confirmation_impressions
                    ),
                    "minimum_confirmation_users": protocol.minimum_confirmation_users,
                    "bootstrap_units": ["whole impression", "whole user cluster"],
                    "proof_noninferiority_required": True,
                    "attempts_per_immutable_cohort": 1,
                },
                "dev_evaluation_used_for_selection": False,
                "representation_scope": (
                    "article representations are fixed before the split; both "
                    "models fit rules, quantiles, CTVs, vocabularies, and priors "
                    "only on the chronological build prefix; confirmation uses "
                    "prequential causal context snapshots, so an earlier tail "
                    "outcome may appear only in a later tail context"
                ),
                "reuse_policy": (
                    "the immutable cohort fingerprint is consumed by this one "
                    "architecture decision, whether it passes, holds, or errors"
                ),
            }
            if attempt_claimed:
                _finish_confirmation_attempt(
                    cohort_fingerprint, registry, registry_lock, result["status"]
                )
            return result
        except BaseException:
            if attempt_claimed:
                _finish_confirmation_attempt(
                    cohort_fingerprint, registry, registry_lock, "error"
                )
            raise


__all__ = [
    "PUBLIC_BOOTSTRAP_REPETITIONS", "PUBLIC_BOOTSTRAP_SEED",
    "PUBLIC_BUILD_FRACTION", "PUBLIC_MIN_CONFIRMATION_IMPRESSIONS",
    "PUBLIC_MIN_CONFIRMATION_USERS",
    "TrainingConfirmationSplit", "TrainingGateProtocol",
    "compare_confirmation_runs", "confirmation_slate_digest",
    "prepare_training_confirmation", "run_training_confirmation",
]
