"""Leakage-safe repeated-user replay over timestamped recommendation slates.

The global miner model and its PeTTaChainer worker stay frozen.  Each arm owns
only an in-memory private user state.  A complete timestamp group is ranked
before any outcome from that group becomes available, so equal-time slates
cannot leak labels into one another.

This is a logged-slate ranking evaluation.  It measures whether prior logged
interactions improve later ordering inside the historical candidate sets; it
does not estimate counterfactual online CTR.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from ..adapters.mind import (
    history_feature_context,
    prepare_history_feature_workspace,
)
from ..features.lexical_workspace import build_lexical_workspace_facts
from ..features.llm_workspace import build_llm_workspace_facts
from ..features.recency_workspace import build_recency_workspace_facts
from ..features.relational_workspace import (
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
)
from ..features.semantic_workspace import build_semantic_workspace_facts


PRIVATE_NONCLICK_HISTORY_LIMIT = 20
PRIVATE_NONCLICK_GENERALIZATION_WINDOW = 5
_FROZEN_NON_HISTORY_FEATURES = frozenset({
    "ctr_bucket", "freshness_bucket", "position_bucket", "time_bucket",
})
_RELATIONAL_SCOPE_FIELDS = (
    REL_ENTITY_CONTINUITY_SCOPE, REL_CONCEPT_CONTINUITY_SCOPE,
)
_RELATIONAL_PROOF_FIELDS = (
    REL_ENTITY_CONTINUITY_PROOF_IDS, REL_CONCEPT_CONTINUITY_PROOF_IDS,
)


@dataclass
class PrivateReplayState:
    """Private state derived exclusively from completed replay impressions."""

    initial_history: tuple[str, ...]
    clicks: list[str] = field(default_factory=list)
    nonclick_history: list[str] = field(default_factory=list)
    completed_impressions: int = 0
    completed_timestamp_groups: int = 0

    def augmented_history(self, causal_history: Iterable[str]) -> list[str]:
        """Merge replay clicks without duplicating clicks already in source history."""

        history = [str(value) for value in causal_history]
        initial = Counter(self.initial_history)
        current = Counter(history)
        incorporated = Counter({
            article: max(0, current[article] - initial[article])
            for article in current
        })
        consumed: Counter[str] = Counter()
        for article in self.clicks:
            if consumed[article] < incorporated[article]:
                consumed[article] += 1
            else:
                history.append(article)
        return history[-200:]

    def apply_impression(
        self,
        candidates: Sequence[str],
        relevant: set[str],
        *,
        include_nonclicks: bool,
    ) -> None:
        """Apply a complete slate after it has been ranked."""

        for raw_article in candidates:
            article = str(raw_article)
            if article in relevant:
                self.clicks.append(article)
                self.nonclick_history[:] = [
                    item for item in self.nonclick_history if item != article
                ]
            elif include_nonclicks:
                self.nonclick_history[:] = [
                    item for item in self.nonclick_history if item != article
                ]
                self.nonclick_history.append(article)
                del self.nonclick_history[:-PRIVATE_NONCLICK_HISTORY_LIMIT]
        self.completed_impressions += 1


def _timestamp(case: Mapping[str, Any]) -> datetime:
    if case.get("timestamp_source") != "official_mind_behaviors":
        raise ValueError(
            "personalization replay requires official MIND behavior timestamps"
        )
    raw = case.get("timestamp")
    if not isinstance(raw, str):
        raise ValueError("personalization case has no ISO timestamp")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid personalization timestamp: {raw!r}") from exc
    if value.tzinfo is not None:
        value = value.replace(tzinfo=None)
    return value


def chronological_user_cases(
    cases: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Validate and return complete per-user chronological sequences."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for case in cases:
        identity = str(case.get("source_impression_id") or case.get("id") or "")
        user = str(case.get("user") or "")
        if not identity or not user:
            raise ValueError("personalization cases require impression and user IDs")
        if identity in seen:
            raise ValueError(f"duplicate personalization impression: {identity}")
        seen.add(identity)
        _timestamp(case)
        if not isinstance(case.get("history"), list):
            raise ValueError(f"personalization impression {identity} has no history")
        if not isinstance(case.get("candidates"), list):
            raise ValueError(f"personalization impression {identity} has no candidates")
        grouped[user].append(case)
    for user, sequence in grouped.items():
        sequence.sort(key=lambda case: (
            _timestamp(case), int(case.get("source_line_number", 0)),
            str(case.get("source_impression_id") or case.get("id")),
        ))
        if len({_timestamp(case) for case in sequence}) < 2:
            raise ValueError(
                f"personalization user {user!r} has fewer than two timestamp groups"
            )
        supplied = [case.get("user_sequence_index") for case in sequence]
        if all(value is not None for value in supplied) and supplied != list(
            range(len(sequence))
        ):
            raise ValueError(
                f"personalization sequence index disagrees for user {user!r}"
            )
    return dict(grouped)


def _negative_match(lab: Any, state: PrivateReplayState, article_id: str) -> str:
    article_id = str(article_id)
    if article_id in state.nonclick_history:
        return "exact"
    articles = getattr(lab, "_articles")
    article = articles[article_id]
    recent = [
        articles[item] for item in
        state.nonclick_history[-PRIVATE_NONCLICK_GENERALIZATION_WINDOW:]
        if item in articles
    ]
    subcategory = str(article.get("subcategory", "unknown"))
    topic = str(article.get("topic", article.get("category", "unknown")))
    if subcategory != "unknown" and any(
        str(item.get("subcategory", "unknown")) == subcategory
        for item in recent
    ):
        return "subcategory"
    if topic != "unknown" and any(
        str(item.get("topic", item.get("category", "unknown"))) == topic
        for item in recent
    ):
        return "topic"
    return "none"


def _regenerate_relational_contexts(
    lab: Any,
    *,
    user: str,
    candidates: Sequence[str],
    history: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Derive history-bound relation values and proof IDs for this arm."""

    mode = getattr(lab, "config", {}).get(
        "relational_evidence_mode", "disabled"
    )
    if mode == "disabled":
        return {str(article): {} for article in candidates}
    if mode != "chained":
        raise ValueError(
            "personalization replay requires chained relational evidence so "
            "each private history can receive fresh PeTTa proof IDs"
        )
    derive = getattr(lab, "_live_relational_features", None)
    if not callable(derive):
        raise ValueError("lab cannot regenerate relational PeTTa evidence")
    profile = lab.data.get("users", {}).get(user)
    if not isinstance(profile, dict):
        raise ValueError("relational personalization requires a user profile")
    missing = object()
    previous = profile.get("history", missing)
    profile["history"] = list(history)
    try:
        projected = derive(
            user, list(candidates), set(_RELATIONAL_SCOPE_FIELDS)
        )
    finally:
        if previous is missing:
            profile.pop("history", None)
        else:
            profile["history"] = previous
    return {
        str(article): dict(projected.get(str(article), {}))
        for article in candidates
    }


def private_contexts(
    lab: Any,
    case: Mapping[str, Any],
    state: PrivateReplayState,
) -> dict[str, dict[str, Any]]:
    """Rebuild one causal control/private history in the frozen workspace."""

    base_contexts = case.get("candidate_context") or {}
    history = state.augmented_history(case.get("history", ()))
    articles = getattr(lab, "_articles")
    entity_vectors = getattr(lab, "_article_entity_vectors", {})
    text_vectors = getattr(lab, "_article_text_vectors", {})
    workspace = prepare_history_feature_workspace(
        history,
        articles,
        entity_vectors=entity_vectors,
        text_semantic_vectors=text_vectors,
    )
    result: dict[str, dict[str, Any]] = {}
    for raw_article in case.get("candidates", ()):
        article_id = str(raw_article)
        article = articles[article_id]
        # Only source-slate/exposure facts survive from the raw adapter. Every
        # history-dependent fact is rebuilt from this Lab's frozen models for
        # both control and adaptive arms.
        base = {
            key: value
            for key, value in (base_contexts.get(article_id) or {}).items()
            if key in _FROZEN_NON_HISTORY_FEATURES
        }
        dynamic = history_feature_context(
            article,
            history,
            articles,
            entity_vectors=entity_vectors,
            text_semantic_vectors=text_vectors,
            title_idf_model=getattr(lab, "_title_idf_model", {}),
            transition_model=getattr(lab, "data", {}).get(
                "subcategory_transition_model"
            ),
            hour=case.get("hour"),
            workspace=workspace,
        )
        if getattr(lab, "_lexical_idf_model", {}):
            dynamic.update(build_lexical_workspace_facts(
                article,
                (articles.get(item, {}) for item in history),
                lab._lexical_idf_model,
            ))
        if getattr(lab, "_semantic_workspace_model", {}):
            dynamic.update(build_semantic_workspace_facts(
                article_id, history, text_vectors,
                lab._semantic_workspace_model,
            ))
        if getattr(lab, "_recency_workspace", {}):
            dynamic.update(build_recency_workspace_facts(
                article_id, history, text_vectors,
            ))
        if getattr(lab, "_llm_workspace", {}):
            dynamic.update(build_llm_workspace_facts(
                article_id, history,
                getattr(lab, "_llm_article_annotations", {}),
            ))
        for key, value in dynamic.items():
            if key not in _FROZEN_NON_HISTORY_FEATURES and value is not None:
                base[key] = value
        base["recent_negative_match"] = _negative_match(
            lab, state, article_id
        )
        result[article_id] = base
    relational = _regenerate_relational_contexts(
        lab,
        user=str(case.get("user")),
        candidates=[str(value) for value in case.get("candidates", ())],
        history=history,
    )
    for article_id, derived in relational.items():
        for field in (*_RELATIONAL_SCOPE_FIELDS, *_RELATIONAL_PROOF_FIELDS):
            result[article_id].pop(field, None)
        result[article_id].update(derived)
    return result


def _case_auc(lab: Any, rows: Sequence[Mapping[str, Any]], relevant: set[str]):
    positives = [row for row in rows if str(row["article"]["id"]) in relevant]
    negatives = [row for row in rows if str(row["article"]["id"]) not in relevant]
    if not positives or not negatives:
        return None
    signature = getattr(lab, "_ranking_signature", None)
    if signature is None:
        order = {str(row["article"]["id"]): -index for index, row in enumerate(rows)}
        signature = lambda row: order[str(row["article"]["id"])]
    return math.fsum(
        (signature(positive) > signature(negative))
        + 0.5 * (signature(positive) == signature(negative))
        for positive in positives for negative in negatives
    ) / (len(positives) * len(negatives))


def _cluster_interval(
    deltas: Sequence[float], users: Sequence[str], *, seed: int, repetitions: int,
) -> list[float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for delta, user in zip(deltas, users):
        clusters[user].append(delta)
    populations = list(clusters.values())
    local = random.Random(seed)
    estimates = []
    for _repeat in range(repetitions):
        sample = [populations[local.randrange(len(populations))]
                  for _index in populations]
        estimates.append(
            math.fsum(value for cluster in sample for value in cluster)
            / sum(map(len, sample))
        )
    estimates.sort()
    return [
        estimates[int(0.025 * (repetitions - 1))],
        estimates[int(0.975 * (repetitions - 1))],
    ]


def run_prequential_personalization(
    lab: Any,
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    update_mode: str = "click_and_nonclick",
    bootstrap_seed: int = 71,
    bootstrap_repetitions: int = 1000,
) -> dict[str, Any]:
    """Compare a frozen arm with prior-interaction-only private adaptation."""

    if update_mode not in {"click_only", "click_and_nonclick"}:
        raise ValueError(
            "update_mode must be click_only or click_and_nonclick"
        )
    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be at least 100")
    source = list(cases if cases is not None else lab.data.get("tests", ()))
    grouped = chronological_user_cases(source)
    model_before = (
        lab.serving_model().get("model_sha256")
        if callable(getattr(lab, "serving_model", None)) else None
    )
    version_before = getattr(lab, "version", None)
    event_count_before = len(getattr(lab, "data", {}).get("events", ()))
    online_count_before = len(getattr(lab, "_online_events", ()))
    started = time.perf_counter()
    records = []
    scored_candidates = 0
    burn_in_impressions = 0
    reasoner_engines: Counter[str] = Counter()
    score_methods: Counter[str] = Counter()
    rows_with_proofs = 0
    scored_rows = 0

    for user, sequence in sorted(grouped.items()):
        state = PrivateReplayState(tuple(sequence[0].get("history", ())))
        for _timestamp_value, timestamp_cases_iter in itertools.groupby(
            sequence, key=_timestamp
        ):
            timestamp_cases = list(timestamp_cases_iter)
            prior_impressions = state.completed_impressions
            if prior_impressions == 0:
                burn_in_impressions += len(timestamp_cases)
            else:
                for case in timestamp_cases:
                    candidates = [str(value) for value in case["candidates"]]
                    relevant = {str(value) for value in case.get("relevant", ())}
                    frozen_contexts = private_contexts(
                        lab,
                        case,
                        PrivateReplayState(tuple(case.get("history", ()))),
                    )
                    adaptive_contexts = private_contexts(lab, case, state)
                    frozen_rows = lab.score(
                        user, candidates, frozen_contexts, limit=0,
                        cache_result=False,
                    )
                    adaptive_rows = lab.score(
                        user, candidates, adaptive_contexts, limit=0,
                        cache_result=False,
                    )
                    for row in (*frozen_rows, *adaptive_rows):
                        reasoner_engines[str(row.get("engine", "unspecified"))] += 1
                        score_methods[str(
                            row.get("score_method", "unspecified")
                        )] += 1
                        rows_with_proofs += bool(row.get("proofs"))
                        scored_rows += 1
                    frozen_auc = _case_auc(lab, frozen_rows, relevant)
                    adaptive_auc = _case_auc(lab, adaptive_rows, relevant)
                    scored_candidates += len(candidates) * 2
                    if frozen_auc is not None and adaptive_auc is not None:
                        records.append({
                            "id": str(case.get("source_impression_id") or case.get("id")),
                            "user": user,
                            "timestamp": case["timestamp"],
                            "prior_impressions": prior_impressions,
                            "prior_clicks": len(state.clicks),
                            "prior_nonclicks": len(state.nonclick_history),
                            "candidates": len(candidates),
                            "frozen_auc": frozen_auc,
                            "adaptive_auc": adaptive_auc,
                            "delta": adaptive_auc - frozen_auc,
                        })
            # Reveal every outcome only after every slate at this timestamp was
            # ranked. Candidate order is the recorded display order.
            for case in timestamp_cases:
                candidates = [str(value) for value in case["candidates"]]
                relevant = {str(value) for value in case.get("relevant", ())}
                state.apply_impression(
                    candidates,
                    relevant,
                    include_nonclicks=update_mode == "click_and_nonclick",
                )
            state.completed_timestamp_groups += 1

    if not records:
        raise ValueError("personalization cohort has no post-update AUC cases")
    model_after = (
        lab.serving_model().get("model_sha256")
        if callable(getattr(lab, "serving_model", None)) else None
    )
    if (model_before != model_after
            or version_before != getattr(lab, "version", None)
            or event_count_before != len(lab.data.get("events", ()))
            or online_count_before != len(getattr(lab, "_online_events", ()))):
        raise RuntimeError("personalization replay mutated global model or events")

    deltas = [record["delta"] for record in records]
    users = [record["user"] for record in records]
    frozen = [record["frozen_auc"] for record in records]
    adaptive = [record["adaptive_auc"] for record in records]
    per_user: dict[str, list[float]] = defaultdict(list)
    dose: dict[str, list[float]] = defaultdict(list)
    for record in records:
        per_user[record["user"]].append(record["delta"])
        count = record["prior_impressions"]
        dose["1" if count == 1 else "2" if count == 2 else "3_plus"].append(
            record["delta"]
        )
    return {
        "status": "complete",
        "interpretation": (
            "Logged-slate prequential ranking comparison; not a "
            "counterfactual online CTR estimate."
        ),
        "update_mode": update_mode,
        "global_model_frozen": True,
        "model_sha256": model_before,
        "users": len(grouped),
        "burn_in_impressions": burn_in_impressions,
        "evaluated_impressions": len(records),
        "scored_candidates_across_two_arms": scored_candidates,
        "reasoner_audit": {
            "engines": dict(sorted(reasoner_engines.items())),
            "score_methods": dict(sorted(score_methods.items())),
            "scored_rows": scored_rows,
            "rows_with_proofs": rows_with_proofs,
            "proof_coverage": (
                rows_with_proofs / scored_rows if scored_rows else 0.0
            ),
            "all_rows_pettachainer": (
                scored_rows > 0
                and set(reasoner_engines) == {"PeTTaChainer"}
            ),
        },
        "frozen_auc": math.fsum(frozen) / len(frozen),
        "adaptive_auc": math.fsum(adaptive) / len(adaptive),
        "mean_delta": math.fsum(deltas) / len(deltas),
        "user_macro_mean_delta": math.fsum(
            math.fsum(values) / len(values) for values in per_user.values()
        ) / len(per_user),
        "whole_user_cluster_delta_95_ci": _cluster_interval(
            deltas, users, seed=bootstrap_seed,
            repetitions=bootstrap_repetitions,
        ),
        "dose_response": {
            key: {
                "impressions": len(values),
                "mean_delta": math.fsum(values) / len(values),
            }
            for key, values in sorted(dose.items())
        },
        "seconds": time.perf_counter() - started,
        "per_impression": records,
    }


def compare_personalization_modes(
    lab: Any,
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    bootstrap_seed: int = 71,
    bootstrap_repetitions: int = 1000,
) -> dict[str, Any]:
    """Evaluate no-update, click-only, and click-plus-nonclick policies.

    Both adaptive policies are replayed independently from empty private state.
    Their frozen rows must agree impression by impression; this makes the
    no-private-update arm an exact shared control rather than a separately
    sampled estimate.
    """

    source = list(cases if cases is not None else lab.data.get("tests", ()))
    click_only = run_prequential_personalization(
        lab,
        cases=source,
        update_mode="click_only",
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    click_and_nonclick = run_prequential_personalization(
        lab,
        cases=source,
        update_mode="click_and_nonclick",
        bootstrap_seed=bootstrap_seed,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    for result in (click_only, click_and_nonclick):
        audit = result["reasoner_audit"]
        if (not audit["all_rows_pettachainer"]
                or audit["rows_with_proofs"] != audit["scored_rows"]):
            raise RuntimeError(
                "personalization replay left the PeTTaChainer proof path"
            )
    click_control = {
        row["id"]: row["frozen_auc"]
        for row in click_only["per_impression"]
    }
    nonclick_control = {
        row["id"]: row["frozen_auc"]
        for row in click_and_nonclick["per_impression"]
    }
    if click_control != nonclick_control:
        raise RuntimeError(
            "no-private-update control changed between adaptation arms"
        )
    if click_only["model_sha256"] != click_and_nonclick["model_sha256"]:
        raise RuntimeError("serving model changed between adaptation arms")

    def arm(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: result[key]
            for key in (
                "adaptive_auc", "mean_delta", "user_macro_mean_delta",
                "whole_user_cluster_delta_95_ci", "dose_response", "seconds",
                "reasoner_audit",
            )
        }

    return {
        "status": "complete",
        "protocol": "official-time prequential complete-user replay",
        "interpretation": click_only["interpretation"],
        "zero_label_semantics": (
            "MIND label 0 is logged non-click after exposure; it is not an "
            "explicit user Skip action"
        ),
        "global_model_frozen": True,
        "model_sha256": click_only["model_sha256"],
        "users": click_only["users"],
        "burn_in_impressions": click_only["burn_in_impressions"],
        "evaluated_impressions": click_only["evaluated_impressions"],
        "arms": {
            "no_private_updates": {
                "auc": click_only["frozen_auc"],
                "mean_delta_vs_no_private_updates": 0.0,
            },
            "click_only": arm(click_only),
            "click_and_nonclick": arm(click_and_nonclick),
        },
        "control_reconstruction_exact": True,
        "reasoner_path_verified": True,
        "per_impression": {
            "click_only": click_only["per_impression"],
            "click_and_nonclick": click_and_nonclick["per_impression"],
        },
    }


def bind_personalization_cohort(
    workspace_data: dict[str, Any],
    cohort_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach official repeated-user cases to a frozen content workspace.

    The operation is deliberately strict: every history and candidate article
    must already belong to the frozen workspace.  This preserves the exact LLM
    annotation/corpus provenance used by the serving model while allowing an
    independently selected, label-blind longitudinal evaluation cohort.
    """

    tests = copy.deepcopy(list(cohort_data.get("tests", ())))
    articles = {
        str(article["id"]): article for article in workspace_data["articles"]
    }
    required = {
        str(article_id)
        for case in tests
        for article_id in (
            *case.get("history", ()), *case.get("candidates", ())
        )
    }
    missing = sorted(required.difference(articles))
    if missing:
        raise ValueError(
            "personalization cohort is outside the frozen content workspace: "
            + ", ".join(missing[:10])
        )
    cohort_articles = {
        str(article["id"]): article
        for article in cohort_data.get("articles", ())
    }
    identity_fields = (
        "source_id", "topic", "category", "subcategory", "title",
        "abstract", "title_entities", "abstract_entities",
    )
    for article_id in sorted(required):
        cohort_article = cohort_articles.get(article_id)
        if cohort_article is None:
            raise ValueError(
                f"cohort has no metadata for referenced article {article_id}"
            )
        for field in identity_fields:
            if (field in cohort_article
                    and cohort_article.get(field) != articles[article_id].get(field)):
                raise ValueError(
                    f"cohort/workspace article metadata disagree for "
                    f"{article_id}/{field}"
                )

    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()

    model_fields = (
        "title_idf_model", "subcategory_transition_model",
        "lexical_idf_model", "semantic_workspace_model",
    )
    model_hashes = {}
    for field in model_fields:
        cohort_value = cohort_data.get(field) or {}
        workspace_value = workspace_data.get(field) or {}
        if cohort_value != workspace_value:
            raise ValueError(
                f"cohort/workspace feature model mismatch: {field}"
            )
        model_hashes[field] = digest(workspace_value)
    cohort_metadata = cohort_data.get("metadata") or {}
    workspace_metadata = workspace_data.get("metadata") or {}
    for field in (
        "feature_thresholds", "text_embedding_content_sha256",
        "text_embedding_model",
    ):
        if cohort_metadata.get(field) != workspace_metadata.get(field):
            raise ValueError(
                f"cohort/workspace feature provenance mismatch: {field}"
            )
        model_hashes[f"metadata.{field}"] = digest(
            workspace_metadata.get(field)
        )
    cohort_vectors = cohort_data.get("article_entity_vectors") or {}
    workspace_vectors = workspace_data.get("article_entity_vectors") or {}
    maximum_entity_vector_drift = 0.0
    for article_id in sorted(required):
        if article_id not in cohort_vectors:
            continue
        left = cohort_vectors[article_id]
        right = workspace_vectors.get(article_id)
        if (not isinstance(left, (list, tuple))
                or not isinstance(right, (list, tuple))
                or len(left) != len(right)):
            raise ValueError(
                "cohort/workspace entity vector shape mismatch for "
                + article_id
            )
        drift = max(
            (abs(float(one) - float(two))
             for one, two in zip(left, right)),
            default=0.0,
        )
        maximum_entity_vector_drift = max(
            maximum_entity_vector_drift, drift
        )
        # JSON projections produced from the same float32 MIND source can
        # differ by one final float32 rounding bit. The cohort copy is never
        # used for scoring; both arms use the frozen workspace copy below.
        if drift > 2e-7:
            raise ValueError(
                "cohort/workspace entity vector source mismatch for "
                + article_id
            )
    users = workspace_data.get("users")
    if not isinstance(users, dict):
        raise ValueError("frozen workspace users must be a mapping")
    for user, sequence in chronological_user_cases(tests).items():
        first = sequence[0]
        history = [str(value) for value in first.get("history", ())]
        topics = Counter(
            str(articles[item].get("topic", articles[item].get(
                "category", "unknown"
            )))
            for item in history
        )
        users[user] = {
            "topics": [
                topic for topic, _count in sorted(
                    topics.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "recent_subcategories": [
                str(articles[item].get("subcategory", "unknown"))
                for item in history[-5:]
            ],
            "history": history,
        }
    workspace_data["tests"] = tests
    source_metadata = cohort_metadata
    metadata = workspace_data.setdefault("metadata", {})
    for key in (
        "official_behavior_source", "official_behavior_source_sha256",
        "official_behavior_join", "evaluation_chronology",
    ):
        metadata[key] = copy.deepcopy(source_metadata.get(key))
    metadata["personalization_cohort_binding"] = {
        "policy": (
            "label-blind complete-user cohort; every referenced article was "
            "required to exist in the frozen content workspace"
        ),
        "users": len({case["user"] for case in tests}),
        "impressions": len(tests),
        "required_articles": len(required),
        "workspace_extended": False,
        "static_feature_models_exact": True,
        "feature_model_sha256": model_hashes,
        "history_features_rebuilt_from_workspace_for_both_arms": True,
        "cohort_entity_vectors_used_for_scoring": False,
        "maximum_entity_vector_source_drift": maximum_entity_vector_drift,
    }
    return workspace_data


def _publish_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish one immutable, fsynced result artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded official-MIND personalization experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mind")
    parser.add_argument("--training-behaviors")
    parser.add_argument("--evaluation-behaviors")
    parser.add_argument(
        "--cohort-data",
        help=(
            "Prepared JSON/JSON.GZ output from the official behavior join; "
            "avoids rescanning MIND while preserving its audit metadata"
        ),
    )
    parser.add_argument("--serving-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--workspace-data",
        help=(
            "Frozen JSON/JSON.GZ content workspace used by the serving model; "
            "the official repeated-user cases are bound to it strictly"
        ),
    )
    parser.add_argument("--text-embeddings")
    parser.add_argument("--max-train-cases", type=int, default=20_000)
    parser.add_argument("--max-eval-users", type=int, default=5)
    parser.add_argument(
        "--eval-user-cohort", choices=("all", "unseen", "seen"),
        default="all",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-seed", type=int, default=71)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.max_train_cases < 1 or args.max_eval_users < 1:
        parser.error("case and user limits must be positive")
    if args.cohort_data:
        if any((args.mind, args.training_behaviors, args.evaluation_behaviors)):
            parser.error(
                "--cohort-data cannot be combined with raw MIND inputs"
            )
    elif not all((
        args.mind, args.training_behaviors, args.evaluation_behaviors,
    )):
        parser.error(
            "provide --cohort-data or all of --mind, --training-behaviors, "
            "and --evaluation-behaviors"
        )

    # Importing the server normally constructs its interactive singleton.
    # This command creates exactly one explicit, frozen-model Lab instead.
    os.environ.setdefault("RECOMMENDATION_DISABLE_DEFAULT_LAB", "1")
    from ..adapters.mind import load_mind
    from ..app.server import Lab, load_serving_model

    started = time.perf_counter()
    if args.cohort_data:
        cohort_path = Path(args.cohort_data)
        opener = gzip.open if cohort_path.suffix == ".gz" else open
        with opener(cohort_path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = load_mind(
            args.mind,
            max_train_cases=args.max_train_cases,
            max_eval_impressions=None,
            seed=args.seed,
            text_embedding_path=args.text_embeddings,
            training_behaviors_path=args.training_behaviors,
            evaluation_behaviors_path=args.evaluation_behaviors,
            max_eval_users=args.max_eval_users,
            eval_user_cohort=args.eval_user_cohort,
        )
    loaded_seconds = time.perf_counter() - started
    if args.workspace_data:
        workspace_path = Path(args.workspace_data)
        opener = gzip.open if workspace_path.suffix == ".gz" else open
        with opener(workspace_path, "rt", encoding="utf-8") as stream:
            workspace_data = json.load(stream)
        data = bind_personalization_cohort(workspace_data, data)
    model = load_serving_model(args.serving_model)
    lab = Lab(data=data, serving_model=model, serving_only=True)
    try:
        result = compare_personalization_modes(
            lab,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
    finally:
        lab.close()
    metadata = data.get("metadata", {})
    artifact = {
        "schema": "mindplex-symbolic-personalization-replay-v1",
        "result": result,
        "data_audit": {
            "dataset": metadata.get("dataset"),
            "official_behavior_source": metadata.get(
                "official_behavior_source"
            ),
            "official_behavior_source_sha256": metadata.get(
                "official_behavior_source_sha256"
            ),
            "official_behavior_join": metadata.get("official_behavior_join"),
            "evaluation_chronology": metadata.get("evaluation_chronology"),
            "training_chronology": metadata.get("training_chronology"),
            "personalization_cohort_binding": metadata.get(
                "personalization_cohort_binding"
            ),
            "selection_uses_outcomes": False,
        },
        "execution": {
            "dataset_load_seconds": loaded_seconds,
            "total_seconds": time.perf_counter() - started,
            "serving_model_path": str(Path(args.serving_model).resolve()),
            "serving_model_file_sha256": hashlib.sha256(
                Path(args.serving_model).read_bytes()
            ).hexdigest(),
            "source_declared_model_sha256": model.get("model_sha256"),
            "runtime_model_sha256": result["model_sha256"],
            "workspace_data_path": (
                str(Path(args.workspace_data).resolve())
                if args.workspace_data else None
            ),
            "cohort_data_path": (
                str(Path(args.cohort_data).resolve())
                if args.cohort_data else None
            ),
            "cohort_data_file_sha256": (
                hashlib.sha256(Path(args.cohort_data).read_bytes()).hexdigest()
                if args.cohort_data else None
            ),
        },
    }
    digest = _publish_json(Path(args.output), artifact)
    print(json.dumps({
        "status": "complete",
        "output": str(Path(args.output).resolve()),
        "artifact_sha256": digest,
        "model_sha256": result["model_sha256"],
        "users": result["users"],
        "evaluated_impressions": result["evaluated_impressions"],
        "arms": result["arms"],
        "total_seconds": artifact["execution"]["total_seconds"],
    }, allow_nan=False), flush=True)
    return 0


__all__ = [
    "PrivateReplayState",
    "bind_personalization_cohort",
    "chronological_user_cases",
    "compare_personalization_modes",
    "private_contexts",
    "run_prequential_personalization",
]


if __name__ == "__main__":
    raise SystemExit(main())
