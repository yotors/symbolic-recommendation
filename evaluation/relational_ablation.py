"""Run the fixed three-arm relational-evidence ablation.

All arms read the same immutable enriched dataset. Arms A and B deliberately
use the same non-relational feature profiles: their equality is a configuration
inertness and fresh-process reproducibility control, not a knowledge-only
ablation. Arm C changes only the relational
activation mode and the point/pair feature profiles so the bounded relational
conclusion is available to the miner and reasoner.

Each arm is executed by :mod:`recommendation.evaluation.symbolic_experiment`
in a new Python process.  Consequently it constructs a fresh ``Lab`` and a
fresh isolated PeTTaChainer worker; no proof or workspace cache can cross an
arm boundary.  The combined artifact retains the complete symbolic result,
hashes exact input and output orders, performs paired AUC comparisons, and is
published without overwriting an existing file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from ..cli.realtime_probe import ProcessTreeSampler
from ..features.relational_workspace import (
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    RELATIONAL_WORKSPACE_SCHEMA,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
    validate_relational_projection,
)
from .symbolic_compare import compare_artifacts


SCHEMA = "recommendation-relational-ablation-v1"
RELATIONAL_METADATA_KEY = "relational_workspace"
RELATIONAL_LEDGER_KEY = "relational_proof_ledger"
RELATIONAL_FIELD = REL_CONCEPT_CONTINUITY_SCOPE
RELATIONAL_FIELDS = (
    REL_ENTITY_CONTINUITY_SCOPE,
    REL_CONCEPT_CONTINUITY_SCOPE,
)
RELATIONAL_REFERENCE_FIELDS = (
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
)
BASE_POINT_PROFILE = "accuracy_detail"
BASE_PAIR_PROFILE = "llm_conditional_quantile"
RELATIONAL_POINT_PROFILE = "accuracy_detail_relational"
RELATIONAL_PAIR_PROFILE = "llm_conditional_quantile_relational"
ARM_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "A",
        "name": "existing_evidence",
        "relation_condition": (
            "facts_disabled_logically_by_nonrelational_feature_profiles; "
            "the same enriched dataset bytes are retained"
        ),
        "overrides": {
            "relational_evidence_mode": "disabled",
            "feature_profile": BASE_POINT_PROFILE,
            "pair_feature_profile": BASE_PAIR_PROFILE,
        },
    },
    {
        "id": "B",
        "name": "configuration_inertness_control",
        "relation_condition": (
            "the facts_only configuration marker is enabled, while relational "
            "records remain filtered by the same nonrelational feature profiles; "
            "this is not an additional-knowledge arm"
        ),
        "overrides": {
            "relational_evidence_mode": "facts_only",
            "feature_profile": BASE_POINT_PROFILE,
            "pair_feature_profile": BASE_PAIR_PROFILE,
        },
    },
    {
        "id": "C",
        "name": "bounded_relational_conclusion",
        "relation_condition": (
            "the bounded relational conclusion is projected into categorical "
            "point and pair evidence for fpMiner and PeTTaChainer"
        ),
        "overrides": {
            "relational_evidence_mode": "chained",
            "feature_profile": RELATIONAL_POINT_PROFILE,
            "pair_feature_profile": RELATIONAL_PAIR_PROFILE,
        },
    },
)


class RelationalAblationError(RuntimeError):
    """The experiment could not satisfy its fixed comparison protocol."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RelationalAblationError(f"{path}: expected a JSON object")
    return value


def _selected_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    selected = payload.get("result", payload)
    if isinstance(selected, Mapping):
        selected = selected.get("config", selected)
    if not isinstance(selected, Mapping):
        raise RelationalAblationError(
            "selected config source does not contain a config object"
        )
    return dict(selected)


def _evaluation_source(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    found = [
        data[key] for key in (
            "eval_impressions", "evaluation", "tests", "impressions",
        ) if isinstance(data.get(key), list)
    ]
    if len(found) != 1:
        raise RelationalAblationError(
            "dataset must contain exactly one supported evaluation-slate list"
        )
    if any(not isinstance(row, Mapping) for row in found[0]):
        raise RelationalAblationError("evaluation slates must be JSON objects")
    return found[0]


def _case_id(case: Mapping[str, Any]) -> str:
    raw = case.get("source_impression_id") or case.get("id") or case.get("impression_id")
    if raw is None or not str(raw).strip():
        raise RelationalAblationError("evaluation slate lacks an impression ID")
    return str(raw)


def _relation_context_counts(data: Mapping[str, Any]) -> dict[str, Any]:
    """Count the declared categorical relation without inspecting outcomes."""

    split_counts: dict[str, dict[str, Any]] = {}
    sources = [("training", data.get("events", [])),
               ("evaluation", _evaluation_source(data))]
    for split, rows in sources:
        values = {field: Counter() for field in RELATIONAL_FIELDS}
        contexts = 0
        present = Counter()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            candidate_context = row.get("candidate_context")
            if not isinstance(candidate_context, Mapping):
                # A point training event may itself be its candidate context.
                candidate_context = {"_event": row}
            for context in candidate_context.values():
                if not isinstance(context, Mapping):
                    continue
                contexts += 1
                for field in RELATIONAL_FIELDS:
                    value = context.get(field)
                    if value is None:
                        continue
                    present[field] += 1
                    values[field][str(value)] += 1
        split_counts[split] = {
            "candidate_contexts": contexts,
            "fields": {
                field: {
                    "present": present[field],
                    "absent": contexts - present[field],
                    "value_counts": dict(sorted(values[field].items())),
                }
                for field in RELATIONAL_FIELDS
            },
        }
    return split_counts


def _audit_leaves(value: Any, *, path: tuple[str, ...] = ()) -> dict[str, Any]:
    """Retain reported relation hashes, counts, timings and safety checks.

    The projector owns the exact metadata layout.  Matching leaf names rather
    than one private layout lets this runner retain new audit fields without
    weakening the stable hash over the complete metadata object.
    """

    selected: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            selected.update(_audit_leaves(child, path=(*path, str(key))))
        return selected
    if isinstance(value, (list, tuple)):
        # Lists of structured records can be large. Their exact value is bound
        # by the enclosing object hash; retain only scalar audit lists here.
        if all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            lowered = ".".join(path).lower()
            if any(token in lowered for token in (
                "sha256", "hash", "count", "seconds", "timing", "duplicate",
                "future", "leak", "causal", "provenance", "rule", "graph",
            )):
                selected[".".join(path)] = list(value)
        return selected
    lowered = ".".join(path).lower()
    if any(token in lowered for token in (
        "sha256", "hash", "count", "seconds", "timing", "duplicate",
        "future", "leak", "causal", "provenance", "rule", "graph",
    )):
        selected[".".join(path)] = value
    return selected


def _ledger_records(ledger: Any) -> tuple[list[Any], str]:
    if isinstance(ledger, list):
        return list(ledger), "list"
    if isinstance(ledger, Mapping):
        for key in ("records", "entries", "proofs"):
            if isinstance(ledger.get(key), list):
                return list(ledger[key]), f"mapping.{key}"
        return [[str(key), ledger[key]] for key in sorted(ledger, key=str)], "mapping"
    raise RelationalAblationError(
        f"{RELATIONAL_LEDGER_KEY} must be a list or object"
    )


def _ledger_audit(ledger: Any) -> dict[str, Any]:
    records, layout = _ledger_records(ledger)
    record_hashes = [_digest(record) for record in records]
    ids: list[str] = []
    if isinstance(ledger, Mapping) and not any(
        isinstance(ledger.get(key), list) for key in ("records", "entries", "proofs")
    ):
        ids = [str(key) for key in ledger]
    for record in records:
        candidate = record[1] if (
            isinstance(record, list) and len(record) == 2
            and isinstance(record[1], Mapping)
        ) else record
        if not isinstance(candidate, Mapping):
            continue
        for key in ("proof_id", "evidence_id", "id"):
            if candidate.get(key) is not None:
                ids.append(str(candidate[key]))
                break
    return {
        "layout": layout,
        "records": len(records),
        "sha256": _digest(ledger),
        "record_multiset_sha256": _digest(sorted(record_hashes)),
        "duplicate_exact_records": len(record_hashes) - len(set(record_hashes)),
        "identified_records": len(ids),
        "duplicate_primary_ids": len(ids) - len(set(ids)),
    }


def _relational_provenance_audit(data: Mapping[str, Any]) -> dict[str, Any]:
    ledger = data[RELATIONAL_LEDGER_KEY]
    if not isinstance(ledger, Mapping):
        # The canonical projection currently uses an ID-keyed ledger. The
        # generic ledger audit above remains useful for forward-compatible
        # diagnostics, while causal-reference validation fails closed here.
        raise RelationalAblationError("relational proof ledger must be ID-keyed")
    contexts = []
    for event in data.get("events", []):
        if isinstance(event, Mapping):
            contexts.append((
                str(event.get("user") or ""), str(event.get("article") or ""),
                list(event.get("history") or ()), event,
            ))
    for case in _evaluation_source(data):
        user = str(case.get("user") or "")
        history = list(case.get("history") or ())
        candidate_context = case.get("candidate_context") or {}
        if not isinstance(candidate_context, Mapping):
            raise RelationalAblationError("evaluation candidate_context must be an object")
        for candidate, context in candidate_context.items():
            if isinstance(context, Mapping):
                contexts.append((user, str(candidate), history, context))

    references = []
    duplicate_references = 0
    missing_records = 0
    future_or_wrong_history_references = 0
    user_or_candidate_mismatches = 0
    for user, candidate, history, context in contexts:
        context_references = []
        for field in RELATIONAL_REFERENCE_FIELDS:
            raw = context.get(field)
            if not isinstance(raw, list):
                raise RelationalAblationError(
                    "relational candidate context lacks a proof-reference list"
                )
            normalized = [str(value) for value in raw]
            duplicate_references += len(normalized) - len(set(normalized))
            context_references.extend(normalized)
        references.extend(context_references)
        for proof_id in context_references:
            record = ledger.get(proof_id)
            if not isinstance(record, Mapping):
                missing_records += 1
                continue
            position = record.get("history_position")
            if (isinstance(position, bool) or not isinstance(position, int)
                    or not 0 <= position < len(history)
                    or str(history[position]) != str(record.get("history_article_id"))):
                future_or_wrong_history_references += 1
            if (str(record.get("user_id")) != user
                    or str(record.get("candidate_id")) != candidate):
                user_or_candidate_mismatches += 1
    referenced = set(references)
    forbidden = {
        "action", "click", "engagement", "label", "labels", "outcome",
        "relevant", "score",
    }
    forbidden_record_fields = sum(
        bool(forbidden.intersection(record))
        for record in ledger.values() if isinstance(record, Mapping)
    )
    origin_keys = [
        (str(record.get("case_id")), str(record.get("origin_id")))
        for record in ledger.values() if isinstance(record, Mapping)
    ]
    checks = {
        "duplicate_context_proof_references": duplicate_references,
        "missing_ledger_records": missing_records,
        "future_or_wrong_history_references": future_or_wrong_history_references,
        "user_or_candidate_mismatches": user_or_candidate_mismatches,
        "forbidden_outcome_fields_in_ledger": forbidden_record_fields,
        "duplicate_case_origin_records": len(origin_keys) - len(set(origin_keys)),
    }
    result = {
        "candidate_contexts": len(contexts),
        "proof_references": len(references),
        "unique_referenced_proofs": len(referenced),
        "unreferenced_ledger_records": len(set(map(str, ledger)) - referenced),
        "checks": checks,
        "all_checks_passed": all(value == 0 for value in checks.values()),
    }
    if not result["all_checks_passed"]:
        raise RelationalAblationError(
            "relational provenance contains duplicate, missing, future, or "
            "misbound evidence"
        )
    return result


def relational_dataset_audit(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and summarize the immutable relational enrichment."""

    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RelationalAblationError("dataset metadata must be an object")
    workspace = metadata.get(RELATIONAL_METADATA_KEY)
    if not isinstance(workspace, Mapping):
        raise RelationalAblationError(
            f"enriched dataset metadata lacks {RELATIONAL_METADATA_KEY}"
        )
    schema = workspace.get("schema")
    if schema != RELATIONAL_WORKSPACE_SCHEMA:
        raise RelationalAblationError(
            "relational workspace metadata has an unsupported schema"
        )
    if RELATIONAL_LEDGER_KEY not in data:
        raise RelationalAblationError(
            f"enriched dataset lacks {RELATIONAL_LEDGER_KEY}"
        )
    ledger = data[RELATIONAL_LEDGER_KEY]
    try:
        validate_relational_projection(data)
    except (TypeError, ValueError) as exc:
        raise RelationalAblationError(
            f"relational workspace validation failed: {exc}"
        ) from exc
    structural_rule_ids = workspace.get("structural_rule_ids") or []
    if (not isinstance(structural_rule_ids, list)
            or any(not isinstance(value, str) for value in structural_rule_ids)):
        raise RelationalAblationError(
            "relational structural_rule_ids must be a list of strings"
        )
    return {
        "schema": schema,
        "metadata_sha256": _digest(workspace),
        "reported_metadata": dict(workspace),
        "reported_audit_fields": _audit_leaves(workspace),
        "ledger": _ledger_audit(ledger),
        "proof_dependency_graph_sha256": _digest(ledger),
        "structural_rule_count": len(structural_rule_ids),
        "structural_rule_ids_sha256": _digest(structural_rule_ids),
        "provenance": _relational_provenance_audit(data),
        "active_derived_field": RELATIONAL_FIELD,
        "derived_fields": list(RELATIONAL_FIELDS),
        "derived_field_counts": _relation_context_counts(data),
    }


def _case_map(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for case in _evaluation_source(data):
        identity = _case_id(case)
        if identity in result:
            raise RelationalAblationError(
                f"duplicate evaluation impression ID: {identity}"
            )
        result[identity] = case
    return result


def cohort_audit(
    artifact: Mapping[str, Any], data: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one arm to its exact admitted input and ranked output orders."""

    result = artifact.get("result")
    rankings = artifact.get("ranked_impressions")
    if not isinstance(result, Mapping) or not isinstance(rankings, list):
        raise RelationalAblationError(
            "arm artifact must contain a result and captured rankings"
        )
    rows = result.get("auc_per_impression")
    if not isinstance(rows, list) or not rows:
        raise RelationalAblationError("arm has no AUC-eligible impressions")
    cases = _case_map(data)
    records = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RelationalAblationError("AUC impression record must be an object")
        identity = str(row.get("id") or "")
        case = cases.get(identity)
        if case is None:
            raise RelationalAblationError(
                f"arm impression {identity!r} is absent from the dataset"
            )
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rankings):
            raise RelationalAblationError(
                f"arm impression {identity!r} has an invalid ranking index"
            )
        ranked_rows = rankings[index]
        if not isinstance(ranked_rows, list) or any(
            not isinstance(item, Mapping) or item.get("article") is None
            for item in ranked_rows
        ):
            raise RelationalAblationError("captured ranking is malformed")
        ranked = [str(item["article"]) for item in ranked_rows]
        if len(ranked) != len(set(ranked)):
            raise RelationalAblationError(
                f"ranked output contains duplicate candidates for {identity}"
            )
        source = [str(value) for value in case.get("candidates", [])]
        if len(source) != len(set(source)):
            raise RelationalAblationError(
                f"source slate contains duplicate candidates for {identity}"
            )
        ranked_set = set(ranked)
        ordered_admitted = [candidate for candidate in source if candidate in ranked_set]
        if len(ordered_admitted) != len(ranked) or set(ordered_admitted) != ranked_set:
            raise RelationalAblationError(
                f"captured ranking changed the candidate set for {identity}"
            )
        relevant_set = {str(value) for value in case.get("relevant", [])}
        ordered_relevant = [
            candidate for candidate in ordered_admitted if candidate in relevant_set
        ]
        if len(ordered_relevant) != row.get("positives"):
            raise RelationalAblationError(
                f"relevant count disagrees for impression {identity}"
            )
        records.append({
            "index": index,
            "id": identity,
            "candidate_count": len(ordered_admitted),
            "relevant_count": len(ordered_relevant),
            "ordered_slate_sha256": _digest(ordered_admitted),
            "ordered_relevant_sha256": _digest(ordered_relevant),
            "ranked_output_sha256": _digest(ranked),
            "ranked_score_records_sha256": _digest(ranked_rows),
        })
    records.sort(key=lambda item: item["index"])
    return {
        "impressions": len(records),
        "candidates": sum(row["candidate_count"] for row in records),
        "relevant": sum(row["relevant_count"] for row in records),
        "impression_ids_sha256": _digest([row["id"] for row in records]),
        "ordered_slates_sha256": _digest([
            [row["id"], row["ordered_slate_sha256"]] for row in records
        ]),
        "ordered_relevant_sets_sha256": _digest([
            [row["id"], row["ordered_relevant_sha256"]] for row in records
        ]),
        "ranked_outputs_sha256": _digest([
            [row["id"], row["ranked_output_sha256"]] for row in records
        ]),
        "ranked_score_records_sha256": _digest([
            [row["id"], row["ranked_score_records_sha256"]] for row in records
        ]),
        "per_impression": records,
    }


def _rule_audit(artifact: Mapping[str, Any]) -> dict[str, Any]:
    point = artifact.get("point_rules") or []
    pair = artifact.get("pair_rules") or []
    point_sources = artifact.get("compiled_point_channels") or {}
    pair_sources = artifact.get("compiled_pair_rules") or {}

    def premises(rule: Any) -> tuple[tuple[str, str], ...]:
        if not isinstance(rule, Mapping):
            return ()
        raw = rule.get("premises")
        if not isinstance(raw, (list, tuple)):
            return ()
        parsed = []
        for premise in raw:
            if not isinstance(premise, (list, tuple)) or len(premise) != 2:
                continue
            parsed.append((str(premise[0]), str(premise[1])))
        return tuple(parsed)

    def relation_rules(rules: Any, field: str | None = None) -> list[Any]:
        if not isinstance(rules, list):
            raise RelationalAblationError("mined rules must be lists")
        predicates = set()
        for token in (RELATIONAL_FIELDS if field is None else (field,)):
            predicates.update((token, f"pair_{token}"))
        return [
            rule for rule in rules
            if any(predicate in predicates for predicate, _value in premises(rule))
        ]

    def value_summary(
        rules: list[Any], predicate: str, *, proof_positive_values: set[str],
    ) -> dict[str, Any]:
        matched = []
        values = Counter()
        proof_positive = []
        for rule in rules:
            rule_values = {
                value for rule_predicate, value in premises(rule)
                if rule_predicate == predicate
            }
            if not rule_values:
                continue
            matched.append(rule)
            values.update(rule_values)
            if rule_values.intersection(proof_positive_values):
                proof_positive.append(rule)
        return {
            "rules": len(matched),
            "rules_by_premise_value": dict(sorted(values.items())),
            "proof_positive_values": sorted(proof_positive_values),
            "proof_positive_mined_rules": len(proof_positive),
            "proof_positive_mined_rules_sha256": _digest(proof_positive),
            "interpretation": (
                "mined-rule availability only; it does not establish that the "
                "rule fired in an evaluated ranking"
            ),
        }

    relational_point = relation_rules(point)
    relational_pair = relation_rules(pair)
    by_field = {}
    for field in RELATIONAL_FIELDS:
        point_summary = value_summary(
            point, field, proof_positive_values={"older", "recent"},
        )
        pair_summary = value_summary(
            pair, f"pair_{field}", proof_positive_values={"left", "right"},
        )
        by_field[field] = {
            # Preserve the original compact counts for artifact consumers.
            "point_rules": point_summary["rules"],
            "pair_rules": pair_summary["rules"],
            "point": point_summary,
            "pair": pair_summary,
        }
    return {
        "point_rules": len(point),
        "pair_rules": len(pair),
        "point_rules_sha256": _digest(point),
        "pair_rules_sha256": _digest(pair),
        "compiled_point_channels_sha256": _digest(point_sources),
        "compiled_pair_rules_sha256": _digest(pair_sources),
        "relational_point_rules": len(relational_point),
        "relational_pair_rules": len(relational_pair),
        "relational_point_rules_sha256": _digest(relational_point),
        "relational_pair_rules_sha256": _digest(relational_pair),
        "proof_positive_definition": {
            "point": "a premise value of recent or older",
            "pair": (
                "a directional left or right comparison; equal and availability "
                "states do not prove that either candidate has a positive path"
            ),
            "scope": (
                "mined rule inventory, not evaluation-time rule activation"
            ),
        },
        "proof_positive_relational_point_rules": sum(
            item["point"]["proof_positive_mined_rules"]
            for item in by_field.values()
        ),
        "proof_positive_relational_pair_rules": sum(
            item["pair"]["proof_positive_mined_rules"]
            for item in by_field.values()
        ),
        "by_derived_field": by_field,
    }


def _evaluation_point_activation_audit(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit proof-positive relation use in the captured held-out rankings.

    The capture records rule IDs only after the normal PeTTa point-proof path
    has returned. This is therefore an evaluation-time activation audit, not
    merely an inventory of rules that happened to be mined.
    """

    rules = artifact.get("point_rules") or []
    rankings = artifact.get("ranked_impressions")
    if not isinstance(rules, list) or not isinstance(rankings, list):
        raise RelationalAblationError(
            "point activation audit requires rules and captured rankings"
        )
    rules_by_id = {
        str(rule["id"]): rule for rule in rules
        if isinstance(rule, Mapping) and rule.get("id") is not None
    }
    rows = [
        row for slate in rankings if isinstance(slate, list)
        for row in slate if isinstance(row, Mapping)
    ]
    proof_positive_candidates = 0
    candidates_by_scope_value = {
        field: Counter() for field in RELATIONAL_FIELDS
    }
    fired_relational = 0
    fired_proof_positive = 0
    fired_by_field_value = Counter()
    fired_rule_ids = set()
    positive_rule_ids = set()
    for row in rows:
        evidence = row.get("relational_evidence") or {}
        scopes = evidence.get("scopes") or {}
        proof_ids = evidence.get("proof_ids") or {}
        if not isinstance(scopes, Mapping) or not isinstance(proof_ids, Mapping):
            raise RelationalAblationError(
                "captured relational evidence must contain mapping scopes and proof_ids"
            )
        row_is_proof_positive = False
        for field in RELATIONAL_FIELDS:
            value = scopes.get(field)
            if value is not None:
                candidates_by_scope_value[field][str(value)] += 1
            if value in {"recent", "older"}:
                reference_field = (
                    REL_ENTITY_CONTINUITY_PROOF_IDS
                    if field == REL_ENTITY_CONTINUITY_SCOPE
                    else REL_CONCEPT_CONTINUITY_PROOF_IDS
                )
                references = proof_ids.get(reference_field)
                if not isinstance(references, list) or not references:
                    raise RelationalAblationError(
                        "a proof-positive captured relation lacks proof IDs"
                    )
                row_is_proof_positive = True
        proof_positive_candidates += row_is_proof_positive

        raw_ids = row.get("point_rule_ids")
        if not isinstance(raw_ids, list):
            raise RelationalAblationError(
                "captured ranking row lacks point_rule_ids"
            )
        for raw_id in raw_ids:
            rule_id = str(raw_id)
            rule = rules_by_id.get(rule_id)
            if rule is None:
                raise RelationalAblationError(
                    f"captured point proof references unknown rule {rule_id!r}"
                )
            relational_premises = [
                (str(premise[0]), str(premise[1]))
                for premise in rule.get("premises", ())
                if isinstance(premise, (list, tuple)) and len(premise) == 2
                and str(premise[0]) in RELATIONAL_FIELDS
            ]
            if not relational_premises:
                continue
            fired_relational += 1
            fired_rule_ids.add(rule_id)
            positive = False
            for field, value in relational_premises:
                if scopes.get(field) != value:
                    raise RelationalAblationError(
                        "captured point-rule activation disagrees with its relation scope"
                    )
                fired_by_field_value[(field, value)] += 1
                positive = positive or value in {"recent", "older"}
            if positive:
                fired_proof_positive += 1
                positive_rule_ids.add(rule_id)
    return {
        "candidate_rows": len(rows),
        "proof_positive_candidate_rows": proof_positive_candidates,
        "candidate_scope_values": {
            field: dict(sorted(values.items()))
            for field, values in candidates_by_scope_value.items()
        },
        "fired_relational_point_rule_uses": fired_relational,
        "fired_proof_positive_relational_point_rule_uses": fired_proof_positive,
        "fired_relational_point_rule_ids": sorted(fired_rule_ids),
        "fired_proof_positive_relational_point_rule_ids": sorted(
            positive_rule_ids
        ),
        "fired_by_field_and_value": {
            f"{field}={value}": count
            for (field, value), count in sorted(fired_by_field_value.items())
        },
        "unit": "candidate-rule activation after a successful PeTTa point proof",
    }


def _metric_summary(artifact: Mapping[str, Any]) -> dict[str, Any]:
    result = artifact.get("result")
    if not isinstance(result, Mapping):
        raise RelationalAblationError("arm artifact has no result object")
    fields = (
        "auc", "auc_proof_only", "auc_95_ci", "auc_proof_only_95_ci",
        "mrr", "ndcg_at_5", "ndcg_at_10", "cases", "candidates",
        "auc_cases", "proof_coverage", "pairwise_proof_coverage",
        "pairwise_directional_coverage", "seconds", "timings",
        "ranking_workload", "proof_cache", "candidate_retrieval",
    )
    return {key: result[key] for key in fields if key in result}


def _config_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    differences = {}
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            differences[key] = {"left": left.get(key), "right": right.get(key)}
    return differences


def _source_hashes(root: Path) -> dict[str, str]:
    relative_paths = (
        "recommendation/evaluation/relational_ablation.py",
        "recommendation/evaluation/symbolic_experiment.py",
        "recommendation/evaluation/symbolic_compare.py",
        "recommendation/features/relational_workspace.py",
        "recommendation/pipelines/relational_data.py",
        "recommendation/app/server.py",
        "recommendation/miner/fpMiner.metta",
        "PeTTaChainer/pettachainer/pettachainer.py",
        "PeTTaChainer/pettachainer/metta/petta_chainer.metta",
    )
    return {
        relative: _file_sha256(root / relative)
        for relative in relative_paths if (root / relative).is_file()
    }


def _runtime_dependency_provenance() -> dict[str, Any]:
    """Bind the experiment to the modules actually imported by this process."""

    # Match the scorer and projection entry points. The historical launch
    # command may still export ``PYTHONPATH=PeTTa/python:PeTTaChainer``; that
    # checkout is API-incompatible with the installed PeTTaChainer runtime and
    # is deliberately removed before either execution or fingerprinting.
    legacy_petta = (Path(__file__).resolve().parents[2] / "PeTTa" / "python").resolve()
    sys.path[:] = [
        entry for entry in sys.path
        if not entry or Path(entry).resolve() != legacy_petta
    ]

    packages = {
        "petta": {
            "distribution": "petta",
            "modules": ("petta",),
        },
        "janus_swi": {
            "distribution": "janus_swi",
            "modules": ("janus_swi", "janus_swi.janus", "janus_swi._swipl"),
        },
        "pettachainer": {
            "distribution": "PeTTaChainer",
            "modules": ("pettachainer", "pettachainer.pettachainer"),
        },
    }
    result = {}
    for package, specification in packages.items():
        distribution_name = specification["distribution"]
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RelationalAblationError(
                f"active runtime distribution is not installed: {distribution_name}"
            ) from exc
        modules = {}
        for module_name in specification["modules"]:
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise RelationalAblationError(
                    f"active runtime module cannot be imported: {module_name}"
                ) from exc
            origin_value = getattr(module, "__file__", None)
            if not origin_value:
                raise RelationalAblationError(
                    f"active runtime module has no hashable origin: {module_name}"
                )
            origin = Path(origin_value).resolve()
            if not origin.is_file():
                raise RelationalAblationError(
                    f"active runtime module origin is not a file: {origin}"
                )
            modules[module_name] = {
                "origin": str(origin),
                "bytes": origin.stat().st_size,
                "sha256": _file_sha256(origin),
            }
        result[package] = {
            "distribution": str(distribution.metadata.get("Name") or distribution_name),
            "version": distribution.version,
            "modules": modules,
        }
    return result


def _reduced_process_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "root_pid": resources.get("root_pid"),
        "sampling_interval_seconds": resources.get("sampling_interval_seconds"),
        "clock_ticks_per_second": resources.get("clock_ticks_per_second"),
        "sample_count": resources.get("sample_count"),
        "summary": resources.get("summary"),
        "process_commands": resources.get("process_commands"),
    }


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_arm(
    *, spec: Mapping[str, Any], data_path: Path, config_path: Path,
    output_path: Path, evidence_mode: str, eval_limit: int,
    resource_interval: float, timeout_seconds: float, root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overrides = json.dumps(spec["overrides"], sort_keys=True, separators=(",", ":"))
    command = [
        sys.executable, "-m", "recommendation.evaluation.symbolic_experiment",
        "--data", str(data_path), "--evidence-mode", evidence_mode,
        "--config-file", str(config_path), "--config-overrides", overrides,
        "--output", str(output_path), "--force-cold-proof-cache",
        "--capture-rankings",
    ]
    if eval_limit:
        command.extend(("--eval-limit", str(eval_limit)))
    child_times_before = os.times()
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    sampler = ProcessTreeSampler(process.pid, resource_interval)
    sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise RelationalAblationError(
            f"arm {spec['id']} exceeded {timeout_seconds:.3f} seconds"
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    finally:
        resources = sampler.stop()
    elapsed = time.monotonic() - started
    child_times_after = os.times()
    execution = {
        "command": [
            "<temporary-arm-output>" if item == str(output_path) else item
            for item in command
        ],
        "exit_code": process.returncode,
        "wall_seconds": elapsed,
        "child_user_cpu_seconds": (
            child_times_after.children_user - child_times_before.children_user
        ),
        "child_system_cpu_seconds": (
            child_times_after.children_system - child_times_before.children_system
        ),
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "stdout_tail": stdout[-20000:],
        "stderr_tail": stderr[-20000:],
        "resources": _reduced_process_resources(resources),
    }
    if process.returncode != 0:
        raise RelationalAblationError(
            f"arm {spec['id']} failed with exit code {process.returncode}:\n"
            f"{stderr[-4000:]}"
        )
    if not output_path.is_file():
        raise RelationalAblationError(f"arm {spec['id']} produced no artifact")
    execution["artifact_bytes"] = output_path.stat().st_size
    execution["artifact_sha256"] = _file_sha256(output_path)
    return _load_json(output_path), execution


def _strip_rankings(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "ranked_impressions"}


def assemble_report(
    *, data: Mapping[str, Any], dataset_path: Path, dataset_sha256: str,
    config_path: Path, config_sha256: str,
    arm_runs: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    comparison_seed: int, bootstrap_repetitions: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate arm isolation/equality and assemble the combined artifact."""

    if set(arm_runs) != {"A", "B", "C"}:
        raise RelationalAblationError("arm runs must contain exactly A, B and C")
    relation_audit = relational_dataset_audit(data)
    selected_config = _selected_config(config_path)
    arms: dict[str, Any] = {}
    raw_artifacts: dict[str, Mapping[str, Any]] = {}
    specs = {spec["id"]: spec for spec in ARM_SPECS}
    for arm_id in ("A", "B", "C"):
        artifact, execution = arm_runs[arm_id]
        if artifact.get("dataset_sha256") != dataset_sha256:
            raise RelationalAblationError(
                f"arm {arm_id} did not bind the supplied dataset bytes"
            )
        result = artifact.get("result")
        if not isinstance(result, Mapping) or not isinstance(result.get("config"), Mapping):
            raise RelationalAblationError(f"arm {arm_id} lacks its effective config")
        effective_config = result["config"]
        for key, value in specs[arm_id]["overrides"].items():
            if effective_config.get(key) != value:
                raise RelationalAblationError(
                    f"arm {arm_id} did not apply override {key}"
                )
        preserved = {
            key: value for key, value in selected_config.items()
            if key not in specs[arm_id]["overrides"]
        }
        changed_preserved = {
            key: {"selected": value, "effective": effective_config.get(key)}
            for key, value in preserved.items()
            if effective_config.get(key) != value
        }
        if changed_preserved:
            raise RelationalAblationError(
                f"arm {arm_id} changed selected configuration outside its overrides"
            )
        cache = result.get("proof_cache") or {}
        if (cache.get("requested_mode") != "force_cold"
                or cache.get("observed_mode") != "cold"):
            raise RelationalAblationError(f"arm {arm_id} was not proof-cache cold")
        cohort = cohort_audit(artifact, data)
        raw_artifacts[arm_id] = artifact
        arms[arm_id] = {
            "name": specs[arm_id]["name"],
            "relation_condition": specs[arm_id]["relation_condition"],
            "requested_overrides": specs[arm_id]["overrides"],
            "effective_config": dict(effective_config),
            "execution": dict(execution),
            "metrics": _metric_summary(artifact),
            "rules": _rule_audit(artifact),
            "evaluation_relational_activation": {
                "captured_point_proofs": (
                    _evaluation_point_activation_audit(artifact)
                ),
                "engine_workload": result.get(
                    "evaluation_relational_activation_audit",
                    {"status": "not_reported"},
                ),
            },
            "cohort": cohort,
            "symbolic_artifact_without_ranked_rows": _strip_rankings(artifact),
        }

    base_config_differences = _config_differences(
        arms["A"]["effective_config"], arms["B"]["effective_config"],
    )
    challenger_config_differences = _config_differences(
        arms["B"]["effective_config"], arms["C"]["effective_config"],
    )
    if set(base_config_differences) != {"relational_evidence_mode"}:
        raise RelationalAblationError(
            "B must differ from A only by relational_evidence_mode"
        )
    expected_challenger_keys = {
        "feature_profile", "pair_feature_profile", "relational_evidence_mode",
    }
    if set(challenger_config_differences) != expected_challenger_keys:
        raise RelationalAblationError(
            "C must differ from B only by relation mode and point/pair profiles"
        )
    cohort_identity_keys = (
        "impressions", "candidates", "relevant", "impression_ids_sha256",
        "ordered_slates_sha256", "ordered_relevant_sets_sha256",
    )
    cohort_matches = all(
        all(arms[arm]["cohort"][key] == arms["A"]["cohort"][key]
            for key in cohort_identity_keys)
        for arm in ("B", "C")
    )
    if not cohort_matches:
        raise RelationalAblationError("arms did not score identical ordered input slates")

    a_b_exact_rank = (
        arms["A"]["cohort"]["ranked_outputs_sha256"]
        == arms["B"]["cohort"]["ranked_outputs_sha256"]
    )
    a_b_exact_scores = (
        arms["A"]["cohort"]["ranked_score_records_sha256"]
        == arms["B"]["cohort"]["ranked_score_records_sha256"]
    )
    rule_hash_fields = (
        "point_rules_sha256", "pair_rules_sha256",
        "compiled_point_channels_sha256", "compiled_pair_rules_sha256",
    )
    a_b_exact_rules = all(
        arms["A"]["rules"][key] == arms["B"]["rules"][key]
        for key in rule_hash_fields
    )
    if not a_b_exact_rank or not a_b_exact_scores or not a_b_exact_rules:
        raise RelationalAblationError(
            "inert relational-fact control changed rules or ranking output"
        )

    comparisons = {}
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        comparisons[f"{right}_minus_{left}"] = compare_artifacts(
            raw_artifacts[left], raw_artifacts[right], data=data,
            data_sha256=dataset_sha256, seed=comparison_seed,
            repetitions=bootstrap_repetitions,
        )

    engine_activation = arms["C"]["evaluation_relational_activation"].get(
        "engine_workload", {}
    )
    point_positive = int(
        (engine_activation.get("point") or {}).get(
            "proof_positive_relational_rule_activations", 0
        )
    ) if isinstance(engine_activation, Mapping) else 0
    pair_positive = int(
        (engine_activation.get("pair") or {}).get(
            "proof_positive_relational_rule_activations", 0
        )
    ) if isinstance(engine_activation, Mapping) else 0
    proof_gate_complete = bool(
        engine_activation.get("proof_gate_complete")
    ) if isinstance(engine_activation, Mapping) else False
    demonstrated = proof_gate_complete and point_positive + pair_positive > 0

    return {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "protocol": {
            "description": (
                "three fresh-process, force-cold-proof-cache arms on identical "
                "enriched dataset bytes and exact logged slates"
            ),
            "arm_order": ["A", "B", "C"],
            "inert_control": (
                "A and B both filter rel_* fields. B changes only the inert "
                "facts_only configuration marker, so their equality is a "
                "fresh-process determinism/configuration-inertness check. It "
                "is not presented as an additional-knowledge ablation"
            ),
            "changed_in_C": sorted(expected_challenger_keys),
            "fixed": (
                "all other selected configuration, candidate/relevant slates, "
                "dataset bytes, seeds, ranking aggregation and evaluation protocol"
            ),
            "evidence_interpretation": (
                "B-A is an inertness/reproducibility control; C-B and C-A test "
                "the complete relational feature/mining/reasoning pipeline, not "
                "a unique causal contribution of the proof engine alone"
            ),
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_sha256,
            "bytes": dataset_path.stat().st_size,
        },
        "selected_config_source": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
            "selected_config": selected_config,
        },
        "relational_workspace": relation_audit,
        "controls": {
            "same_dataset_sha256": True,
            "same_ordered_input_cohort": cohort_matches,
            "A_B_only_mode_differs": (
                set(base_config_differences) == {"relational_evidence_mode"}
            ),
            "A_B_config_differences": base_config_differences,
            "A_B_ranked_output_exact": a_b_exact_rank,
            "A_B_ranked_score_records_exact": a_b_exact_scores,
            "A_B_rule_artifacts_exact": a_b_exact_rules,
            "B_C_config_differences": challenger_config_differences,
            "all_arms_fresh_process": True,
            "all_arms_force_cold_proof_cache": True,
        },
        "reasoning_contribution_evidence": {
            "status": (
                "proof_positive_activation_demonstrated"
                if demonstrated else
                "no_proof_positive_activation_demonstrated"
            ),
            "proof_gate_complete": proof_gate_complete,
            "proof_positive_point_rule_activations": point_positive,
            "proof_positive_pair_rule_activations": pair_positive,
            "interpretation": (
                "Counts are evaluation-time applicable PeTTa-proved channels. "
                "They establish that chained evidence reached the ranking "
                "pipeline, but a weaker fired variant may still be discarded "
                "by dependency-max fusion and is not necessarily a retained "
                "numerical contribution. Accuracy causality is evaluated only "
                "for the complete C-minus-A pipeline."
            ),
        },
        "arms": arms,
        "comparisons": comparisons,
        "bootstrap": {
            "seed": comparison_seed,
            "repetitions": bootstrap_repetitions,
            "paired_impression_and_user_cluster": True,
        },
        "runtime": dict(runtime),
    }


def _atomic_publish_json(target: Path, payload: Mapping[str, Any]) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError("output must be new; existing artifacts are not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_three_arm_ablation(
    *, data_path: Path, config_path: Path, output_path: Path,
    evidence_mode: str = "llm_workspace", eval_limit: int = 0,
    comparison_seed: int = 37, bootstrap_repetitions: int = 2000,
    resource_interval: float = 0.25, arm_timeout_seconds: float = 7200.0,
) -> dict[str, Any]:
    orchestration_started = time.monotonic()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("output must be new; existing artifacts are not overwritten")
    if eval_limit < 0:
        raise ValueError("eval_limit must be nonnegative")
    if evidence_mode != "llm_workspace":
        raise ValueError(
            "the canonical-concept relational ablation requires llm_workspace"
        )
    if bootstrap_repetitions < 100:
        raise ValueError("bootstrap_repetitions must be at least 100")
    if resource_interval <= 0 or arm_timeout_seconds <= 0:
        raise ValueError("resource interval and arm timeout must be positive")
    data_path = data_path.resolve()
    config_path = config_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = _load_json(data_path)
    dataset_sha256 = _file_sha256(data_path)
    config_sha256 = _file_sha256(config_path)
    # Fail before the expensive arms if enrichment is absent or malformed.
    relational_dataset_audit(data)
    root = Path(__file__).resolve().parents[2]
    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "workspace_source_sha256": _source_hashes(root),
        "active_runtime_dependencies": _runtime_dependency_provenance(),
        "started_at": _utc_now(),
        "evidence_mode": evidence_mode,
        "eval_limit": eval_limit,
        "resource_sampling_interval_seconds": resource_interval,
        "arm_timeout_seconds": arm_timeout_seconds,
        "child_python_hash_seed": "0",
    }
    arm_runs: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    with tempfile.TemporaryDirectory(
        dir=output_path.parent, prefix=f".{output_path.name}.arms."
    ) as temporary:
        arm_directory = Path(temporary)
        for spec in ARM_SPECS:
            if (_file_sha256(data_path) != dataset_sha256
                    or _file_sha256(config_path) != config_sha256):
                raise RelationalAblationError(
                    "dataset or selected config changed between experiment arms"
                )
            arm_path = arm_directory / f"arm_{spec['id']}.json"
            artifact, execution = _run_arm(
                spec=spec, data_path=data_path, config_path=config_path,
                output_path=arm_path, evidence_mode=evidence_mode,
                eval_limit=eval_limit, resource_interval=resource_interval,
                timeout_seconds=arm_timeout_seconds, root=root,
            )
            arm_runs[spec["id"]] = (artifact, execution)
            result = artifact["result"]
            print(json.dumps({
                "stage": "arm_complete", "arm": spec["id"],
                "auc": result.get("auc"),
                "auc_proof_only": result.get("auc_proof_only"),
                "wall_seconds": execution["wall_seconds"],
            }), flush=True)
        if (_file_sha256(data_path) != dataset_sha256
                or _file_sha256(config_path) != config_sha256):
            raise RelationalAblationError(
                "dataset or selected config changed during the experiment"
            )
        source_hashes_after = _source_hashes(root)
        runtime["workspace_source_sha256_after"] = source_hashes_after
        runtime["workspace_sources_unchanged"] = (
            source_hashes_after == runtime["workspace_source_sha256"]
        )
        if not runtime["workspace_sources_unchanged"]:
            raise RelationalAblationError(
                "audited implementation sources changed between experiment arms"
            )
        dependencies_after = _runtime_dependency_provenance()
        runtime["active_runtime_dependencies_after"] = dependencies_after
        runtime["active_runtime_dependencies_unchanged"] = (
            dependencies_after == runtime["active_runtime_dependencies"]
        )
        if not runtime["active_runtime_dependencies_unchanged"]:
            raise RelationalAblationError(
                "active imported runtime dependencies changed between experiment arms"
            )
        runtime["finished_arms_at"] = _utc_now()
        report = assemble_report(
            data=data, dataset_path=data_path, dataset_sha256=dataset_sha256,
            config_path=config_path, config_sha256=config_sha256,
            arm_runs=arm_runs, comparison_seed=comparison_seed,
            bootstrap_repetitions=bootstrap_repetitions, runtime=runtime,
        )
    report["runtime"]["finished_at"] = _utc_now()
    report["runtime"]["total_wall_seconds"] = (
        time.monotonic() - orchestration_started
    )
    _atomic_publish_json(output_path, report)
    print(json.dumps({
        "stage": "complete", "output": str(output_path),
        "output_sha256": _file_sha256(output_path),
        "auc": {
            arm: report["arms"][arm]["metrics"].get("auc")
            for arm in ("A", "B", "C")
        },
        "delta_C_minus_B": report["comparisons"]["C_minus_B"]["metrics"]["served"],
    }), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence-mode",
        choices=("llm_workspace",),
        default="llm_workspace",
    )
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--comparison-seed", type=int, default=37)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--resource-interval", type=float, default=0.25)
    parser.add_argument("--arm-timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    try:
        run_three_arm_ablation(
            data_path=args.data, config_path=args.config_file,
            output_path=args.output, evidence_mode=args.evidence_mode,
            eval_limit=args.eval_limit, comparison_seed=args.comparison_seed,
            bootstrap_repetitions=args.bootstrap_repetitions,
            resource_interval=args.resource_interval,
            arm_timeout_seconds=args.arm_timeout_seconds,
        )
    except (ValueError, OSError, json.JSONDecodeError, RelationalAblationError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
