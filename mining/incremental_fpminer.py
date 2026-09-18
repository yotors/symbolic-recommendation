"""Exact append-incremental support state for the MeTTa ``fpMiner``.

The executable miner lives in :mod:`recommendation/miner/fpMiner.metta` and
is evaluated by PeTTa.  A persistent data AtomSpace alone is not enough to
make mining incremental: evaluating ``frequency-pattern-miner`` on that space
still enumerates and counts every structure again.

This module keeps the sufficient statistics of each fixed-combination mining
plan.  The first call (and every mutation/removal fallback) asks the real
MeTTa miner for both closed outcomes at support one.  On a strict append it
asks the same miner to inspect only the appended cases and adds those support
deltas transactionally.  An unchanged workspace requires no miner query.

The support-one seed is exact, not an approximation.  If a target conjunction
has support at least ``min_support``, every atom in that conjunction also has
support at least ``min_support``.  Consequently fpMiner's candidate-level
support pruning cannot remove a conjunction which survives the final target
support filter.  Retaining all support-one conjunctions lets callers change
the threshold later without losing a previously hidden pattern.

Only discovery/support accumulation is incremental here.  A caller may still
perform full-population calibration, validation, or a separate target-aware
search after this component returns.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from .petta_workspace import PeTTaExecutor


IncrementalMode = Literal[
    "full_seed", "rebuilt_full", "reconfigured_full",
    "retention_expired_full", "delta_updated", "reused"
]
DEFAULT_MAX_TRACKED_PATTERNS_PER_PLAN = 250_000
DEFAULT_MAX_CACHED_PLANS = 512
DEFAULT_MAX_CASES_PER_PLAN = 1_000_000
Premise = tuple[str, str]
PatternKey = tuple[Premise, ...]


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_]+")
_SAFE_SYMBOL = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$.-]*$")
_SAFE_SPACE = re.compile(r"^&[A-Za-z_][A-Za-z0-9_.-]*$")
_ENGAGEMENT_FACT = re.compile(
    r'^\(\s*engagement\s+([^\s()]+)\s+("(?:\\.|[^"\\])*")\s*\)$'
)
_CASE_VALUE_FACT = re.compile(
    r'^\(\s*([^\s()]+)\s+([^\s()]+)\s+("(?:\\.|[^"\\])*")\s*\)$'
)
_FACT_HEAD = re.compile(r'^\(\s*([^\s()]+)')


@dataclass(frozen=True, slots=True)
class ParsedSupport:
    """One unique target conjunction emitted by ``fpMiner.metta``."""

    premises: PatternKey
    support: int


@dataclass(frozen=True, slots=True)
class IncrementalMiningAudit:
    """Auditable description of one exact support-state transition."""

    plan: str
    mode: IncrementalMode
    total_cases: int
    previous_cases: int
    appended_cases: int
    changed_cases: int
    removed_cases: int
    tracked_patterns: int
    output_patterns: int
    full_miner_calls: int
    delta_miner_calls: int
    researched_cases: int
    full_space: str
    delta_space: str | None
    min_support: int
    pattern_limit: int
    cached_plan_limit: int
    cached_plans_after_commit: int
    case_limit: int
    retention: dict[str, object] | None
    previous_retained_units: int
    appended_retained_units: int
    expired_retained_units: int
    expiration_forced_full_rebuild: bool
    rebuild_reason: str | None
    support_floor: int = 1
    exact: bool = True

    @property
    def miner_calls(self) -> int:
        return self.full_miner_calls + self.delta_miner_calls

    @property
    def full_structure_research(self) -> bool:
        return self.mode in {
            "full_seed", "rebuilt_full", "reconfigured_full",
            "retention_expired_full",
        }

    @property
    def incremental_support_update(self) -> bool:
        return self.mode == "delta_updated"

    def as_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan,
            "mode": self.mode,
            "total_cases": self.total_cases,
            "previous_cases": self.previous_cases,
            "appended_cases": self.appended_cases,
            "changed_cases": self.changed_cases,
            "removed_cases": self.removed_cases,
            "tracked_patterns": self.tracked_patterns,
            "output_patterns": self.output_patterns,
            "full_miner_calls": self.full_miner_calls,
            "delta_miner_calls": self.delta_miner_calls,
            "miner_calls": self.miner_calls,
            "researched_cases": self.researched_cases,
            "full_structure_research": self.full_structure_research,
            "incremental_support_update": self.incremental_support_update,
            "full_space": self.full_space,
            "delta_space": self.delta_space,
            "min_support": self.min_support,
            "pattern_limit": self.pattern_limit,
            "cached_plan_limit": self.cached_plan_limit,
            "cached_plans_after_commit": self.cached_plans_after_commit,
            "case_limit": self.case_limit,
            "retention": self.retention,
            "previous_retained_units": self.previous_retained_units,
            "appended_retained_units": self.appended_retained_units,
            "expired_retained_units": self.expired_retained_units,
            "expiration_forced_full_rebuild": (
                self.expiration_forced_full_rebuild
            ),
            "rebuild_reason": self.rebuild_reason,
            "support_floor": self.support_floor,
            "exact": self.exact,
            "executor": "recommendation/miner/fpMiner.metta via PeTTa",
            "support_state": "transactional_host_sufficient_statistics",
        }


@dataclass(frozen=True, slots=True)
class IncrementalMiningResult:
    """Rendered fpMiner-compatible forms and their transition audit."""

    output: tuple[str, ...]
    audit: IncrementalMiningAudit


@dataclass(frozen=True, slots=True)
class _PatternCounts:
    premises: PatternKey
    target: int
    other: int


@dataclass(frozen=True, slots=True)
class _PlanState:
    shape: tuple[int, tuple[str, ...], str, str]
    case_hashes: dict[str, str]
    total_cases: int
    target_cases: int
    patterns: dict[PatternKey, _PatternCounts]
    retained_units: frozenset[str] | None


@dataclass(slots=True)
class _DeltaSpace:
    name: str
    populated: bool = False
    dirty: bool = False


def _single_expression(source: str) -> str:
    """Return one normalized parenthesized expression or reject commands."""

    if not isinstance(source, str):
        raise TypeError("case facts must be strings")
    value = source.strip()
    if not value or not value.startswith("(") or value.startswith("!("):
        raise ValueError("case fact must be one parenthesized MeTTa expression")

    depth = 0
    quoted = False
    escaped = False
    closed_at: int | None = None
    for index, character in enumerate(value):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("case fact has unbalanced parentheses")
            if depth == 0:
                closed_at = index
                break
    if quoted or depth != 0 or closed_at is None:
        raise ValueError("case fact has an unterminated string or expression")
    if value[closed_at + 1 :].strip():
        raise ValueError("case fact must contain exactly one MeTTa expression")
    return value


def _balanced_forms(text: str, head: str = "supportOf") -> tuple[str, ...]:
    forms: list[str] = []
    for match in re.finditer(r"\(" + re.escape(head) + r"\b", text):
        start = match.start()
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    forms.append(text[start : index + 1])
                    break
    return tuple(forms)


def parse_fpminer_supports(
    raw: Iterable[object],
    *,
    features: Iterable[str],
    expected_target: str,
    expected_depth: int,
) -> dict[PatternKey, ParsedSupport]:
    """Parse and strictly deduplicate one target pass from ``fpMiner.metta``.

    PeTTa may alpha-rename ``$x`` and may serialize the same logical result
    more than once.  Premises are canonicalized by the caller's feature order;
    duplicates are accepted only when their support is identical.
    """

    selected = tuple(features)
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError("features must be a non-empty sequence without duplicates")
    if not isinstance(expected_depth, int) or isinstance(expected_depth, bool):
        raise ValueError("expected_depth must be an integer")
    if expected_depth < 2:
        raise ValueError("expected_depth must be at least two")
    order = {predicate: index for index, predicate in enumerate(selected)}
    alternatives = "|".join(
        re.escape(predicate)
        for predicate in sorted((*selected, "engagement"), key=len, reverse=True)
    )
    clause_re = re.compile(
        rf'\(({alternatives})\s+([^\s()]+)\s+("(?:\\.|[^"\\])*")\)'
    )
    support_re = re.compile(r"\)\s+([0-9]+)\s*\)$")
    parsed: dict[PatternKey, ParsedSupport] = {}
    text = " ".join(str(value) for value in raw)
    for form in _balanced_forms(text):
        clauses = clause_re.findall(form)
        targets: list[str] = []
        premises: list[Premise] = []
        variables: set[str] = set()
        for predicate, variable, encoded_value in clauses:
            try:
                value = json.loads(encoded_value)
            except json.JSONDecodeError as exc:
                raise ValueError("fpMiner emitted a non-JSON string value") from exc
            if not isinstance(value, str):
                raise ValueError("fpMiner values must be strings")
            variables.add(variable)
            if predicate == "engagement":
                targets.append(value)
            else:
                premises.append((predicate, value))
        if not targets:
            continue
        if len(targets) != 1 or targets[0] != expected_target:
            raise ValueError("fpMiner output contains an unexpected target")
        if len(variables) != 1:
            raise ValueError("fpMiner output does not join clauses on one case variable")
        if len(premises) != expected_depth - 1:
            raise ValueError("fpMiner output has an unexpected conjunction depth")
        if len({predicate for predicate, _value in premises}) != len(premises):
            raise ValueError("fpMiner output repeats a premise predicate")
        canonical = tuple(
            sorted(premises, key=lambda item: (order[item[0]], item[1]))
        )
        support_match = support_re.search(form)
        if support_match is None:
            raise ValueError("fpMiner supportOf form has no integer support")
        support = int(support_match.group(1))
        if support < 1:
            raise ValueError("support-one fpMiner output must have positive support")
        previous = parsed.get(canonical)
        if previous is not None and previous.support != support:
            raise ValueError("duplicate fpMiner pattern has conflicting support")
        parsed[canonical] = ParsedSupport(canonical, support)
    return parsed


class IncrementalFpMinerCache:
    """Incrementally maintain exact support for fixed fpMiner plans.

    ``full_space`` must already contain exactly ``cases``.  Each case must have
    one generated ``(engagement CASE \"target-or-other\")`` fact.  The cache
    independently fingerprints every complete case, making state transitions
    recoverable even when a prior caller appended the full AtomSpace but failed
    before committing support state.
    """

    def __init__(
        self,
        petta: PeTTaExecutor,
        *,
        namespace: str = "recommendation",
        batch_size: int = 1000,
        max_tracked_patterns_per_plan: int = DEFAULT_MAX_TRACKED_PATTERNS_PER_PLAN,
        max_cached_plans: int = DEFAULT_MAX_CACHED_PLANS,
        max_cases_per_plan: int = DEFAULT_MAX_CASES_PER_PLAN,
        lock: threading.RLock | None = None,
    ) -> None:
        if not hasattr(petta, "process_metta_string"):
            raise TypeError("petta must provide process_metta_string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for label, value in (
            ("max_tracked_patterns_per_plan", max_tracked_patterns_per_plan),
            ("max_cached_plans", max_cached_plans),
            ("max_cases_per_plan", max_cases_per_plan),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        clean = _SAFE_NAME.sub("_", str(namespace)).strip("_")
        if not clean:
            raise ValueError("namespace must contain a letter, digit, or underscore")
        self._petta = petta
        self._namespace = clean[:24]
        self._instance = uuid.uuid4().hex[:12]
        self._batch_size = batch_size
        self._max_tracked_patterns_per_plan = max_tracked_patterns_per_plan
        self._max_cached_plans = max_cached_plans
        self._max_cases_per_plan = max_cases_per_plan
        self._lock = lock or threading.RLock()
        self._states: dict[str, _PlanState] = {}
        self._delta_spaces: dict[str, _DeltaSpace] = {}
        self._sequence = 0

    @staticmethod
    def _validate_plan(plan: str) -> str:
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("plan must be a non-empty string")
        return plan.strip()

    @staticmethod
    def _validate_shape(
        features: Iterable[str], depth: int, target: str, other: str
    ) -> tuple[int, tuple[str, ...], str, str]:
        selected = tuple(features)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("features must be non-empty and unique")
        if any(not _SAFE_SYMBOL.fullmatch(feature) for feature in selected):
            raise ValueError("features must be safe MeTTa symbols")
        if "engagement" in selected:
            raise ValueError("engagement is added by the miner and cannot be a feature")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 2:
            raise ValueError("depth must be an integer of at least two")
        if depth - 1 > len(selected):
            raise ValueError("depth requires more distinct predicates than features")
        if not isinstance(target, str) or not target or not isinstance(other, str) or not other:
            raise ValueError("target and other must be non-empty strings")
        if target == other:
            raise ValueError("target and other outcomes must differ")
        return depth, selected, target, other

    @staticmethod
    def _normalize_cases(
        cases: Mapping[str, Iterable[str]], features: Iterable[str],
        target: str, other: str,
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str], dict[str, str]]:
        if not isinstance(cases, Mapping) or not cases:
            raise ValueError("cases must be a non-empty mapping")
        canonical: dict[str, tuple[str, ...]] = {}
        hashes: dict[str, str] = {}
        labels: dict[str, str] = {}
        selected = frozenset(features)
        for raw_case_id, raw_facts in cases.items():
            if not isinstance(raw_case_id, str) or not _SAFE_SYMBOL.fullmatch(raw_case_id):
                raise ValueError("case IDs must be safe MeTTa symbols")
            if isinstance(raw_facts, (str, bytes)):
                raise TypeError("each case must contain an iterable of facts")
            facts = tuple(sorted({_single_expression(fact) for fact in raw_facts}))
            if not facts:
                raise ValueError("each case must contain at least one fact")
            outcomes: list[str] = []
            for fact in facts:
                head_match = _FACT_HEAD.match(fact)
                if head_match is not None and head_match.group(1) in selected:
                    feature_match = _CASE_VALUE_FACT.fullmatch(fact)
                    if feature_match is None:
                        raise ValueError(
                            "selected-feature facts must have predicate, case ID, "
                            "and JSON-string value"
                        )
                    if feature_match.group(2) != raw_case_id:
                        raise ValueError(
                            "selected-feature fact case does not match its mapping key"
                        )
                match = _ENGAGEMENT_FACT.fullmatch(fact)
                if match is None:
                    continue
                if match.group(1) != raw_case_id:
                    raise ValueError("engagement fact case does not match its mapping key")
                try:
                    value = json.loads(match.group(2))
                except json.JSONDecodeError as exc:
                    raise ValueError("engagement outcome must be a JSON string") from exc
                if not isinstance(value, str):
                    raise ValueError("engagement outcome must be a string")
                outcomes.append(value)
            if len(outcomes) != 1 or outcomes[0] not in {target, other}:
                raise ValueError(
                    "each case must contain exactly one target-or-other engagement fact"
                )
            payload = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
            canonical[raw_case_id] = facts
            hashes[raw_case_id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            labels[raw_case_id] = outcomes[0]
        return canonical, hashes, labels

    @staticmethod
    def _validate_parameters(full_space: str, min_support: int, evidence_k: float) -> float:
        if not isinstance(full_space, str) or not _SAFE_SPACE.fullmatch(full_space):
            raise ValueError("full_space must be a safe bound PeTTa space symbol")
        if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
            raise ValueError("min_support must be a positive integer")
        if isinstance(evidence_k, bool) or not isinstance(evidence_k, (int, float)):
            raise ValueError("evidence_k must be finite and positive")
        normalized_k = float(evidence_k)
        if not math.isfinite(normalized_k) or normalized_k <= 0.0:
            raise ValueError("evidence_k must be finite and positive")
        return normalized_k

    def _execute(self, source: str) -> Any:
        return self._petta.process_metta_string(source)

    def _new_delta_space(self, plan: str) -> _DeltaSpace:
        self._sequence += 1
        digest = hashlib.sha256(plan.encode("utf-8")).hexdigest()[:12]
        name = (
            f"&rec_inc_{self._namespace}_{self._instance}_"
            f"{self._sequence}_{digest}"
        )
        self._execute(f"!(bind! {name} (new-space))")
        delta = _DeltaSpace(name=name)
        self._delta_spaces[plan] = delta
        return delta

    def _clear_space(self, space: str) -> None:
        self._execute(
            f"!(let $atom (superpose (collapse (match {space} $x $x))) "
            f"(remove-atom {space} $atom))"
        )

    def _prepare_delta_space(
        self, plan: str, case_ids: Iterable[str], cases: Mapping[str, tuple[str, ...]]
    ) -> str:
        delta = self._delta_spaces.get(plan)
        if delta is None:
            delta = self._new_delta_space(plan)
        try:
            if delta.populated or delta.dirty:
                self._clear_space(delta.name)
            additions = [
                f"(add-atom {delta.name} {fact})"
                for case_id in sorted(case_ids)
                for fact in cases[case_id]
            ]
            for start in range(0, len(additions), self._batch_size):
                batch = additions[start : start + self._batch_size]
                self._execute(f"!(superpose ({' '.join(batch)}))")
        except BaseException:
            delta.dirty = True
            raise
        delta.populated = True
        delta.dirty = False
        return delta.name

    def _mine_outcome(
        self,
        space: str,
        *,
        features: tuple[str, ...],
        depth: int,
        outcome: str,
        evidence_k: float,
    ) -> dict[PatternKey, ParsedSupport]:
        encoded_target = json.dumps(outcome, ensure_ascii=False)
        raw = self._execute(
            f"!(frequency-pattern-miner {space} 1 {depth} "
            f"{encoded_target} {evidence_k:.17g})"
        )
        if raw is None:
            values: tuple[object, ...] = ()
        elif isinstance(raw, (str, bytes)):
            values = (raw,)
        else:
            try:
                values = tuple(raw)
            except TypeError:
                values = (raw,)
        return parse_fpminer_supports(
            values,
            features=features,
            expected_target=outcome,
            expected_depth=depth,
        )

    @staticmethod
    def _combine_full(
        target_rows: Mapping[PatternKey, ParsedSupport],
        other_rows: Mapping[PatternKey, ParsedSupport],
    ) -> dict[PatternKey, _PatternCounts]:
        result: dict[PatternKey, _PatternCounts] = {}
        for key in target_rows.keys() | other_rows.keys():
            target_support = target_rows[key].support if key in target_rows else 0
            other_support = other_rows[key].support if key in other_rows else 0
            premises = (
                target_rows[key].premises if key in target_rows else other_rows[key].premises
            )
            result[key] = _PatternCounts(premises, target_support, other_support)
        return result

    @staticmethod
    def _merge_delta(
        previous: Mapping[PatternKey, _PatternCounts],
        target_rows: Mapping[PatternKey, ParsedSupport],
        other_rows: Mapping[PatternKey, ParsedSupport],
    ) -> dict[PatternKey, _PatternCounts]:
        result = dict(previous)
        for key in target_rows.keys() | other_rows.keys():
            old = result.get(key)
            target_support = (old.target if old else 0) + (
                target_rows[key].support if key in target_rows else 0
            )
            other_support = (old.other if old else 0) + (
                other_rows[key].support if key in other_rows else 0
            )
            premises = (
                old.premises if old else
                target_rows[key].premises if key in target_rows else
                other_rows[key].premises
            )
            result[key] = _PatternCounts(premises, target_support, other_support)
        return result

    @staticmethod
    def _validate_counts(
        patterns: Mapping[PatternKey, _PatternCounts],
        *,
        total_cases: int,
        target_cases: int,
    ) -> None:
        other_cases = total_cases - target_cases
        for counts in patterns.values():
            if not (0 <= counts.target <= target_cases):
                raise ValueError("fpMiner target support exceeds the target population")
            if not (0 <= counts.other <= other_cases):
                raise ValueError("fpMiner other support exceeds the other population")
            if counts.target + counts.other < 1:
                raise ValueError("tracked fpMiner pattern has zero antecedent support")

    @staticmethod
    def _render(
        state: _PlanState, *, min_support: int, evidence_k: float
    ) -> tuple[str, ...]:
        depth, _features, target, _other = state.shape
        del depth  # The stored premises already encode and validate the depth.
        rows: list[tuple[int, float, PatternKey, str]] = []
        for key, counts in state.patterns.items():
            joint = counts.target
            if joint < min_support:
                continue
            antecedent = counts.target + counts.other
            outside = state.total_cases - antecedent
            outside_target = state.target_cases - joint
            valid = (
                0 <= joint <= antecedent
                and joint <= state.target_cases
                and 0 <= outside_target <= outside
            )
            if not valid:
                raise ValueError("incremental contingency is internally inconsistent")
            strength = joint / antecedent
            confidence = antecedent / (antecedent + evidence_k)
            negative_strength = outside_target / outside if outside else 0.0
            negative_confidence = outside / (outside + evidence_k) if outside else 0.0
            clauses = " ".join(
                f"({predicate} $x {json.dumps(value, ensure_ascii=False)})"
                for predicate, value in counts.premises
            )
            form = (
                f"(supportOf (({clauses} "
                f"(engagement $x {json.dumps(target, ensure_ascii=False)})) "
                f"(CTV (STV {strength!r} {confidence!r}) "
                f"(STV {negative_strength!r} {negative_confidence!r}))) "
                f"{joint})"
            )
            rows.append((joint, strength, key, form))
        rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return tuple(item[3] for item in rows)

    def mine(
        self,
        *,
        plan: str,
        full_space: str,
        cases: Mapping[str, Iterable[str]],
        features: Iterable[str],
        depth: int,
        min_support: int,
        evidence_k: float,
        target: str = "click",
        other: str = "skip",
        retention_units: Iterable[str] | None = None,
        retention_audit: Mapping[str, object] | None = None,
    ) -> IncrementalMiningResult:
        """Mine or update one plan and return fpMiner-compatible support forms."""

        normalized_plan = self._validate_plan(plan)
        shape = self._validate_shape(features, depth, target, other)
        normalized_k = self._validate_parameters(full_space, min_support, evidence_k)
        canonical, hashes, labels = self._normalize_cases(
            cases, shape[1], target, other
        )
        total_cases = len(canonical)
        if total_cases > self._max_cases_per_plan:
            raise MemoryError(
                "incremental fpMiner case limit exceeded: "
                f"{total_cases} > {self._max_cases_per_plan}"
            )
        target_cases = sum(value == target for value in labels.values())
        if retention_units is None:
            normalized_retention_units = None
            if retention_audit is not None:
                raise ValueError("retention_audit requires retention_units")
        else:
            normalized_retention_units = frozenset(retention_units)
            if any(not isinstance(unit, str) or not unit
                   for unit in normalized_retention_units):
                raise ValueError("retention unit IDs must be non-empty strings")
            if retention_audit is None:
                raise ValueError("retention_units require retention_audit")
            declared = retention_audit.get("retained_units")
            if declared != len(normalized_retention_units):
                raise ValueError(
                    "retention audit retained-unit count disagrees with membership"
                )
        retention_metadata = (
            dict(retention_audit) if retention_audit is not None else None
        )

        with self._lock:
            previous = self._states.get(normalized_plan)
            if previous is None and len(self._states) >= self._max_cached_plans:
                raise MemoryError(
                    "incremental fpMiner cached-plan limit reached; prune unused "
                    "plans before creating another"
                )
            previous_cases = previous.total_cases if previous else 0
            old_hashes = previous.case_hashes if previous else {}
            old_ids = set(old_hashes)
            desired_ids = set(hashes)
            changed_ids = {
                case_id
                for case_id in old_ids & desired_ids
                if old_hashes[case_id] != hashes[case_id]
            }
            removed_ids = old_ids - desired_ids
            appended_ids = desired_ids - old_ids
            shape_changed = previous is not None and previous.shape != shape
            old_retention_units = (
                previous.retained_units if previous is not None else None
            )
            retention_contract_changed = (
                previous is not None
                and ((old_retention_units is None)
                     != (normalized_retention_units is None))
            )
            expired_retention_units = (
                old_retention_units - normalized_retention_units
                if (old_retention_units is not None
                    and normalized_retention_units is not None)
                else frozenset()
            )
            appended_retention_units = (
                normalized_retention_units - old_retention_units
                if (old_retention_units is not None
                    and normalized_retention_units is not None)
                else (normalized_retention_units or frozenset())
                if previous is None else frozenset()
            )

            full_calls = delta_calls = 0
            delta_space: str | None = None
            if previous is None:
                mode: IncrementalMode = "full_seed"
                rebuild_reason = "initial_seed"
            elif shape_changed or retention_contract_changed:
                mode = "reconfigured_full"
                rebuild_reason = (
                    "plan_shape_changed" if shape_changed
                    else "retention_contract_changed"
                )
            elif expired_retention_units:
                # Never subtract supports approximately. Mine the complete
                # retained AtomSpace so every vanished causal unit is removed
                # from every conjunction and both CTV branches exactly.
                mode = "retention_expired_full"
                rebuild_reason = "causal_units_expired_exact_full_rebuild"
            elif changed_ids or removed_ids:
                mode = "rebuilt_full"
                rebuild_reason = "case_changed_or_removed"
            elif appended_ids:
                mode = "delta_updated"
                rebuild_reason = None
            else:
                mode = "reused"
                rebuild_reason = None

            if mode in {
                "full_seed", "rebuilt_full", "reconfigured_full",
                "retention_expired_full",
            }:
                target_rows = self._mine_outcome(
                    full_space, features=shape[1], depth=depth,
                    outcome=target, evidence_k=normalized_k,
                )
                other_rows = self._mine_outcome(
                    full_space, features=shape[1], depth=depth,
                    outcome=other, evidence_k=normalized_k,
                )
                full_calls = 2
                patterns = self._combine_full(target_rows, other_rows)
                researched_cases = total_cases
            elif mode == "delta_updated":
                # Prepare/query first and mutate only a local copy.  Neither a
                # failed PeTTa pass nor a parse error can partially commit the
                # cumulative support state.
                delta_space = self._prepare_delta_space(
                    normalized_plan, appended_ids, canonical
                )
                target_rows = self._mine_outcome(
                    delta_space, features=shape[1], depth=depth,
                    outcome=target, evidence_k=normalized_k,
                )
                other_rows = self._mine_outcome(
                    delta_space, features=shape[1], depth=depth,
                    outcome=other, evidence_k=normalized_k,
                )
                delta_calls = 2
                patterns = self._merge_delta(
                    previous.patterns, target_rows, other_rows
                )
                researched_cases = len(appended_ids)
            else:
                patterns = previous.patterns
                researched_cases = 0

            self._validate_counts(
                patterns, total_cases=total_cases, target_cases=target_cases
            )
            if len(patterns) > self._max_tracked_patterns_per_plan:
                # Support-one storage is required for exact threshold changes,
                # so silently truncating would corrupt future mining. Fail the
                # staged build before its transactional commit instead.
                raise MemoryError(
                    "incremental fpMiner pattern limit exceeded: "
                    f"{len(patterns)} > {self._max_tracked_patterns_per_plan}"
                )
            next_state = _PlanState(
                shape=shape,
                case_hashes=dict(hashes),
                total_cases=total_cases,
                target_cases=target_cases,
                patterns=dict(patterns),
                retained_units=normalized_retention_units,
            )
            # Assignment is the transaction commit.  All fallible mining,
            # parsing, merging and validation has completed above.
            self._states[normalized_plan] = next_state
            output = self._render(
                next_state, min_support=min_support, evidence_k=normalized_k
            )
            audit = IncrementalMiningAudit(
                plan=normalized_plan,
                mode=mode,
                total_cases=total_cases,
                previous_cases=previous_cases,
                appended_cases=len(appended_ids),
                changed_cases=len(changed_ids),
                removed_cases=len(removed_ids),
                tracked_patterns=len(patterns),
                output_patterns=len(output),
                full_miner_calls=full_calls,
                delta_miner_calls=delta_calls,
                researched_cases=researched_cases,
                full_space=full_space,
                delta_space=delta_space,
                min_support=min_support,
                pattern_limit=self._max_tracked_patterns_per_plan,
                cached_plan_limit=self._max_cached_plans,
                cached_plans_after_commit=len(self._states),
                case_limit=self._max_cases_per_plan,
                retention=retention_metadata,
                previous_retained_units=len(old_retention_units or ()),
                appended_retained_units=len(appended_retention_units),
                expired_retained_units=len(expired_retention_units),
                expiration_forced_full_rebuild=bool(expired_retention_units),
                rebuild_reason=rebuild_reason,
            )
            return IncrementalMiningResult(output=output, audit=audit)

    def invalidate(self, plan: str) -> bool:
        """Forget one support snapshot so its next call performs a full seed."""

        normalized = self._validate_plan(plan)
        with self._lock:
            return self._states.pop(normalized, None) is not None

    def prune(self, keep: Iterable[str] = ()) -> tuple[str, ...]:
        """Forget unused plans and empty their reusable delta AtomSpaces."""

        keep_set = {self._validate_plan(plan) for plan in keep}
        with self._lock:
            removed: list[str] = []
            for plan in sorted((set(self._states) | set(self._delta_spaces)) - keep_set):
                delta = self._delta_spaces.get(plan)
                if delta is not None and delta.populated:
                    try:
                        self._clear_space(delta.name)
                    except BaseException:
                        delta.dirty = True
                        raise
                self._states.pop(plan, None)
                self._delta_spaces.pop(plan, None)
                removed.append(plan)
            return tuple(removed)

    def plans(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._states))


__all__ = [
    "DEFAULT_MAX_CACHED_PLANS",
    "DEFAULT_MAX_CASES_PER_PLAN",
    "DEFAULT_MAX_TRACKED_PATTERNS_PER_PLAN",
    "IncrementalFpMinerCache",
    "IncrementalMiningAudit",
    "IncrementalMiningResult",
    "IncrementalMode",
    "ParsedSupport",
    "parse_fpminer_supports",
]
