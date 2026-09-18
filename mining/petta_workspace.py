"""Persistent, delta-synchronised AtomSpaces for PeTTa mining plans.

The frequent-pattern miner operates over a named PeTTa ``Space``.  Rebuilding
that space before every mining query needlessly re-sends every unchanged case.
This module keeps one process-local AtomSpace per mining plan and synchronises
closed cases by stable case identifier and a canonical fact hash.

This is deliberately a *workspace* cache, not an incremental pattern miner.
Appending cases avoids repeated AtomSpace ingestion.  The subsequent MeTTa
``frequency-pattern-miner`` query still recomputes pattern support until its
search/state machinery is made incremental as well.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol


SyncMode = Literal["created", "reused", "appended", "rebuilt"]
DEFAULT_MAX_CACHED_PLANS = 512
DEFAULT_MAX_CASES_PER_PLAN = 1_000_000
DEFAULT_MAX_FACTS_PER_PLAN = 10_000_000


class PeTTaExecutor(Protocol):
    """Minimum PeTTa surface required by :class:`PeTTaWorkspaceCache`."""

    def process_metta_string(self, metta_code: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class WorkspaceSync:
    """Auditable result of synchronising one mining plan."""

    plan: str
    space: str
    mode: SyncMode
    reused: int
    appended: int
    total: int
    removed: int = 0
    changed: int = 0
    appended_facts: int = 0
    total_facts: int = 0
    case_limit: int = DEFAULT_MAX_CASES_PER_PLAN
    fact_limit: int = DEFAULT_MAX_FACTS_PER_PLAN
    cached_plan_limit: int = DEFAULT_MAX_CACHED_PLANS
    cached_plans_after_commit: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan,
            "space": self.space,
            "mode": self.mode,
            "reused": self.reused,
            "appended": self.appended,
            "total": self.total,
            "removed": self.removed,
            "changed": self.changed,
            "appended_facts": self.appended_facts,
            "total_facts": self.total_facts,
            "case_limit": self.case_limit,
            "fact_limit": self.fact_limit,
            "cached_plan_limit": self.cached_plan_limit,
            "cached_plans_after_commit": self.cached_plans_after_commit,
        }


@dataclass(slots=True)
class _Workspace:
    space: str
    case_hashes: dict[str, str]
    dirty: bool = False


_SAFE_NAMESPACE = re.compile(r"[^A-Za-z0-9_]+")


def _single_expression(source: str) -> str:
    """Validate and normalize one parenthesized MeTTa fact expression.

    The cache receives generated facts, but still refuses commands or multiple
    expressions so a malformed value cannot escape its enclosing ``add-atom``.
    Parentheses inside quoted strings are ignored and ordinary backslash string
    escapes are supported.
    """

    if not isinstance(source, str):
        raise TypeError("case facts must be strings")
    value = source.strip()
    if not value or not value.startswith("("):
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


def _case_hash(facts: tuple[str, ...]) -> str:
    payload = json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PeTTaWorkspaceCache:
    """Keep append-friendly PeTTa mining spaces for independent plans.

    ``plan`` identifies the complete mining projection (for example point/topic
    depth 2 or pair/semantic depth 3). ``cases`` maps stable case IDs to all raw
    facts belonging to that closed case. Existing cases may only be reused when
    their canonical hashes are unchanged. Any deletion or mutation rebuilds the
    plan's space; a strict unchanged prefix/subset appends only new cases.

    State is persistent for the lifetime of this object and PeTTa process. It is
    not a disk checkpoint and is intentionally separate from PeTTaChainer's
    compiled proof workspace.
    """

    def __init__(
        self,
        petta: PeTTaExecutor,
        *,
        namespace: str = "recommendation",
        batch_size: int = 1000,
        max_cached_plans: int = DEFAULT_MAX_CACHED_PLANS,
        max_cases_per_plan: int = DEFAULT_MAX_CASES_PER_PLAN,
        max_facts_per_plan: int = DEFAULT_MAX_FACTS_PER_PLAN,
        lock: threading.RLock | None = None,
    ) -> None:
        if not hasattr(petta, "process_metta_string"):
            raise TypeError("petta must provide process_metta_string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        for label, value in (
            ("max_cached_plans", max_cached_plans),
            ("max_cases_per_plan", max_cases_per_plan),
            ("max_facts_per_plan", max_facts_per_plan),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        clean_namespace = _SAFE_NAMESPACE.sub("_", str(namespace)).strip("_")
        if not clean_namespace:
            raise ValueError("namespace must contain a letter, digit, or underscore")

        self._petta = petta
        self._namespace = clean_namespace[:24]
        self._instance = uuid.uuid4().hex[:12]
        self._batch_size = batch_size
        self._max_cached_plans = max_cached_plans
        self._max_cases_per_plan = max_cases_per_plan
        self._max_facts_per_plan = max_facts_per_plan
        self._lock = lock or threading.RLock()
        self._workspaces: dict[str, _Workspace] = {}
        self._sequence = 0

    @staticmethod
    def _validate_plan(plan: str) -> str:
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("plan must be a non-empty string")
        return plan.strip()

    def _normalize_cases(
        self,
        cases: Mapping[str, Iterable[str]],
    ) -> tuple[dict[str, tuple[str, ...]], dict[str, str], int]:
        if not isinstance(cases, Mapping):
            raise TypeError("cases must map stable case IDs to fact iterables")
        if len(cases) > self._max_cases_per_plan:
            raise MemoryError(
                "PeTTa workspace case limit exceeded: "
                f"{len(cases)} > {self._max_cases_per_plan}"
            )
        canonical: dict[str, tuple[str, ...]] = {}
        submitted_facts = 0
        for case_number, (raw_case_id, facts) in enumerate(cases.items(), start=1):
            # Do not trust a custom Mapping's reported length.  The iteration
            # guard keeps case state bounded even when ``len`` under-reports.
            if case_number > self._max_cases_per_plan:
                raise MemoryError(
                    "PeTTa workspace case limit exceeded: "
                    f"> {self._max_cases_per_plan}"
                )
            if not isinstance(raw_case_id, str) or not raw_case_id.strip():
                raise ValueError("case IDs must be non-empty strings")
            case_id = raw_case_id.strip()
            if case_id in canonical:
                raise ValueError(f"duplicate normalized case ID: {case_id}")
            if isinstance(facts, (str, bytes)):
                raise TypeError("case facts must be an iterable of MeTTa expressions")
            normalized_facts: set[str] = set()
            for fact in facts:
                submitted_facts += 1
                if submitted_facts > self._max_facts_per_plan:
                    raise MemoryError(
                        "PeTTa workspace fact limit exceeded: "
                        f"> {self._max_facts_per_plan}"
                    )
                normalized_facts.add(_single_expression(fact))
            if not normalized_facts:
                raise ValueError("each mining case must contain at least one fact")
            canonical[case_id] = tuple(sorted(normalized_facts))
        hashes = {case_id: _case_hash(facts) for case_id, facts in canonical.items()}
        total_facts = sum(len(facts) for facts in canonical.values())
        return canonical, hashes, total_facts

    def _new_space(self, plan: str) -> str:
        self._sequence += 1
        digest = hashlib.sha256(plan.encode("utf-8")).hexdigest()[:12]
        return (
            f"&rec_mine_{self._namespace}_{self._instance}_"
            f"{self._sequence}_{digest}"
        )

    def _execute(self, source: str) -> Any:
        return self._petta.process_metta_string(source)

    def _bind(self, space: str) -> None:
        self._execute(f"!(bind! {space} (new-space))")

    def _clear_space(self, space: str) -> None:
        self._execute(
            f"!(let $atom (superpose (collapse (match {space} $x $x))) "
            f"(remove-atom {space} $atom))"
        )

    def _append_facts(
        self,
        space: str,
        case_ids: Iterable[str],
        cases: Mapping[str, tuple[str, ...]],
    ) -> int:
        appended = 0
        batch: list[str] = []
        for case_id in sorted(case_ids):
            for fact in cases[case_id]:
                batch.append(f"(add-atom {space} {fact})")
                if len(batch) == self._batch_size:
                    self._execute(f"!(superpose ({' '.join(batch)}))")
                    appended += len(batch)
                    batch.clear()
        if batch:
            self._execute(f"!(superpose ({' '.join(batch)}))")
            appended += len(batch)
        return appended

    def sync(
        self,
        plan: str,
        cases: Mapping[str, Iterable[str]],
    ) -> WorkspaceSync:
        """Synchronise ``cases`` and return the space plus reuse audit."""

        plan = self._validate_plan(plan)
        # Reject a new plan before walking or materializing its potentially
        # large case input.  The check is repeated below while holding the lock
        # so concurrent callers cannot over-commit the plan budget.
        with self._lock:
            if (plan not in self._workspaces
                    and len(self._workspaces) >= self._max_cached_plans):
                raise MemoryError(
                    "PeTTa workspace cached-plan limit reached; prune unused "
                    "plans before creating another"
                )
        canonical, desired_hashes, total_facts = self._normalize_cases(cases)
        with self._lock:
            workspace = self._workspaces.get(plan)
            created = workspace is None
            if workspace is None:
                if len(self._workspaces) >= self._max_cached_plans:
                    raise MemoryError(
                        "PeTTa workspace cached-plan limit reached; prune unused "
                        "plans before creating another"
                    )
                workspace = _Workspace(self._new_space(plan), {})
                self._workspaces[plan] = workspace
                try:
                    self._bind(workspace.space)
                except BaseException:
                    self._workspaces.pop(plan, None)
                    raise

            old_hashes = workspace.case_hashes
            old_ids = set(old_hashes)
            desired_ids = set(desired_hashes)
            changed_ids = {
                case_id
                for case_id in old_ids & desired_ids
                if old_hashes[case_id] != desired_hashes[case_id]
            }
            removed_ids = old_ids - desired_ids
            rebuild = workspace.dirty or bool(changed_ids or removed_ids)

            if rebuild:
                mode: SyncMode = "rebuilt"
                append_ids = desired_ids
                reused = 0
            else:
                append_ids = desired_ids - old_ids
                reused = len(old_ids)
                if created:
                    mode = "created"
                elif append_ids:
                    mode = "appended"
                else:
                    mode = "reused"

            appended_facts = 0
            try:
                if rebuild:
                    self._clear_space(workspace.space)
                appended_facts = self._append_facts(
                    workspace.space, append_ids, canonical
                )
            except BaseException:
                # A batched mutation can fail after partial application. Never
                # trust the old hashes after that; the next sync must rebuild.
                if created:
                    # A failed initial stage has no committed cache state to
                    # preserve.  Best-effort clearing avoids leaking partially
                    # populated PeTTa atoms; forgetting it releases plan budget.
                    try:
                        self._clear_space(workspace.space)
                    except BaseException:
                        pass
                    if self._workspaces.get(plan) is workspace:
                        self._workspaces.pop(plan, None)
                else:
                    workspace.dirty = True
                raise

            workspace.case_hashes = desired_hashes
            workspace.dirty = False
            return WorkspaceSync(
                plan=plan,
                space=workspace.space,
                mode=mode,
                reused=reused,
                appended=len(append_ids),
                total=len(desired_ids),
                removed=len(removed_ids),
                changed=len(changed_ids),
                appended_facts=appended_facts,
                total_facts=total_facts,
                case_limit=self._max_cases_per_plan,
                fact_limit=self._max_facts_per_plan,
                cached_plan_limit=self._max_cached_plans,
                cached_plans_after_commit=len(self._workspaces),
            )

    def discard(self, plan: str, *, expected_space: str | None = None) -> bool:
        """Clear and forget one plan, optionally guarded by its staged space.

        ``expected_space`` makes downstream rollback safe against deleting a
        newer workspace for the same plan.  A clearing failure retains a dirty
        entry so a later discard/prune can retry instead of silently orphaning
        a populated process-global space.
        """

        plan = self._validate_plan(plan)
        if expected_space is not None and not isinstance(expected_space, str):
            raise TypeError("expected_space must be a string or None")
        with self._lock:
            workspace = self._workspaces.get(plan)
            if workspace is None:
                return False
            if expected_space is not None and workspace.space != expected_space:
                return False
            try:
                self._clear_space(workspace.space)
            except BaseException:
                workspace.dirty = True
                raise
            if self._workspaces.get(plan) is workspace:
                del self._workspaces[plan]
                return True
            return False

    def rollback(self, staged: WorkspaceSync) -> bool:
        """Discard the exact workspace represented by a rejected sync result."""

        if not isinstance(staged, WorkspaceSync):
            raise TypeError("staged must be a WorkspaceSync")
        return self.discard(staged.plan, expected_space=staged.space)

    def clear(self, plan: str) -> int:
        """Empty one bound plan space, retaining it for future appends."""

        plan = self._validate_plan(plan)
        with self._lock:
            workspace = self._workspaces.get(plan)
            if workspace is None:
                return 0
            previous = len(workspace.case_hashes)
            try:
                self._clear_space(workspace.space)
            except BaseException:
                workspace.dirty = True
                raise
            workspace.case_hashes = {}
            workspace.dirty = False
            return previous

    def prune(self, keep: Iterable[str] = ()) -> tuple[str, ...]:
        """Clear and forget every cached plan not present in ``keep``."""

        keep_set = {self._validate_plan(plan) for plan in keep}
        with self._lock:
            removed: list[str] = []
            for plan in sorted(set(self._workspaces) - keep_set):
                workspace = self._workspaces[plan]
                try:
                    self._clear_space(workspace.space)
                except BaseException:
                    workspace.dirty = True
                    raise
                del self._workspaces[plan]
                removed.append(plan)
            return tuple(removed)

    def space_for(self, plan: str) -> str | None:
        """Return the currently bound space name without creating it."""

        plan = self._validate_plan(plan)
        with self._lock:
            workspace = self._workspaces.get(plan)
            return workspace.space if workspace is not None else None

    def plans(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._workspaces))


__all__ = [
    "DEFAULT_MAX_CACHED_PLANS",
    "DEFAULT_MAX_CASES_PER_PLAN",
    "DEFAULT_MAX_FACTS_PER_PLAN",
    "PeTTaExecutor",
    "PeTTaWorkspaceCache",
    "SyncMode",
    "WorkspaceSync",
]
