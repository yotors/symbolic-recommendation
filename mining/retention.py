"""Exact, bounded retention of complete causal mining units.

The unit is supplied by the caller (normally one closed impression).  This
module deliberately knows nothing about a particular dataset schema.  It
keeps a newest suffix of units, never individual rows from a unit, so a
retained mining snapshot cannot contain half of an impression.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar


DEFAULT_RETENTION_MAX_UNITS = 25_000
DEFAULT_RETENTION_MAX_CASES = 250_000
HARD_RETENTION_MAX_UNITS = 1_000_000
HARD_RETENTION_MAX_CASES = 1_000_000

T = TypeVar("T")


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fingerprint(unit_ids: Iterable[str]) -> str:
    payload = json.dumps(
        tuple(unit_ids), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RetentionAudit:
    """JSON-safe metadata for one immutable retained snapshot."""

    max_units: int
    max_cases: int
    source_units: int
    source_cases: int
    retained_units: int
    retained_cases: int
    expired_units: int
    expired_cases: int
    unit_limit_applied: bool
    case_limit_applied: bool
    retained_membership_sha256: str
    policy: str = "newest_complete_unit_suffix_v1"
    atomic_unit: str = "caller_supplied_causal_unit"
    exact: bool = True
    split_units: bool = False
    expiration_update: str = "exact_full_rebuild"

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "atomic_unit": self.atomic_unit,
            "max_units": self.max_units,
            "max_cases": self.max_cases,
            "source_units": self.source_units,
            "source_cases": self.source_cases,
            "retained_units": self.retained_units,
            "retained_cases": self.retained_cases,
            "expired_units": self.expired_units,
            "expired_cases": self.expired_cases,
            "unit_limit_applied": self.unit_limit_applied,
            "case_limit_applied": self.case_limit_applied,
            "retained_membership_sha256": self.retained_membership_sha256,
            "exact": self.exact,
            "split_units": self.split_units,
            "expiration_update": self.expiration_update,
        }


@dataclass(frozen=True, slots=True)
class RetainedWindow(Generic[T]):
    """Retained records plus ordered membership needed by exact fpMiner state."""

    records: tuple[T, ...]
    unit_ids: tuple[str, ...]
    audit: RetentionAudit


def retain_complete_units(
    records: Iterable[tuple[str, T]],
    *,
    max_units: int = DEFAULT_RETENTION_MAX_UNITS,
    max_cases: int = DEFAULT_RETENTION_MAX_CASES,
) -> RetainedWindow[T]:
    """Keep the newest whole units within both finite resource bounds.

    Unit recency is its final record's position.  This remains deterministic
    when records from a unit are interleaved.  If the newest unit alone cannot
    fit, the snapshot is rejected instead of silently splitting that unit.
    Once an older unit would cross the case bound, all still-older units are
    expired as well, preserving a true suffix rather than cherry-picking.
    """

    max_units = _positive_integer("max_units", max_units)
    max_cases = _positive_integer("max_cases", max_cases)
    materialized: list[tuple[str, T]] = []
    counts: dict[str, int] = {}
    last_position: dict[str, int] = {}
    for position, (raw_unit_id, record) in enumerate(records):
        if not isinstance(raw_unit_id, str) or not raw_unit_id:
            raise ValueError("unit IDs must be non-empty strings")
        unit_id = raw_unit_id
        materialized.append((unit_id, record))
        counts[unit_id] = counts.get(unit_id, 0) + 1
        last_position[unit_id] = position

    ordered_units = tuple(sorted(counts, key=last_position.__getitem__))
    selected_reversed: list[str] = []
    retained_cases = 0
    case_limit_applied = False
    for unit_id in reversed(ordered_units):
        unit_cases = counts[unit_id]
        if not selected_reversed and unit_cases > max_cases:
            raise MemoryError(
                "newest causal unit exceeds retention case limit; refusing "
                "to split an atomic unit"
            )
        if len(selected_reversed) >= max_units:
            break
        if retained_cases + unit_cases > max_cases:
            case_limit_applied = True
            break
        selected_reversed.append(unit_id)
        retained_cases += unit_cases

    retained_units_ordered = tuple(reversed(selected_reversed))
    retained_set = frozenset(retained_units_ordered)
    retained_records = tuple(
        record for unit_id, record in materialized if unit_id in retained_set
    )
    source_cases = len(materialized)
    source_units = len(ordered_units)
    audit = RetentionAudit(
        max_units=max_units,
        max_cases=max_cases,
        source_units=source_units,
        source_cases=source_cases,
        retained_units=len(retained_units_ordered),
        retained_cases=len(retained_records),
        expired_units=source_units - len(retained_units_ordered),
        expired_cases=source_cases - len(retained_records),
        unit_limit_applied=source_units > len(retained_units_ordered)
        and len(retained_units_ordered) == max_units,
        case_limit_applied=case_limit_applied,
        retained_membership_sha256=_fingerprint(retained_units_ordered),
    )
    return RetainedWindow(
        records=retained_records,
        unit_ids=retained_units_ordered,
        audit=audit,
    )


__all__ = [
    "DEFAULT_RETENTION_MAX_CASES",
    "DEFAULT_RETENTION_MAX_UNITS",
    "HARD_RETENTION_MAX_CASES",
    "HARD_RETENTION_MAX_UNITS",
    "RetainedWindow",
    "RetentionAudit",
    "retain_complete_units",
]
