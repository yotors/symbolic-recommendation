"""Typed boundary records for PeTTa conditional mining."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

AtomicValue = str | int | float | bool | None


def _atomic(value: object, name: str) -> AtomicValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return 0.0 if value == 0.0 else value
    raise TypeError(f"{name} must be a finite JSON scalar")


def _value_key(value: AtomicValue) -> tuple[str, str]:
    if value is None:
        return ("none", "")
    if isinstance(value, bool):
        return ("bool", "1" if value else "0")
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", value.hex())
    return ("str", value)


def _number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}")
    return result


@dataclass(frozen=True, eq=False)
class Atom:
    predicate: str
    value: AtomicValue

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, str) or not self.predicate:
            raise ValueError("atom predicate must be non-empty")
        object.__setattr__(self, "value", _atomic(self.value, "atom value"))

    @property
    def sort_key(self) -> tuple[str, tuple[str, str]]:
        return (self.predicate, _value_key(self.value))

    def __hash__(self) -> int:
        return hash((self.predicate, type(self.value), self.value))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Atom)
            and self.predicate == other.predicate
            and type(self.value) is type(other.value)
            and self.value == other.value
        )

    def as_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "value": self.value}


@dataclass(frozen=True)
class WeightedContingency:
    tp: float
    fp: float
    fn: float
    tn: float

    @property
    def support(self) -> float:
        return self.tp + self.fp

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CountContingency:
    tp: int
    fp: int
    fn: int
    tn: int

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class ConditionalMiningLimitError(ValueError):
    """A declared mining bound was exceeded before returning partial rules."""


@dataclass(frozen=True)
class FpMinerUnary:
    rule_id: str
    predicate: str
    value: AtomicValue
    target: AtomicValue
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("fpMiner rule_id must be non-empty")
        atom = Atom(self.predicate, self.value)
        object.__setattr__(self, "value", atom.value)
        object.__setattr__(self, "target", _atomic(self.target, "target"))
        if not isinstance(self.source, str) or not self.source.replace("\\", "/").endswith(
            "fpMiner.metta"
        ):
            raise ValueError("fpMiner source must end with fpMiner.metta")

    @property
    def atom(self) -> Atom:
        return Atom(self.predicate, self.value)


@dataclass(frozen=True)
class ConditionalMiningConfig:
    min_support: float = 10.0
    fold_min_support: float = 3.0
    max_depth: int = 2
    top_k: int = 100
    min_usable_folds: int = 2
    min_effect: float = 0.0
    min_incremental_effect: float = 0.0
    max_rows: int = 250_000
    max_values_per_predicate: int = 16
    max_generated_candidates: int = 200_000

    def __post_init__(self) -> None:
        for name, positive in (
            ("min_support", True),
            ("fold_min_support", True),
            ("min_effect", False),
            ("min_incremental_effect", False),
        ):
            object.__setattr__(self, name, _number(
                name, getattr(self, name), positive=positive
            ))
        for name in (
            "max_depth", "top_k", "min_usable_folds", "max_rows",
            "max_values_per_predicate", "max_generated_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_depth != 2:
            raise ValueError("the champion conditional miner has depth 2")


@dataclass(frozen=True)
class ConditionalFoldStatistics:
    fold: AtomicValue
    weighted: WeightedContingency
    counts: CountContingency
    support: float
    coverage: float
    base_rate: float
    precision: float
    effect: float
    parent_precision: float
    incremental_effect: float
    wracc: float
    incremental_wracc: float
    usable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "weighted": self.weighted.as_dict(),
            "counts": self.counts.as_dict(),
        }


@dataclass(frozen=True)
class ConditionalPattern:
    premises: tuple[Atom, ...]
    fpminer_seed_rule_ids: tuple[str, ...]
    evidence_lineages: tuple[str, ...]
    semantic_lineages: tuple[str, ...]
    context_lineages: tuple[str, ...]
    lineage_signature: str
    variant_signature: str
    weighted: WeightedContingency
    counts: CountContingency
    support: float
    coverage: float
    base_rate: float
    precision: float
    effect: float
    parent_precision: float
    incremental_effect: float
    wracc: float
    incremental_wracc: float
    stable_effect: float
    stable_incremental_effect: float
    robust_incremental_wracc: float
    usable_fold_count: int
    fold_statistics: tuple[ConditionalFoldStatistics, ...]

    @property
    def depth(self) -> int:
        return len(self.premises)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "premises": [atom.as_dict() for atom in self.premises],
            "depth": self.depth,
            "weighted": self.weighted.as_dict(),
            "counts": self.counts.as_dict(),
            "fold_statistics": [fold.as_dict() for fold in self.fold_statistics],
            "fpminer_seed_rule_ids": list(self.fpminer_seed_rule_ids),
            "evidence_lineages": list(self.evidence_lineages),
            "semantic_lineages": list(self.semantic_lineages),
            "context_lineages": list(self.context_lineages),
        }


@dataclass(frozen=True)
class ConditionalMiningResult:
    patterns: tuple[ConditionalPattern, ...]
    config: ConditionalMiningConfig
    audit: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "patterns": [pattern.as_dict() for pattern in self.patterns],
            "config": dict(self.config.__dict__),
            "audit": self.audit.as_dict(),
        }


__all__ = [
    "Atom", "ConditionalFoldStatistics", "ConditionalMiningConfig",
    "ConditionalMiningLimitError", "ConditionalMiningResult",
    "ConditionalPattern", "CountContingency", "FpMinerUnary",
    "WeightedContingency",
]
