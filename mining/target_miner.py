"""Target-aware, dataset-agnostic symbolic conjunction mining.

The live recommendation lab's original MeTTa miner enumerates fixed-size
feature/outcome combinations and evaluates target quality afterwards.  This
module provides the complementary supervised search primitive: the target is
kept outside the antecedent, patterns are expanded one predicate at a time,
and each atom owns a vertical integer-bitset cover.

The module intentionally has no dependency on MIND, ``Lab``, MeTTa, or
PeTTaChainer.  Point and pair adapters only need to provide:

* one mapping of bounded categorical facts per case;
* a binary target per case;
* optional non-negative case weights; and
* optional fold labels.

``max_depth`` always means *antecedent predicate count*.  The target is not an
item in the pattern, which avoids the easy off-by-one ambiguity in a miner
where ``engagement`` is one of the combined atoms.

Search modes
------------

``exhaustive=False`` uses safe upper bounds for positive and negative WRAcc to
prune a branch once no descendant can enter the requested top-k.  Setting
``exhaustive=True`` disables only quality-bound pruning; anti-monotone support
pruning and the one-value-per-predicate constraint remain exact.  The two
modes therefore provide a convenient correctness oracle on small workspaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence


AtomicValue = str | int | float | bool | None
Objective = Literal["positive", "negative", "absolute"]


def _normalize_atomic(value: object, *, field_name: str) -> AtomicValue:
    """Validate one deterministic, JSON-compatible categorical value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must not contain NaN or infinity")
        # Keep hashing, equality, and sorting consistent for signed zero.
        return 0.0 if value == 0.0 else value
    raise TypeError(
        f"{field_name} must use only str, int, float, bool, or None values"
    )


def _atomic_key(value: AtomicValue) -> tuple[str, str]:
    """Return a total ordering that keeps values of different types distinct."""

    if value is None:
        return ("0:none", "")
    if isinstance(value, bool):
        return ("1:bool", "1" if value else "0")
    if isinstance(value, int):
        # A sign and zero-padded magnitude make integer ordering deterministic.
        sign = "0" if value < 0 else "1"
        return ("2:int", f"{sign}:{abs(value):040d}")
    if isinstance(value, float):
        return ("3:float", value.hex())
    return ("4:str", value)


@dataclass(frozen=True, eq=False)
class Atom:
    """One categorical antecedent atom.

    Equality is type-sensitive, so the categorical values ``True``, ``1`` and
    ``1.0`` remain distinct even though Python normally considers them equal.
    """

    predicate: str
    value: AtomicValue

    def __post_init__(self) -> None:
        if not isinstance(self.predicate, str) or not self.predicate:
            raise ValueError("atom predicate must be a non-empty string")
        normalized = _normalize_atomic(self.value, field_name="atom value")
        object.__setattr__(self, "value", normalized)

    @property
    def sort_key(self) -> tuple[str, tuple[str, str]]:
        return (self.predicate, _atomic_key(self.value))

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
    """Weighted target contingency table for one antecedent cover."""

    tp: float
    fp: float
    fn: float
    tn: float

    @property
    def total(self) -> float:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def support(self) -> float:
        return self.tp + self.fp

    def as_dict(self) -> dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass(frozen=True)
class CountContingency:
    """Unweighted row counts retained beside weighted sufficient statistics."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def support(self) -> int:
        return self.tp + self.fp

    def as_dict(self) -> dict[str, int]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn}


@dataclass(frozen=True)
class FoldStatistics:
    """Optional within-fold target statistics for an antecedent."""

    fold: AtomicValue
    weighted: WeightedContingency
    counts: CountContingency
    base_rate: float | None
    precision: float | None
    effect: float | None
    wracc: float
    usable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "weighted": self.weighted.as_dict(),
            "counts": self.counts.as_dict(),
            "base_rate": self.base_rate,
            "precision": self.precision,
            "effect": self.effect,
            "wracc": self.wracc,
            "usable": self.usable,
        }


@dataclass(frozen=True)
class MinedPattern:
    """One antecedent-only conjunction and its auditable target statistics."""

    premises: tuple[Atom, ...]
    weighted: WeightedContingency
    counts: CountContingency
    coverage: float
    precision: float | None
    base_rate: float
    wracc: float
    objective_score: float
    positive_wracc_upper_bound: float
    negative_wracc_upper_bound: float
    fold_statistics: tuple[FoldStatistics, ...] = ()
    stable_positive_effect: float | None = None
    stable_negative_effect: float | None = None

    @property
    def depth(self) -> int:
        return len(self.premises)

    def as_dict(self) -> dict[str, Any]:
        return {
            "premises": [atom.as_dict() for atom in self.premises],
            "depth": self.depth,
            "weighted": self.weighted.as_dict(),
            "counts": self.counts.as_dict(),
            "coverage": self.coverage,
            "precision": self.precision,
            "base_rate": self.base_rate,
            "wracc": self.wracc,
            "objective_score": self.objective_score,
            "positive_wracc_upper_bound": self.positive_wracc_upper_bound,
            "negative_wracc_upper_bound": self.negative_wracc_upper_bound,
            "fold_statistics": [fold.as_dict() for fold in self.fold_statistics],
            "stable_positive_effect": self.stable_positive_effect,
            "stable_negative_effect": self.stable_negative_effect,
        }


@dataclass(frozen=True)
class TargetMinerConfig:
    """Search controls.

    ``min_support`` and ``fold_min_support`` are expressed in the same weighted
    mass units as the supplied case weights.  ``min_wracc`` is a non-negative
    threshold on the selected objective: signed positive WRAcc, the magnitude
    of negative WRAcc, or absolute WRAcc.
    """

    min_support: float = 1.0
    max_depth: int = 2
    top_k: int | None = 100
    objective: Objective = "absolute"
    min_wracc: float = 0.0
    exhaustive: bool = False
    fold_min_support: float = 0.0

    def __post_init__(self) -> None:
        for name, value, strictly_positive in (
            ("min_support", self.min_support, True),
            ("min_wracc", self.min_wracc, False),
            ("fold_min_support", self.fold_min_support, False),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite number")
            number = float(value)
            if not math.isfinite(number) or number < 0 or (strictly_positive and number == 0):
                qualifier = "positive" if strictly_positive else "non-negative"
                raise ValueError(f"{name} must be a finite {qualifier} number")
            object.__setattr__(self, name, number)
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            raise ValueError("max_depth must be a positive integer")
        if self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
                raise ValueError("top_k must be a positive integer or None")
            if self.top_k < 1:
                raise ValueError("top_k must be a positive integer or None")
        if self.objective not in {"positive", "negative", "absolute"}:
            raise ValueError("objective must be positive, negative, or absolute")
        if not isinstance(self.exhaustive, bool):
            raise ValueError("exhaustive must be boolean")


@dataclass(frozen=True)
class MiningAudit:
    """Deterministic counters describing the performed search."""

    row_count: int
    predicate_count: int
    candidate_atom_count: int
    frequent_atom_count: int
    atom_support_pruned: int
    nodes_generated: int
    nodes_evaluated: int
    node_support_pruned: int
    bound_checks: int
    bound_pruned: int
    positive_bound_below_threshold: int
    negative_bound_below_threshold: int
    eligible_patterns: int
    output_patterns: int
    max_depth_nodes: int
    cover_cache_hits: int
    cover_cache_misses: int
    total_weight: float
    positive_weight: float
    negative_weight: float
    final_score_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MiningResult:
    """Deterministically ordered patterns plus search provenance."""

    patterns: tuple[MinedPattern, ...]
    frequent_atoms: tuple[Atom, ...]
    config: TargetMinerConfig
    audit: MiningAudit

    def as_dict(self) -> dict[str, Any]:
        return {
            "patterns": [pattern.as_dict() for pattern in self.patterns],
            "frequent_atoms": [atom.as_dict() for atom in self.frequent_atoms],
            "config": dict(self.config.__dict__),
            "audit": self.audit.as_dict(),
        }


@dataclass(frozen=True)
class _CoveredMass:
    positive_weight: float
    negative_weight: float
    positive_count: int
    negative_count: int

    @property
    def support(self) -> float:
        return self.positive_weight + self.negative_weight


@dataclass
class _MutableAudit:
    nodes_generated: int = 0
    nodes_evaluated: int = 0
    node_support_pruned: int = 0
    bound_checks: int = 0
    bound_pruned: int = 0
    positive_bound_below_threshold: int = 0
    negative_bound_below_threshold: int = 0
    eligible_patterns: int = 0
    max_depth_nodes: int = 0
    cover_cache_hits: int = 0
    cover_cache_misses: int = 0


class _CoverMassCache:
    """Compute weighted mass directly from vertical integer bitsets."""

    def __init__(
        self,
        targets: Sequence[bool],
        weights: Sequence[float],
        audit: _MutableAudit,
    ) -> None:
        self._targets = targets
        self._weights = weights
        self._audit = audit
        self._cache: dict[int, _CoveredMass] = {}

    def get(self, cover: int) -> _CoveredMass:
        cached = self._cache.get(cover)
        if cached is not None:
            self._audit.cover_cache_hits += 1
            return cached
        self._audit.cover_cache_misses += 1
        # ``math.fsum`` keeps the sufficient statistics stable when callers
        # present the same weighted cases in a different row order.  Ordinary
        # ``+=`` is observably order-sensitive for mixtures such as 1e16 and
        # unit weights, which could otherwise perturb WRAcc and top-k pruning.
        positive_weights: list[float] = []
        negative_weights: list[float] = []
        positive_count = negative_count = 0
        remaining = cover
        while remaining:
            bit = remaining & -remaining
            index = bit.bit_length() - 1
            if self._targets[index]:
                positive_weights.append(self._weights[index])
                positive_count += 1
            else:
                negative_weights.append(self._weights[index])
                negative_count += 1
            remaining ^= bit
        result = _CoveredMass(
            positive_weight=math.fsum(positive_weights),
            negative_weight=math.fsum(negative_weights),
            positive_count=positive_count,
            negative_count=negative_count,
        )
        self._cache[cover] = result
        return result


def _clamped_difference(total: float, part: float) -> float:
    """Avoid tiny negative residuals caused only by floating summation order."""

    difference = total - part
    tolerance = 1e-12 * max(1.0, abs(total), abs(part))
    return 0.0 if difference < 0.0 and difference >= -tolerance else difference


def _objective_score(wracc: float, objective: Objective) -> float:
    if objective == "positive":
        return wracc
    if objective == "negative":
        return -wracc
    return abs(wracc)


def _bound_strictly_below(bound: float, threshold: float) -> bool:
    """Conservatively compare floating bounds without pruning a rounded tie."""

    tolerance = 1e-12 * max(1.0, abs(bound), abs(threshold))
    return bound < threshold - tolerance


def _pattern_rank(pattern: MinedPattern) -> tuple[Any, ...]:
    """Stable top-k order: quality, sign, simplicity, then lexical premises."""

    return (
        -pattern.objective_score,
        -pattern.wracc,  # Prefer a positive rule on an exact absolute tie.
        pattern.depth,
        tuple(atom.sort_key for atom in pattern.premises),
    )


class TargetAwareConjunctionMiner:
    """Mine supervised antecedent conjunctions from bounded categorical rows."""

    def __init__(self, config: TargetMinerConfig | None = None) -> None:
        self.config = config or TargetMinerConfig()

    def mine(
        self,
        rows: Iterable[Mapping[str, AtomicValue]],
        targets: Iterable[bool | int],
        *,
        weights: Iterable[float] | None = None,
        folds: Iterable[AtomicValue] | None = None,
    ) -> MiningResult:
        """Mine patterns and return sufficient statistics plus audit counters.

        A row is a mapping and therefore already has at most one value for each
        predicate.  Missing predicates simply contribute no atom.  Zero-weight
        rows remain visible in raw counts but cannot satisfy weighted support.
        """

        materialized_rows = list(rows)
        raw_targets = list(targets)
        if not materialized_rows:
            raise ValueError("rows must not be empty")
        if len(raw_targets) != len(materialized_rows):
            raise ValueError("targets length must equal rows length")
        normalized_targets: list[bool] = []
        for index, value in enumerate(raw_targets):
            if value is True or value == 1:
                normalized_targets.append(True)
            elif value is False or value == 0:
                normalized_targets.append(False)
            else:
                raise ValueError(f"target {index} must be boolean or 0/1")

        if weights is None:
            normalized_weights = [1.0] * len(materialized_rows)
        else:
            raw_weights = list(weights)
            if len(raw_weights) != len(materialized_rows):
                raise ValueError("weights length must equal rows length")
            normalized_weights = []
            for index, value in enumerate(raw_weights):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"weight {index} must be finite and non-negative")
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    raise ValueError(f"weight {index} must be finite and non-negative")
                normalized_weights.append(number)
        total_weight = math.fsum(normalized_weights)
        if total_weight <= 0.0:
            raise ValueError("at least one row must have positive weight")

        normalized_folds: list[AtomicValue] | None = None
        if folds is not None:
            raw_folds = list(folds)
            if len(raw_folds) != len(materialized_rows):
                raise ValueError("folds length must equal rows length")
            normalized_folds = [
                _normalize_atomic(value, field_name=f"fold {index}")
                for index, value in enumerate(raw_folds)
            ]

        covers: dict[Atom, int] = {}
        predicates: set[str] = set()
        for row_index, row in enumerate(materialized_rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"row {row_index} must be a mapping")
            for predicate, raw_value in row.items():
                if not isinstance(predicate, str) or not predicate:
                    raise ValueError(
                        f"row {row_index} predicates must be non-empty strings"
                    )
                value = _normalize_atomic(
                    raw_value, field_name=f"row {row_index} value for {predicate}"
                )
                atom = Atom(predicate, value)
                predicates.add(predicate)
                covers[atom] = covers.get(atom, 0) | (1 << row_index)

        mutable_audit = _MutableAudit()
        mass_cache = _CoverMassCache(
            normalized_targets, normalized_weights, mutable_audit
        )
        candidate_atoms = sorted(covers, key=lambda atom: atom.sort_key)
        frequent_atoms = tuple(
            atom
            for atom in candidate_atoms
            if mass_cache.get(covers[atom]).support >= self.config.min_support
        )
        atom_support_pruned = len(candidate_atoms) - len(frequent_atoms)
        frequent_covers = tuple(covers[atom] for atom in frequent_atoms)

        total_positive_weight = math.fsum(
            weight
            for target, weight in zip(normalized_targets, normalized_weights)
            if target
        )
        total_negative_weight = _clamped_difference(
            total_weight, total_positive_weight
        )
        total_positive_count = sum(normalized_targets)
        total_negative_count = len(normalized_targets) - total_positive_count
        base_rate = total_positive_weight / total_weight

        fold_groups: tuple[tuple[AtomicValue, int, _CoveredMass], ...] = ()
        if normalized_folds is not None:
            grouped: dict[tuple[str, str], tuple[AtomicValue, int]] = {}
            for index, fold in enumerate(normalized_folds):
                key = _atomic_key(fold)
                previous = grouped.get(key)
                cover = (previous[1] if previous else 0) | (1 << index)
                grouped[key] = (fold, cover)
            fold_groups = tuple(
                (fold, cover, mass_cache.get(cover))
                for _key, (fold, cover) in sorted(grouped.items())
            )

        best: list[MinedPattern] = []

        def current_threshold() -> float:
            threshold = self.config.min_wracc
            if self.config.top_k is not None and len(best) >= self.config.top_k:
                threshold = max(threshold, best[-1].objective_score)
            return threshold

        def make_pattern(
            premise_indexes: tuple[int, ...], cover: int, mass: _CoveredMass
        ) -> MinedPattern:
            weighted = WeightedContingency(
                tp=mass.positive_weight,
                fp=mass.negative_weight,
                fn=_clamped_difference(total_positive_weight, mass.positive_weight),
                tn=_clamped_difference(total_negative_weight, mass.negative_weight),
            )
            counts = CountContingency(
                tp=mass.positive_count,
                fp=mass.negative_count,
                fn=total_positive_count - mass.positive_count,
                tn=total_negative_count - mass.negative_count,
            )
            support = mass.support
            coverage = support / total_weight
            precision = mass.positive_weight / support if support > 0.0 else None
            wracc = (
                mass.positive_weight / total_weight - coverage * base_rate
            )
            positive_upper_bound = (
                mass.positive_weight / total_weight * (1.0 - base_rate)
            )
            negative_upper_bound = (
                mass.negative_weight / total_weight * base_rate
            )

            per_fold: list[FoldStatistics] = []
            usable_effects: list[float] = []
            for fold, fold_cover, fold_total_mass in fold_groups:
                covered = mass_cache.get(cover & fold_cover)
                fold_total = fold_total_mass.support
                fold_support = covered.support
                fold_base = (
                    fold_total_mass.positive_weight / fold_total
                    if fold_total > 0.0
                    else None
                )
                fold_precision = (
                    covered.positive_weight / fold_support
                    if fold_support > 0.0
                    else None
                )
                effect = (
                    fold_precision - fold_base
                    if fold_precision is not None and fold_base is not None
                    else None
                )
                fold_wracc = (
                    fold_support / fold_total * effect
                    if effect is not None and fold_total > 0.0
                    else 0.0
                )
                usable = (
                    effect is not None
                    and fold_support >= self.config.fold_min_support
                )
                if usable:
                    usable_effects.append(effect)
                per_fold.append(
                    FoldStatistics(
                        fold=fold,
                        weighted=WeightedContingency(
                            tp=covered.positive_weight,
                            fp=covered.negative_weight,
                            fn=_clamped_difference(
                                fold_total_mass.positive_weight,
                                covered.positive_weight,
                            ),
                            tn=_clamped_difference(
                                fold_total_mass.negative_weight,
                                covered.negative_weight,
                            ),
                        ),
                        counts=CountContingency(
                            tp=covered.positive_count,
                            fp=covered.negative_count,
                            fn=(
                                fold_total_mass.positive_count
                                - covered.positive_count
                            ),
                            tn=(
                                fold_total_mass.negative_count
                                - covered.negative_count
                            ),
                        ),
                        base_rate=fold_base,
                        precision=fold_precision,
                        effect=effect,
                        wracc=fold_wracc,
                        usable=usable,
                    )
                )
            stable_positive = min(usable_effects) if usable_effects else None
            stable_negative = (
                min(-effect for effect in usable_effects)
                if usable_effects
                else None
            )
            return MinedPattern(
                premises=tuple(frequent_atoms[index] for index in premise_indexes),
                weighted=weighted,
                counts=counts,
                coverage=coverage,
                precision=precision,
                base_rate=base_rate,
                wracc=wracc,
                objective_score=_objective_score(wracc, self.config.objective),
                positive_wracc_upper_bound=positive_upper_bound,
                negative_wracc_upper_bound=negative_upper_bound,
                fold_statistics=tuple(per_fold),
                stable_positive_effect=stable_positive,
                stable_negative_effect=stable_negative,
            )

        def retain(pattern: MinedPattern) -> None:
            if pattern.objective_score < self.config.min_wracc:
                return
            mutable_audit.eligible_patterns += 1
            best.append(pattern)
            best.sort(key=_pattern_rank)
            if self.config.top_k is not None and len(best) > self.config.top_k:
                del best[self.config.top_k :]

        def visit(
            premise_indexes: tuple[int, ...],
            used_predicates: frozenset[str],
            start_index: int,
            cover: int,
            mass: _CoveredMass,
        ) -> None:
            pattern = make_pattern(premise_indexes, cover, mass)
            retain(pattern)
            if len(premise_indexes) >= self.config.max_depth:
                mutable_audit.max_depth_nodes += 1
                return

            if not self.config.exhaustive:
                threshold = current_threshold()
                mutable_audit.bound_checks += 1
                positive_low = _bound_strictly_below(
                    pattern.positive_wracc_upper_bound, threshold
                )
                negative_low = _bound_strictly_below(
                    pattern.negative_wracc_upper_bound, threshold
                )
                mutable_audit.positive_bound_below_threshold += int(positive_low)
                mutable_audit.negative_bound_below_threshold += int(negative_low)
                if self.config.objective == "positive":
                    prune = positive_low
                elif self.config.objective == "negative":
                    prune = negative_low
                else:
                    prune = positive_low and negative_low
                if prune:
                    mutable_audit.bound_pruned += 1
                    return

            for atom_index in range(start_index, len(frequent_atoms)):
                atom = frequent_atoms[atom_index]
                if atom.predicate in used_predicates:
                    continue
                mutable_audit.nodes_generated += 1
                child_cover = cover & frequent_covers[atom_index]
                child_mass = mass_cache.get(child_cover)
                mutable_audit.nodes_evaluated += 1
                if child_mass.support < self.config.min_support:
                    mutable_audit.node_support_pruned += 1
                    continue
                visit(
                    (*premise_indexes, atom_index),
                    used_predicates | {atom.predicate},
                    atom_index + 1,
                    child_cover,
                    child_mass,
                )

        all_rows_cover = (1 << len(materialized_rows)) - 1
        all_rows_mass = mass_cache.get(all_rows_cover)
        for atom_index, atom in enumerate(frequent_atoms):
            mutable_audit.nodes_generated += 1
            cover = frequent_covers[atom_index]
            mass = mass_cache.get(cover)
            mutable_audit.nodes_evaluated += 1
            # Atom-level filtering already guarantees this, but retaining the
            # guard keeps the root loop correct if storage changes later.
            if mass.support < self.config.min_support:
                mutable_audit.node_support_pruned += 1
                continue
            visit((atom_index,), frozenset({atom.predicate}), atom_index + 1, cover, mass)

        best.sort(key=_pattern_rank)
        final_threshold = self.config.min_wracc
        if self.config.top_k is not None and len(best) >= self.config.top_k:
            final_threshold = max(final_threshold, best[-1].objective_score)
        audit = MiningAudit(
            row_count=len(materialized_rows),
            predicate_count=len(predicates),
            candidate_atom_count=len(candidate_atoms),
            frequent_atom_count=len(frequent_atoms),
            atom_support_pruned=atom_support_pruned,
            nodes_generated=mutable_audit.nodes_generated,
            nodes_evaluated=mutable_audit.nodes_evaluated,
            node_support_pruned=mutable_audit.node_support_pruned,
            bound_checks=mutable_audit.bound_checks,
            bound_pruned=mutable_audit.bound_pruned,
            positive_bound_below_threshold=(
                mutable_audit.positive_bound_below_threshold
            ),
            negative_bound_below_threshold=(
                mutable_audit.negative_bound_below_threshold
            ),
            eligible_patterns=mutable_audit.eligible_patterns,
            output_patterns=len(best),
            max_depth_nodes=mutable_audit.max_depth_nodes,
            cover_cache_hits=mutable_audit.cover_cache_hits,
            cover_cache_misses=mutable_audit.cover_cache_misses,
            total_weight=total_weight,
            positive_weight=total_positive_weight,
            negative_weight=total_negative_weight,
            final_score_threshold=final_threshold,
        )
        # Assert the root statistics while they are still cheap to inspect. This
        # also prevents an accidentally unused all-row cover from hiding a mass
        # accounting regression.
        if not math.isclose(all_rows_mass.support, total_weight):
            raise AssertionError("vertical cover mass does not match total case weight")
        return MiningResult(
            patterns=tuple(best),
            frequent_atoms=frequent_atoms,
            config=self.config,
            audit=audit,
        )


def mine_target_patterns(
    rows: Iterable[Mapping[str, AtomicValue]],
    targets: Iterable[bool | int],
    *,
    weights: Iterable[float] | None = None,
    folds: Iterable[AtomicValue] | None = None,
    config: TargetMinerConfig | None = None,
) -> MiningResult:
    """Convenience functional API shared by point and pair adapters."""

    return TargetAwareConjunctionMiner(config).mine(
        rows, targets, weights=weights, folds=folds
    )


__all__ = [
    "Atom",
    "CountContingency",
    "FoldStatistics",
    "MinedPattern",
    "MiningAudit",
    "MiningResult",
    "TargetAwareConjunctionMiner",
    "TargetMinerConfig",
    "WeightedContingency",
    "mine_target_patterns",
]
