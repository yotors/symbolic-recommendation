"""Bounded target-aware expansion of real-miner rules with LLM evidence.

This module is deliberately independent of MIND, ``Lab`` and PeTTaChainer.
It consumes *training cases only* and returns auditable antecedent structures;
it neither compiles rules nor scores candidates.  The intended hybrid is::

    real MeTTa fpMiner unary rules
        -> this constrained conditional expansion
        -> population CTV calibration
        -> PeTTaChainer proof execution

Every returned conjunction:

* contains an atom discovered by the real fpMiner;
* contains both an LLM-derived predicate and a structured-context predicate;
* has exact weighted support and contingency statistics;
* improves on every immediate parent on the complete training population; and
* has the same positive incremental direction in enough temporal folds.

The caller declares predicate roles and evidence lineages.  Consequently this
code contains no dataset taxonomy, concept names, user identifiers, or MIND
field assumptions.  Missing evidence is omitted instead of becoming a useful
``None`` category.  Candidate generation is protected by explicit vocabulary,
row, depth, and combination caps and fails closed when a cap is exceeded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .target_miner import Atom, CountContingency, WeightedContingency


AtomicValue = str | int | float | bool | None
CaseValue = AtomicValue | Sequence[AtomicValue] | set[AtomicValue] | frozenset[AtomicValue]

_FORBIDDEN_TARGET_PREDICATES = frozenset({
    "action", "clicked", "engagement", "is_click", "label", "outcome", "target",
})


class ConditionalMiningLimitError(ValueError):
    """Raised before a partial result can escape an exceeded search bound."""


def _typed_value_key(value: AtomicValue) -> tuple[str, str]:
    if value is None:
        return ("none", "")
    if isinstance(value, bool):
        return ("bool", "1" if value else "0")
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", value.hex())
    return ("str", value)


def _fold_key(value: AtomicValue) -> tuple[str, str]:
    # Reuse Atom's public typed ordering without depending on target_miner's
    # private helpers.
    return _typed_value_key(Atom("fold", value).value)


def _validate_number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    invalid = not math.isfinite(number) or number < 0.0 or (positive and number == 0.0)
    if invalid:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return number


@dataclass(frozen=True)
class FpMinerUnary:
    """One unary rule parsed from the real MeTTa fpMiner output.

    ``source`` is checked rather than silently accepting an arbitrary host-side
    seed.  The lab currently records these rules as
    ``recommendation/miner/fpMiner.metta``.
    """

    rule_id: str
    predicate: str
    value: AtomicValue
    target: AtomicValue
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("fpMiner unary rule_id must be non-empty")
        atom = Atom(self.predicate, self.value)
        object.__setattr__(self, "value", atom.value)
        if not isinstance(self.source, str) or not self.source.replace("\\", "/").endswith(
            "fpMiner.metta"
        ):
            raise ValueError("fpMiner unary source must end with fpMiner.metta")
        # Validate the target as a bounded atomic value too.
        object.__setattr__(self, "target", Atom("target", self.target).value)

    @property
    def atom(self) -> Atom:
        return Atom(self.predicate, self.value)


@dataclass(frozen=True)
class ConditionalMiningConfig:
    """Dataset-independent safety and stability controls."""

    min_support: float = 10.0
    fold_min_support: float = 3.0
    max_depth: int = 3
    top_k: int = 100
    min_usable_folds: int = 2
    min_effect: float = 0.0
    min_incremental_effect: float = 0.0
    max_rows: int = 250_000
    max_values_per_predicate: int = 16
    max_frequent_atoms: int = 96
    max_generated_candidates: int = 200_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_support", _validate_number(
            "min_support", self.min_support, positive=True
        ))
        object.__setattr__(self, "fold_min_support", _validate_number(
            "fold_min_support", self.fold_min_support, positive=True
        ))
        object.__setattr__(self, "min_effect", _validate_number(
            "min_effect", self.min_effect
        ))
        object.__setattr__(self, "min_incremental_effect", _validate_number(
            "min_incremental_effect", self.min_incremental_effect
        ))
        for name in (
            "max_depth", "top_k", "min_usable_folds", "max_rows",
            "max_values_per_predicate", "max_frequent_atoms",
            "max_generated_candidates",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_depth < 2 or self.max_depth > 4:
            raise ValueError("max_depth must be between 2 and 4")


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
            "fold": self.fold,
            "weighted": self.weighted.as_dict(),
            "counts": self.counts.as_dict(),
            "support": self.support,
            "coverage": self.coverage,
            "base_rate": self.base_rate,
            "precision": self.precision,
            "effect": self.effect,
            "parent_precision": self.parent_precision,
            "incremental_effect": self.incremental_effect,
            "wracc": self.wracc,
            "incremental_wracc": self.incremental_wracc,
            "usable": self.usable,
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
    covered_row_indexes: tuple[int, ...]

    @property
    def depth(self) -> int:
        return len(self.premises)

    def as_dict(self, *, include_covered_rows: bool = False) -> dict[str, Any]:
        result = {
            "premises": [atom.as_dict() for atom in self.premises],
            "depth": self.depth,
            "fpminer_seed_rule_ids": list(self.fpminer_seed_rule_ids),
            "evidence_lineages": list(self.evidence_lineages),
            "semantic_lineages": list(self.semantic_lineages),
            "context_lineages": list(self.context_lineages),
            "lineage_signature": self.lineage_signature,
            "variant_signature": self.variant_signature,
            "weighted": self.weighted.as_dict(),
            "counts": self.counts.as_dict(),
            "support": self.support,
            "coverage": self.coverage,
            "base_rate": self.base_rate,
            "precision": self.precision,
            "effect": self.effect,
            "parent_precision": self.parent_precision,
            "incremental_effect": self.incremental_effect,
            "wracc": self.wracc,
            "incremental_wracc": self.incremental_wracc,
            "stable_effect": self.stable_effect,
            "stable_incremental_effect": self.stable_incremental_effect,
            "robust_incremental_wracc": self.robust_incremental_wracc,
            "usable_fold_count": self.usable_fold_count,
            "fold_statistics": [fold.as_dict() for fold in self.fold_statistics],
        }
        if include_covered_rows:
            result["covered_row_indexes"] = list(self.covered_row_indexes)
        return result


@dataclass(frozen=True)
class ConditionalMiningAudit:
    row_count: int
    fold_count: int
    semantic_predicate_count: int
    context_predicate_count: int
    candidate_atom_count: int
    frequent_atom_count: int
    support_pruned_atoms: int
    input_fpminer_unaries: int
    usable_fpminer_seed_atoms: int
    generated_candidates: int
    repeated_predicate_rejections: int
    role_rejections: int
    seed_rejections: int
    support_rejections: int
    insufficient_fold_rejections: int
    effect_rejections: int
    incremental_effect_rejections: int
    unstable_effect_rejections: int
    unstable_incremental_rejections: int
    eligible_patterns: int
    output_patterns: int
    total_weight: float
    positive_weight: float

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ConditionalMiningResult:
    patterns: tuple[ConditionalPattern, ...]
    frequent_atoms: tuple[Atom, ...]
    config: ConditionalMiningConfig
    audit: ConditionalMiningAudit

    def as_dict(self) -> dict[str, Any]:
        return {
            "patterns": [pattern.as_dict() for pattern in self.patterns],
            "frequent_atoms": [atom.as_dict() for atom in self.frequent_atoms],
            "config": dict(self.config.__dict__),
            "audit": self.audit.as_dict(),
        }


@dataclass(frozen=True)
class _Mass:
    positive_weight: float
    negative_weight: float
    positive_count: int
    negative_count: int

    @property
    def support(self) -> float:
        return self.positive_weight + self.negative_weight


def _bounded_materialize(values: Iterable[Any], *, name: str, limit: int) -> list[Any]:
    materialized = []
    for value in values:
        if len(materialized) >= limit:
            raise ConditionalMiningLimitError(f"{name} exceeds max_rows={limit}")
        materialized.append(value)
    return materialized


def _as_atoms(predicate: str, raw: CaseValue) -> tuple[Atom, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple, set, frozenset)):
        values = raw
    else:
        values = (raw,)
    unique = {Atom(predicate, value) for value in values if value is not None}
    return tuple(sorted(unique, key=lambda atom: atom.sort_key))


def _pattern_sort_key(pattern: ConditionalPattern) -> tuple[Any, ...]:
    return (
        -pattern.robust_incremental_wracc,
        -pattern.incremental_wracc,
        -pattern.wracc,
        -pattern.stable_incremental_effect,
        pattern.depth,
        tuple(atom.sort_key for atom in pattern.premises),
    )


def mine_conditional_llm_patterns(
    training_rows: Iterable[Mapping[str, CaseValue]],
    targets: Iterable[bool | int],
    *,
    fpminer_unaries: Iterable[FpMinerUnary],
    positive_target: AtomicValue,
    semantic_predicates: Iterable[str],
    context_predicates: Iterable[str],
    predicate_lineages: Mapping[str, str],
    weights: Iterable[float] | None = None,
    folds: Iterable[AtomicValue],
    config: ConditionalMiningConfig | None = None,
) -> ConditionalMiningResult:
    """Mine stable conditional expansions anchored in real fpMiner unaries.

    Inputs must be causal training snapshots.  Evaluation rows are intentionally
    absent from the API.  Semantic predicates may be multi-valued (for example,
    several grounded concepts); structured context predicates must be scalar so
    a rule cannot express contradictory context values.
    """

    selected_config = config or ConditionalMiningConfig()
    rows = _bounded_materialize(
        training_rows, name="training_rows", limit=selected_config.max_rows
    )
    if not rows:
        raise ValueError("training_rows must not be empty")
    raw_targets = _bounded_materialize(
        targets, name="targets", limit=selected_config.max_rows
    )
    if len(raw_targets) != len(rows):
        raise ValueError("targets length must equal training_rows length")
    normalized_targets = []
    for index, target in enumerate(raw_targets):
        if target is True or target == 1:
            normalized_targets.append(True)
        elif target is False or target == 0:
            normalized_targets.append(False)
        else:
            raise ValueError(f"target {index} must be boolean or 0/1")

    if weights is None:
        normalized_weights = [1.0] * len(rows)
    else:
        raw_weights = _bounded_materialize(
            weights, name="weights", limit=selected_config.max_rows
        )
        if len(raw_weights) != len(rows):
            raise ValueError("weights length must equal training_rows length")
        normalized_weights = [
            _validate_number(f"weight {index}", value)
            for index, value in enumerate(raw_weights)
        ]
    total_weight = math.fsum(normalized_weights)
    if total_weight <= 0.0:
        raise ValueError("at least one training row must have positive weight")

    raw_folds = _bounded_materialize(folds, name="folds", limit=selected_config.max_rows)
    if len(raw_folds) != len(rows):
        raise ValueError("folds length must equal training_rows length")
    normalized_folds = [Atom("fold", fold).value for fold in raw_folds]
    distinct_fold_values: dict[tuple[str, str], AtomicValue] = {}
    for fold in normalized_folds:
        distinct_fold_values[_fold_key(fold)] = fold
    distinct_folds = tuple(
        distinct_fold_values[key] for key in sorted(distinct_fold_values)
    )
    if len(distinct_folds) < selected_config.min_usable_folds:
        raise ValueError(
            "folds contain fewer distinct values than min_usable_folds"
        )

    semantic = frozenset(semantic_predicates)
    context = frozenset(context_predicates)
    for role, predicates in (("semantic", semantic), ("context", context)):
        if not predicates or any(not isinstance(item, str) or not item for item in predicates):
            raise ValueError(f"{role}_predicates must contain non-empty strings")
    overlap = semantic & context
    if overlap:
        raise ValueError(f"predicate roles must be disjoint: {sorted(overlap)!r}")
    active_predicates = semantic | context
    forbidden = sorted(
        predicate for predicate in active_predicates
        if predicate.casefold() in _FORBIDDEN_TARGET_PREDICATES
    )
    if forbidden:
        raise ValueError(f"target-like predicates are forbidden in antecedents: {forbidden!r}")
    missing_lineages = sorted(
        predicate for predicate in active_predicates
        if not isinstance(predicate_lineages.get(predicate), str)
        or not predicate_lineages[predicate]
    )
    if missing_lineages:
        raise ValueError(f"missing evidence lineage for predicates: {missing_lineages!r}")

    normalized_rows: list[frozenset[Atom]] = []
    covers: dict[Atom, int] = {}
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"training row {row_index} must be a mapping")
        target_like = [key for key in row if isinstance(key, str)
                       and key.casefold() in _FORBIDDEN_TARGET_PREDICATES]
        if target_like:
            raise ValueError(
                f"training row {row_index} contains target-like predicates: {target_like!r}"
            )
        atoms: set[Atom] = set()
        for predicate in active_predicates:
            if predicate not in row:
                continue
            values = _as_atoms(predicate, row[predicate])
            if predicate in context and len(values) > 1:
                raise ValueError(
                    f"context predicate {predicate!r} must be scalar in row {row_index}"
                )
            atoms.update(values)
        frozen = frozenset(atoms)
        normalized_rows.append(frozen)
        for atom in frozen:
            covers[atom] = covers.get(atom, 0) | (1 << row_index)

    mass_cache: dict[int, _Mass] = {}

    def mass(cover: int) -> _Mass:
        cached = mass_cache.get(cover)
        if cached is not None:
            return cached
        positive_weights: list[float] = []
        negative_weights: list[float] = []
        positive_count = negative_count = 0
        remaining = cover
        while remaining:
            bit = remaining & -remaining
            index = bit.bit_length() - 1
            if normalized_targets[index]:
                positive_weights.append(normalized_weights[index])
                positive_count += 1
            else:
                negative_weights.append(normalized_weights[index])
                negative_count += 1
            remaining ^= bit
        result = _Mass(
            positive_weight=math.fsum(positive_weights),
            negative_weight=math.fsum(negative_weights),
            positive_count=positive_count,
            negative_count=negative_count,
        )
        mass_cache[cover] = result
        return result

    candidate_atoms = sorted(covers, key=lambda atom: atom.sort_key)
    frequent_atoms = tuple(
        atom for atom in candidate_atoms
        if mass(covers[atom]).support >= selected_config.min_support
    )
    by_predicate: dict[str, int] = {}
    for atom in frequent_atoms:
        by_predicate[atom.predicate] = by_predicate.get(atom.predicate, 0) + 1
    excessive_values = {
        predicate: count for predicate, count in by_predicate.items()
        if count > selected_config.max_values_per_predicate
    }
    if excessive_values:
        raise ConditionalMiningLimitError(
            "frequent vocabulary exceeds max_values_per_predicate: "
            + json.dumps(excessive_values, sort_keys=True)
        )
    if len(frequent_atoms) > selected_config.max_frequent_atoms:
        raise ConditionalMiningLimitError(
            f"frequent atoms {len(frequent_atoms)} exceed "
            f"max_frequent_atoms={selected_config.max_frequent_atoms}"
        )

    positive_target_atom = Atom("target", positive_target).value
    seeds = list(fpminer_unaries)
    if not seeds:
        raise ValueError("at least one real fpMiner unary is required")
    seed_rule_ids: dict[Atom, set[str]] = {}
    seen_rule_ids: set[str] = set()
    for seed in seeds:
        if not isinstance(seed, FpMinerUnary):
            raise TypeError("fpminer_unaries must contain FpMinerUnary values")
        if seed.rule_id in seen_rule_ids:
            raise ValueError(f"duplicate fpMiner rule_id: {seed.rule_id}")
        seen_rule_ids.add(seed.rule_id)
        if type(seed.target) is type(positive_target_atom) and seed.target == positive_target_atom:
            if seed.predicate in active_predicates:
                seed_rule_ids.setdefault(seed.atom, set()).add(seed.rule_id)
    frequent_set = set(frequent_atoms)
    usable_seed_atoms = frequent_set & set(seed_rule_ids)
    if not usable_seed_atoms:
        raise ValueError(
            "no real fpMiner unary for the positive target survives the bounded vocabulary"
        )

    total_positive_weight = math.fsum(
        weight for target, weight in zip(normalized_targets, normalized_weights) if target
    )
    total_negative_weight = total_weight - total_positive_weight
    base_rate = total_positive_weight / total_weight
    all_cover = (1 << len(rows)) - 1
    total_positive_count = sum(normalized_targets)
    total_negative_count = len(rows) - total_positive_count
    fold_covers_by_key: dict[tuple[str, str], int] = {}
    for index, fold in enumerate(normalized_folds):
        key = _fold_key(fold)
        fold_covers_by_key[key] = fold_covers_by_key.get(key, 0) | (1 << index)
    fold_totals = {key: mass(cover) for key, cover in fold_covers_by_key.items()}

    def precision(covered: _Mass, default: float) -> float:
        return covered.positive_weight / covered.support if covered.support else default

    def intersect(premises: Sequence[Atom]) -> int:
        result = all_cover
        for atom in premises:
            result &= covers[atom]
        return result

    generated = repeated_rejections = role_rejections = seed_rejections = 0
    support_rejections = insufficient_folds = effect_rejections = 0
    incremental_rejections = unstable_effect = unstable_incremental = 0
    eligible: list[ConditionalPattern] = []
    tolerance = 1e-12

    for depth in range(2, selected_config.max_depth + 1):
        for premises in itertools.combinations(frequent_atoms, depth):
            generated += 1
            if generated > selected_config.max_generated_candidates:
                raise ConditionalMiningLimitError(
                    "conditional search exceeds max_generated_candidates="
                    f"{selected_config.max_generated_candidates}"
                )
            predicates = {atom.predicate for atom in premises}
            if len(predicates) != depth:
                repeated_rejections += 1
                continue
            if not predicates.intersection(semantic) or not predicates.intersection(context):
                role_rejections += 1
                continue
            premise_seed_atoms = usable_seed_atoms.intersection(premises)
            if not premise_seed_atoms:
                seed_rejections += 1
                continue
            cover = intersect(premises)
            covered = mass(cover)
            if covered.support < selected_config.min_support:
                support_rejections += 1
                continue

            immediate_parent_covers = [
                intersect((*premises[:index], *premises[index + 1 :]))
                for index in range(depth)
            ]
            candidate_precision = precision(covered, base_rate)
            parent_precision = max(
                [base_rate]
                + [precision(mass(parent_cover), base_rate)
                   for parent_cover in immediate_parent_covers]
            )
            effect = candidate_precision - base_rate
            incremental_effect = candidate_precision - parent_precision
            coverage = covered.support / total_weight
            wracc = coverage * effect
            incremental_wracc = coverage * incremental_effect

            fold_statistics: list[ConditionalFoldStatistics] = []
            for fold in distinct_folds:
                key = _fold_key(fold)
                fold_cover = fold_covers_by_key[key]
                fold_total = fold_totals[key]
                fold_covered = mass(cover & fold_cover)
                fold_support = fold_covered.support
                fold_base_rate = precision(fold_total, base_rate)
                fold_precision = precision(fold_covered, fold_base_rate)
                parent_fold_precision = max(
                    [fold_base_rate]
                    + [precision(mass(parent_cover & fold_cover), fold_base_rate)
                       for parent_cover in immediate_parent_covers]
                )
                fold_effect = fold_precision - fold_base_rate
                fold_incremental = fold_precision - parent_fold_precision
                fold_coverage = (
                    fold_support / fold_total.support if fold_total.support else 0.0
                )
                fold_statistics.append(ConditionalFoldStatistics(
                    fold=fold,
                    weighted=WeightedContingency(
                        tp=fold_covered.positive_weight,
                        fp=fold_covered.negative_weight,
                        fn=fold_total.positive_weight - fold_covered.positive_weight,
                        tn=fold_total.negative_weight - fold_covered.negative_weight,
                    ),
                    counts=CountContingency(
                        tp=fold_covered.positive_count,
                        fp=fold_covered.negative_count,
                        fn=fold_total.positive_count - fold_covered.positive_count,
                        tn=fold_total.negative_count - fold_covered.negative_count,
                    ),
                    support=fold_support,
                    coverage=fold_coverage,
                    base_rate=fold_base_rate,
                    precision=fold_precision,
                    effect=fold_effect,
                    parent_precision=parent_fold_precision,
                    incremental_effect=fold_incremental,
                    wracc=fold_coverage * fold_effect,
                    incremental_wracc=fold_coverage * fold_incremental,
                    usable=fold_support >= selected_config.fold_min_support,
                ))
            usable_folds = [fold for fold in fold_statistics if fold.usable]
            if len(usable_folds) < selected_config.min_usable_folds:
                insufficient_folds += 1
                continue
            if effect <= selected_config.min_effect + tolerance:
                effect_rejections += 1
                continue
            if incremental_effect <= selected_config.min_incremental_effect + tolerance:
                incremental_rejections += 1
                continue
            stable_effect_value = min(fold.effect for fold in usable_folds)
            if stable_effect_value <= selected_config.min_effect + tolerance:
                unstable_effect += 1
                continue
            stable_incremental_value = min(
                fold.incremental_effect for fold in usable_folds
            )
            if stable_incremental_value <= selected_config.min_incremental_effect + tolerance:
                unstable_incremental += 1
                continue
            robust_incremental_wracc = min(
                fold.incremental_wracc for fold in usable_folds
            )

            lineages = tuple(sorted({predicate_lineages[atom.predicate] for atom in premises}))
            semantic_lineages = tuple(sorted({
                predicate_lineages[atom.predicate]
                for atom in premises if atom.predicate in semantic
            }))
            context_lineages = tuple(sorted({
                predicate_lineages[atom.predicate]
                for atom in premises if atom.predicate in context
            }))
            lineage_payload = json.dumps(lineages, separators=(",", ":"), ensure_ascii=False)
            premise_payload = json.dumps(
                [(atom.predicate, *_typed_value_key(atom.value)) for atom in premises],
                separators=(",", ":"), ensure_ascii=False,
            )
            lineage_signature = hashlib.sha256(lineage_payload.encode("utf-8")).hexdigest()[:16]
            variant_signature = hashlib.sha256(
                (lineage_payload + "\n" + premise_payload).encode("utf-8")
            ).hexdigest()[:16]
            matched_indexes = tuple(
                index for index in range(len(rows)) if cover & (1 << index)
            )
            matching_seed_ids = tuple(sorted({
                rule_id for atom in premise_seed_atoms for rule_id in seed_rule_ids[atom]
            }))
            eligible.append(ConditionalPattern(
                premises=premises,
                fpminer_seed_rule_ids=matching_seed_ids,
                evidence_lineages=lineages,
                semantic_lineages=semantic_lineages,
                context_lineages=context_lineages,
                lineage_signature=lineage_signature,
                variant_signature=variant_signature,
                weighted=WeightedContingency(
                    tp=covered.positive_weight,
                    fp=covered.negative_weight,
                    fn=total_positive_weight - covered.positive_weight,
                    tn=total_negative_weight - covered.negative_weight,
                ),
                counts=CountContingency(
                    tp=covered.positive_count,
                    fp=covered.negative_count,
                    fn=total_positive_count - covered.positive_count,
                    tn=total_negative_count - covered.negative_count,
                ),
                support=covered.support,
                coverage=coverage,
                base_rate=base_rate,
                precision=candidate_precision,
                effect=effect,
                parent_precision=parent_precision,
                incremental_effect=incremental_effect,
                wracc=wracc,
                incremental_wracc=incremental_wracc,
                stable_effect=stable_effect_value,
                stable_incremental_effect=stable_incremental_value,
                robust_incremental_wracc=robust_incremental_wracc,
                usable_fold_count=len(usable_folds),
                fold_statistics=tuple(fold_statistics),
                covered_row_indexes=matched_indexes,
            ))

    eligible.sort(key=_pattern_sort_key)
    patterns = tuple(eligible[:selected_config.top_k])
    audit = ConditionalMiningAudit(
        row_count=len(rows),
        fold_count=len(distinct_folds),
        semantic_predicate_count=len(semantic),
        context_predicate_count=len(context),
        candidate_atom_count=len(candidate_atoms),
        frequent_atom_count=len(frequent_atoms),
        support_pruned_atoms=len(candidate_atoms) - len(frequent_atoms),
        input_fpminer_unaries=len(seeds),
        usable_fpminer_seed_atoms=len(usable_seed_atoms),
        generated_candidates=generated,
        repeated_predicate_rejections=repeated_rejections,
        role_rejections=role_rejections,
        seed_rejections=seed_rejections,
        support_rejections=support_rejections,
        insufficient_fold_rejections=insufficient_folds,
        effect_rejections=effect_rejections,
        incremental_effect_rejections=incremental_rejections,
        unstable_effect_rejections=unstable_effect,
        unstable_incremental_rejections=unstable_incremental,
        eligible_patterns=len(eligible),
        output_patterns=len(patterns),
        total_weight=total_weight,
        positive_weight=total_positive_weight,
    )
    return ConditionalMiningResult(
        patterns=patterns,
        frequent_atoms=frequent_atoms,
        config=selected_config,
        audit=audit,
    )


__all__ = [
    "ConditionalFoldStatistics",
    "ConditionalMiningAudit",
    "ConditionalMiningConfig",
    "ConditionalMiningLimitError",
    "ConditionalMiningResult",
    "ConditionalPattern",
    "FpMinerUnary",
    "mine_conditional_llm_patterns",
]
