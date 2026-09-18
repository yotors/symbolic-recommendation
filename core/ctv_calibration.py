"""Impression-weighted calibration for symbolic recommendation rules.

The module is deliberately independent from :mod:`recommendation.app.server`. It
turns closed, pre-outcome rule observations into a two-branch CTV while keeping
three quantities separate:

* conditional strength -- the target rate when the rule does/does not match;
* epistemic confidence -- derived from distinct-impression effective support;
* applicability -- how much of the scored population had sufficient evidence.

Every impression has total mass one, irrespective of its number of candidates
or oriented pairs.  Optional per-row weights are normalized *within* the
impression, making the API suitable for point cases, pair cases, or future
propensity-weighted observations without allowing a large slate to dominate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Hashable, Iterable


DEFAULT_EVIDENCE_K = 800.0


@dataclass(frozen=True, slots=True)
class CTVObservation:
    """One closed rule observation from a point or pair impression.

    ``matched`` says whether the antecedent is true, while ``target`` is the
    closed outcome (for example click or left-wins).  An inapplicable row has
    missing/non-comparable premise evidence: it contributes to the coverage
    denominator but is excluded from both logical CTV branches.

    ``weight`` is a non-negative within-impression relative weight.  Absolute
    scales do not matter because weights are normalized per impression.
    """

    impression_id: Hashable
    matched: bool
    target: bool
    applicable: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.impression_id is None:
            raise ValueError("impression_id cannot be None")
        try:
            hash(self.impression_id)
        except TypeError as exc:
            raise ValueError("impression_id must be hashable") from exc
        for name in ("matched", "target", "applicable"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if isinstance(self.weight, bool) or not isinstance(self.weight, Real):
            raise ValueError("weight must be a finite non-negative number")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be a finite non-negative number")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class BranchCTV:
    """One CTV branch and its independent evidence audit."""

    strength: float
    confidence: float
    weighted_support: float
    weighted_target_support: float
    effective_impressions: float
    distinct_impressions: int
    raw_support: int
    raw_target_support: int
    defined: bool

    @property
    def stv(self) -> tuple[float, float]:
        return self.strength, self.confidence


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    """Population coverage, deliberately not folded into CTV confidence."""

    weighted_mass: float
    weighted_fraction: float
    raw_count: int
    raw_fraction: float
    distinct_impressions: int


@dataclass(frozen=True, slots=True)
class RawContingency:
    """Unweighted observation counts retained for reproducibility/audit."""

    matched_target: int
    matched_non_target: int
    unmatched_target: int
    unmatched_non_target: int
    inapplicable_target: int
    inapplicable_non_target: int

    @property
    def observations(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True, slots=True)
class WeightedContingency:
    """The same table after every impression has been normalized to mass one."""

    matched_target: float
    matched_non_target: float
    unmatched_target: float
    unmatched_non_target: float
    inapplicable_target: float
    inapplicable_non_target: float

    @property
    def mass(self) -> float:
        return math.fsum(asdict(self).values())


@dataclass(frozen=True, slots=True)
class CTVCalibration:
    """Complete result for one antecedent/target calibration."""

    rule_kind: str
    evidence_k: float
    impressions: int
    observations: int
    target_base_rate: float
    applicable_target_base_rate: float
    positive: BranchCTV
    negative: BranchCTV
    applicability: CoverageAudit
    activation: CoverageAudit
    activation_given_applicable: float
    raw: RawContingency
    weighted: WeightedContingency

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe nested audit record."""

        return asdict(self)

    def metta_ctv(self) -> str:
        """Render only the two branch STVs in PeTTa-compatible CTV syntax."""

        return (
            f"(CTV (STV {self.positive.strength:.12g} "
            f"{self.positive.confidence:.12g}) "
            f"(STV {self.negative.strength:.12g} "
            f"{self.negative.confidence:.12g}))"
        )


def _effective_impressions(masses: Iterable[float]) -> float:
    positive = [float(value) for value in masses if value > 0.0]
    total = math.fsum(positive)
    squared = math.fsum(value * value for value in positive)
    return total * total / squared if squared else 0.0


def _confidence(effective_impressions: float, evidence_k: float) -> float:
    if effective_impressions <= 0.0:
        return 0.0
    return effective_impressions / (effective_impressions + evidence_k)


def reencode_ctv_confidence(
    calibration: CTVCalibration,
    *,
    evidence_k: float = DEFAULT_EVIDENCE_K,
) -> CTVCalibration:
    """Re-encode branch evidence using another confidence convention.

    Strengths, support, coverage, and effective-impression counts are left
    unchanged.  Only ``confidence = n_eff / (n_eff + K)`` and the recorded
    ``evidence_k`` change.  This lets a configurable host-side selection
    reliability use one ``K`` while every CTV crossing the PeTTa boundary is
    encoded with PeTTa's fixed evidence convention.
    """

    if not isinstance(calibration, CTVCalibration):
        raise ValueError("calibration must be a CTVCalibration")
    if isinstance(evidence_k, bool) or not isinstance(evidence_k, Real):
        raise ValueError("evidence_k must be a finite positive number")
    evidence_k = float(evidence_k)
    if not math.isfinite(evidence_k) or evidence_k <= 0.0:
        raise ValueError("evidence_k must be a finite positive number")
    return replace(
        calibration,
        evidence_k=evidence_k,
        positive=replace(
            calibration.positive,
            confidence=_confidence(
                calibration.positive.effective_impressions, evidence_k
            ),
        ),
        negative=replace(
            calibration.negative,
            confidence=_confidence(
                calibration.negative.effective_impressions, evidence_k
            ),
        ),
    )


def _coverage(
    *,
    weighted_mass: float,
    total_mass: float,
    raw_count: int,
    total_count: int,
    per_impression_mass: dict[Hashable, float],
) -> CoverageAudit:
    return CoverageAudit(
        weighted_mass=weighted_mass,
        weighted_fraction=weighted_mass / total_mass if total_mass else 0.0,
        raw_count=raw_count,
        raw_fraction=raw_count / total_count if total_count else 0.0,
        distinct_impressions=sum(value > 0.0 for value in per_impression_mass.values()),
    )


def calibrate_ctv(
    observations: Iterable[CTVObservation],
    *,
    evidence_k: float = DEFAULT_EVIDENCE_K,
    rule_kind: str = "generic",
) -> CTVCalibration:
    """Calculate an impression-macro, two-branch CTV.

    The positive branch estimates ``P(target | matched)``; the negative branch
    estimates ``P(target | applicable and not matched)``.  If a branch has no
    support, it receives the applicable population base rate with confidence
    zero and ``defined=False``.

    Branch confidence is ``n_eff / (n_eff + evidence_k)``, where ``n_eff`` is
    Kish effective sample size over the branch's aggregate mass in each
    distinct impression.  Coverage never multiplies this confidence.
    """

    if isinstance(evidence_k, bool) or not isinstance(evidence_k, Real):
        raise ValueError("evidence_k must be a finite positive number")
    evidence_k = float(evidence_k)
    if not math.isfinite(evidence_k) or evidence_k <= 0.0:
        raise ValueError("evidence_k must be a finite positive number")
    if not isinstance(rule_kind, str) or not rule_kind.strip():
        raise ValueError("rule_kind must be a non-empty string")

    rows = list(observations)
    if not rows:
        raise ValueError("at least one observation is required")
    if not all(isinstance(row, CTVObservation) for row in rows):
        raise ValueError("observations must contain CTVObservation values")

    by_impression: dict[Hashable, list[CTVObservation]] = defaultdict(list)
    for row in rows:
        by_impression[row.impression_id].append(row)
    totals = {
        impression_id: math.fsum(row.weight for row in group)
        for impression_id, group in by_impression.items()
    }
    empty = [impression_id for impression_id, total in totals.items() if total <= 0.0]
    if empty:
        raise ValueError("every impression must have positive total weight")

    raw_values = {
        "matched_target": 0,
        "matched_non_target": 0,
        "unmatched_target": 0,
        "unmatched_non_target": 0,
        "inapplicable_target": 0,
        "inapplicable_non_target": 0,
    }
    weighted_values = {key: 0.0 for key in raw_values}
    applicable_by_impression: dict[Hashable, float] = defaultdict(float)
    matched_by_impression: dict[Hashable, float] = defaultdict(float)
    unmatched_by_impression: dict[Hashable, float] = defaultdict(float)

    for row in rows:
        normalized = row.weight / totals[row.impression_id]
        if not row.applicable:
            key = "inapplicable_target" if row.target else "inapplicable_non_target"
        elif row.matched:
            key = "matched_target" if row.target else "matched_non_target"
            applicable_by_impression[row.impression_id] += normalized
            matched_by_impression[row.impression_id] += normalized
        else:
            key = "unmatched_target" if row.target else "unmatched_non_target"
            applicable_by_impression[row.impression_id] += normalized
            unmatched_by_impression[row.impression_id] += normalized
        raw_values[key] += 1
        weighted_values[key] += normalized

    raw = RawContingency(**raw_values)
    weighted = WeightedContingency(**weighted_values)
    impression_count = len(by_impression)
    total_mass = float(impression_count)
    applicable_mass = math.fsum(applicable_by_impression.values())
    matched_mass = math.fsum(matched_by_impression.values())
    unmatched_mass = math.fsum(unmatched_by_impression.values())
    matched_target_mass = weighted.matched_target
    unmatched_target_mass = weighted.unmatched_target
    all_target_mass = math.fsum((
        weighted.matched_target,
        weighted.unmatched_target,
        weighted.inapplicable_target,
    ))
    applicable_target_mass = matched_target_mass + unmatched_target_mass
    target_base_rate = all_target_mass / total_mass if total_mass else 0.0
    applicable_base_rate = (
        applicable_target_mass / applicable_mass
        if applicable_mass else target_base_rate
    )

    matched_effective = _effective_impressions(matched_by_impression.values())
    unmatched_effective = _effective_impressions(unmatched_by_impression.values())
    positive = BranchCTV(
        strength=(matched_target_mass / matched_mass
                  if matched_mass else applicable_base_rate),
        confidence=_confidence(matched_effective, evidence_k),
        weighted_support=matched_mass,
        weighted_target_support=matched_target_mass,
        effective_impressions=matched_effective,
        distinct_impressions=sum(value > 0.0 for value in matched_by_impression.values()),
        raw_support=raw.matched_target + raw.matched_non_target,
        raw_target_support=raw.matched_target,
        defined=matched_mass > 0.0,
    )
    negative = BranchCTV(
        strength=(unmatched_target_mass / unmatched_mass
                  if unmatched_mass else applicable_base_rate),
        confidence=_confidence(unmatched_effective, evidence_k),
        weighted_support=unmatched_mass,
        weighted_target_support=unmatched_target_mass,
        effective_impressions=unmatched_effective,
        distinct_impressions=sum(value > 0.0 for value in unmatched_by_impression.values()),
        raw_support=raw.unmatched_target + raw.unmatched_non_target,
        raw_target_support=raw.unmatched_target,
        defined=unmatched_mass > 0.0,
    )
    applicability = _coverage(
        weighted_mass=applicable_mass,
        total_mass=total_mass,
        raw_count=(raw.matched_target + raw.matched_non_target
                   + raw.unmatched_target + raw.unmatched_non_target),
        total_count=len(rows),
        per_impression_mass=applicable_by_impression,
    )
    activation = _coverage(
        weighted_mass=matched_mass,
        total_mass=total_mass,
        raw_count=raw.matched_target + raw.matched_non_target,
        total_count=len(rows),
        per_impression_mass=matched_by_impression,
    )
    return CTVCalibration(
        rule_kind=rule_kind.strip(),
        evidence_k=evidence_k,
        impressions=impression_count,
        observations=len(rows),
        target_base_rate=target_base_rate,
        applicable_target_base_rate=applicable_base_rate,
        positive=positive,
        negative=negative,
        applicability=applicability,
        activation=activation,
        activation_given_applicable=(matched_mass / applicable_mass
                                     if applicable_mass else 0.0),
        raw=raw,
        weighted=weighted,
    )


__all__ = [
    "DEFAULT_EVIDENCE_K",
    "BranchCTV",
    "CTVCalibration",
    "CTVObservation",
    "CoverageAudit",
    "RawContingency",
    "WeightedContingency",
    "calibrate_ctv",
    "reencode_ctv_confidence",
]
