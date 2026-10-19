"""Deserialize complete-population rule calibration computed by MeTTa."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from .petta_workspace import PeTTaExecutor
from .rule_parser import STV_RE, balanced_forms


_SAFE_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_COUNTS_RE = re.compile(
    rf"\(Counts\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\)"
)
_METRICS_RE = re.compile(
    rf"\(Metrics\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})\)"
)
_ID_RE = re.compile(r"^\(PointCalibration\s+([A-Za-z_][A-Za-z0-9_]*)\b")


@dataclass(frozen=True, slots=True)
class PointCalibration:
    rule_id: str
    strength: float
    confidence: float
    negative_strength: float
    negative_confidence: float
    joint_support: int
    antecedent_support: int
    population_size: int
    target_support: int
    base_rate: float
    lift: float
    specificity: int
    quality: float

    def as_rule_update(self) -> dict[str, int | float]:
        return {
            "strength": self.strength,
            "confidence": self.confidence,
            "negative_strength": self.negative_strength,
            "negative_confidence": self.negative_confidence,
            "support": self.joint_support,
            "joint_support": self.joint_support,
            "antecedent_support": self.antecedent_support,
            "lift": self.lift,
            "specificity": self.specificity,
            "quality": self.quality,
        }

    def for_rule(self, rule_id: str) -> "PointCalibration":
        if not _SAFE_SYMBOL.fullmatch(rule_id):
            raise ValueError("rule IDs must be safe MeTTa symbols")
        return replace(self, rule_id=rule_id)


def _integer(value: str, label: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"MeTTa calibration returned a non-integer {label}")
    return int(number)


def parse_point_calibrations(
    raw: Iterable[object], *, expected_ids: Iterable[str] = (),
) -> dict[str, PointCalibration]:
    """Parse and validate MeTTa's data-only ``PointCalibration`` forms."""

    expected = frozenset(expected_ids)
    parsed: dict[str, PointCalibration] = {}
    for form in balanced_forms(" ".join(map(str, raw)), head="PointCalibration"):
        identity = _ID_RE.match(form)
        counts = _COUNTS_RE.search(form)
        metrics = _METRICS_RE.search(form)
        tvs = STV_RE.findall(form)
        if identity is None or counts is None or metrics is None or len(tvs) != 2:
            raise ValueError("malformed PointCalibration returned by MeTTa")
        rule_id = identity.group(1)
        if rule_id in parsed:
            raise ValueError(f"duplicate MeTTa calibration for {rule_id}")
        strength, confidence = map(float, tvs[0])
        negative_strength, negative_confidence = map(float, tvs[1])
        joint, antecedent, population, target = (
            _integer(value, label)
            for value, label in zip(
                counts.groups(), ("joint support", "antecedent support", "population", "target support")
            )
        )
        base_rate, lift, specificity_value, quality = map(float, metrics.groups())
        specificity = _integer(str(specificity_value), "specificity")
        numbers = (
            strength, confidence, negative_strength, negative_confidence,
            base_rate, lift, quality,
        )
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("MeTTa calibration returned a non-finite value")
        outside = population - antecedent
        outside_target = target - joint
        if not (
            0 <= joint <= antecedent <= population
            and joint <= target <= population
            and 0 <= outside_target <= outside
            and specificity >= 1
            and quality >= 0.0
        ):
            raise ValueError("MeTTa calibration returned an inconsistent contingency")
        parsed[rule_id] = PointCalibration(
            rule_id, strength, confidence, negative_strength,
            negative_confidence, joint, antecedent, population, target,
            base_rate, lift, specificity, quality,
        )
    if expected and set(parsed) != set(expected):
        missing = sorted(expected - parsed.keys())
        unexpected = sorted(parsed.keys() - expected)
        raise ValueError(
            "MeTTa calibration result does not match requested rules: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return parsed


def _validate_request(space: str, evidence_k: float, batch_size: int) -> None:
    if not space.startswith("&") or not _SAFE_SYMBOL.fullmatch(space[1:]):
        raise ValueError("space must be a safe named MeTTa space")
    if not math.isfinite(evidence_k) or evidence_k <= 0:
        raise ValueError("evidence_k must be finite and positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _rule_pattern(rule: Mapping[str, object], target: str) -> tuple[str, str]:
    rule_id = str(rule["id"])
    if not _SAFE_SYMBOL.fullmatch(rule_id):
        raise ValueError("rule IDs must be safe MeTTa symbols")
    clauses = []
    for predicate, value in tuple(rule["premises"]):
        if not _SAFE_SYMBOL.fullmatch(str(predicate)):
            raise ValueError("premise predicates must be safe MeTTa symbols")
        clauses.append(
            f"({predicate} $case {json.dumps(str(value), ensure_ascii=False)})"
        )
    if not clauses:
        raise ValueError("a calibration rule must contain premises")
    clauses.append(f'(engagement $case {json.dumps(target, ensure_ascii=False)})')
    return rule_id, f"(, {' '.join(clauses)})"


def _run_queries(
    petta: PeTTaExecutor, calls: Sequence[str], expected_ids: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, PointCalibration], int]:
    raw: list[object] = []
    for start in range(0, len(calls), batch_size):
        batch = calls[start:start + batch_size]
        result = petta.process_metta_string(
            f"!(collapse (superpose ({' '.join(batch)})))"
        )
        if result is None:
            continue
        if isinstance(result, (str, bytes)):
            raw.append(result)
        else:
            try:
                raw.extend(result)
            except TypeError:
                raw.append(result)
    query_count = math.ceil(len(calls) / batch_size)
    return parse_point_calibrations(raw, expected_ids=expected_ids), query_count


def calibrate_point_rules(
    petta: PeTTaExecutor,
    *,
    space: str,
    rules: Sequence[Mapping[str, object]],
    evidence_k: float,
    target: str = "click",
    batch_size: int = 128,
) -> tuple[dict[str, PointCalibration], int]:
    """Ask MeTTa to calibrate rule structures against ``space`` exactly."""

    _validate_request(space, evidence_k, batch_size)
    calls: list[str] = []
    expected_ids: list[str] = []
    for rule in rules:
        rule_id, pattern = _rule_pattern(rule, target)
        calls.append(
            f"(recommendation-calibrate-rule {rule_id} {pattern} {space} "
            f"{json.dumps(target, ensure_ascii=False)} {evidence_k:.17g})"
        )
        expected_ids.append(rule_id)
    return _run_queries(petta, calls, expected_ids, batch_size)


def update_point_rules(
    petta: PeTTaExecutor,
    *,
    delta_space: str,
    rules: Sequence[Mapping[str, object]],
    previous: Mapping[str, PointCalibration],
    evidence_k: float,
    target: str = "click",
    batch_size: int = 128,
) -> tuple[dict[str, PointCalibration], int]:
    """Have MeTTa add an append-only delta to prior sufficient statistics."""

    expected = {str(rule["id"]) for rule in rules}
    if set(previous) != expected:
        raise ValueError("previous calibrations must match the updated rules")
    _validate_request(delta_space, evidence_k, batch_size)
    calls: list[str] = []
    expected_ids: list[str] = []
    for rule in rules:
        rule_id, pattern = _rule_pattern(rule, target)
        prior = previous[rule_id]
        calls.append(
            f"(recommendation-update-calibration {rule_id} {pattern} "
            f"{delta_space} {json.dumps(target, ensure_ascii=False)} "
            f"{evidence_k:.17g} {prior.joint_support} "
            f"{prior.antecedent_support} {prior.population_size} "
            f"{prior.target_support})"
        )
        expected_ids.append(rule_id)
    return _run_queries(petta, calls, expected_ids, batch_size)


__all__ = [
    "PointCalibration", "calibrate_point_rules", "parse_point_calibrations",
    "update_point_rules",
]
