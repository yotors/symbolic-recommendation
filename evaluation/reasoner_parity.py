"""Matched direct-versus-PeTTa semantic parity for point and pair channels.

This evaluator answers one deliberately narrow question: for the active,
compiler-issued pair-proof topology, can a transparent rule-table evaluator
reproduce antecedent activation, PeTTa root STVs, and the dependency margins
used by the host ranker on the *same* grounded cases?

It is not an accuracy ablation and it cannot attribute an AUC change to the
reasoner.  A successful report establishes semantic parity for the supported
shallow topology only.  The evaluator fails closed when it encounters a rule
shape whose result cannot be reconstructed without general reasoning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any


REPORT_KIND = "pair_reasoner_semantic_parity_v2"
INTERPRETATION = (
    "Independent matched semantic parity only: categorical case facts and all "
    "active PairSignal roots are loaded into a disposable PeTTaChainer worker, "
    "while a separate direct evaluator reconstructs activation and STVs from "
    "the compiler contract. A full-model match requires every compiled proof "
    "channel to be activated by the cohort; otherwise a clean result is a "
    "sampled-channel match. It is not an accuracy causal ablation and does not "
    "measure a unique AUC contribution."
)
SUPPORTED_TOPOLOGY = "isolated_extensional_pair_channel_v1"
SUPPORTED_POINT_TOPOLOGY = "isolated_extensional_point_channel_v1"
_HARD_MAX_CASES = 4096
_HARD_MAX_RULES = 1024
_HARD_MAX_QUERY_ROOTS = 32768
_HARD_MAX_ERROR_DETAILS = 256
_AUDIT_FACT_BATCH = 1000
_AUDIT_QUERY_BATCH = 256
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CTV_AT_END = re.compile(
    rf"\(CTV \(STV ({_NUMBER}) ({_NUMBER})\) "
    rf"\(STV ({_NUMBER}) ({_NUMBER})\)\)\)$"
)
_PROOF_STV = re.compile(rf"\(STV\s+({_NUMBER})\s+({_NUMBER})\)")
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PREDICATE = re.compile(r"[a-z][a-z0-9_]*")


class UnsupportedParityTopology(ValueError):
    """The active model is outside the evaluator's exact safety contract."""


def _unsupported(message: str) -> UnsupportedParityTopology:
    return UnsupportedParityTopology(f"unsupported pair parity topology: {message}")


def _finite_unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _unsupported(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise _unsupported(f"{field} must be finite and in [0, 1]")
    return result


def _positive_integer(value: Any, field: str, hard_limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if value > hard_limit:
        raise ValueError(f"{field} cannot exceed the hard limit {hard_limit}")
    return value


def _ideal_clip(strength: float) -> float:
    return max(0.000001, min(0.999999, strength))


def _confidence_to_count(confidence: float) -> float:
    return confidence * 800.0 / (1.0 - min(confidence, 0.9999))


def _ideal_variance(strength: float, confidence: float) -> float:
    clipped = _ideal_clip(strength)
    return clipped * (1.0 - clipped) / (_confidence_to_count(confidence) + 1.0)


def _confidence_from_variance(strength: float, variance: float) -> float:
    if variance <= 0.0:
        return 0.9999
    clipped = _ideal_clip(strength)
    maximum = clipped * (1.0 - clipped)
    bounded = min(variance, maximum)
    count = maximum / bounded - 1.0
    return max(0.000001, count / (count + 800.0))


def _and_formula(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    """Exact Python transcription of PeTTaChainer's current AndFormula."""

    left_strength, left_confidence = left
    right_strength, right_confidence = right
    strength = left_strength * right_strength
    if left_confidence <= 0.0 or right_confidence <= 0.0:
        return strength, 0.0
    left_variance = _ideal_variance(left_strength, left_confidence)
    right_variance = _ideal_variance(right_strength, right_confidence)
    variance = (
        left_variance * right_variance
        + left_variance * right_strength * right_strength
        + left_strength * left_strength * right_variance
    )
    return strength, _confidence_from_variance(strength, variance)


def _ctv_modus_ponens(
    antecedent: tuple[float, float],
    positive: tuple[float, float],
    negative: tuple[float, float],
) -> tuple[float, float]:
    """Exact Python transcription of CTVModusPonensFormula for STV input."""

    antecedent_strength, antecedent_confidence = antecedent
    positive_strength, positive_confidence = positive
    negative_strength, negative_confidence = negative
    strength = (
        positive_strength * antecedent_strength
        + negative_strength * (1.0 - antecedent_strength)
    )
    positive_variance = _ideal_variance(positive_strength, positive_confidence)
    negative_variance = _ideal_variance(negative_strength, negative_confidence)
    antecedent_variance = _ideal_variance(
        antecedent_strength, antecedent_confidence
    )
    complement = 1.0 - antecedent_strength
    branch_difference = positive_strength - negative_strength
    variance = (
        antecedent_strength * antecedent_strength * positive_variance
        + complement * complement * negative_variance
        + branch_difference * branch_difference * antecedent_variance
        + antecedent_variance * (positive_variance + negative_variance)
    )
    return strength, _confidence_from_variance(strength, variance)


def _margin(
    truth_value: tuple[float, float], transform: str, power: float
) -> float:
    strength, confidence = truth_value
    posterior = 0.5 + confidence * (strength - 0.5)
    if transform == "log_odds":
        bounded = max(1e-9, min(1.0 - 1e-9, posterior))
        value = math.log(bounded / (1.0 - bounded))
    else:
        value = 2.0 * posterior - 1.0
    if value:
        value = math.copysign(abs(value) ** power, value)
    return value


def _proof_truth_value(proof: str) -> tuple[float, float]:
    matches = _PROOF_STV.findall(proof)
    if not matches:
        raise ValueError("PeTTa proof contains no STV")
    truth_value = tuple(map(float, matches[-1]))
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in truth_value):
        raise ValueError("PeTTa proof root STV is not finite in [0, 1]")
    return truth_value  # type: ignore[return-value]


def _quoted_atoms(proof: str) -> set[str]:
    atoms: set[str] = set()
    for token in re.findall(r'"(?:[^"\\]|\\.)*"', proof):
        try:
            decoded = json.loads(token)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, str):
            atoms.add(decoded)
    return atoms


def _source_ctv(source: str, prefix: str) -> tuple[tuple[float, float], tuple[float, float]]:
    match = _CTV_AT_END.search(source)
    if match is None or source[: match.start()] != prefix:
        raise _unsupported("compiled mined-rule source does not match its declared premises")
    values = tuple(float(value) for value in match.groups())
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise _unsupported("compiled CTV fields must be finite and in [0, 1]")
    return (values[0], values[1]), (values[2], values[3])


def _compiled_channels(lab: Any, max_rules: int) -> tuple[list[dict[str, Any]], str]:
    config = getattr(lab, "config", None)
    if not isinstance(config, Mapping):
        raise _unsupported("lab config is unavailable")
    if config.get("pair_aggregation") != "proof_margin":
        raise _unsupported("only isolated proof_margin channels are reconstructable")
    transform = config.get("pair_margin_transform")
    if transform not in {"linear", "log_odds"}:
        raise _unsupported("pair_margin_transform must be linear or log_odds")
    power = config.get("pair_margin_power")
    if isinstance(power, bool) or not isinstance(power, (int, float)):
        raise _unsupported("pair_margin_power must be numeric")
    if not math.isfinite(float(power)) or float(power) <= 0.0:
        raise _unsupported("pair_margin_power must be finite and positive")

    rules = getattr(lab, "pair_rules", None)
    sources = getattr(lab, "_pair_rule_sources", None)
    if not isinstance(rules, list) or not rules:
        raise _unsupported("the active model has no pair rules")
    if len(rules) > max_rules:
        raise _unsupported(f"active rule count {len(rules)} exceeds bound {max_rules}")
    if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
        raise _unsupported("compiled pair-rule sources are unavailable")
    if len(sources) != len(set(sources)):
        raise _unsupported("compiled pair-rule sources contain duplicates")

    source_hashes = {
        hashlib.sha256(source.encode("utf-8")).hexdigest(): source
        for source in sources
    }
    channels: list[dict[str, Any]] = []
    roots: set[tuple[str, str]] = set()
    variants: set[str] = set()
    consumed: set[str] = set()
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            raise _unsupported("pair rule is not a mapping")
        dependency = raw_rule.get("dependency_id")
        variant = raw_rule.get("variant_id")
        channel = raw_rule.get("proof_channel_id")
        if not all(isinstance(value, str) and _SYMBOL.fullmatch(value)
                   for value in (dependency, variant, channel)):
            raise _unsupported("dependency, variant, and channel IDs must be safe symbols")
        if channel == dependency or channel != variant:
            raise _unsupported("each proof channel must be an isolated variant")
        root = dependency, channel
        if root in roots or variant in variants:
            raise _unsupported(f"proof channel {root!r} has multiple producers")
        roots.add(root)
        variants.add(variant)

        raw_premises = raw_rule.get("premises")
        if not isinstance(raw_premises, (tuple, list)) or not raw_premises:
            raise _unsupported(f"channel {channel!r} has no categorical premises")
        premises: list[tuple[str, str]] = []
        seen_predicates: set[str] = set()
        for item in raw_premises:
            if (not isinstance(item, (tuple, list)) or len(item) != 2
                    or not isinstance(item[0], str)
                    or not _PREDICATE.fullmatch(item[0])
                    or not isinstance(item[1], str)):
                raise _unsupported(f"channel {channel!r} has a non-categorical premise")
            predicate, value = item
            if predicate in seen_predicates:
                raise _unsupported(f"channel {channel!r} repeats predicate {predicate!r}")
            seen_predicates.add(predicate)
            premises.append((predicate, value))

        contract = raw_rule.get("proof_factorization")
        if (not isinstance(contract, Mapping)
                or contract.get("schema") != SUPPORTED_TOPOLOGY
                or contract.get("case_variable") != "$pair"
                or contract.get("premise_tv") != [1.0, 1.0]
                or contract.get("single_channel_producer") is not True
                or contract.get("dependency_id") != dependency
                or contract.get("proof_channel_id") != channel
                or contract.get("premises") != [list(item) for item in premises]):
            raise _unsupported(f"channel {channel!r} lacks the exact factorization contract")
        source_digest = contract.get("rule_source_sha256")
        if not isinstance(source_digest, str) or source_digest not in source_hashes:
            raise _unsupported(f"channel {channel!r} source hash is not active")
        variant_source = source_hashes[source_digest]

        terms = [
            f"({predicate.title()} $pair {json.dumps(value)})"
            for predicate, value in premises
        ]
        premise_source = terms[0] if len(terms) == 1 else f"(And {' '.join(terms)})"
        prefix = (
            f"(: {variant} (Implication {premise_source} "
            f"(MinedPairPreference $pair {json.dumps(dependency)} "
            f"{json.dumps(channel)})) "
        )
        positive, negative = _source_ctv(variant_source, prefix)
        expected_positive = (
            _finite_unit(raw_rule.get("proof_strength", raw_rule.get("strength")),
                         f"{channel}.positive_strength"),
            _finite_unit(raw_rule.get("proof_confidence", raw_rule.get("confidence")),
                         f"{channel}.positive_confidence"),
        )
        if "proof_strength" in raw_rule:
            expected_negative = (0.5, 0.0)
        else:
            expected_negative = (
                _finite_unit(raw_rule.get("negative_strength"),
                             f"{channel}.negative_strength"),
                _finite_unit(raw_rule.get("negative_confidence"),
                             f"{channel}.negative_confidence"),
            )
        if positive != expected_positive or negative != expected_negative:
            raise _unsupported(f"channel {channel!r} metadata disagrees with compiled CTV")

        quoted_dependency = re.escape(json.dumps(dependency))
        quoted_channel = re.escape(json.dumps(channel))
        decision_pattern = re.compile(
            rf"^\(: pair_decision_rule_\d+ \(Implication "
            rf"\(MinedPairPreference \$pair {quoted_dependency} {quoted_channel}\) "
            rf"\(PairSignal \$pair {quoted_dependency} {quoted_channel}\)\) "
            rf"\(CTV \(STV 1\.0 1\.0\) \(STV 0\.0 1\.0\)\)\)$"
        )
        bridge_pattern = re.compile(
            rf"^\(: \(no_inverse pair_merge_rule_\d+\) \(Implication "
            rf"\(PairSignal \$pair {quoted_dependency} {quoted_channel}\) "
            rf"\(PairWin \$pair\)\) "
            rf"\(CTV \(STV 1\.0 1\.0\) \(STV 0\.0 1\.0\)\)\)$"
        )
        decisions = [source for source in sources if decision_pattern.fullmatch(source)]
        bridges = [source for source in sources if bridge_pattern.fullmatch(source)]
        if len(decisions) != 1 or len(bridges) != 1:
            raise _unsupported(f"channel {channel!r} lacks unique identity adapters")
        consumed.update((variant_source, decisions[0], bridges[0]))

        antecedent = (1.0, 1.0)
        for _ in premises[1:]:
            antecedent = _and_formula(antecedent, (1.0, 1.0))
        mined_truth_value = _ctv_modus_ponens(antecedent, positive, negative)
        root_truth_value = _ctv_modus_ponens(
            mined_truth_value, (1.0, 1.0), (0.0, 1.0)
        )
        channels.append({
            "dependency_id": dependency,
            "proof_channel_id": channel,
            "premises": tuple(premises),
            "positive_ctv": positive,
            "negative_ctv": negative,
            "expected_root_stv": root_truth_value,
            "source_sha256": source_digest,
        })

    if consumed != set(sources):
        raise _unsupported("compiled source set contains an unrecognized rule topology")
    channels.sort(key=lambda item: (item["dependency_id"], item["proof_channel_id"]))
    model_digest = hashlib.sha256(
        "\n".join(sorted(sources)).encode("utf-8")
    ).hexdigest()
    return channels, model_digest


def _compiled_point_channels(
    lab: Any, max_rules: int
) -> tuple[list[dict[str, Any]], str]:
    """Validate the compiler-issued isolated point proof channels."""

    rules = getattr(lab, "mined_rules", None)
    sources = getattr(lab, "_point_channel_sources", None)
    if not isinstance(rules, list) or not rules:
        raise _unsupported("the active model has no point rules")
    if len(rules) > max_rules:
        raise _unsupported(
            f"active point-rule count {len(rules)} exceeds bound {max_rules}"
        )
    if (not isinstance(sources, list) or len(sources) != 2 * len(rules)
            or not all(isinstance(source, str) for source in sources)):
        raise _unsupported("compiled isolated point-channel sources are unavailable")
    if len(sources) != len(set(sources)):
        raise _unsupported("compiled point-rule sources contain duplicates")

    source_set = set(sources)
    consumed: set[str] = set()
    statement_ids: set[str] = set()
    channels_seen: set[str] = set()
    channels: list[dict[str, Any]] = []
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            raise _unsupported("point rule is not a mapping")
        identifier = raw_rule.get("id")
        variant = raw_rule.get("point_variant_id")
        decision = raw_rule.get("point_decision_id")
        channel = raw_rule.get("point_proof_channel_id")
        if (not all(isinstance(value, str) and _SYMBOL.fullmatch(value)
                    for value in (identifier, variant, decision, channel))
                or len({identifier, variant, decision}) != 3
                or any(value in statement_ids for value in (
                    identifier, variant, decision
                ))
                or channel in channels_seen):
            raise _unsupported(
                "point rule, variant, decision, and channel IDs must be "
                "unique safe symbols"
            )
        statement_ids.update((identifier, variant, decision))
        channels_seen.add(channel)
        target = raw_rule.get("target")
        if target != "click":
            raise _unsupported("only the compiled click point target is supported")
        raw_premises = raw_rule.get("premises")
        if not isinstance(raw_premises, (tuple, list)) or not raw_premises:
            raise _unsupported(f"point rule {identifier!r} has no premises")
        premises: list[tuple[str, str]] = []
        predicates: set[str] = set()
        for item in raw_premises:
            if (not isinstance(item, (tuple, list)) or len(item) != 2
                    or not isinstance(item[0], str)
                    or not _PREDICATE.fullmatch(item[0])
                    or not isinstance(item[1], str)):
                raise _unsupported(
                    f"point rule {identifier!r} has a non-categorical premise"
                )
            predicate, value = item
            if predicate in predicates:
                raise _unsupported(
                    f"point rule {identifier!r} repeats predicate {predicate!r}"
                )
            predicates.add(predicate)
            premises.append((predicate, value))
        positive = (
            _finite_unit(raw_rule.get("strength"), f"{identifier}.strength"),
            _finite_unit(
                raw_rule.get("confidence"), f"{identifier}.confidence"
            ),
        )
        negative = (
            _finite_unit(
                raw_rule.get("negative_strength"),
                f"{identifier}.negative_strength",
            ),
            _finite_unit(
                raw_rule.get("negative_confidence"),
                f"{identifier}.negative_confidence",
            ),
        )
        terms = [
            f"({predicate.title()} $case {json.dumps(value)})"
            for predicate, value in premises
        ]
        premise_source = (
            terms[0] if len(terms) == 1 else f"(And {' '.join(terms)})"
        )
        mined_signal = (
            f'(MinedPointPreference $case {json.dumps(identifier)} '
            f'{json.dumps(channel)})'
        )
        point_signal = (
            f'(PointSignal $case {json.dumps(identifier)} '
            f'{json.dumps(channel)})'
        )
        expected_variant_source = (
            f'(: {variant} (Implication {premise_source} '
            f'{mined_signal}) '
            f'(CTV (STV {positive[0]} {positive[1]}) '
            f'(STV {negative[0]} {negative[1]})))'
        )
        expected_decision_source = (
            f'(: {decision} (Implication {mined_signal} {point_signal}) '
            '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
        )
        contract = raw_rule.get("point_proof_factorization")
        if (not isinstance(contract, Mapping)
                or contract.get("schema") != SUPPORTED_POINT_TOPOLOGY
                or contract.get("case_variable") != "$case"
                or contract.get("premise_tv") != [1.0, 1.0]
                or contract.get("single_channel_producer") is not True
                or contract.get("rule_id") != identifier
                or contract.get("variant_id") != variant
                or contract.get("decision_id") != decision
                or contract.get("proof_channel_id") != channel
                or contract.get("premises") != [list(item) for item in premises]
                or contract.get("rule_source_sha256") != hashlib.sha256(
                    expected_variant_source.encode("utf-8")
                ).hexdigest()
                or contract.get("decision_source_sha256") != hashlib.sha256(
                    expected_decision_source.encode("utf-8")
                ).hexdigest()):
            raise _unsupported(
                f"point rule {identifier!r} lacks its exact factorization contract"
            )
        if (expected_variant_source not in source_set
                or expected_decision_source not in source_set):
            raise _unsupported(
                f"point rule {identifier!r} metadata disagrees with compiled channel sources"
            )
        consumed.update((expected_variant_source, expected_decision_source))
        antecedent = (1.0, 1.0)
        for _premise in premises[1:]:
            antecedent = _and_formula(antecedent, (1.0, 1.0))
        mined_truth_value = _ctv_modus_ponens(
            antecedent, positive, negative
        )
        channels.append({
            "rule_id": identifier,
            "variant_id": variant,
            "decision_id": decision,
            "proof_channel_id": channel,
            "premises": tuple(premises),
            "root_stv": _ctv_modus_ponens(
                mined_truth_value, (1.0, 1.0), (0.0, 1.0)
            ),
            "source_sha256": contract["rule_source_sha256"],
            "decision_source_sha256": contract["decision_source_sha256"],
        })
    if consumed != source_set:
        raise _unsupported("compiled point-channel source set contains unknown rules")
    channels.sort(key=lambda item: item["rule_id"])
    digest = hashlib.sha256(
        "\n".join(sorted(sources)).encode("utf-8")
    ).hexdigest()
    return channels, digest


class DirectPointReconstructor:
    """Lazy transparent evaluator for compiler-issued categorical point rules."""

    def __init__(self, lab: Any, *, max_rules: int = 256):
        max_rules = _positive_integer(max_rules, "max_rules", _HARD_MAX_RULES)
        channels, digest = _compiled_point_channels(lab, max_rules)
        self.channels = tuple(channels)
        self.model_sha256 = digest
        self.supported_topology = SUPPORTED_POINT_TOPOLOGY

    def reconstruct(self, attrs: Mapping[str, str]) -> list[dict[str, Any]]:
        if not isinstance(attrs, Mapping):
            raise ValueError("candidate attributes must be a mapping")
        return [
            {
                "rule_id": channel["rule_id"],
                "variant_id": channel["variant_id"],
                "decision_id": channel["decision_id"],
                "proof_channel_id": channel["proof_channel_id"],
                "premises": [list(item) for item in channel["premises"]],
                "root_stv": list(channel["root_stv"]),
                "source_sha256": channel["source_sha256"],
                "decision_source_sha256": channel["decision_source_sha256"],
            }
            for channel in self.channels
            if all(
                attrs.get(predicate) == value
                for predicate, value in channel["premises"]
            )
        ]

    def proof_rows(self, case: str, attrs: Mapping[str, str]) -> list[str]:
        if not isinstance(case, str) or not _SYMBOL.fullmatch(case):
            raise ValueError("direct reconstruction case must be a safe symbol")
        return [
            f'(direct-reconstruction (PointSignal {case} '
            f'{json.dumps(row["rule_id"])} '
            f'{json.dumps(row["proof_channel_id"])}) '
            f'(by {row["variant_id"]}) (by {row["decision_id"]}) '
            f'(STV {row["root_stv"][0]} {row["root_stv"][1]}))'
            for row in self.reconstruct(attrs)
        ]


class DirectPairReconstructor:
    """Compile the supported pair-rule table once and evaluate cases lazily.

    This object is intentionally independent of PeTTa proof retrieval and of
    the serving path's activation cache.  Besides powering the bounded parity
    audit, it powers the opt-in matched execution ablation that replaces PeTTa
    proofs with direct channel rows while keeping candidates and fusion fixed.
    """

    def __init__(self, lab: Any, *, max_rules: int = 256):
        max_rules = _positive_integer(max_rules, "max_rules", _HARD_MAX_RULES)
        channels, model_digest = _compiled_channels(lab, max_rules)
        config = getattr(lab, "config", {})
        self.channels = tuple(channels)
        self.model_sha256 = model_digest
        self.supported_topology = SUPPORTED_TOPOLOGY
        self.margin_transform = str(config["pair_margin_transform"])
        self.margin_power = float(config["pair_margin_power"])
        self.dependencies = tuple(sorted({
            str(channel["dependency_id"]) for channel in channels
        }))

    def reconstruct(self, attrs: Mapping[str, str]) -> dict[str, Any]:
        """Return active proof-channel and dependency rows for one fact map."""

        if not isinstance(attrs, Mapping):
            raise ValueError("pair attributes must be a mapping")
        channel_rows: list[dict[str, Any]] = []
        for channel in self.channels:
            if not all(
                attrs.get(predicate) == value
                for predicate, value in channel["premises"]
            ):
                continue
            truth_value = channel["expected_root_stv"]
            channel_rows.append({
                "dependency_id": channel["dependency_id"],
                "proof_channel_id": channel["proof_channel_id"],
                "premises": [list(item) for item in channel["premises"]],
                "root_stv": list(truth_value),
                "margin": _margin(
                    truth_value, self.margin_transform, self.margin_power
                ),
                "source_sha256": channel["source_sha256"],
            })
        dependencies: dict[str, dict[str, Any]] = {}
        for dependency in self.dependencies:
            rows = [
                row for row in channel_rows
                if row["dependency_id"] == dependency
            ]
            if rows:
                dependencies[dependency] = max(
                    rows, key=lambda row: abs(float(row["margin"]))
                )
        return {
            "active_channels": channel_rows,
            "selected_dependencies": dependencies,
        }

    def proof_rows(self, case: str, attrs: Mapping[str, str]) -> list[str]:
        """Lazily encode direct channel rows for the existing fusion contract.

        These are diagnostic records, not PeTTa proofs.  They carry the same
        root STV and dependency/channel identity so the matched execution
        ablation can reuse the production tournament/fusion code unchanged.
        """

        if not isinstance(case, str) or not _SYMBOL.fullmatch(case):
            raise ValueError("direct reconstruction case must be a safe symbol")
        result = self.reconstruct(attrs)
        proofs = []
        for row in result["active_channels"]:
            strength, confidence = row["root_stv"]
            proofs.append(
                f'(direct-reconstruction (PairSignal {case} '
                f'{json.dumps(row["dependency_id"])} '
                f'{json.dumps(row["proof_channel_id"])}) '
                f'(STV {strength} {confidence}))'
            )
        return proofs


def _normalize_specs(
    specs: Sequence[tuple[str, Mapping[str, str]]], max_cases: int
) -> list[tuple[str, dict[str, str]]]:
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise ValueError("specs must be a sequence of (case, attributes) pairs")
    normalized: list[tuple[str, dict[str, str]]] = []
    seen: dict[str, dict[str, str]] = {}
    for item in specs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("each pair spec must contain a case and attributes")
        case, raw_attrs = item
        if not isinstance(case, str) or not _SYMBOL.fullmatch(case):
            raise ValueError("pair case IDs must be safe symbols")
        if not isinstance(raw_attrs, Mapping):
            raise ValueError(f"attributes for {case!r} must be a mapping")
        attrs: dict[str, str] = {}
        for predicate, value in raw_attrs.items():
            if (not isinstance(predicate, str) or not _PREDICATE.fullmatch(predicate)
                    or not isinstance(value, str)):
                raise ValueError(
                    f"attributes for {case!r} must be categorical string facts"
                )
            attrs[predicate] = value
        previous = seen.get(case)
        if previous is not None:
            if previous != attrs:
                raise ValueError(f"duplicate case {case!r} has conflicting attributes")
            continue
        seen[case] = attrs
        normalized.append((case, attrs))
        if len(normalized) > max_cases:
            raise ValueError(f"unique pair case count exceeds bound {max_cases}")
    if not normalized:
        raise ValueError("at least one pair spec is required")
    return normalized


def _create_isolated_audit_engine(*, default_timeout_seconds: float) -> Any:
    """Construct the disposable runtime lazily to avoid a module import cycle."""

    from ..app.server import IsolatedPeTTaChainer

    return IsolatedPeTTaChainer(
        default_timeout_seconds=default_timeout_seconds,
    )


def _audit_case_id(
    ordinal: int,
    case: str,
    attrs: Mapping[str, str],
    model_digest: str,
) -> str:
    """Return a safe, audit-only identity unrelated to the live proof cache."""

    payload = json.dumps(
        {
            "ordinal": ordinal,
            "source_case": case,
            "attributes": dict(sorted(attrs.items())),
            "model_sha256": model_digest,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.blake2s(payload.encode("utf-8"), digest_size=16).hexdigest()
    return f"parity_audit_{ordinal}_{digest}"


def _isolated_pair_signal_proofs(
    lab: Any,
    channels: Sequence[Mapping[str, Any]],
    specs: Sequence[tuple[str, Mapping[str, str]]],
    model_digest: str,
) -> tuple[dict[tuple[str, str], list[str]], dict[str, Any], dict[str, str]]:
    """Query every case/channel root in a fresh PeTTaChainer process.

    The host's optimized proof path deliberately is not called here.  All
    categorical facts relevant to an active rule source are grounded on fresh
    identities, and PeTTa itself decides which explicit ``PairSignal`` roots
    have proofs.  The worker is always closed and is never installed on the
    supplied lab, so neither live facts nor proof caches can be changed.
    """

    roots = len(channels) * len(specs)
    if roots > _HARD_MAX_QUERY_ROOTS:
        raise _unsupported(
            "audit pair case x channel root count "
            f"{roots} exceeds bound {_HARD_MAX_QUERY_ROOTS}"
        )
    config = getattr(lab, "config", {})
    if not isinstance(config, Mapping):
        raise _unsupported("lab config is unavailable")
    raw_timeout = config.get(
        "benchmark_reasoner_timeout_seconds",
        config.get("serving_reasoner_timeout_seconds", 30.0),
    )
    if (isinstance(raw_timeout, bool)
            or not isinstance(raw_timeout, (int, float))
            or not math.isfinite(float(raw_timeout))
            or float(raw_timeout) <= 0.0):
        raise _unsupported("reasoner audit timeout must be finite and positive")
    timeout = float(raw_timeout)
    raw_steps = config.get("pair_chain_steps", 12)
    if (isinstance(raw_steps, bool) or not isinstance(raw_steps, int)
            or raw_steps <= 0):
        raise _unsupported("pair_chain_steps must be a positive integer")
    raw_batch = config.get("query_batch_size", _AUDIT_QUERY_BATCH)
    if (isinstance(raw_batch, bool) or not isinstance(raw_batch, int)
            or raw_batch <= 0):
        raise _unsupported("query_batch_size must be a positive integer")
    query_batch = min(raw_batch, _AUDIT_QUERY_BATCH)

    relevant_predicates = sorted({
        predicate
        for channel in channels
        for predicate, _value in channel["premises"]
    })
    audit_ids: dict[str, str] = {}
    audit_id_to_source: dict[str, str] = {}
    facts: list[str] = []
    for ordinal, (case, attrs) in enumerate(specs, 1):
        audit_id = _audit_case_id(ordinal, case, attrs, model_digest)
        if audit_id in audit_id_to_source:
            raise RuntimeError(f"audit case identity collision at {audit_id}")
        audit_ids[case] = audit_id
        audit_id_to_source[audit_id] = case
        for predicate in relevant_predicates:
            if predicate not in attrs:
                continue
            value = attrs[predicate]
            facts.append(
                f'(: fact_{audit_id}_{predicate} '
                f'({predicate.title()} {audit_id} {json.dumps(value)}) '
                '(STV 1.0 1.0))'
            )

    requests: list[tuple[str, str, str, str]] = []
    for case, _attrs in specs:
        audit_id = audit_ids[case]
        for channel in channels:
            dependency = str(channel["dependency_id"])
            channel_id = str(channel["proof_channel_id"])
            query = (
                f'(: $proof (PairSignal {audit_id} '
                f'{json.dumps(dependency)} {json.dumps(channel_id)}) $tv)'
            )
            requests.append((case, dependency, channel_id, query))

    sources = getattr(lab, "_pair_rule_sources", None)
    if not isinstance(sources, list) or not all(
        isinstance(source, str) for source in sources
    ):
        raise _unsupported("compiled pair-rule sources are unavailable")
    proofs: dict[tuple[str, str], list[str]] = {}
    query_calls = 0
    engine = _create_isolated_audit_engine(default_timeout_seconds=timeout)
    try:
        engine.replace(list(sources))
        for offset in range(0, len(facts), _AUDIT_FACT_BATCH):
            engine.add_atoms_no_check(
                facts[offset:offset + _AUDIT_FACT_BATCH],
                timeout_sec=timeout,
            )
        for offset in range(0, len(requests), query_batch):
            batch = requests[offset:offset + query_batch]
            results = engine.query_many(
                [request[3] for request in batch],
                steps=max(10, raw_steps) * len(batch),
                timeout_sec=timeout,
            )
            query_calls += 1
            if (not isinstance(results, list) or len(results) != len(batch)):
                raise RuntimeError(
                    "isolated PeTTa parity query returned a malformed batch"
                )
            for request, result in zip(batch, results):
                case, _dependency, channel_id, _query = request
                if (not isinstance(result, list)
                        or not all(isinstance(item, str) for item in result)):
                    raise RuntimeError(
                        "isolated PeTTa parity query returned malformed proofs "
                        f"for {case!r}/{channel_id!r}"
                    )
                proofs[(case, channel_id)] = result
    finally:
        engine.close()

    stats = {
        "worker_kind": "disposable_isolated_pettachainer",
        "live_worker_mutated": False,
        "live_proof_cache_mutated": False,
        "host_optimized_proof_path_calls": 0,
        "grounded_cases": len(specs),
        "grounded_relevant_facts": len(facts),
        "queried_pair_signal_roots": len(requests),
        "compiled_roots_queried_per_case": len(channels),
        "query_batch_size": query_batch,
        "pettachainer_query_calls": query_calls,
    }
    return proofs, stats, audit_ids


def _difference(
    expected: tuple[float, float], actual: tuple[float, float]
) -> tuple[float, float, float]:
    strength = abs(expected[0] - actual[0])
    confidence = abs(expected[1] - actual[1])
    return strength, confidence, max(strength, confidence)


def evaluate_pair_reasoner_parity(
    lab: Any,
    specs: Sequence[tuple[str, Mapping[str, str]]],
    *,
    max_cases: int = 128,
    max_rules: int = 256,
    absolute_tolerance: float = 1e-8,
    max_error_details: int = 64,
) -> dict[str, Any]:
    """Compare lazy direct reconstruction with an isolated PeTTa execution.

    The function accepts an already-built lab only as an immutable source of
    compiler-issued rules/configuration.  It neither mines nor changes model
    configuration, candidate facts, the live worker, or any live proof cache.
    """

    max_cases = _positive_integer(max_cases, "max_cases", _HARD_MAX_CASES)
    max_rules = _positive_integer(max_rules, "max_rules", _HARD_MAX_RULES)
    max_error_details = _positive_integer(
        max_error_details, "max_error_details", _HARD_MAX_ERROR_DETAILS
    )
    if (isinstance(absolute_tolerance, bool)
            or not isinstance(absolute_tolerance, (int, float))
            or not math.isfinite(float(absolute_tolerance))
            or not 0.0 <= float(absolute_tolerance) <= 1e-3):
        raise ValueError("absolute_tolerance must be finite in [0, 1e-3]")
    tolerance = float(absolute_tolerance)
    normalized = _normalize_specs(specs, max_cases)
    reconstructor = DirectPairReconstructor(lab, max_rules=max_rules)
    channels = list(reconstructor.channels)
    model_digest = reconstructor.model_sha256
    transform = reconstructor.margin_transform
    power = reconstructor.margin_power
    relevant_predicates = {
        predicate
        for channel in channels
        for predicate, _value in channel["premises"]
    }
    direct_by_case = {
        case: reconstructor.reconstruct(attrs) for case, attrs in normalized
    }
    proof_map, audit_stats, audit_ids = _isolated_pair_signal_proofs(
        lab, channels, normalized, model_digest
    )
    channel_by_id = {item["proof_channel_id"]: item for item in channels}
    all_channel_ids = set(channel_by_id)
    dependency_order = list(reconstructor.dependencies)

    details: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total_error_count = 0

    def add_error(payload: dict[str, Any]) -> None:
        nonlocal total_error_count
        total_error_count += 1
        if len(errors) < max_error_details:
            errors.append(payload)

    expected_activation_count = 0
    exercised_channel_ids: set[str] = set()
    proved_expected_count = 0
    unexpected_activation_count = 0
    exact_case_matches = 0
    stv_comparisons = 0
    stv_matches = 0
    strength_errors: list[float] = []
    confidence_errors: list[float] = []
    margin_comparisons = 0
    margin_matches = 0
    margin_errors: list[float] = []

    observed_activation_count = 0
    for case, attrs in normalized:
        expected_channel_rows = direct_by_case[case]["active_channels"]
        expected_ids = {
            row["proof_channel_id"] for row in expected_channel_rows
        }
        exercised_channel_ids.update(expected_ids)
        expected_activation_count += len(expected_ids)
        actual_by_channel: dict[str, list[dict[str, Any]]] = {}
        unattributed: list[str] = []
        returned_ids: set[str] = set()
        for channel in channels:
            dependency = channel["dependency_id"]
            channel_id = channel["proof_channel_id"]
            raw_proofs = proof_map.get((case, channel_id))
            if not isinstance(raw_proofs, list) or not all(
                isinstance(item, str) for item in raw_proofs
            ):
                raise RuntimeError(
                    f"isolated PeTTa proof list for {case!r}/{channel_id!r} "
                    "is malformed"
                )
            if raw_proofs:
                returned_ids.add(channel_id)
            for proof in raw_proofs:
                digest = hashlib.sha256(proof.encode("utf-8")).hexdigest()
                quoted = _quoted_atoms(proof)
                if dependency not in quoted or channel_id not in quoted:
                    unattributed.append(digest)
                    add_error({
                        "kind": "queried_root_identity_mismatch",
                        "case": case,
                        "proof_channel_id": channel_id,
                        "proof_sha256": digest,
                        "quoted_atoms": sorted(quoted),
                    })
                    continue
                try:
                    truth_value = _proof_truth_value(proof)
                except ValueError as exc:
                    add_error({
                        "kind": "invalid_proof_stv", "case": case,
                        "proof_channel_id": channel_id,
                        "message": str(exc),
                    })
                    continue
                actual_by_channel.setdefault(channel_id, []).append({
                    "stv": truth_value,
                    "margin": _margin(truth_value, transform, power),
                    "proof_sha256": digest,
                })
        actual_ids = returned_ids
        observed_activation_count += len(actual_ids)
        proved_expected_count += len(expected_ids.intersection(actual_ids))
        unexpected = sorted(actual_ids.difference(expected_ids))
        missing = sorted(expected_ids.difference(actual_ids))
        unexpected_activation_count += len(unexpected)
        topology_match = not missing and not unexpected and not unattributed
        if topology_match:
            exact_case_matches += 1
        for channel_id in missing:
            add_error({
                "kind": "missing_expected_proof", "case": case,
                "proof_channel_id": channel_id,
            })
        for channel_id in unexpected:
            add_error({
                "kind": "unexpected_proof", "case": case,
                "proof_channel_id": channel_id,
            })

        actual_channel_rows: list[dict[str, Any]] = []
        actual_selected: dict[str, dict[str, Any]] = {}
        for channel_id, candidates in actual_by_channel.items():
            actual = max(candidates, key=lambda item: abs(item["margin"]))
            actual_selected[channel_id] = actual
            channel = channel_by_id[channel_id]
            actual_channel_rows.append({
                "dependency_id": channel["dependency_id"],
                "proof_channel_id": channel_id,
                "root_stv": list(actual["stv"]),
                "margin": actual["margin"],
                "proof_count": len(candidates),
                "proof_sha256": actual["proof_sha256"],
            })
        for expected_row in expected_channel_rows:
            channel_id = expected_row["proof_channel_id"]
            expected_tv = tuple(expected_row["root_stv"])
            actual = actual_selected.get(channel_id)
            if actual is None:
                continue
            strength_error, confidence_error, maximum_error = _difference(
                expected_tv, actual["stv"]
            )
            stv_comparisons += 1
            strength_errors.append(strength_error)
            confidence_errors.append(confidence_error)
            if maximum_error <= tolerance:
                stv_matches += 1
            else:
                add_error({
                    "kind": "root_stv_mismatch", "case": case,
                    "proof_channel_id": channel_id,
                    "expected": list(expected_tv), "actual": list(actual["stv"]),
                    "max_absolute_error": maximum_error,
                })

        expected_dependencies: dict[str, dict[str, Any]] = {}
        actual_dependencies: dict[str, dict[str, Any]] = {}
        for dependency in dependency_order:
            direct = [row for row in expected_channel_rows
                      if row["dependency_id"] == dependency]
            if direct:
                expected_dependencies[dependency] = max(
                    direct, key=lambda row: abs(row["margin"])
                )
            inferred = [row for row in actual_channel_rows
                        if row["dependency_id"] == dependency]
            if inferred:
                actual_dependencies[dependency] = max(
                    inferred, key=lambda row: abs(row["margin"])
                )
        dependency_rows: list[dict[str, Any]] = []
        for dependency in sorted(set(expected_dependencies) | set(actual_dependencies)):
            direct = expected_dependencies.get(dependency)
            inferred = actual_dependencies.get(dependency)
            row: dict[str, Any] = {
                "dependency_id": dependency,
                "expected": None if direct is None else {
                    "selected_channel": direct["proof_channel_id"],
                    "root_stv": direct["root_stv"],
                    "margin": direct["margin"],
                },
                "actual": None if inferred is None else {
                    "selected_channel": inferred["proof_channel_id"],
                    "root_stv": inferred["root_stv"],
                    "margin": inferred["margin"],
                },
            }
            if direct is not None and inferred is not None:
                error = abs(float(direct["margin"]) - float(inferred["margin"]))
                row["margin_absolute_error"] = error
                margin_comparisons += 1
                margin_errors.append(error)
                if error <= tolerance:
                    margin_matches += 1
                else:
                    add_error({
                        "kind": "dependency_margin_mismatch", "case": case,
                        "dependency_id": dependency,
                        "expected": direct["margin"], "actual": inferred["margin"],
                        "absolute_error": error,
                    })
            dependency_rows.append(row)

        grounded = sorted(
            [predicate, attrs[predicate]]
            for predicate in relevant_predicates if predicate in attrs
        )
        details.append({
            "case": case,
            "isolated_audit_case": audit_ids[case],
            "grounded_relevant_facts": grounded,
            "topology_match": topology_match,
            "expected_active_channels": expected_channel_rows,
            "actual_proof_channels": sorted(
                actual_channel_rows, key=lambda row: row["proof_channel_id"]
            ),
            "missing_channels": missing,
            "unexpected_channels": unexpected,
            "unattributed_proof_sha256": unattributed,
            "dependencies": dependency_rows,
        })

    cases = len(normalized)
    if expected_activation_count == 0 and observed_activation_count == 0:
        raise ValueError(
            "parity cohort is not evaluable because neither the direct "
            "reconstructor nor isolated PeTTa activated a pair-rule channel"
        )
    coverage = (
        proved_expected_count / expected_activation_count
        if expected_activation_count else None
    )
    sampled_semantic_parity = (
        total_error_count == 0
        and exact_case_matches == cases
        and stv_matches == stv_comparisons == expected_activation_count
        and margin_matches == margin_comparisons
    )
    compiled_channel_count = len(channels)
    exercised_channel_count = len(exercised_channel_ids)
    channel_coverage = exercised_channel_count / compiled_channel_count
    full_model_semantic_parity = (
        sampled_semantic_parity
        and exercised_channel_count == compiled_channel_count
    )
    parity_scope = (
        "full_model" if exercised_channel_count == compiled_channel_count
        else "sampled_channels"
    )
    parity_status = (
        "full_model_match" if full_model_semantic_parity
        else "sampled_match" if sampled_semantic_parity
        else "mismatch"
    )
    return {
        "report_kind": REPORT_KIND,
        "interpretation": INTERPRETATION,
        # ``semantic_parity`` is deliberately strict: it is true only after
        # every compiled proof channel has been exercised successfully.  The
        # sampled result remains separately available without overstating its
        # model coverage.
        "semantic_parity": full_model_semantic_parity,
        "full_model_semantic_parity": full_model_semantic_parity,
        "sampled_semantic_parity": sampled_semantic_parity,
        "parity_scope": parity_scope,
        "status": parity_status,
        "supported_topology": SUPPORTED_TOPOLOGY,
        "model_sha256": model_digest,
        "configuration": {
            "pair_aggregation": "proof_margin",
            "pair_margin_transform": transform,
            "pair_margin_power": power,
            "absolute_tolerance": tolerance,
            "max_cases": max_cases,
            "max_rules": max_rules,
            "direct_evaluator": "DirectPairReconstructor",
            "pettachainer_evaluator": (
                "disposable isolated worker querying every PairSignal root"
            ),
        },
        "summary": {
            "cases": cases,
            "active_rules": len(channels),
            "compiled_channels": compiled_channel_count,
            "exercised_channels": exercised_channel_count,
            "unexercised_channels": sorted(
                all_channel_ids.difference(exercised_channel_ids)
            ),
            "compiled_channel_coverage": channel_coverage,
            **audit_stats,
            "exact_case_matches": exact_case_matches,
            "case_match_rate": exact_case_matches / cases,
            "expected_channel_activations": expected_activation_count,
            "isolated_pettachainer_channel_activations": (
                observed_activation_count
            ),
            "proved_expected_channels": proved_expected_count,
            "expected_activation_coverage": coverage,
            "unexpected_channel_activations": unexpected_activation_count,
            "root_stv_comparisons": stv_comparisons,
            "root_stv_matches": stv_matches,
            "root_stv_match_rate": (
                stv_matches / stv_comparisons if stv_comparisons else None
            ),
            "max_strength_absolute_error": max(strength_errors, default=0.0),
            "max_confidence_absolute_error": max(confidence_errors, default=0.0),
            "dependency_margin_comparisons": margin_comparisons,
            "dependency_margin_matches": margin_matches,
            "dependency_margin_match_rate": (
                margin_matches / margin_comparisons if margin_comparisons else None
            ),
            "max_dependency_margin_absolute_error": max(margin_errors, default=0.0),
            "error_count": total_error_count,
            "error_details_returned": len(errors),
        },
        "errors": errors,
        "cases": details,
    }


__all__ = [
    "DirectPairReconstructor",
    "DirectPointReconstructor",
    "INTERPRETATION",
    "REPORT_KIND",
    "SUPPORTED_POINT_TOPOLOGY",
    "SUPPORTED_TOPOLOGY",
    "UnsupportedParityTopology",
    "evaluate_pair_reasoner_parity",
]
