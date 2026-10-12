"""Thin host boundary for the MeTTa conditional champion miner."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .conditional_model import (
    Atom,
    ConditionalFoldStatistics,
    ConditionalMiningConfig,
    ConditionalMiningLimitError,
    ConditionalMiningResult,
    ConditionalPattern,
    CountContingency,
    FpMinerUnary,
    WeightedContingency,
)


class _PeTTaRuntime(Protocol):
    def process_metta_string(self, source: str) -> Any: ...


_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_SPACE = re.compile(r"^&[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN = re.compile(r'\s*(\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+)')


@dataclass(frozen=True)
class PeTTaConditionalAudit:
    row_count: int
    fold_count: int
    input_fpminer_unaries: int
    usable_fpminer_seed_atoms: int
    context_atom_count: int
    generated_candidates: int
    eligible_patterns: int
    output_patterns: int
    total_weight: float
    positive_weight: float
    executor: str = "recommendation/miner/conditionalMiner.metta via PeTTa"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _atom_source(value: object) -> str:
    if value is None or isinstance(value, (str, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise TypeError("MeTTa atom values must be JSON scalars")


def _mass_table(
    petta: _PeTTaRuntime,
    space: str,
    target: object,
    cells: Iterable[tuple[str, tuple[str, ...], object | None]],
) -> str:
    """Batch PeTTa mass queries and label their returned sufficient statistics."""

    pending = tuple(cells)
    rows: list[str] = []
    for offset in range(0, len(pending), 128):
        batch = pending[offset:offset + 128]
        queries = []
        for _key, clauses, fold in batch:
            scoped = clauses + (() if fold is None else (
                f'(case-fold "$x" {_atom_source(fold)})',
            ))
            queries.append(
                "!(conditional-case-mass "
                f"{space} ({' '.join(scoped)}) {_atom_source(target)})"
            )
        raw = petta.process_metta_string(" ".join(queries))
        values = list(raw if isinstance(raw, (list, tuple)) else [raw])
        if len(values) != len(batch):
            raise ValueError("PeTTa returned an incomplete conditional mass table")
        for (key, _clauses, fold), value in zip(batch, values):
            text = str(value)
            parsed = _forms(text)
            if len(parsed) != 1 or _find(parsed, "Mass") is None:
                raise ValueError("PeTTa returned an invalid conditional mass")
            scope = "AllFolds" if fold is None else _atom_source(fold)
            rows.append(f"(Cell {key} {scope} {text})")
    return f"({' '.join(rows)})"


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    offset = 0
    while offset < len(text):
        match = _TOKEN.match(text, offset)
        if match is None:
            if not text[offset:].strip():
                break
            raise ValueError(f"invalid MeTTa result near offset {offset}")
        tokens.append(match.group(1))
        offset = match.end()
    return tokens


def _forms(text: str) -> list[Any]:
    tokens = _tokens(text)
    position = 0

    def parse() -> Any:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("unexpected end of MeTTa result")
        token = tokens[position]
        position += 1
        if token == "(":
            values = []
            while position < len(tokens) and tokens[position] != ")":
                values.append(parse())
            if position >= len(tokens):
                raise ValueError("unclosed MeTTa result")
            position += 1
            return values
        if token == ")":
            raise ValueError("unexpected closing parenthesis")
        if token.startswith('"'):
            return json.loads(token)
        if token in {"true", "True"}:
            return True
        if token in {"false", "False"}:
            return False
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    parsed = []
    while position < len(tokens):
        parsed.append(parse())
    return parsed


def _find(node: Any, head: str) -> list[Any] | None:
    if isinstance(node, list):
        if node and node[0] == head:
            return node
        for child in node:
            found = _find(child, head)
            if found is not None:
                return found
    return None


def _child(node: list[Any], head: str) -> list[Any]:
    for value in node[1:]:
        if isinstance(value, list) and value and value[0] == head:
            return value
    raise ValueError(f"MeTTa result is missing {head}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _decode_stats(node: list[Any]) -> dict[str, Any]:
    total = _child(node, "Total")
    matched = _child(node, "Matched")
    rates = _child(node, "Rates")
    if len(total) != 5 or len(matched) != 5 or len(rates) != 10:
        raise ValueError("invalid MeTTa conditional statistics")
    total_positive, total_negative = map(
        lambda item: _number(item, "total weight"), total[1:3]
    )
    total_positive_count = _integer(total[3], "total positive count")
    total_negative_count = _integer(total[4], "total negative count")
    positive, negative = map(
        lambda item: _number(item, "matched weight"), matched[1:3]
    )
    positive_count = _integer(matched[3], "matched positive count")
    negative_count = _integer(matched[4], "matched negative count")
    rate_values = [_number(item, "conditional rate") for item in rates[1:]]
    return {
        "weighted": WeightedContingency(
            positive, negative,
            total_positive - positive, total_negative - negative,
        ),
        "counts": CountContingency(
            positive_count, negative_count,
            total_positive_count - positive_count,
            total_negative_count - negative_count,
        ),
        "support": rate_values[0],
        "coverage": rate_values[1],
        "base_rate": rate_values[2],
        "precision": rate_values[3],
        "effect": rate_values[4],
        "parent_precision": rate_values[5],
        "incremental_effect": rate_values[6],
        "wracc": rate_values[7],
        "incremental_wracc": rate_values[8],
    }


def _decode_pattern(node: list[Any]) -> ConditionalPattern:
    premises_node = _child(node, "Premises")
    premises = tuple(sorted((
        Atom(str(clause[0]), clause[2])
        for clause in premises_node[1:]
        if isinstance(clause, list) and len(clause) == 3
    ), key=lambda atom: atom.sort_key))
    if len(premises) != 2:
        raise ValueError("MeTTa conditional rule must contain two premises")
    seed_value = _child(node, "Seed")[1]
    seed_ids = tuple(str(item) for item in (
        seed_value if isinstance(seed_value, list) else [seed_value]
    ))
    lineage_node = _child(node, "Lineages")
    semantic_lineages = (str(lineage_node[1]),)
    context_lineages = (str(lineage_node[2]),)
    lineages = tuple(sorted(set((*semantic_lineages, *context_lineages))))
    global_stats = _decode_stats(_child(node, "Stats"))
    stable = _child(node, "Stable")
    folds_container = _child(node, "Folds")
    fold_nodes = folds_container[1] if len(folds_container) == 2 else []
    fold_statistics = []
    for fold_node in fold_nodes:
        if not isinstance(fold_node, list) or len(fold_node) != 4:
            raise ValueError("invalid MeTTa fold statistic")
        fold_stats = _decode_stats(fold_node[2])
        fold_statistics.append(ConditionalFoldStatistics(
            fold=fold_node[1],
            weighted=fold_stats["weighted"],
            counts=fold_stats["counts"],
            support=fold_stats["support"],
            coverage=fold_stats["coverage"],
            base_rate=fold_stats["base_rate"],
            precision=fold_stats["precision"],
            effect=fold_stats["effect"],
            parent_precision=fold_stats["parent_precision"],
            incremental_effect=fold_stats["incremental_effect"],
            wracc=fold_stats["wracc"],
            incremental_wracc=fold_stats["incremental_wracc"],
            usable=fold_node[3] is True,
        ))
    lineage_payload = json.dumps(lineages, separators=(",", ":"), ensure_ascii=False)
    premise_payload = json.dumps(
        [(atom.predicate, *atom.sort_key[1]) for atom in premises],
        separators=(",", ":"), ensure_ascii=False,
    )
    return ConditionalPattern(
        premises=premises,
        fpminer_seed_rule_ids=seed_ids,
        evidence_lineages=lineages,
        semantic_lineages=semantic_lineages,
        context_lineages=context_lineages,
        lineage_signature=hashlib.sha256(
            lineage_payload.encode("utf-8")
        ).hexdigest()[:16],
        variant_signature=hashlib.sha256(
            (lineage_payload + "\n" + premise_payload).encode("utf-8")
        ).hexdigest()[:16],
        **global_stats,
        stable_effect=_number(stable[1], "stable effect"),
        stable_incremental_effect=_number(stable[2], "stable incremental effect"),
        robust_incremental_wracc=_number(stable[3], "robust incremental WRAcc"),
        usable_fold_count=_integer(stable[4], "usable fold count"),
        fold_statistics=tuple(fold_statistics),
    )


def mine_petta_conditional_patterns(
    petta: _PeTTaRuntime,
    space: str,
    *,
    row_count: int,
    fpminer_unaries: Iterable[FpMinerUnary],
    positive_target: object,
    context_predicates: Iterable[str],
    predicate_lineages: Mapping[str, str],
    folds: Iterable[object],
    config: ConditionalMiningConfig,
) -> ConditionalMiningResult:
    """Execute and decode the two-premise conditional miner in PeTTa."""

    if not _SPACE.fullmatch(space):
        raise ValueError("space must be a safe bound PeTTa space symbol")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 1:
        raise ValueError("row_count must be a positive integer")
    if row_count > config.max_rows:
        raise ConditionalMiningLimitError(
            f"row_count exceeds max_rows={config.max_rows}"
        )
    if config.max_depth != 2:
        raise ValueError("the champion PeTTa conditional miner requires max_depth=2")
    positive = Atom("target", positive_target).value
    grouped: dict[Atom, set[str]] = {}
    seen_rule_ids: set[str] = set()
    for seed in fpminer_unaries:
        if not isinstance(seed, FpMinerUnary):
            raise TypeError("fpminer_unaries must contain FpMinerUnary values")
        if seed.rule_id in seen_rule_ids:
            raise ValueError(f"duplicate fpMiner rule_id: {seed.rule_id}")
        seen_rule_ids.add(seed.rule_id)
        if type(seed.target) is not type(positive) or seed.target != positive:
            continue
        grouped.setdefault(seed.atom, set()).add(seed.rule_id)
    if not grouped:
        raise ValueError("at least one fpMiner unary for the target is required")
    seed_forms = []
    seed_records = []
    for index, atom in enumerate(sorted(grouped, key=lambda item: item.sort_key)):
        if not _SYMBOL.fullmatch(atom.predicate):
            raise ValueError(f"unsafe seed predicate: {atom.predicate!r}")
        lineage = predicate_lineages.get(atom.predicate)
        if not isinstance(lineage, str) or not lineage:
            raise ValueError(f"missing lineage for {atom.predicate!r}")
        identifiers = " ".join(
            _atom_source(value) for value in sorted(grouped[atom])
        )
        seed_key = f"seed_{index}"
        seed_forms.append(
            f'(Seed {seed_key} ({identifiers}) '
            f'({atom.predicate} "$x" {_atom_source(atom.value)}) '
            f'{_atom_source(lineage)})'
        )
        seed_records.append((seed_key, atom))
    context_forms = []
    context_names = sorted(set(context_predicates))
    if not context_names:
        raise ValueError("at least one context predicate is required")
    for predicate in context_names:
        if not _SYMBOL.fullmatch(predicate):
            raise ValueError(f"unsafe context predicate: {predicate!r}")
        lineage = predicate_lineages.get(predicate)
        if not isinstance(lineage, str) or not lineage:
            raise ValueError(f"missing lineage for {predicate!r}")
        context_forms.append(
            f'(Context {predicate} {_atom_source(lineage)})'
        )
    fold_values = sorted(
        {Atom("fold", value).value for value in folds},
        key=lambda value: Atom("fold", value).sort_key,
    )
    if len(fold_values) < config.min_usable_folds:
        raise ValueError("insufficient distinct folds")

    raw_atoms = petta.process_metta_string(f"!(collapse (absSpeCan {space} 1))")
    atom_text = " ".join(map(
        str, raw_atoms if isinstance(raw_atoms, (list, tuple)) else [raw_atoms]
    ))
    atom_forms = _forms(atom_text)
    candidates = atom_forms[0] if len(atom_forms) == 1 else []
    context_atoms = tuple(sorted({
        Atom(str(item[0]), item[2])
        for item in candidates
        if isinstance(item, list) and len(item) == 3
        and item[0] in context_names
    }, key=lambda item: item.sort_key))
    value_counts = {
        predicate: sum(atom.predicate == predicate for atom in context_atoms)
        for predicate in context_names
    }
    excessive = {
        predicate: count for predicate, count in value_counts.items()
        if count > config.max_values_per_predicate
    }
    if excessive:
        raise ConditionalMiningLimitError(
            f"context vocabulary exceeds max_values_per_predicate: {excessive}"
        )
    generated_candidates = len(seed_records) * len(context_atoms)
    if generated_candidates > config.max_generated_candidates:
        raise ConditionalMiningLimitError(
            f"PeTTa generated {generated_candidates} candidates; limit is "
            f"{config.max_generated_candidates}"
        )
    scopes = (None, *fold_values)
    population_table = _mass_table(
        petta, space, positive, (
            ("Population", (), fold) for fold in scopes
        ),
    )
    semantic_table = _mass_table(
        petta, space, positive, (
            (
                f"(Semantic {seed_key})",
                (f'({atom.predicate} "$x" {_atom_source(atom.value)})',),
                fold,
            )
            for seed_key, atom in seed_records for fold in scopes
        ),
    )
    context_table = _mass_table(
        petta, space, positive, (
            (
                f"(Context {atom.predicate} {_atom_source(atom.value)})",
                (f'({atom.predicate} "$x" {_atom_source(atom.value)})',),
                fold,
            )
            for atom in context_atoms for fold in scopes
        ),
    )
    candidate_table = _mass_table(
        petta, space, positive, (
            (
                f"(Candidate {seed_key} {context.predicate} "
                f"{_atom_source(context.value)})",
                (
                    f'({semantic.predicate} "$x" '
                    f'{_atom_source(semantic.value)})',
                    f'({context.predicate} "$x" '
                    f'{_atom_source(context.value)})',
                ),
                fold,
            )
            for seed_key, semantic in seed_records
            for context in context_atoms
            for fold in scopes
        ),
    )
    source = (
        f'!(conditional-frequency-pattern-miner '
        f"{population_table} {semantic_table} {context_table} {candidate_table} "
        f"({' '.join(seed_forms)}) ({' '.join(context_forms)}) "
        f"({' '.join(_atom_source(value) for value in fold_values)}) "
        f"{config.min_support:.17g} "
        f"{config.fold_min_support:.17g} {config.min_usable_folds} "
        f"{config.min_effect:.17g} {config.min_incremental_effect:.17g} "
        f"{config.top_k} {config.max_generated_candidates})"
    )
    raw = petta.process_metta_string(source)
    text = " ".join(map(str, raw if isinstance(raw, (list, tuple)) else [raw]))
    parsed = _forms(text)
    limit = _find(parsed, "MiningLimit")
    if limit is not None:
        raise ConditionalMiningLimitError(
            f"PeTTa generated {limit[1]} candidates; limit is {limit[2]}"
        )
    result = _find(parsed, "ConditionalMining")
    if result is None:
        raise ValueError("PeTTa returned no conditional mining result")
    audit_node = _child(result, "Audit")
    if len(audit_node) != 6:
        raise ValueError("invalid PeTTa conditional audit")
    mass = audit_node[5]
    if not isinstance(mass, list) or len(mass) != 5 or mass[0] != "Mass":
        raise ValueError("invalid PeTTa population mass")
    rule_container = result[2] if len(result) == 3 else []
    rule_nodes = [
        item for item in rule_container
        if isinstance(item, list) and item and item[0] == "ConditionalRule"
    ]
    patterns = tuple(_decode_pattern(node) for node in rule_nodes)
    total_weight = _number(mass[1], "positive weight") + _number(
        mass[2], "negative weight"
    )
    audit = PeTTaConditionalAudit(
        row_count=row_count,
        fold_count=len(fold_values),
        input_fpminer_unaries=sum(len(values) for values in grouped.values()),
        usable_fpminer_seed_atoms=len(grouped),
        context_atom_count=_integer(audit_node[2], "context atom count"),
        generated_candidates=_integer(audit_node[1], "generated candidates"),
        eligible_patterns=_integer(audit_node[3], "eligible patterns"),
        output_patterns=_integer(audit_node[4], "output patterns"),
        total_weight=total_weight,
        positive_weight=_number(mass[1], "positive weight"),
    )
    return ConditionalMiningResult(
        patterns=patterns,
        config=config,
        audit=audit,
    )


__all__ = ["PeTTaConditionalAudit", "mine_petta_conditional_patterns"]
