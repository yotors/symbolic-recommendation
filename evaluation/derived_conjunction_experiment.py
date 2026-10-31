"""Benchmark MDL-selected derived-feature conjunctions on frozen MIND."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import resource
import time

from .connected_miner_experiment import (
    IsolatedPeTTaChainer,
    PeTTa,
    impression_auc,
    midranks,
)
from ..mining.rule_parser import parse_rules, proof_tv


FEATURES = (
    "topic",
    "recent_affinity",
    "long_affinity",
    "entity_overlap",
    "entity_overlap_detail",
    "history_topic_count_bucket",
    "recent_topic_count_bucket",
)
POSITIVE = frozenset({"click", "like", "complete"})
CASE_RE = re.compile(r"\beval_case_[0-9]+_[0-9]+\b")
SCORE_RE = re.compile(
    r"\(DerivedScore\s+(eval_case_[0-9]+_[0-9]+)\s+"
    r"([0-9.eE+-]+)\)"
)


def categorical_facts(case: str, context: dict) -> list[str]:
    return [
        f"({feature} {case} {json.dumps(value)})"
        for feature in FEATURES
        if isinstance((value := context.get(feature)), str)
    ]


def load_atoms(runtime: PeTTa, space: str, facts: list[str]) -> float:
    started = time.perf_counter()
    for offset in range(0, len(facts), 1000):
        additions = " ".join(
            f"(add-atom {space} {fact})"
            for fact in facts[offset:offset + 1000]
        )
        runtime.process_metta_string(f"!(superpose ({additions}))")
    return time.perf_counter() - started


def training_facts(data: dict) -> tuple[list[str], list[str]]:
    complete: list[str] = []
    positive: list[str] = []
    for index, event in enumerate(data["events"]):
        case = f"train_case_{index}"
        local = categorical_facts(case, event)
        outcome = "click" if event.get("action") in POSITIVE else "skip"
        local.append(f"(engagement {case} {json.dumps(outcome)})")
        complete.extend(local)
        if outcome == "click":
            positive.extend(local)
    return complete, positive


def evaluation_facts(data: dict):
    facts: list[str] = []
    cases: list[list[tuple[str, str]]] = []
    case_index: dict[str, tuple[int, str]] = {}
    for impression_index, test in enumerate(data["tests"]):
        local: list[tuple[str, str]] = []
        contexts = test["candidate_context"]
        for candidate_index, article in enumerate(test["candidates"]):
            article = str(article)
            case = f"eval_case_{impression_index}_{candidate_index}"
            facts.extend(categorical_facts(case, contexts[article]))
            local.append((article, case))
            case_index[case] = (impression_index, article)
        cases.append(local)
    return facts, cases, case_index


def mine_rules(
    runtime: PeTTa, complete: list[str], positive: list[str], *,
    min_support: int, max_conjuncts: int, evidence_k: float,
    length_cost: float, max_patterns: int,
):
    runtime.process_metta_string("!(bind! &derived_train (new-space))")
    runtime.process_metta_string("!(bind! &derived_positive (new-space))")
    complete_seconds = load_atoms(runtime, "&derived_train", complete)
    positive_seconds = load_atoms(runtime, "&derived_positive", positive)
    schemas = "(" + " ".join(
        f"({feature} exposure value value)" for feature in FEATURES
    ) + ")"
    query = (
        "!(connected-frequency-pattern-miner-range-prefiltered "
        f"&derived_train &derived_positive {min_support} 2 "
        f"{max_conjuncts + 1} \"click\" {evidence_k} {length_cost} "
        f"{schemas} () {max_patterns})"
    )
    started = time.perf_counter()
    raw = runtime.process_metta_string(query)
    mining_seconds = time.perf_counter() - started
    text = " ".join(map(str, raw))
    if "MiningLimit" in text:
        raise RuntimeError(f"derived conjunction search exceeded its bound: {text}")
    rules = parse_rules(raw, features=FEATURES)
    for index, rule in enumerate(rules, 1):
        rule["id"] = f"derived_{index:04d}"
        rule["specificity"] = len(rule["premises"])
    return rules, {
        "complete_fact_load_seconds": complete_seconds,
        "positive_fact_load_seconds": positive_seconds,
        "mining_seconds": mining_seconds,
    }


def premise_source(rule: dict, case: str = "$case") -> list[str]:
    return [
        f"({predicate} {case} {json.dumps(value)})"
        for predicate, value in rule["premises"]
    ]


def compiled_rule(rule: dict) -> str:
    clauses = premise_source(rule)
    premise = clauses[0] if len(clauses) == 1 else f"(And {' '.join(clauses)})"
    positive = rule["discovery_ctv"]["positive"]
    negative = rule["discovery_ctv"]["negative"]
    return (
        f"(: {rule['id']} (Implication {premise} "
        f"(DerivedSignal $case {json.dumps(rule['id'])})) "
        f"(CTV (STV {positive['strength']} {positive['confidence']}) "
        f"(STV {negative['strength']} {negative['confidence']})))"
    )


def rule_pattern(rule: dict) -> str:
    clauses = premise_source(rule)
    return clauses[0] if len(clauses) == 1 else f"(, {' '.join(clauses)})"


def match_rules(
    runtime: PeTTa, facts: list[str], rules: list[dict],
    case_index: dict[str, tuple[int, str]],
):
    runtime.process_metta_string("!(bind! &derived_eval (new-space))")
    load_seconds = load_atoms(runtime, "&derived_eval", facts)
    started = time.perf_counter()
    matches: dict[str, set[str]] = {}
    for rule in rules:
        raw = runtime.process_metta_string(
            "!(unique-atom (collapse "
            f"(match &derived_eval {rule_pattern(rule)} $case)))"
        )
        found = set(CASE_RE.findall(" ".join(map(str, raw))))
        if not found <= case_index.keys():
            raise RuntimeError(f"{rule['id']}: PeTTa returned unknown cases")
        matches[rule["id"]] = found
    return matches, {
        "evaluation_fact_load_seconds": load_seconds,
        "rule_match_seconds": time.perf_counter() - started,
    }


def prove_rules(
    rules: list[dict], matches: dict[str, set[str]], *, base_rate: float,
):
    active = [rule for rule in rules if matches[rule["id"]]]
    engine = IsolatedPeTTaChainer(startup_timeout_seconds=180)
    timings: dict[str, float | int] = {}
    try:
        started = time.perf_counter()
        engine.replace([compiled_rule(rule) for rule in active])
        timings["reasoner_start_seconds"] = time.perf_counter() - started
        facts: set[str] = set()
        representatives: list[tuple[dict, str]] = []
        for rule in active:
            case = min(matches[rule["id"]])
            representatives.append((rule, case))
            facts.update(premise_source(rule, case))
        named = [
            f"(: derived_fact_{index} {fact} (STV 1.0 1.0))"
            for index, fact in enumerate(sorted(facts))
        ]
        started = time.perf_counter()
        engine.add_atoms_no_check(named, timeout_sec=180)
        timings["reasoner_fact_seconds"] = time.perf_counter() - started
        queries = [
            f"(: $proof (DerivedSignal {case} {json.dumps(rule['id'])}) $tv)"
            for rule, case in representatives
        ]
        started = time.perf_counter()
        results = engine.query_many(
            queries, steps=max(12, 12 * len(queries)), timeout_sec=180,
        )
        timings["proof_query_seconds"] = time.perf_counter() - started
        evidence: list[str] = []
        proved = 0
        for (rule, _case), proofs in zip(representatives, results):
            if not proofs:
                raise RuntimeError(f"{rule['id']}: no PeTTaChainer proof")
            values = {
                tuple(round(value, 12) for value in proof_tv(proof))
                for proof in proofs
            }
            if len(values) != 1:
                raise RuntimeError(f"{rule['id']}: inconsistent proof STVs")
            strength, confidence = next(iter(values))
            posterior = base_rate + confidence * (strength - base_rate)
            quality = (
                confidence * abs(strength - base_rate)
                * math.log1p(rule["support"])
                * (1.0 + 0.1 * (rule["specificity"] - 1))
            )
            for case in matches[rule["id"]]:
                evidence.append(
                    f"(RuleEvidence {case} {rule['id']} "
                    f"{rule['specificity']} {quality} {posterior})"
                )
            rule["proof_stv"] = [strength, confidence]
            rule["posterior"] = posterior
            rule["quality"] = quality
            proved += 1
        timings.update(
            compiled_rules=len(active), proved_rules=proved,
            factorized_source_facts=len(named),
            grounded_rule_evidence=len(evidence),
        )
        return evidence, timings
    finally:
        engine.close()


def aggregate_in_petta(
    runtime: PeTTa, evidence: list[str], case_index: dict, *,
    base_rate: float, max_conjuncts: int, root: Path,
):
    runtime.load_metta_file(str(root / "miner/derivedConjunctionPolicy.metta"))
    runtime.process_metta_string("!(bind! &derived_policy (new-space))")
    policy_facts = [f"(CandidateCase {case})" for case in case_index]
    policy_facts.extend(evidence)
    load_seconds = load_atoms(runtime, "&derived_policy", policy_facts)
    scores: dict[int, dict[str, float]] = {}
    started = time.perf_counter()
    for depth in range(1, max_conjuncts + 1):
        raw = runtime.process_metta_string(
            f"!(derived-case-scores &derived_policy {depth} {base_rate})"
        )
        found = {
            case: float(value)
            for case, value in SCORE_RE.findall(" ".join(map(str, raw)))
        }
        if found.keys() != case_index.keys():
            missing = len(case_index.keys() - found.keys())
            raise RuntimeError(f"PeTTa policy omitted {missing} candidate cases")
        scores[depth] = found
    return scores, {
        "policy_fact_load_seconds": load_seconds,
        "policy_score_seconds": time.perf_counter() - started,
    }


def baseline_scores(rows: list[dict]) -> dict[str, tuple[float, ...]]:
    return {
        str(row["article"]): (
            float(row.get("ranking_score", row.get("score", 0.0))),
            float(row.get("pairwise_score", 0.0) or 0.0),
            float(row.get("score", 0.0)),
            float(row.get("stv", {}).get("strength", 0.0)),
            float(row.get("stv", {}).get("confidence", 0.0)),
        )
        for row in rows
    }


def paired_interval(
    rows: list[dict], control: str, challenger: str, *, repetitions: int = 5000,
    seed: int = 7,
) -> list[float]:
    deltas = [row[challenger] - row[control] for row in rows]
    generator = random.Random(seed)
    count = len(deltas)
    estimates = sorted(
        math.fsum(deltas[generator.randrange(count)] for _ in range(count))
        / count
        for _ in range(repetitions)
    )
    return [
        estimates[int(0.025 * repetitions)],
        estimates[int(0.975 * repetitions) - 1],
    ]


def evaluate(data: dict, baseline: dict, cases, case_scores, max_conjuncts: int):
    baseline_by_id = {
        str(row["id"]): ranking
        for row, ranking in zip(
            baseline["result"]["auc_per_impression"],
            baseline["ranked_impressions"],
        )
    }
    per_impression: list[dict] = []
    for index, (test, local) in enumerate(zip(data["tests"], cases)):
        identity = str(test.get("source_impression_id") or test["id"])
        champion = baseline_scores(baseline_by_id[identity])
        candidates = [article for article, _case in local]
        if set(champion) != set(candidates):
            raise RuntimeError("baseline candidate slate differs")
        positives = set(map(str, test["relevant"]))
        row = {"id": identity, "champion": impression_auc(champion, positives)}
        derived_by_depth: dict[int, dict[str, float]] = {}
        for depth in range(1, max_conjuncts + 1):
            derived = {
                article: case_scores[depth][case]
                for article, case in local
            }
            derived_by_depth[depth] = derived
            row[f"derived_depth_{depth}"] = impression_auc(derived, positives)
        derived = derived_by_depth[max_conjuncts]
        tie_break = {
            article: (*champion[article], derived[article])
            for article in candidates
        }
        champion_rank = midranks(champion)
        derived_rank = midranks(derived)
        fusion = {
            article: (champion_rank[article] + derived_rank[article]) / 2
            for article in candidates
        }
        row["champion_tie_break"] = impression_auc(tie_break, positives)
        row["balanced_fusion"] = impression_auc(fusion, positives)
        per_impression.append(row)
    names = ["champion", "champion_tie_break", "balanced_fusion"] + [
        f"derived_depth_{depth}" for depth in range(1, max_conjuncts + 1)
    ]
    metrics = {
        name: math.fsum(row[name] for row in per_impression) / len(per_impression)
        for name in names
    }
    metrics["depth_deltas_vs_unary"] = {
        str(depth): metrics[f"derived_depth_{depth}"] - metrics["derived_depth_1"]
        for depth in range(2, max_conjuncts + 1)
    }
    metrics["paired_95_intervals"] = {
        f"depth_{depth}_vs_unary": paired_interval(
            per_impression, "derived_depth_1", f"derived_depth_{depth}",
        )
        for depth in range(2, max_conjuncts + 1)
    }
    metrics["paired_95_intervals"].update({
        "tie_break_vs_champion": paired_interval(
            per_impression, "champion", "champion_tie_break",
        ),
        "fusion_vs_champion": paired_interval(
            per_impression, "champion", "balanced_fusion",
        ),
    })
    return metrics, per_impression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-support", type=int, default=16)
    parser.add_argument("--max-conjuncts", type=int, default=5)
    parser.add_argument("--evidence-k", type=float, default=800.0)
    parser.add_argument("--length-cost", type=float, default=0.5)
    parser.add_argument("--max-patterns", type=int, default=100_000)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    started = time.perf_counter()
    data_path = Path(args.data)
    with gzip.open(data_path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    baseline = json.loads(Path(args.baseline_artifact).read_text())
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if baseline.get("dataset_sha256") != digest:
        parser.error("baseline artifact was produced from different dataset bytes")
    complete, positive = training_facts(data)
    eval_facts, cases, case_index = evaluation_facts(data)
    build_seconds = time.perf_counter() - started
    root = Path(__file__).resolve().parents[1]
    runtime = PeTTa()
    runtime.load_metta_file(str(root / "miner/helpers.metta"))
    runtime.load_metta_file(str(root / "miner/fpMiner.metta"))
    rules, mining_timings = mine_rules(
        runtime, complete, positive, min_support=args.min_support,
        max_conjuncts=args.max_conjuncts, evidence_k=args.evidence_k,
        length_cost=args.length_cost, max_patterns=args.max_patterns,
    )
    print(json.dumps({
        "stage": "mined", "rules": len(rules),
        "specificities": Counter(rule["specificity"] for rule in rules),
        **mining_timings,
    }), flush=True)
    matches, match_timings = match_rules(runtime, eval_facts, rules, case_index)
    base_rate = sum(event.get("action") in POSITIVE for event in data["events"]) / len(data["events"])
    evidence, proof_timings = prove_rules(rules, matches, base_rate=base_rate)
    scores, policy_timings = aggregate_in_petta(
        runtime, evidence, case_index, base_rate=base_rate,
        max_conjuncts=args.max_conjuncts, root=root,
    )
    metrics, per_impression = evaluate(
        data, baseline, cases, scores, args.max_conjuncts,
    )
    expected = float(baseline["result"]["auc_proof_only"])
    if abs(metrics["champion"] - expected) > 0.00005:
        raise RuntimeError("frozen champion reconstruction failed")
    artifact = {
        "schema": "derived-conjunction-mdl-evaluation-v1",
        "dataset_sha256": digest,
        "protocol": {
            "features": FEATURES,
            "target": "click",
            "min_support": args.min_support,
            "max_conjuncts": args.max_conjuncts,
            "evidence_k": args.evidence_k,
            "length_cost": args.length_cost,
            "max_patterns": args.max_patterns,
            "mining": "PeTTa target-prefiltered Apriori conjunction expansion",
            "reasoning": "PeTTaChainer factorized proofs",
            "aggregation": "PeTTa hierarchical specificity backoff",
        },
        "facts": {
            "training": len(complete), "positive_training": len(positive),
            "evaluation": len(eval_facts), "candidate_cases": len(case_index),
        },
        "rules": rules,
        "rule_specificities": dict(Counter(
            rule["specificity"] for rule in rules
        )),
        "metrics": metrics,
        "performance": {
            "fact_build_seconds": build_seconds,
            **mining_timings, **match_timings, **proof_timings,
            **policy_timings,
            "total_seconds": time.perf_counter() - started,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
        "per_impression": per_impression,
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({
        "stage": "complete", "output": str(output),
        "metrics": metrics, "performance": artifact["performance"],
    }), flush=True)


if __name__ == "__main__":
    main()
