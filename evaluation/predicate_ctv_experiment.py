"""Chronological parent-aware rule and predicate selection experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .connected_miner_experiment import IsolatedPeTTaChainer, PeTTa, impression_auc
from .derived_conjunction_experiment import (
    FEATURES, POSITIVE, baseline_scores, compiled_rule, evaluation_facts, load_atoms,
    match_rules, mine_rules, paired_interval, premise_source,
    training_facts,
)
from ..mining.rule_parser import STV_RE, proof_tv


SCORE_RE = re.compile(
    r"\(PredicateScore\s+(eval_case_[0-9]+_[0-9]+)\s+([0-9.eE+-]+)\)"
)


def split_training(events: list[dict]):
    """Split complete impressions chronologically without splitting timestamps."""
    groups: dict[str, list[dict]] = {}
    for event in events:
        groups.setdefault(str(event["impression"]), []).append(event)
    ordered = sorted(
        groups.values(), key=lambda group: (
            str(group[0]["timestamp"]), str(group[0]["impression"]),
        ),
    )
    for group in ordered:
        if len({str(row["timestamp"]) for row in group}) != 1:
            raise ValueError("an impression has conflicting timestamps")
        if len({str(row["article"]) for row in group}) != len(group):
            raise ValueError("an impression has duplicate candidates")
    count = len(ordered)
    first, second = int(count * 0.6), int(count * 0.8)
    while first < count and ordered[first][0]["timestamp"] == ordered[first - 1][0]["timestamp"]:
        first += 1
    second = max(second, first + 1)
    while second < count and ordered[second][0]["timestamp"] == ordered[second - 1][0]["timestamp"]:
        second += 1
    if first <= 0 or second >= count:
        raise ValueError("not enough chronological impressions")
    return ordered[:first], ordered[first:second], ordered[second:]


def impression_tests(groups: list[list[dict]]) -> list[dict]:
    tests = []
    for group in groups:
        positives = [str(row["article"]) for row in group if row.get("action") in POSITIVE]
        if not positives or len(positives) == len(group):
            continue
        tests.append({
            "id": str(group[0]["impression"]),
            "user": str(group[0]["user"]),
            "timestamp": str(group[0]["timestamp"]),
            "candidates": [str(row["article"]) for row in group],
            "candidate_context": {str(row["article"]): row for row in group},
            "relevant": positives,
        })
    return tests


def parent_filter(runtime: PeTTa, rules: list[dict], root: Path, cost: float):
    runtime.load_metta_file(str(root / "miner/parentAwareSelection.metta"))
    started = time.perf_counter()
    for rule in rules:
        clauses = premise_source(rule)
        pattern = f'(, {" ".join(clauses)} (engagement $case "click"))'
        raw = runtime.process_metta_string(
            f'!(parent-minimum-gain &derived_train {pattern} '
            f'(engagement $case "click") {cost})'
        )
        if len(raw) != 1:
            raise RuntimeError(f"{rule['id']}: invalid parent MDL result {raw}")
        gain = float(str(raw[0]))
        if not math.isfinite(gain):
            raise RuntimeError(f"{rule['id']}: non-finite parent MDL result")
        rule["parent_mdl_gain"] = gain
        rule["parent_accepted"] = gain > 0
    return time.perf_counter() - started


def prepare_policy(runtime, root, rules, matches, case_index, base):
    runtime.load_metta_file(str(root / "miner/predicateRankingPolicy.metta"))
    runtime.process_metta_string("!(bind! &predicate_policy (new-space))")
    facts = [f"(CandidateCase {case})" for case in case_index]
    for rule in rules:
        if "proof_stv" not in rule:
            continue
        strength, confidence = rule["proof_stv"]
        families = "(" + " ".join(p for p, _ in rule["premises"]) + ")"
        accepted = "True" if rule["parent_accepted"] else "False"
        meta = (
            f"(RuleMeta {rule['id']} {rule['specificity']} "
            f"(predicate-rule-quality {strength} {confidence} {rule['support']} "
            f"{rule['specificity']} {base}) "
            f"{strength} {confidence} {accepted} {families})"
        )
        # Reduce the quality expression before inserting the atom.
        runtime.process_metta_string(f"!(let $fact {meta} (add-atom &predicate_policy $fact))")
        facts.extend(f"(RuleFired {case} {rule['id']})" for case in matches[rule["id"]])
    return load_atoms(runtime, "&predicate_policy", facts), len(facts)


def prove_templates(rules, matches):
    """Prove one representative grounding per rule with certain premises."""
    active = [rule for rule in rules if matches[rule["id"]]]
    engine = IsolatedPeTTaChainer(startup_timeout_seconds=180)
    try:
        started = time.perf_counter()
        engine.replace([compiled_rule(rule) for rule in active])
        startup = time.perf_counter() - started
        representatives = [(rule, min(matches[rule["id"]])) for rule in active]
        facts = {fact for rule, case in representatives for fact in premise_source(rule, case)}
        named = [f"(: derived_fact_{i} {fact} (STV 1.0 1.0))" for i, fact in enumerate(sorted(facts))]
        started = time.perf_counter()
        engine.add_atoms_no_check(named, timeout_sec=180)
        insertion = time.perf_counter() - started
        queries = [f"(: $proof (DerivedSignal {case} {json.dumps(rule['id'])}) $tv)" for rule, case in representatives]
        started = time.perf_counter()
        results = engine.query_many(queries, steps=max(12, 12 * len(queries)), timeout_sec=180)
        if len(results) != len(representatives):
            raise RuntimeError("PeTTaChainer returned an incomplete proof batch")
        for (rule, _case), proofs in zip(representatives, results):
            if any(STV_RE.search(proof) is None for proof in proofs):
                raise RuntimeError(f"{rule['id']}: proof has no STV")
            values = {tuple(round(value, 12) for value in proof_tv(proof)) for proof in proofs}
            if len(values) != 1:
                raise RuntimeError(f"{rule['id']}: missing or inconsistent proof STV")
            rule["proof_stv"] = list(values.pop())
        return {"reasoner_start_seconds": startup, "reasoner_fact_seconds": insertion,
                "proof_query_seconds": time.perf_counter() - started,
                "compiled_rules": len(active), "proved_rules": len(results),
                "factorized_source_facts": len(named),
                "grounded_rule_evidence": sum(len(matches[rule["id"]]) for rule in active)}
    finally:
        engine.close()


def policy_scores(runtime, case_index, *, parent, excluded, base, mode):
    raw = runtime.process_metta_string(
        f"!(predicate-rank-scores &predicate_policy {'True' if parent else 'False'} "
        f"{excluded} {base} {mode})"
    )
    found = SCORE_RE.findall(" ".join(map(str, raw)))
    scores = {
        case: float(value)
        for case, value in found
    }
    if len(found) != len(scores) or scores.keys() != case_index.keys():
        raise RuntimeError("PeTTa policy did not return exactly one score per case")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values()):
        raise RuntimeError("invalid probability score")
    return scores


def per_impression(tests, cases, scores, offset):
    rows = []
    for index, test in enumerate(tests):
        positives = set(map(str, test["relevant"]))
        values = {article: scores[case] for article, case in cases[offset + index]}
        auc = impression_auc(values, positives)
        brier = math.fsum((value - (article in positives)) ** 2 for article, value in values.items()) / len(values)
        logloss = math.fsum(
            -math.log(max(1e-12, value if article in positives else 1 - value))
            for article, value in values.items()
        ) / len(values)
        rows.append({"id": test["id"], "user": test["user"], "auc": auc, "brier": brier, "logloss": logloss})
    return rows


def summary(rows):
    return {
        "impressions": len(rows),
        "users": len({row["user"] for row in rows}),
        **{name: math.fsum(row[name] for row in rows) / len(rows) for name in ("auc", "brier", "logloss")},
    }


def compare(left, right):
    if [row["id"] for row in left] != [row["id"] for row in right]:
        raise RuntimeError("paired evaluation cohorts differ")
    pairs = [{"left": a["auc"], "right": b["auc"]} for a, b in zip(left, right)]
    return {
        "auc_delta": summary(right)["auc"] - summary(left)["auc"],
        "paired_impression_bootstrap_95": paired_interval(pairs, "left", "right"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--max-conjuncts", type=int, default=4)
    parser.add_argument("--length-cost", type=float, default=0.5)
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
    mining, selection, holdout = split_training(data["events"])
    mining_events = [event for group in mining for event in group]
    selection_tests, holdout_tests = impression_tests(selection), impression_tests(holdout)
    public_tests = data["tests"]
    tests = selection_tests + holdout_tests + public_tests
    if not selection_tests or not holdout_tests:
        raise RuntimeError("empty train selection or holdout split")
    complete, positive = training_facts({"events": mining_events})
    facts, cases, case_index = evaluation_facts({"tests": tests})
    base = sum(event.get("action") in POSITIVE for event in mining_events) / len(mining_events)
    root = Path(__file__).resolve().parents[1]
    runtime = PeTTa()
    runtime.load_metta_file(str(root / "miner/helpers.metta"))
    runtime.load_metta_file(str(root / "miner/fpMiner.metta"))
    rules, timings = mine_rules(
        runtime, complete, positive, min_support=args.min_support,
        max_conjuncts=args.max_conjuncts, evidence_k=800.0,
        length_cost=args.length_cost, max_patterns=100_000,
    )
    if not rules:
        raise RuntimeError("no rules mined")
    timings["parent_selection_seconds"] = parent_filter(runtime, rules, root, args.length_cost)
    print(json.dumps({"stage": "mined", "rules": len(rules), "parent_accepted": sum(r["parent_accepted"] for r in rules), "depths": Counter(r["specificity"] for r in rules), **timings}), flush=True)
    matches, match_timings = match_rules(runtime, facts, rules, case_index)
    # The existing proof executor verifies one representative grounding per
    # rule. Certain premises make its STV identical for equivalent groundings.
    proof_timings = prove_templates(rules, matches)
    print(json.dumps({"stage": "proved", **match_timings, **proof_timings}), flush=True)
    policy_load, policy_facts = prepare_policy(runtime, root, rules, matches, case_index, base)
    print(json.dumps({"stage": "policy_loaded", "facts": policy_facts, "seconds": policy_load}), flush=True)
    variants = [(parent, mode, "none") for parent in (False, True) for mode in ("raw", "shrink")]
    scores_by_variant = {}
    selection_rows = {}
    for variant in variants:
        parent, mode, excluded = variant
        scores = policy_scores(runtime, case_index, parent=parent, excluded=excluded, base=base, mode=mode)
        scores_by_variant[variant] = scores
        selection_rows[variant] = per_impression(selection_tests, cases, scores, 0)
        print(json.dumps({"stage": "selection_variant", "variant": variant, **summary(selection_rows[variant])}), flush=True)
    winner = max(variants, key=lambda key: (summary(selection_rows[key])["auc"], tuple(map(str, key))))
    exclusions = [(*winner[:2], feature) for feature in FEATURES]
    for variant in exclusions:
        scores = policy_scores(runtime, case_index, parent=variant[0], excluded=variant[2], base=base, mode=variant[1])
        scores_by_variant[variant] = scores
        selection_rows[variant] = per_impression(selection_tests, cases, scores, 0)
    # Selection is exploratory. No model is promoted by this experiment.
    selected = max([winner, *exclusions], key=lambda key: (summary(selection_rows[key])["auc"], tuple(map(str, key))))
    holdout_offset = len(selection_tests)
    public_offset = holdout_offset + len(holdout_tests)
    reference = (False, "shrink", "none")
    holdout_ref = per_impression(holdout_tests, cases, scores_by_variant[reference], holdout_offset)
    holdout_selected = per_impression(holdout_tests, cases, scores_by_variant[selected], holdout_offset)
    public_ref = per_impression(public_tests, cases, scores_by_variant[reference], public_offset)
    public_selected = per_impression(public_tests, cases, scores_by_variant[selected], public_offset)
    baseline_by_id = {str(row["id"]): ranking for row, ranking in zip(baseline["result"]["auc_per_impression"], baseline["ranked_impressions"])}
    champion_rows = []
    for test in public_tests:
        identity = str(test.get("source_impression_id") or test["id"])
        champion = baseline_scores(baseline_by_id[identity])
        if set(champion) != set(map(str, test["candidates"])):
            raise RuntimeError("champion slate mismatch")
        champion_rows.append({"id": test["id"], "user": test["user"], "auc": impression_auc(champion, set(map(str, test["relevant"])))})
    champion_auc = math.fsum(row["auc"] for row in champion_rows) / len(champion_rows)
    if abs(champion_auc - float(baseline["result"]["auc_proof_only"])) > 0.00005:
        raise RuntimeError("champion reconstruction failed")
    constant_scores = {case: base for case in case_index}
    artifact = {
        "schema": "predicate-ctv-selection-v1", "dataset_sha256": digest,
        "protocol": {"split": "chronological complete training impressions 60/20/20; equal timestamps stay together", "development_set": "500 repeatedly inspected MIND impressions; not a fresh final test", "selection": "AUC selects parent/raw-or-shrink policy and at most one excluded predicate family; no production promotion", "point_probabilities": "proof STV strength or confidence-shrunk strength; measured at candidate level with impression-macro weighting", "features": FEATURES, "min_support": args.min_support, "max_conjuncts": args.max_conjuncts, "length_cost": args.length_cost, "selected_variant": selected},
        "splits": {name: {"impressions": len(groups), "cases": sum(map(len, groups)), "first": groups[0][0]["timestamp"], "last": groups[-1][0]["timestamp"]} for name, groups in (("mining", mining), ("selection", selection), ("holdout", holdout))},
        "rules": {"total": len(rules), "parent_accepted": sum(r["parent_accepted"] for r in rules), "by_depth": dict(Counter(r["specificity"] for r in rules)), "parent_by_depth": dict(Counter(r["specificity"] for r in rules if r["parent_accepted"])), "items": rules},
        "selection": {str(key): summary(rows) for key, rows in selection_rows.items()},
        "holdout": {"reference": summary(holdout_ref), "selected": summary(holdout_selected), "constant_train_prior": summary(per_impression(holdout_tests, cases, constant_scores, holdout_offset)), "selected_vs_reference": compare(holdout_ref, holdout_selected)},
        "mind_development": {"reference": summary(public_ref), "selected": summary(public_selected), "constant_train_prior": summary(per_impression(public_tests, cases, constant_scores, public_offset)), "champion_auc": champion_auc, "selected_vs_reference": compare(public_ref, public_selected)},
        "performance": {**timings, **match_timings, **proof_timings, "policy_load_seconds": policy_load, "total_seconds": time.perf_counter() - started},
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "output": str(output), "selected": selected, "holdout": artifact["holdout"], "mind_development": artifact["mind_development"], "performance": artifact["performance"]}), flush=True)


if __name__ == "__main__":
    main()
