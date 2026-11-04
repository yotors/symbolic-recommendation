"""Select parent-aware MDL on MIND and test transfer on disjoint EB-NeRD."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import time
import traceback

from .connected_miner_experiment import PeTTa
from .derived_conjunction_experiment import (
    evaluation_facts, load_atoms, match_rules, mine_rules, premise_source,
    training_facts,
)
from .predicate_ctv_experiment import (
    POSITIVE, SCORE_RE, impression_tests, per_impression, prove_templates,
    split_training, summary,
)


CUTOFFS = (-1.0, 0.0, 0.05, 0.1, 0.25, 0.5)


def read_dataset(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def disjoint_tests(fresh, previous):
    """Exclude earlier benchmark IDs before examining validation labels."""
    old_ids = {str(test["source_impression_id"]) for test in previous}
    eligible = [test for test in fresh if str(test["source_impression_id"]) not in old_ids]
    overlap = len(fresh) - len(eligible)
    tests = [test for test in eligible
             if 0 < len(test["relevant"]) < len(test["candidates"])]
    identifiers = {str(test["source_impression_id"]) for test in tests}
    if len(identifiers) != len(tests):
        raise ValueError("fresh evaluation impressions have duplicate IDs")
    return tests, {"overlap_excluded": overlap,
                   "ineligible_excluded": len(eligible) - len(tests)}


def parent_bounds(runtime, rules, root):
    runtime.load_metta_file(str(root / "miner/parentAwareSelection.metta"))
    started = time.perf_counter()
    for rule in rules:
        clauses = premise_source(rule)
        pattern = f'(, {" ".join(clauses)} (engagement $case "click"))'
        raw = runtime.process_metta_string(
            f'!(parent-minimum-penalty-bound &derived_train {pattern} '
            '(engagement $case "click"))'
        )
        if len(raw) != 1:
            raise RuntimeError(f"{rule['id']}: invalid parent bound: {raw}")
        bound = float(str(raw[0]))
        if not math.isfinite(bound):
            raise RuntimeError(f"{rule['id']}: non-finite parent bound")
        rule["parent_bound"] = bound
    return time.perf_counter() - started


def prepare_policy(runtime, rules, matches, case_index, base, root):
    runtime.load_metta_file(str(root / "miner/predicateRankingPolicy.metta"))
    runtime.load_metta_file(str(root / "miner/adaptiveParentPolicy.metta"))
    runtime.process_metta_string("!(bind! &adaptive_policy (new-space))")
    facts = [f"(CandidateCase {case})" for case in case_index]
    for rule in rules:
        if "proof_stv" not in rule:
            continue
        strength, confidence = rule["proof_stv"]
        families = "(" + " ".join(p for p, _ in rule["premises"]) + ")"
        atom = (
            f"(RuleMeta {rule['id']} {rule['specificity']} "
            f"(predicate-rule-quality {strength} {confidence} {rule['support']} "
            f"{rule['specificity']} {base}) "
            f"{strength} {confidence} {rule['parent_bound']} {families})"
        )
        runtime.process_metta_string(
            f"!(let $fact {atom} (add-atom &adaptive_policy $fact))"
        )
        facts.extend(f"(RuleFired {case} {rule['id']})" for case in matches[rule["id"]])
    return load_atoms(runtime, "&adaptive_policy", facts), len(facts)


def scores(runtime, cases, cutoff, base):
    raw = runtime.process_metta_string(
        f"!(adaptive-parent-scores &adaptive_policy {cutoff} {base} shrink)"
    )
    found = SCORE_RE.findall(" ".join(map(str, raw)))
    result = {key: float(value) for key, value in found}
    if len(found) != len(result) or result.keys() != cases.keys():
        raise RuntimeError("PeTTa policy returned missing or duplicate scores")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in result.values()):
        raise RuntimeError("PeTTa policy returned an invalid score")
    return result


def user_interval(reference, candidate, *, seed=7, repetitions=5000):
    if [r["id"] for r in reference] != [r["id"] for r in candidate]:
        raise RuntimeError("evaluation cohorts differ")
    by_user = defaultdict(list)
    for left, right in zip(reference, candidate):
        if left["user"] != right["user"]:
            raise RuntimeError("paired user mismatch")
        by_user[left["user"]].append(right["auc"] - left["auc"])
    groups = list(by_user.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        estimates.append(sum(map(sum, selected)) / sum(map(len, selected)))
    estimates.sort()
    return [estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions) - 1]]


def compare(reference, candidate):
    return {
        "delta_auc": summary(candidate)["auc"] - summary(reference)["auc"],
        "user_cluster_bootstrap_95": user_interval(reference, candidate),
    }


def run_dataset(train_events, tests, *, cutoffs, min_support, max_conjuncts, length_cost):
    started = time.perf_counter()
    complete, positive = training_facts({"events": train_events})
    facts, local, case_index = evaluation_facts({"tests": tests})
    base = sum(row.get("action") in POSITIVE for row in train_events) / len(train_events)
    root = Path(__file__).resolve().parents[1]
    runtime = PeTTa()
    runtime.load_metta_file(str(root / "miner/helpers.metta"))
    runtime.load_metta_file(str(root / "miner/fpMiner.metta"))
    rules, mining = mine_rules(
        runtime, complete, positive, min_support=min_support,
        max_conjuncts=max_conjuncts, evidence_k=800.0,
        length_cost=length_cost, max_patterns=100_000,
    )
    if not rules:
        raise RuntimeError("no rules mined")
    parent_seconds = parent_bounds(runtime, rules, root)
    counts = {
        str(cutoff): dict(Counter(
            rule["specificity"] for rule in rules
            if cutoff < 0 or rule["parent_bound"] > cutoff
        ))
        for cutoff in cutoffs
    }
    print(json.dumps({"stage": "mined", "rules": len(rules), "accepted_by_depth": counts, "mining": mining, "parent_seconds": parent_seconds}), flush=True)
    matches, matching = match_rules(runtime, facts, rules, case_index)
    proof = prove_templates(rules, matches)
    print(json.dumps({"stage": "proved", **matching, **proof}), flush=True)
    load_seconds, fact_count = prepare_policy(runtime, rules, matches, case_index, base, root)
    print(json.dumps({"stage": "policy_loaded", "facts": fact_count, "seconds": load_seconds}), flush=True)
    rows = {}
    for cutoff in cutoffs:
        values = scores(runtime, case_index, cutoff, base)
        rows[cutoff] = per_impression(tests, local, values, 0)
        print(json.dumps({"stage": "scored", "cutoff": cutoff, **summary(rows[cutoff])}), flush=True)
    return rows, {
        "rules": len(rules), "accepted_by_depth": counts,
        "exact_duplicate_children": sum(rule["specificity"] > 1 and rule["parent_bound"] < 0 for rule in rules),
        "rule_bounds": [{"id": rule["id"], "premises": rule["premises"], "bound": rule["parent_bound"]} for rule in rules],
        "base_rate": base, "performance": {**mining, "parent_seconds": parent_seconds, **matching, **proof, "policy_load_seconds": load_seconds, "total_seconds": time.perf_counter() - started},
    }


def _isolated_worker(connection, train_events, tests, options):
    try:
        connection.send(("ok", run_dataset(train_events, tests, **options)))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def run_isolated(train_events, tests, **options):
    """PeTTa spaces are process-global: never reuse a process across datasets."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_worker, args=(child, train_events, tests, options))
    process.start()
    child.close()
    try:
        while not parent.poll(30):
            if not process.is_alive():
                raise RuntimeError(f"dataset worker exited with code {process.exitcode}")
        status, result = parent.recv()
        process.join(10)
        if status != "ok":
            raise RuntimeError(f"dataset worker failed:\n{result}")
        if process.exitcode != 0:
            raise RuntimeError(f"dataset worker exited with code {process.exitcode}")
        return result
    finally:
        parent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mind", required=True)
    parser.add_argument("--ebnerd-fresh", required=True)
    parser.add_argument("--ebnerd-previous", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--max-conjuncts", type=int, default=4)
    parser.add_argument("--length-cost", type=float, default=0.5)
    parser.add_argument("--resume-mind-checkpoint")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    mind = read_dataset(args.mind)
    mining, selection, holdout = split_training(mind["events"])
    selection_tests, holdout_tests = impression_tests(selection), impression_tests(holdout)
    source_root = Path(__file__).resolve().parents[1]
    identity = {
        "mind_sha256": hashlib.sha256(Path(args.mind).read_bytes()).hexdigest(),
        "min_support": args.min_support, "max_conjuncts": args.max_conjuncts,
        "length_cost": args.length_cost, "cutoffs": list(CUTOFFS),
        "source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                          for name in ("miner/fpMiner.metta", "miner/conjunctionExpansion.metta",
                                       "miner/helpers.metta", "miner/parentAwareSelection.metta",
                                       "miner/adaptiveParentPolicy.metta", "miner/predicateRankingPolicy.metta",
                                       "evaluation/adaptive_parent_experiment.py",
                                       "evaluation/predicate_ctv_experiment.py",
                                       "evaluation/derived_conjunction_experiment.py",
                                       "evaluation/connected_miner_experiment.py",
                                       "mining/rule_parser.py", "app/reasoner.py")},
    }
    if args.resume_mind_checkpoint:
        checkpoint = json.loads(Path(args.resume_mind_checkpoint).read_text())
        if checkpoint["identity"] != identity:
            raise RuntimeError("MIND checkpoint data, configuration or source fingerprint mismatch")
        selected = float(checkpoint["selected_cutoff"])
        mind_result = checkpoint["mind"]
    else:
        mind_rows, mind_audit = run_isolated(
            [row for group in mining for row in group],
            selection_tests + holdout_tests, cutoffs=CUTOFFS,
            min_support=args.min_support, max_conjuncts=args.max_conjuncts,
            length_cost=args.length_cost,
        )
        n = len(selection_tests)
        selection_metrics = {str(c): summary(rows[:n]) for c, rows in mind_rows.items()}
        # Deterministic tie-break prefers simpler/stricter policies.
        selected = max(CUTOFFS, key=lambda c: (selection_metrics[str(c)]["auc"], c))
        mind_ref, mind_selected = mind_rows[-1.0][n:], mind_rows[selected][n:]
        mind_result = {"selection": selection_metrics,
                       "later_development": {"reference": summary(mind_ref),
                                             "selected": summary(mind_selected),
                                             "comparison": compare(mind_ref, mind_selected)},
                       "audit": mind_audit}
        checkpoint_path = output.with_name(output.stem + ".mind-checkpoint.json")
        with checkpoint_path.open("x", encoding="utf-8") as stream:
            json.dump({"identity": identity, "selected_cutoff": selected, "mind": mind_result}, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"stage": "mind_selected", "cutoff": selected,
                          "checkpoint": str(checkpoint_path),
                          "selection": selection_metrics,
                          "later_development": mind_result["later_development"]["comparison"]}), flush=True)

    fresh = read_dataset(args.ebnerd_fresh)
    previous = read_dataset(args.ebnerd_previous)
    fresh_tests, exclusions = disjoint_tests(fresh["tests"], previous["tests"])
    if len(fresh_tests) < 500:
        raise RuntimeError("fresh EB-NeRD holdout has fewer than 500 eligible impressions")
    # Report fixed looser controls as diagnostics, without reselecting on EB.
    eb_cutoffs = tuple(dict.fromkeys((-1.0, 0.0, 0.05, 0.5, selected)))
    eb_rows, eb_audit = run_isolated(
        fresh["events"], fresh_tests, cutoffs=eb_cutoffs,
        min_support=args.min_support, max_conjuncts=args.max_conjuncts,
        length_cost=args.length_cost,
    )
    eb_ref, eb_selected = eb_rows[-1.0], eb_rows[selected]
    artifact = {
        "schema": "adaptive-parent-mdl-v1",
        "protocol": {
            "parent_cutoffs": CUTOFFS,
            "rule_discovery_global_mdl_cost": args.length_cost,
            "parent_cost_selected_on": "previously inspected MIND 60/20/20 training selection split",
            "mind_later": "previously inspected development split, not fresh confirmation",
            "ebnerd": "seed-43 validation impressions excluding IDs in previous seed-7 EB-NeRD dataset; threshold locked from MIND; no EB validation labels used for model selection",
            "model": "pointwise best applicable PeTTaChainer proof with PeTTa hierarchical backoff; not the pairwise champion",
            "min_support": args.min_support, "max_conjuncts": args.max_conjuncts,
            "confidence_interval": "bootstrap users with replacement, retain all their impressions",
            "input_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (("mind", args.mind), ("ebnerd_fresh", args.ebnerd_fresh), ("ebnerd_previous", args.ebnerd_previous))},
        },
        "selected_cutoff": selected,
        "mind": mind_result,
        "ebnerd": {"previous_ids_excluded": exclusions["overlap_excluded"],
                   "ineligible_excluded": exclusions["ineligible_excluded"],
                   "fresh_impressions": len(fresh_tests),
                   "reference": summary(eb_ref), "selected": summary(eb_selected),
                   "strict": summary(eb_rows[0.5]),
                   "diagnostic_cutoffs": {str(c): {"metrics": summary(eb_rows[c]),
                                                    "vs_reference": compare(eb_ref, eb_rows[c])}
                                          for c in eb_cutoffs if c not in (-1.0, selected)},
                   "selected_vs_reference": compare(eb_ref, eb_selected),
                   "selected_vs_strict": compare(eb_rows[0.5], eb_selected),
                   "audit": eb_audit},
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"stage": "complete", "selected_cutoff": selected,
                      "mind_later": artifact["mind"]["later_development"],
                      "ebnerd": {k: v for k, v in artifact["ebnerd"].items() if k != "audit"},
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
