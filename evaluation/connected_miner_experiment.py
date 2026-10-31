"""Benchmark connected fpMiner rules and PeTTaChainer proofs on MIND."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import resource
import sys
import time

# PeTTaChainer requires the compatible PeTTa wheel in its virtualenv.  The
# workspace's older source runtime is useful for its own tests but cannot parse
# the chainer's lambda syntax; mirror the production server's path boundary.
from ..paths import WORKSPACE_ROOT

_LEGACY_PETTA = (WORKSPACE_ROOT / "PeTTa" / "python").resolve()
sys.path[:] = [
    entry for entry in sys.path
    if not entry or Path(entry).resolve() != _LEGACY_PETTA
]
sys.path.insert(0, str(WORKSPACE_ROOT / "PeTTaChainer"))

from ..app.reasoner import IsolatedPeTTaChainer
from ..mining.rule_parser import proof_tv
from petta import PeTTa


FAMILIES = {
    "topic": ("CandidateTopic", "SnapshotTopic", "topic"),
    "subcategory": (
        "CandidateSubcategory", "SnapshotSubcategory", "subcategory",
    ),
    "recent_subcategory": (
        "CandidateSubcategory", "RecentSnapshotSubcategory", "subcategory",
    ),
    "format": ("CandidateFormat", "SnapshotFormat", "format"),
    "entity": ("CandidateEntity", "SnapshotEntity", "entity"),
}
POSITIVE = frozenset({"click", "like", "complete"})
STV = re.compile(
    r"\(STV\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\)"
)


def symbol(prefix: str, value: object) -> str:
    digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return f"{prefix}_{digest}"


def article_values(article: dict, family: str) -> tuple[str, ...]:
    if family == "entity":
        values = {
            str(value).upper()
            for key in ("title_entities", "abstract_entities")
            for value in article.get(key, ())
            if isinstance(value, str) and value
        }
    else:
        field = FAMILIES[family][2]
        value = article.get(field)
        values = {str(value)} if value not in (None, "") else set()
    return tuple(sorted(symbol(family, value) for value in values))


def relation_facts(data: dict, *, evaluation_limit: int = 0):
    articles = {str(row["id"]): row for row in data["articles"]}
    observations = {
        family: {
            article_id: article_values(article, family)
            for article_id, article in articles.items()
        }
        for family in FAMILIES
    }

    def snapshot_facts(snapshot: str, history: tuple[str, ...]):
        result = set()
        for family, (_candidate, snapshot_predicate, _field) in FAMILIES.items():
            selected = history[-5:] if family == "recent_subcategory" else history
            values = {
                value for article_id in selected
                for value in observations[family].get(article_id, ())
            }
            result.update(
                f"({snapshot_predicate} {snapshot} {value})"
                for value in values
            )
        return result

    def candidate_facts(case: str, snapshot: str, article_id: str):
        article = symbol("article", article_id)
        result = {
            f"(Candidate {case} {article})",
            f"(ExposureSnapshot {case} {snapshot})",
        }
        for family, (predicate, _snapshot, _field) in FAMILIES.items():
            result.update(
                f"({predicate} {article} {value})"
                for value in observations[family].get(article_id, ())
            )
        return result

    training = set()
    built_snapshots = set()
    for index, event in enumerate(data["events"]):
        history = tuple(map(str, event.get("history", ())))
        snapshot = symbol(
            "train_snapshot",
            (event.get("user"), event.get("impression"), history),
        )
        if snapshot not in built_snapshots:
            built_snapshots.add(snapshot)
            training.update(snapshot_facts(snapshot, history))
        case = f"train_exposure_{index}"
        training.update(candidate_facts(case, snapshot, str(event["article"])))
        outcome = "click" if event.get("action") in POSITIVE else "skip"
        training.add(f'(engagement {case} "{outcome}")')

    tests = list(data.get("tests", ()))
    if evaluation_limit:
        tests = tests[:evaluation_limit]
    evaluation = set()
    cases = []
    for impression_index, impression in enumerate(tests):
        history = tuple(map(str, impression.get("history", ())))
        snapshot = symbol(
            "eval_snapshot",
            (impression.get("user"), impression.get("id"), history),
        )
        evaluation.update(snapshot_facts(snapshot, history))
        local = []
        for candidate_index, article_id in enumerate(impression["candidates"]):
            case = f"eval_case_{impression_index}_{candidate_index}"
            article_id = str(article_id)
            evaluation.update(candidate_facts(case, snapshot, article_id))
            local.append((article_id, case))
        cases.append(local)
    return training, evaluation, tests, cases


def mine_families(
    facts: set[str], *, min_support: int, evidence_k: float,
    length_cost: float, max_patterns: int,
):
    runtime = PeTTa()
    root = Path(__file__).resolve().parents[1]
    runtime.load_metta_file(str(root / "miner" / "helpers.metta"))
    runtime.load_metta_file(str(root / "miner" / "fpMiner.metta"))
    runtime.process_metta_string("!(bind! &connected_benchmark (new-space))")
    ordered = sorted(facts)
    started = time.perf_counter()
    for offset in range(0, len(ordered), 1000):
        additions = " ".join(
            f"(add-atom &connected_benchmark {fact})"
            for fact in ordered[offset:offset + 1000]
        )
        runtime.process_metta_string(f"!(superpose ({additions}))")
    load_seconds = time.perf_counter() - started

    mined = {}
    for family, (candidate, snapshot, role) in FAMILIES.items():
        schemas = (
            "((Candidate exposure article join) "
            "(ExposureSnapshot exposure snapshot join) "
            f"({candidate} article {role} join) "
            f"({snapshot} snapshot {role} join))"
        )
        required = f"(Candidate ExposureSnapshot {candidate} {snapshot})"
        query = (
            "!(connected-frequency-pattern-miner-range "
            f"&connected_benchmark {min_support} 2 5 \"click\" "
            f"{evidence_k} {length_cost} {schemas} {required} {max_patterns})"
        )
        started = time.perf_counter()
        raw = runtime.process_metta_string(query)
        seconds = time.perf_counter() - started
        text = " ".join(map(str, raw))
        count = text.count("supportOf")
        if count > 1:
            raise RuntimeError(f"{family}: expected at most one structural rule")
        if count:
            tvs = STV.findall(text)
            if len(tvs) != 2 or any(name not in text for name in required[1:-1].split()):
                raise RuntimeError(f"{family}: invalid connected rule")
            positive = tuple(map(float, tvs[0]))
            negative = tuple(map(float, tvs[1]))
            support_match = re.search(r"\)\s+([0-9]+)\)\s*\)?$", text)
            if support_match is None:
                raise RuntimeError(f"{family}: missing support")
            mined[family] = {
                "positive_stv": positive,
                "negative_stv": negative,
                "support": int(support_match.group(1)),
                "source": text,
                "seconds": seconds,
            }
        else:
            mined[family] = {"rejected_by_mdl": True, "seconds": seconds}
    return mined, load_seconds


def compile_rule(family: str, rule: dict) -> str:
    candidate, snapshot, _role = FAMILIES[family]
    strength, confidence = rule["positive_stv"]
    negative_strength, negative_confidence = rule["negative_stv"]
    premise = (
        "(And (Candidate $case $article) "
        f"({candidate} $article $value) "
        "(ExposureSnapshot $case $snapshot) "
        f"({snapshot} $snapshot $value))"
    )
    return (
        f"(: connected_{family} (Implication {premise} "
        f'(RelationalSignal $case "{family}")) '
        f"(CTV (STV {strength} {confidence}) "
        f"(STV {negative_strength} {negative_confidence})))"
    )


def prove(
    facts: set[str], cases: list[list[tuple[str, str]]], rules: dict,
    *, base_rate: float, batch_size: int,
):
    accepted = {name: rule for name, rule in rules.items() if "source" in rule}
    case_index = {
        case: (impression_index, article_id)
        for impression_index, local in enumerate(cases)
        for article_id, case in local
    }
    prefilter = PeTTa()
    prefilter.process_metta_string("!(bind! &connected_eval (new-space))")
    ordered_facts = sorted(facts)
    started = time.perf_counter()
    for offset in range(0, len(ordered_facts), 1000):
        additions = " ".join(
            f"(add-atom &connected_eval {fact})"
            for fact in ordered_facts[offset:offset + 1000]
        )
        prefilter.process_metta_string(f"!(superpose ({additions}))")
    prefilter_load_seconds = time.perf_counter() - started
    matched = {}
    representative = {}
    started = time.perf_counter()
    for family in accepted:
        candidate, snapshot, _role = FAMILIES[family]
        pattern = (
            "(, (Candidate $case $article) "
            f"({candidate} $article $value) "
            "(ExposureSnapshot $case $snapshot) "
            f"({snapshot} $snapshot $value))"
        )
        raw = prefilter.process_metta_string(
            "!(unique-atom (collapse "
            f"(match &connected_eval {pattern} "
            "(Grounding $case $article $snapshot $value))))"
        )
        groundings = re.findall(
            r"\(Grounding\s+(eval_case_[0-9]+_[0-9]+)\s+"
            r"([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\)",
            " ".join(map(str, raw)),
        )
        found = {row[0] for row in groundings}
        if not found <= case_index.keys():
            raise RuntimeError(f"{family}: symbolic prefilter returned unknown cases")
        matched[family] = found
        if groundings:
            representative[family] = sorted(groundings)[0]
    prefilter_seconds = time.perf_counter() - started

    engine = IsolatedPeTTaChainer(startup_timeout_seconds=180)
    timings = {
        "prefilter_fact_load_seconds": prefilter_load_seconds,
        "prefilter_seconds": prefilter_seconds,
    }
    try:
        started = time.perf_counter()
        engine.replace([compile_rule(name, rule) for name, rule in accepted.items()])
        timings["reasoner_start_seconds"] = time.perf_counter() - started
        proof_facts = set()
        for family, (case, article, snapshot_atom, value) in representative.items():
            candidate, snapshot_predicate, _role = FAMILIES[family]
            proof_facts.update((
                f"(Candidate {case} {article})",
                f"({candidate} {article} {value})",
                f"(ExposureSnapshot {case} {snapshot_atom})",
                f"({snapshot_predicate} {snapshot_atom} {value})",
            ))
        named = [
            f"(: connected_eval_fact_{index} {fact} (STV 1.0 1.0))"
            for index, fact in enumerate(sorted(proof_facts))
        ]
        started = time.perf_counter()
        for offset in range(0, len(named), 1000):
            engine.add_atoms_no_check(named[offset:offset + 1000], timeout_sec=180)
        timings["fact_insertion_seconds"] = time.perf_counter() - started

        scores = defaultdict(dict)
        for case, (impression_index, article_id) in case_index.items():
            for family, rule in accepted.items():
                strength, confidence = rule["negative_stv"]
                scores[(impression_index, article_id)][family] = (
                    base_rate + confidence * (strength - base_rate)
                )
        roots = [
            (*case_index[case], case, family)
            for family, (case, _article, _snapshot, _value)
            in representative.items()
        ]
        proof_roots = proof_hits = 0
        started = time.perf_counter()
        for offset in range(0, len(roots), batch_size):
            batch = roots[offset:offset + batch_size]
            queries = [
                f'(: $proof (RelationalSignal {case} "{family}") $tv)'
                for _index, _article, case, family in batch
            ]
            results = engine.query_many(
                queries, steps=12 * len(queries), timeout_sec=180,
            )
            proof_roots += len(batch)
            for (_index, _article, _case, family), proofs in zip(batch, results):
                rule = accepted[family]
                if proofs:
                    values = {tuple(round(value, 12) for value in proof_tv(proof))
                              for proof in proofs}
                    if len(values) != 1:
                        raise RuntimeError(
                            f"{family}: one rule produced inconsistent path STVs"
                        )
                    strength, confidence = next(iter(values))
                    proof_hits += 1
                else:
                    raise RuntimeError(
                        f"{family}: PeTTa-matched root produced no PeTTaChainer proof"
                    )
                posterior = base_rate + confidence * (strength - base_rate)
                for case in matched[family]:
                    index, article = case_index[case]
                    scores[(index, article)][family] = posterior
        timings["proof_query_seconds"] = time.perf_counter() - started
        timings["proof_roots"] = proof_roots
        timings["proof_hits"] = proof_hits
        timings["factorized_matched_groundings"] = sum(map(len, matched.values()))
        timings["factorized_source_facts"] = len(named)
        return scores, timings, tuple(accepted)
    finally:
        engine.close()


def midranks(scores: dict[str, float]) -> dict[str, float]:
    grouped = defaultdict(list)
    for key, value in scores.items():
        grouped[value].append(key)
    result = {}
    offset = 0
    denominator = max(1, len(scores) - 1)
    for value in sorted(grouped):
        keys = grouped[value]
        rank = (offset + (len(keys) - 1) / 2) / denominator
        result.update({key: rank for key in keys})
        offset += len(keys)
    return result


def impression_auc(scores: dict[str, object], positives: set[str]) -> float:
    positive = [scores[key] for key in scores if key in positives]
    negative = [scores[key] for key in scores if key not in positives]
    if not positive or not negative:
        raise ValueError("AUC requires positive and negative candidates")
    return math.fsum(
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positive for right in negative
    ) / (len(positive) * len(negative))


def paired_bootstrap(
    rows: list[dict], challenger: str, *, repetitions: int = 5_000,
    seed: int = 7,
) -> dict:
    """Bootstrap the macro-AUC delta by resampling complete impressions."""
    deltas = [row[challenger] - row["baseline"] for row in rows]
    generator = random.Random(seed)
    count = len(deltas)
    estimates = sorted(
        math.fsum(deltas[generator.randrange(count)] for _ in range(count))
        / count
        for _ in range(repetitions)
    )
    return {
        "repetitions": repetitions,
        "seed": seed,
        "delta_95_percentile_interval": [
            estimates[int(0.025 * repetitions)],
            estimates[int(0.975 * repetitions) - 1],
        ],
    }


def evaluate(
    tests: list[dict], cases: list[list[tuple[str, str]]], scores: dict,
    accepted: tuple[str, ...], baseline_artifact: dict,
):
    auc_rows = baseline_artifact["result"]["auc_per_impression"]
    rankings = baseline_artifact["ranked_impressions"]
    baseline_by_id = {
        str(row["id"]): ranking for row, ranking in zip(auc_rows, rankings)
    }
    if len(tests) != len(cases):
        raise ValueError("baseline and relational evaluation cohorts differ")
    per_impression = []
    for index, (test, local) in enumerate(zip(tests, cases)):
        identity = str(test.get("source_impression_id") or test["id"])
        baseline_rows = baseline_by_id.get(identity)
        if baseline_rows is None:
            raise ValueError("baseline impression is missing")
        candidates = [article for article, _case in local]
        baseline = {
            str(row["article"]): (
                float(row.get("ranking_score", row.get("score", 0.0))),
                float(row.get("pairwise_score", 0.0) or 0.0),
                float(row.get("score", 0.0)),
                float(row.get("stv", {}).get("strength", 0.0)),
                float(row.get("stv", {}).get("confidence", 0.0)),
            )
            for row in baseline_rows
        }
        if set(candidates) != set(baseline):
            raise ValueError("baseline candidate slate differs")
        family_ranks = []
        for family in accepted:
            family_ranks.append(midranks({
                article: scores[(index, article)][family]
                for article in candidates
            }))
        relational = {
            article: math.fsum(rank[article] for rank in family_ranks)
                     / len(family_ranks)
            for article in candidates
        }
        baseline_rank = midranks(baseline)
        relation_rank = midranks(relational)
        balanced = {
            article: (baseline_rank[article] + relation_rank[article]) / 2
            for article in candidates
        }
        tie_break = {
            article: (*baseline[article], relational[article])
            for article in candidates
        }
        positives = set(map(str, test["relevant"]))
        per_impression.append({
            "id": identity,
            "baseline": impression_auc(baseline, positives),
            "relational_only": impression_auc(relational, positives),
            "champion_tie_break": impression_auc(tie_break, positives),
            "balanced_rank_fusion": impression_auc(balanced, positives),
        })
    metrics = {
        name: math.fsum(row[name] for row in per_impression) / len(per_impression)
        for name in (
            "baseline", "relational_only", "champion_tie_break",
            "balanced_rank_fusion",
        )
    }
    metrics["tie_break_delta"] = metrics["champion_tie_break"] - metrics["baseline"]
    metrics["balanced_fusion_delta"] = (
        metrics["balanced_rank_fusion"] - metrics["baseline"]
    )
    metrics["paired_bootstrap"] = {
        name: paired_bootstrap(per_impression, name)
        for name in ("champion_tie_break", "balanced_rank_fusion")
    }
    return metrics, per_impression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=16)
    parser.add_argument("--evidence-k", type=float, default=800.0)
    parser.add_argument("--length-cost", type=float, default=0.5)
    parser.add_argument("--max-patterns", type=int, default=10_000)
    parser.add_argument("--query-batch", type=int, default=512)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")

    started = time.perf_counter()
    path = Path(args.data)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    baseline = json.loads(Path(args.baseline_artifact).read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if baseline.get("dataset_sha256") != digest:
        parser.error("baseline artifact was produced from different dataset bytes")
    training, evaluation, tests, cases = relation_facts(
        data, evaluation_limit=args.eval_limit,
    )
    built_seconds = time.perf_counter() - started
    print(json.dumps({
        "stage": "facts", "training": len(training),
        "evaluation": len(evaluation), "seconds": built_seconds,
    }), flush=True)

    rules, miner_load_seconds = mine_families(
        training, min_support=args.min_support, evidence_k=args.evidence_k,
        length_cost=args.length_cost, max_patterns=args.max_patterns,
    )
    print(json.dumps({"stage": "mined", "rules": rules}), flush=True)
    base_rate = sum(event.get("action") in POSITIVE for event in data["events"]) / len(
        data["events"]
    )
    proof_scores, proof_timings, accepted = prove(
        evaluation, cases, rules, base_rate=base_rate,
        batch_size=args.query_batch,
    )
    metrics, per_impression = evaluate(
        tests, cases, proof_scores, accepted, baseline,
    )
    expected_baseline = float(baseline["result"]["auc_proof_only"])
    if args.eval_limit == 0 and abs(
        metrics["baseline"] - expected_baseline
    ) > 0.00005:
        raise RuntimeError(
            "baseline reconstruction differs from the frozen artifact: "
            f"{metrics['baseline']} != {expected_baseline}"
        )
    artifact = {
        "schema": "connected-miner-mind-evaluation-v1",
        "dataset_sha256": digest,
        "protocol": {
            "target": "click", "min_support": args.min_support,
            "evidence_k": args.evidence_k,
            "length_cost": args.length_cost,
            "length_cost_basis": "BIC: 0.5 * complexity * ln(N + 1)",
            "depth_range": [2, 5],
            "families": list(FAMILIES),
            "aggregation": "equal-family impression midranks",
            "reasoner": "PeTTaChainer",
        },
        "facts": {"training": len(training), "evaluation": len(evaluation)},
        "rules": rules,
        "accepted_families": accepted,
        "metrics": metrics,
        "performance": {
            "data_and_fact_build_seconds": built_seconds,
            "miner_fact_load_seconds": miner_load_seconds,
            "family_mining_seconds": {
                name: rule["seconds"] for name, rule in rules.items()
            },
            **proof_timings,
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
