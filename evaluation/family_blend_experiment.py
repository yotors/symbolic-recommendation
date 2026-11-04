"""Evaluate frozen champion family-rank weights on MIND development slates."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .connected_miner_experiment import PeTTa, impression_auc
from .derived_conjunction_experiment import baseline_scores, paired_interval


BLEND_RE = re.compile(r"\(Blend\s+(item_[0-9]+)\s+([0-9.eE+-]+)\)")


def petta_blend(runtime, candidates, weight):
    output = {}
    for offset in range(0, len(candidates), 1000):
        batch = candidates[offset:offset + 1000]
        expressions = " ".join(
            f"(Blend {case} (family-blend {champion} {structured} {semantic} {weight}))"
            for case, champion, structured, semantic in batch
        )
        raw = runtime.process_metta_string(f"!(collapse (superpose ({expressions})))")
        for case, value in BLEND_RE.findall(" ".join(map(str, raw))):
            output[case] = round(float(value), 8)
    if len(output) != len(candidates):
        raise RuntimeError("PeTTa returned an incomplete family blend")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists")
    started = time.perf_counter()
    data_path = Path(args.data)
    with gzip.open(data_path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    baseline = json.loads(Path(args.baseline_artifact).read_text())
    if baseline.get("dataset_sha256") != hashlib.sha256(data_path.read_bytes()).hexdigest():
        parser.error("baseline dataset fingerprint mismatch")
    by_id = {str(row["id"]): ranking for row, ranking in zip(baseline["result"]["auc_per_impression"], baseline["ranked_impressions"])}
    tests = data["tests"]
    cases = []
    impression_candidates = []
    for test in tests:
        identity = str(test.get("source_impression_id") or test["id"])
        rows = by_id[identity]
        baseline_map = baseline_scores(rows)
        if set(baseline_map) != set(map(str, test["candidates"])):
            raise RuntimeError("champion slate mismatch")
        local = []
        for row in rows:
            family = row.get("pairwise_family_rank_scores", {})
            if set(family) != {"structured_symbolic", "text_semantic"}:
                raise RuntimeError("unexpected champion evidence families")
            case = f"item_{len(cases)}"
            cases.append((case, float(row["ranking_score"]), float(family["structured_symbolic"]), float(family["text_semantic"])))
            local.append((str(row["article"]), case, baseline_map[str(row["article"])]))
        impression_candidates.append(local)
    runtime = PeTTa()
    runtime.load_metta_file(str(Path(__file__).resolve().parents[1] / "miner/familyBlend.metta"))
    weights = [index / 10 for index in range(11)]
    per_weight = {}
    for weight in weights:
        blended = petta_blend(runtime, cases, weight)
        aucs = []
        for test, local in zip(tests, impression_candidates):
            score = {article: (blended[case], *baseline[1:]) for article, case, baseline in local}
            aucs.append(impression_auc(score, set(map(str, test["relevant"]))))
        per_weight[weight] = aucs
    champion = [impression_auc(baseline_scores(by_id[str(test.get("source_impression_id") or test["id"])]), set(map(str, test["relevant"]))) for test in tests]
    reconstructed = math.fsum(per_weight[0.5]) / len(tests)
    champion_auc = math.fsum(champion) / len(tests)
    if abs(champion_auc - float(baseline["result"]["auc_proof_only"])) > 0.00005:
        raise RuntimeError("frozen champion reconstruction failed")
    if abs(reconstructed - champion_auc) > 0.00005:
        raise RuntimeError(f"equal-family blend reconstruction failed: {reconstructed} vs {champion_auc}")
    midpoint = len(tests) // 2
    averages = {str(w): math.fsum(values[:midpoint]) / midpoint for w, values in per_weight.items()}
    selected = max(weights, key=lambda w: (averages[str(w)], -abs(w - 0.5)))
    heldout = per_weight[selected][midpoint:]
    control = champion[midpoint:]
    pairs = [{"control": a, "selected": b} for a, b in zip(control, heldout)]
    artifact = {
        "schema": "family-blend-development-v1",
        "protocol": "Frozen champion proof-family ranks; candidate scores are anchored to stored champion ranks to preserve exact ties despite rounded family ranks. Other weights are approximations at the stored precision. Select weight on first half of previously inspected MIND development impressions in source order and report second half. Source order is not verified chronology; this is not a fresh test.",
        "selection_aucs": averages,
        "selected_structured_weight": selected,
        "heldout": {"impressions": len(heldout), "champion_auc": math.fsum(control) / len(control), "selected_auc": math.fsum(heldout) / len(heldout), "delta": math.fsum(b-a for a,b in zip(control,heldout)) / len(heldout), "paired_impression_bootstrap_95": paired_interval(pairs, "control", "selected")},
        "full_champion_auc": champion_auc,
        "full_equal_blend_auc": reconstructed,
        "full_selected_auc": math.fsum(per_weight[selected]) / len(tests),
        "total_seconds": time.perf_counter() - started,
    }
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact), flush=True)


if __name__ == "__main__":
    main()
