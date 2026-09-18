"""Reproducible real-miner/real-PeTTa evaluation with explicit evidence policy.

Dataset construction is separate so development and a fresh confirmation cohort
can share exactly the same causal training population and frozen lexical model.
Each JSON output includes exact per-impression scores and compiled rule texts.
Strict-symbolic remains the default. Semantic-workspace mode permits a frozen
content encoder only for fact construction, never as an alternative ranker.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

# This command creates its explicit dataset-backed Lab inside ``main``.  Do not
# pay for, mine, or leak the interactive fixture singleton merely by importing
# the module or asking argparse for ``--help``.
os.environ.setdefault("RECOMMENDATION_DISABLE_DEFAULT_LAB", "1")
from ..app.server import Lab


def evaluation_source(data):
    return next((data[key] for key in ("eval_impressions","evaluation","tests","impressions")
                 if isinstance(data.get(key),list)),[])


def stability_audit(result, data, *, seed=37):
    tests={str(t.get("source_impression_id") or t.get("id")):t
           for t in evaluation_source(data)}
    users=defaultdict(list)
    slices=defaultdict(list)
    train_users={e["user"] for e in data["events"]}
    for row in result["auc_per_impression"]:
        test=tests[row["id"]]
        auc=row["auc_proof_only"]
        users[test["user"]].append(auc)
        history=len(test.get("history",[]))
        bucket="cold" if history==0 else "short" if history<=5 else "medium" if history<=20 else "long"
        slices[f"history_{bucket}"].append(auc)
        slices["seen_user" if test["user"] in train_users else "new_user"].append(auc)
        block=min(2,row["index"]*3//max(1,result["cases"]))
        slices[f"source_order_third_{block+1}"].append(auc)
    rng=random.Random(seed)
    user_values=list(users.values())
    means=[]
    if user_values:
        for _ in range(1000):
            sampled=[user_values[rng.randrange(len(user_values))] for _ in user_values]
            means.append(math.fsum(math.fsum(v) for v in sampled)/sum(map(len,sampled)))
    means.sort()
    return {
        "scope":"single-dataset subgroup diagnostics; not evidence of cross-dataset stability",
        "unique_users":len(users),
        "proof_auc_user_cluster_95_ci":([means[24],means[974]] if means else None),
        "slices":{key:{"impressions":len(v),"auc_proof_only":math.fsum(v)/len(v)}
                  for key,v in sorted(slices.items())},
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--evidence-mode",choices=("strict_symbolic","semantic_workspace","legacy_snapshot","llm_workspace"),
                        default="strict_symbolic")
    parser.add_argument("--config",default="{}",help="JSON configuration overrides")
    parser.add_argument("--config-file",help="Frozen configuration or selection/experiment artifact")
    parser.add_argument("--config-overrides",default="{}",help="Explicit JSON overrides on the selected frozen configuration")
    parser.add_argument("--output",required=True)
    parser.add_argument("--eval-limit",type=int,default=0)
    parser.add_argument("--fusion-grid",default="",help="Development only: reuse mined proof snapshot for comma-separated family fusion modes")
    parser.add_argument(
        "--force-cold-proof-cache",action="store_true",
        help=(
            "Clear proof-result caches before the primary benchmark. Fusion "
            "ablations then reuse that exact proof snapshot so only their "
            "post-proof aggregation changes."
        ),
    )
    parser.add_argument(
        "--matched-direct-reasoner-ablation",action="store_true",
        help=(
            "Run the opt-in direct channel reconstructor beside the primary "
            "PeTTaChainer result as an execution-parity diagnostic."
        ),
    )
    parser.add_argument("--capture-rankings",action="store_true",
                        help="Export returned proof-derived candidate ranks for auditable development diagnostics")
    args=parser.parse_args()
    if Path(args.output).exists():
        parser.error("output must be a new artifact; existing experiments are not overwritten")
    path=Path(args.data)
    with (gzip.open(path,"rt") if path.suffix==".gz" else path.open()) as stream:
        data=json.load(stream)
    semantic=args.evidence_mode=="semantic_workspace"
    legacy=args.evidence_mode=="legacy_snapshot"
    llm=args.evidence_mode=="llm_workspace"
    if llm and not data.get("metadata",{}).get("llm_workspace"):
        parser.error("llm_workspace requires a prepared llm_data snapshot")
    if semantic and (not data.get("semantic_workspace_model")
                     or not data.get("metadata",{}).get("semantic_workspace")):
        parser.error("semantic_workspace requires a prepared semantic_data snapshot")
    config=json.loads(args.config)
    if args.config_file:
        if config:
            parser.error("use either --config or --config-file")
        saved=json.loads(Path(args.config_file).read_text())
        config=saved.get("result",saved).get("config",saved)
    overrides=json.loads(args.config_overrides)
    if not isinstance(config,dict) or not isinstance(overrides,dict):
        parser.error("configuration and overrides must be JSON objects")
    config={**config,**overrides}
    print(json.dumps({"stage":"start","evidence_mode":args.evidence_mode,
                      "config":config,"training":len(data["events"]),
                      "evaluation":len(evaluation_source(data))}),flush=True)
    start=time.monotonic()
    lab=Lab(data,symbolic_only=not (semantic or legacy or llm),config=config)
    captured=[]
    if args.capture_rankings:
        original_rank=lab._pairwise_rank
        def capture_rank(*rank_args,**rank_kwargs):
            rows=original_rank(*rank_args,**rank_kwargs)
            captured.append([
                {"article":row["article"]["id"],
                 **{key:row[key] for key in (
                     "ranking_score","score","stv","pairwise_score",
                     "pairwise_rank_score","pairwise_margin_score",
                     "pointwise_rank_score","pairwise_family_rank_scores",
                     "relational_evidence",
                 ) if key in row}}
                 | {"point_rule_ids": sorted(
                     str(rule["id"]) for rule in row.get("rules", ())
                     if isinstance(rule, dict) and rule.get("id") is not None
                 )}
                for row in rows
            ])
            return rows
        lab._pairwise_rank=capture_rank
    try:
        print(json.dumps({"stage":"mined","seconds":time.monotonic()-start,
                          "point_rules":len(lab.mined_rules),"pair_rules":len(lab.pair_rules),
                          "premises":[r["premises"] for r in lab.pair_rules]}),flush=True)
        result=lab.benchmark({
            "remine":False,
            "eval_case_limit":args.eval_limit,
            "force_cold_proof_cache":args.force_cold_proof_cache,
            "matched_direct_reasoner_ablation":(
                args.matched_direct_reasoner_ablation
            ),
        })
        artifact={
            "policy":("llm_workspace; frozen article annotations supply observations only; actual miner and PeTTa proofs rank"
                      if llm else
                      "semantic_workspace; frozen content embeddings supply facts only; actual miner and PeTTa proofs rank"
                      if semantic else
                      "legacy_snapshot; preserve historical content/entity observations; actual miner and PeTTa proofs rank"
                      if legacy else "symbolic_only; no NN, embeddings, source entity annotations or NL2PLN"),
            "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "dataset_metadata":data.get("metadata",{}),
            "result":result,
            "stability":stability_audit(result,data),
            "point_rules":lab.mined_rules,
            "pair_rules":lab.pair_rules,
            "compiled_point_channels":lab._point_channel_sources,
            "compiled_pair_rules":lab._pair_rule_sources,
            "total_seconds":time.monotonic()-start,
            "python":sys.version,
        }
        if args.capture_rankings:
            artifact["ranked_impressions"]=list(captured)
        ablations=[]
        for fusion in filter(None,args.fusion_grid.split(",")):
            if fusion==result["config"]["pair_family_fusion"]:
                continue
            trial=lab.benchmark({"remine":False,"eval_case_limit":args.eval_limit,
                                 "pair_family_fusion":fusion})
            ablations.append({"result":trial,"stability":stability_audit(trial,data)})
            print(json.dumps({"stage":"fusion_ablation","fusion":fusion,
                              "auc":trial["auc"],"auc_proof_only":trial["auc_proof_only"]}),flush=True)
        artifact["ranking_ablations"]=ablations
        target=Path(args.output)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=target.parent,
                                             prefix=f".{target.name}.",delete=False) as stream:
                temporary=Path(stream.name)
                json.dump(artifact,stream,indent=2,default=str,allow_nan=False)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary,target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(json.dumps({"stage":"complete","output":str(target),
                          **{k:result[k] for k in ("auc","auc_proof_only","auc_95_ci",
                             "mrr","ndcg_at_5","ndcg_at_10","seconds","candidates")},
                          "stability":artifact["stability"]}),flush=True)
    finally:
        lab.engine.close()


if __name__=="__main__":
    main()
