"""Run one fpMiner/PeTTaChainer configuration on a frozen training-only split.

This runner is intentionally incapable of reading the public development
impressions. It converts the chronological tail of the training log into
complete validation slates and logs macro impression AUC for autoresearch.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import time


for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "2")
os.environ.setdefault("RECOMMENDATION_DISABLE_DEFAULT_LAB", "1")


def _load_json(path: Path) -> dict:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact_config(path: Path) -> dict:
    saved = _load_json(path)
    config = saved.get("result", saved).get("config", saved)
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a model configuration")
    return dict(config)


def _publish_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, default=str, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--overrides", default="{}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        parser.error("--output must name a new artifact")
    try:
        overrides = json.loads(args.overrides)
    except json.JSONDecodeError as exc:
        parser.error(f"--overrides is not valid JSON: {exc}")
    if not isinstance(overrides, dict):
        parser.error("--overrides must be a JSON object")
    if not 0.0 < args.build_fraction < 1.0:
        parser.error("--build-fraction must be between zero and one")

    random.seed(args.seed)
    data_path = Path(args.data)
    data = _load_json(data_path)
    base_config = _artifact_config(Path(args.base_config))
    config = {**base_config, **overrides, "random_seed": args.seed}

    from ..app.server import CONTEXT_FEATURES, POSITIVE, Lab
    from .training_gate import _validated_records, prepare_training_confirmation

    split = prepare_training_confirmation(
        data,
        context_features=CONTEXT_FEATURES,
        positive_actions=POSITIVE,
        build_fraction=args.build_fraction,
    )
    print(json.dumps({
        "stage": "start",
        "metric": "val_auc",
        "split": split.audit,
        "overrides": overrides,
    }), flush=True)

    started = time.monotonic()
    lab = None
    tracker = None
    try:
        try:
            import openscience_track as tracker
        except ImportError:
            tracker = None
        if tracker is not None:
            tracker.init()

        lab = Lab(data=split.data, symbolic_only=False, config=config)
        result = lab.benchmark({
            "remine": False,
            "max_candidates": 0,
            "eval_case_limit": 0,
            "reasoner_parity_mode": "off",
        })
        _validated_records(result, split.expected, "autoresearch validation")
        metrics = {
            "val_auc": float(result["auc"]),
            "val_auc_proof_only": float(result["auc_proof_only"]),
            "val_mrr": float(result["mrr"]),
            "val_ndcg_at_5": float(result["ndcg_at_5"]),
            "val_ndcg_at_10": float(result["ndcg_at_10"]),
            "proof_coverage": float(result["proof_coverage"]),
            "pairwise_proof_coverage": float(result["pairwise_proof_coverage"]),
        }
        artifact = {
            "schema": "recommendation-autoresearch-training-tail-v1",
            "policy": (
                "LLM article annotations are frozen content observations only; "
                "fpMiner rules and PeTTaChainer proofs determine every rank"
            ),
            "public_dev_evaluated": False,
            "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "base_config": str(Path(args.base_config)),
            "overrides": overrides,
            "split": split.audit,
            "metrics": metrics,
            "result": result,
            "point_rules": lab.mined_rules,
            "pair_rules": lab.pair_rules,
            "startup_point_mining": lab.last_mining,
            "startup_pair_mining": lab.last_pair_mining,
            "total_seconds": time.monotonic() - started,
        }
        _publish_new(output_path, artifact)
        if tracker is not None:
            tracker.log(metrics, step=1)
            for key, value in metrics.items():
                tracker.summary[key] = value
        print(json.dumps({
            "stage": "complete", "output": str(output_path), **metrics,
            "seconds": time.monotonic() - started,
        }), flush=True)
    finally:
        if lab is not None and lab.engine is not None:
            lab.engine.close()
        if tracker is not None:
            tracker.finish()


if __name__ == "__main__":
    main()
