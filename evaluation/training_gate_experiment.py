"""Run one immutable train-prefix/train-tail architecture decision.

This runner never evaluates the dataset's public development/test slates.  It
exists so the statistically guarded training confirmation can be reproduced
without starting the browser server or enabling its administrative endpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
import tempfile
import time

# The experiment constructs its explicit full-data Lab below. Avoid paying for
# the convenience fixture singleton that interactive imports use.
os.environ.setdefault("RECOMMENDATION_DISABLE_DEFAULT_LAB", "1")
from ..app.server import CONTEXT_FEATURES, POSITIVE, Lab
from .training_gate import (
    PUBLIC_BUILD_FRACTION,
    PUBLIC_MIN_CONFIRMATION_IMPRESSIONS,
    PUBLIC_MIN_CONFIRMATION_USERS,
    prepare_training_confirmation,
)
from ..paths import GATE_CLAIM_DIR, WORKSPACE_ROOT


_ATTEMPT_SCHEMA = "recommendation-training-confirmation-attempt-v2"
_COHORT_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINAL_STATUSES = frozenset({"pass", "hold", "error"})
_ATTEMPT_ORIGINS = frozenset({"pre_run", "backfill"})
_REGISTRY_DIRECTORY = GATE_CLAIM_DIR

_AUDITED_SOURCE_PATHS = (
    "recommendation/mining/conditional_llm_mining.py",
    "recommendation/core/ctv_calibration.py",
    "recommendation/integrations/engine.py",
    "recommendation/app/server.py",
    "recommendation/features/lexical_workspace.py",
    "recommendation/features/llm_workspace.py",
    "recommendation/adapters/mind.py",
    "recommendation/core/multi_interest.py",
    "recommendation/features/recency_workspace.py",
    "recommendation/features/semantic_workspace.py",
    "recommendation/core/symbolic.py",
    "recommendation/pipelines/symbolic_data.py",
    "recommendation/mining/target_miner.py",
    "recommendation/features/text_embeddings.py",
    "recommendation/evaluation/training_gate.py",
    "recommendation/paths.py",
    "recommendation/miner/fpMiner.metta",
    "PeTTa/python/petta/__init__.py",
    "PeTTaChainer/pettachainer/pettachainer.py",
    "PeTTaChainer/pettachainer/metta/petta_chainer.metta",
)


def _load_json(path: Path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def _artifact_config(path: Path) -> dict:
    payload = _load_json(path)
    config = payload.get("result", payload).get("config", payload)
    if not isinstance(config, dict):
        raise ValueError("champion config artifact does not contain a config object")
    return config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attempt_path(registry_directory: Path, cohort_fingerprint: str) -> Path:
    if not _COHORT_FINGERPRINT.fullmatch(cohort_fingerprint):
        raise ValueError("cohort fingerprint must be 64 lowercase hexadecimal characters")
    return registry_directory / f"{cohort_fingerprint}.json"


def _prepare_registry_directory(registry_directory: Path) -> Path:
    registry_directory = Path(registry_directory)
    if registry_directory.is_symlink():
        raise RuntimeError("durable attempt registry must not be a symbolic link")
    try:
        registry_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError("cannot create durable attempt registry") from exc
    if not registry_directory.is_dir():
        raise RuntimeError("durable attempt registry is not a directory")
    return registry_directory


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _locked_registry(registry_directory: Path):
    registry_directory = _prepare_registry_directory(registry_directory)
    lock_path = registry_directory / ".lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield registry_directory
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _record_bytes(payload: dict) -> bytes:
    return (json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while recording durable attempt")
        offset += written


def _read_attempt(path: Path, cohort_fingerprint: str) -> dict:
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise OSError("attempt record is not a regular file")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "durable attempt registry record is corrupt; the cohort remains consumed"
        ) from exc
    expected_keys = {
        "schema", "cohort_fingerprint", "status", "origin", "artifact_sha256",
        "claimed_at", "finalized_at",
    }
    status = payload.get("status") if isinstance(payload, dict) else None
    artifact_sha256 = (
        payload.get("artifact_sha256") if isinstance(payload, dict) else None
    )
    if (not isinstance(payload, dict)
            or set(payload) != expected_keys
            or payload.get("schema") != _ATTEMPT_SCHEMA
            or payload.get("cohort_fingerprint") != cohort_fingerprint
            or status not in ({"running"} | _FINAL_STATUSES)
            or payload.get("origin") not in _ATTEMPT_ORIGINS
            or not isinstance(payload.get("claimed_at"), str)
            or not payload["claimed_at"]
            or (status == "running" and payload.get("finalized_at") is not None)
            or (status != "running"
                and (not isinstance(payload.get("finalized_at"), str)
                     or not payload["finalized_at"]))
            or (status in {"running", "error"} and artifact_sha256 is not None)
            or (status in {"pass", "hold"}
                and (not isinstance(artifact_sha256, str)
                     or not _ARTIFACT_SHA256.fullmatch(artifact_sha256)))):
        raise RuntimeError(
            "durable attempt registry record is corrupt; the cohort remains consumed"
        )
    return payload


def _claim_attempt_record(
    registry_directory: Path, cohort_fingerprint: str, *, origin: str,
) -> dict:
    """Atomically consume a cohort before mining begins.

    A process killed after this function leaves either a valid ``running``
    record or a corrupt-but-present record. Both states permanently fail
    closed for that cohort.
    """

    if origin not in _ATTEMPT_ORIGINS:
        raise ValueError("durable attempt origin must be pre_run or backfill")
    with _locked_registry(registry_directory) as directory:
        path = _attempt_path(directory, cohort_fingerprint)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        payload = {
            "schema": _ATTEMPT_SCHEMA,
            "cohort_fingerprint": cohort_fingerprint,
            "status": "running",
            "origin": origin,
            "artifact_sha256": None,
            "claimed_at": _utc_now(),
            "finalized_at": None,
        }
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            # Validate for auditability, but never recycle an existing record.
            existing = _read_attempt(path, cohort_fingerprint)
            raise ValueError(
                "this immutable training confirmation cohort is already consumed "
                f"(durable status: {existing['status']})"
            ) from None
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, _record_bytes(payload))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(directory)
        return payload


def _claim_attempt(registry_directory: Path, cohort_fingerprint: str) -> dict:
    """Claim a live attempt; callers cannot relabel it as a backfill."""
    return _claim_attempt_record(
        registry_directory, cohort_fingerprint, origin="pre_run"
    )


def _finalize_attempt(
    registry_directory: Path,
    cohort_fingerprint: str,
    status: str,
    *,
    artifact_sha256: str | None = None,
) -> dict:
    if status not in _FINAL_STATUSES:
        raise ValueError("durable attempt status must be pass, hold, or error")
    if status == "error":
        if artifact_sha256 is not None:
            raise ValueError("an error attempt must not bind an artifact hash")
    elif (not isinstance(artifact_sha256, str)
          or not _ARTIFACT_SHA256.fullmatch(artifact_sha256)):
        raise ValueError("a pass or hold attempt must bind a SHA-256 artifact hash")
    with _locked_registry(registry_directory) as directory:
        path = _attempt_path(directory, cohort_fingerprint)
        if not path.exists():
            raise RuntimeError(
                "durable attempt registry lost its claim; refusing to finalize"
            )
        current = _read_attempt(path, cohort_fingerprint)
        if current["status"] != "running":
            if (current["status"] == status
                    and current["artifact_sha256"] == artifact_sha256):
                return current
            raise RuntimeError("durable attempt registry record is already finalized")
        final = {
            **current,
            "status": status,
            "artifact_sha256": artifact_sha256,
            "finalized_at": _utc_now(),
        }
        temporary = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=directory, prefix=f".{cohort_fingerprint}.", suffix=".tmp",
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                _write_all(descriptor, _record_bytes(final))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, path)
            temporary = None
            _fsync_directory(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return final


def _preflight_new_output(path: Path) -> None:
    if os.path.lexists(path):
        raise ValueError("output must be a new artifact; existing results are immutable")


def _publish_new(path: Path, payload: dict) -> str:
    _preflight_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, default=str, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        artifact_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
        # Hard-link publication fails instead of replacing an artifact that
        # appeared after the initial existence check.
        os.link(temporary, path)
        _fsync_directory(path.parent)
        return artifact_sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _runtime_provenance() -> dict:
    """Record selected core-source and runtime identities without secrets."""

    project_root = WORKSPACE_ROOT
    source_sha256 = {}
    for relative in _AUDITED_SOURCE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise RuntimeError(f"audited runtime source is missing: {relative}")
        source_sha256[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for distribution in ("janus-swi", "numpy"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "schema": "recommendation-training-runtime-provenance-v1",
        "selected_core_source_sha256": source_sha256,
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_full_version": sys.version,
            "python_cache_tag": sys.implementation.cache_tag,
            "operating_system": platform.system(),
            "machine": platform.machine(),
            "packages": packages,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--champion-config", required=True)
    parser.add_argument("--challenger-config", required=True,
                        help="JSON object containing only declared architecture overrides")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--evidence-mode",
        choices=("strict_symbolic", "semantic_workspace", "legacy_snapshot", "llm_workspace"),
        default="strict_symbolic",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    _preflight_new_output(output_path)
    data = _load_json(data_path)
    challenger = json.loads(args.challenger_config)
    if not isinstance(challenger, dict) or not challenger:
        parser.error("--challenger-config must be a non-empty JSON object")
    if args.evidence_mode == "llm_workspace" and not (
            data.get("metadata", {}).get("llm_workspace")):
        parser.error("llm_workspace requires a prepared LLM projection")
    if args.evidence_mode == "semantic_workspace" and not (
            data.get("metadata", {}).get("semantic_workspace")):
        parser.error("semantic_workspace requires a prepared semantic projection")
    symbolic_only = args.evidence_mode == "strict_symbolic"
    champion_config = _artifact_config(Path(args.champion_config))
    gate_data = data
    if symbolic_only:
        # Match Lab.__init__ exactly without constructing its PeTTa worker
        # before the durable claim.
        from ..pipelines.symbolic_data import strip_neural_evidence
        gate_data = strip_neural_evidence(data)
    split = prepare_training_confirmation(
        gate_data,
        context_features=CONTEXT_FEATURES,
        positive_actions=POSITIVE,
        build_fraction=PUBLIC_BUILD_FRACTION,
    )
    if len(split.expected) < PUBLIC_MIN_CONFIRMATION_IMPRESSIONS:
        parser.error(
            "training confirmation needs at least "
            f"{PUBLIC_MIN_CONFIRMATION_IMPRESSIONS} eligible tail impressions; "
            f"found {len(split.expected)}"
        )
    if split.audit["confirmation_users"] < PUBLIC_MIN_CONFIRMATION_USERS:
        parser.error(
            "training confirmation needs at least "
            f"{PUBLIC_MIN_CONFIRMATION_USERS} distinct tail users"
        )
    cohort_fingerprint = split.audit["cohort_fingerprint"]
    _claim_attempt(_REGISTRY_DIRECTORY, cohort_fingerprint)
    print(json.dumps({
        "stage": "start",
        "policy": "one immutable chronological training confirmation",
        "events": len(data.get("events", ())),
        "cohort_fingerprint": cohort_fingerprint,
        "challenger_overrides": challenger,
    }), flush=True)

    started = time.monotonic()
    lab = None
    finalized = False
    try:
        lab = Lab(data=data, symbolic_only=symbolic_only, config=champion_config)
        result = lab.training_confirmation({
            "challenger_config": challenger,
            "min_delta": args.min_delta,
            "require_proof_noninferiority": True,
            "promote": False,
        })
        if result.get("cohort_fingerprint") != cohort_fingerprint:
            raise RuntimeError(
                "durable claim and training-gate cohort fingerprints diverged"
            )
        artifact = {
            "schema": "recommendation-training-confirmation-artifact-v2",
            "data_path": str(data_path.resolve()),
            "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "public_dev_evaluated": False,
            "result": result,
            "total_seconds": time.monotonic() - started,
            "python": sys.version,
            "runtime_provenance": _runtime_provenance(),
        }
        artifact_sha256 = _publish_new(output_path, artifact)
        _finalize_attempt(
            _REGISTRY_DIRECTORY,
            cohort_fingerprint,
            result["status"],
            artifact_sha256=artifact_sha256,
        )
        finalized = True
        print(json.dumps({
            "stage": "complete",
            "output": str(output_path),
            "status": result["status"],
            "decision": result["decision"],
            "mean_auc_delta": result["mean_auc_delta"],
            "delta_95_ci": result["delta_95_ci"],
            "mean_proof_only_auc_delta": result["mean_proof_only_auc_delta"],
            "proof_only_delta_95_ci": result["proof_only_delta_95_ci"],
            "seconds": time.monotonic() - started,
        }), flush=True)
    except BaseException:
        if not finalized:
            try:
                _finalize_attempt(_REGISTRY_DIRECTORY, cohort_fingerprint, "error")
            except BaseException:
                # Never delete or recycle the original running claim. A
                # corrupt/unfinalized record remains a consumed cohort.
                pass
        raise
    finally:
        if lab is not None and lab.engine is not None:
            lab.engine.close()


if __name__ == "__main__":
    main()
