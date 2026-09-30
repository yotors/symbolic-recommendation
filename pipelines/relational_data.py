"""Project exact causal entity/concept continuity proofs into an immutable replay.

The input must already contain an explicit oldest-to-newest history on every
training exposure and evaluation impression.  Engagement outcomes are
preserved byte-for-byte in the copy but are never passed to the relational
feature builder or PeTTaChainer. PeTTaChainer composes two structural rules
for exact entity continuity and three for canonical-concept continuity (the
extra rule binds the source annotation to its canonical ID); the resulting
candidate observation can subsequently be offered to fpMiner. Every
label-free expected origin/value path is submitted as a separate proof root;
a partial wildcard result is never accepted as complete candidate evidence.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback

from ..features.relational_workspace import (
    CANONICAL_CONCEPT_BRIDGE_RULE_ID,
    ENTITY_PATH_CASE_NODE_RULE_ID,
    ENTITY_PATH_COUNT_RULE_ID,
    ENTITY_PATH_NONE_RULE_ID,
    ENTITY_PATH_ONE_RULE_ID,
    ENTITY_PATH_ORIGIN_NODE_RULE_ID,
    ENTITY_PATH_TWO_PLUS_RULE_ID,
    RELATIONAL_PROJECTION_SCHEMA,
    RELATIONAL_STRUCTURAL_RULES,
    RELATIONAL_WORKSPACE_FEATURES,
    RELATIONAL_WORKSPACE_SCHEMA,
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
    REL_ENTITY_PATH_MULTIPLICITY,
    REL_ENTITY_PATH_MULTIPLICITY_PROOF_IDS,
    build_relational_plans,
    relational_context_observations_sha256,
    reduce_concept_relational_proofs,
    reduce_entity_path_multiplicity_proofs,
    reduce_relational_proofs,
    relational_safety_audit,
    validate_relational_projection,
)
from .recency_data import _evaluation, _history, _read
from ..paths import WORKSPACE_ROOT


# FoldAll must enumerate the complete origin subgraph before its categorical
# result is meaningful.  Give every categorical query its own fixed expansion
# budget; sharing a query_many budget lets one candidate starve another.
ENTITY_PATH_MULTIPLICITY_QUERY_STEPS = 2_000


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _new_reasoner():
    # Match the active lab's runtime isolation: the legacy source checkout is
    # incompatible with the installed PeTTaChainer runtime and must not shadow
    # the virtualenv package when callers still export the historical
    # ``PYTHONPATH=PeTTa/python`` command.
    legacy = (WORKSPACE_ROOT / "PeTTa" / "python").resolve()
    sys.path[:] = [
        entry for entry in sys.path
        if not entry or Path(entry).resolve() != legacy
    ]
    chainer_root = str(WORKSPACE_ROOT / "PeTTaChainer")
    if chainer_root not in sys.path:
        sys.path.insert(0, chainer_root)
    try:
        from pettachainer import PeTTaChainer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "PeTTaChainer is required; use the PeTTaChainer virtual environment"
        ) from exc
    return PeTTaChainer()


def _isolated_query_workspace_process(connection, statements, requests):
    """Execute arbitrary proof requests on one short-lived shared engine."""

    try:
        engine = _new_reasoner()
        engine.add_atoms_no_check(list(statements))
        results = [
            engine.query_many(list(queries), steps=steps, timeout_sec=0)
            for queries, steps in requests
        ]
        connection.send(("ok", results))
    except BaseException as exc:  # pragma: no cover - exercised by parent
        connection.send((
            "error", exc.__class__.__name__, str(exc), traceback.format_exc(),
        ))
    finally:
        connection.close()


def _run_isolated_query_workspace(statements, requests):
    """Run real PeTTa requests in one spawned workspace, then retire Janus.

    Integration tests use this seam when several queries must share one
    AtomSpace.  The spawned process prevents PeTTa's process-global runtime
    from colliding with a server worker imported earlier in the test process.
    """

    normalized_statements = tuple(str(statement) for statement in statements)
    normalized_requests = tuple(
        (tuple(str(query) for query in queries), _positive_integer(steps, "steps"))
        for queries, steps in requests
    )
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_query_workspace_process,
        args=(child, normalized_statements, normalized_requests),
        name="relational-petta-one-shot-workspace", daemon=False,
    )
    try:
        process.start()
        child.close()
        try:
            payload = parent.recv()
        except EOFError as exc:
            raise RuntimeError(
                "PeTTa one-shot workspace exited without returning a result"
            ) from exc
    finally:
        parent.close()
        process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            "PeTTa one-shot workspace exited unexpectedly "
            f"({process.exitcode})"
        )
    if not payload or payload[0] != "ok":
        _status, error_type, message, error_traceback = payload
        raise RuntimeError(
            f"PeTTa one-shot workspace failed [{error_type}]: "
            f"{message}\n{error_traceback}"
        )
    return payload[1]


def _article_mapping(data):
    raw = data.get("articles")
    if not isinstance(raw, list):
        raise ValueError("dataset articles must be a list")
    articles = {}
    for index, article in enumerate(raw):
        if not isinstance(article, Mapping):
            raise ValueError(f"article {index} must be a mapping")
        identifier = article.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"article {index} requires a nonempty ID")
        if identifier in articles:
            raise ValueError(f"duplicate article ID: {identifier}")
        articles[identifier] = article
    return articles


def _contexts(data, evaluation_key):
    """Yield (context, candidate, history, user, split) without any labels."""

    impression_histories = {}
    events = data.get("events")
    if not isinstance(events, list):
        raise ValueError("dataset events must be a list")
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"training row {index} must be a mapping")
        candidate, user = event.get("article"), event.get("user")
        if not isinstance(candidate, str) or not candidate or not isinstance(user, str) or not user:
            raise ValueError(f"training row {index} requires user and article IDs")
        history = _history(event, f"training row {index}")
        identity = (user, event.get("impression"))
        if identity in impression_histories and impression_histories[identity] != history:
            raise ValueError("one training impression cannot contain different preceding histories")
        impression_histories[identity] = history
        yield event, candidate, history, user, "training"

    for index, case in enumerate(data[evaluation_key]):
        if not isinstance(case, Mapping):
            raise ValueError(f"evaluation impression {index} must be a mapping")
        user = case.get("user")
        history = _history(case, f"evaluation impression {index}")
        candidates = case.get("candidates")
        contexts = case.get("candidate_context")
        if not isinstance(user, str) or not user or not isinstance(candidates, list):
            raise ValueError(f"evaluation impression {index} requires user and candidates")
        if not isinstance(contexts, Mapping) or any(
            not isinstance(contexts.get(str(candidate)), Mapping) for candidate in candidates
        ):
            raise ValueError("each evaluation candidate requires its original recorded context")
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate:
                raise ValueError("evaluation candidate IDs must be nonempty strings")
            yield contexts[str(candidate)], candidate, history, user, "evaluation"


_STATEMENT_NAME = re.compile(r"^\s*\(:\s+([A-Za-z][A-Za-z0-9_]*)\s")


def _require_unique_statement_names(statements):
    """Reject a named statement that has two meanings before partitioning."""

    by_name = {}
    for statement in statements:
        match = _STATEMENT_NAME.match(statement)
        if match is None:
            raise ValueError("relational workspace contains an unnamed statement")
        previous = by_name.setdefault(match.group(1), statement)
        if previous != statement:
            raise ValueError(
                f"relational statement name collision: {match.group(1)}"
            )


def _root_batch_shards(indexed_roots, query_batch_size, shard_root_size):
    """Group unchanged query batches into bounded isolated workspaces."""

    query_batches = tuple(
        tuple(indexed_roots[start:start + query_batch_size])
        for start in range(0, len(indexed_roots), query_batch_size)
    )
    batches_per_shard = max(1, shard_root_size // query_batch_size)
    return tuple(
        tuple(item for batch in query_batches[start:start + batches_per_shard]
              for item in batch)
        for start in range(0, len(query_batches), batches_per_shard)
    )


def _reasoner_query_batches(indexed_roots, query_batch_size):
    """Batch exact roots while isolating every FoldAll category root."""

    pending_exact = []
    for item in indexed_roots:
        kind = item[3]
        if kind == "entity_path_multiplicity":
            if pending_exact:
                yield tuple(pending_exact)
                pending_exact.clear()
            yield (item,)
            continue
        pending_exact.append(item)
        if len(pending_exact) == query_batch_size:
            yield tuple(pending_exact)
            pending_exact.clear()
    if pending_exact:
        yield tuple(pending_exact)


def _run_reasoner_shard(
    indexed_roots, engine, *, add_batch_size, query_batch_size, query_steps,
):
    """Execute one root-batch-preserving proof shard."""

    statements = set(RELATIONAL_STRUCTURAL_RULES)
    for _root_index, plan, _query, _kind, _diagnostic in indexed_roots:
        statements.update(plan.statements)
    ordered_statements = sorted(statements)
    ordered_source_statements = sorted(
        statements - set(RELATIONAL_STRUCTURAL_RULES)
    )

    insertion_started = time.perf_counter()
    for start in range(0, len(ordered_statements), add_batch_size):
        engine.add_atoms_no_check(
            ordered_statements[start:start + add_batch_size]
        )
    insertion_seconds = time.perf_counter() - insertion_started

    roots = list(indexed_roots)
    root_results = []
    query_batches = 0
    exact_query_calls = 0
    multiplicity_query_calls = 0
    multiplicity_no_path_query_calls = 0
    multiplicity_positive_path_query_calls = 0
    proof_rows_returned = 0
    query_seconds = 0.0
    exact_query_seconds = 0.0
    multiplicity_query_seconds = 0.0
    multiplicity_no_path_query_seconds = 0.0
    multiplicity_positive_path_query_seconds = 0.0
    for batch in _reasoner_query_batches(roots, query_batch_size):
        query_batches += 1
        category = batch[0][3] == "entity_path_multiplicity"
        if category and len(batch) != 1:
            raise RuntimeError(
                "entity-path multiplicity queries must execute independently"
            )
        steps = (
            ENTITY_PATH_MULTIPLICITY_QUERY_STEPS
            if category else query_steps * len(batch)
        )
        query_started = time.perf_counter()
        results = engine.query_many(
            [query for _root_index, _plan, query, _kind, _diagnostic in batch],
            steps=steps, timeout_sec=0,
        )
        elapsed = time.perf_counter() - query_started
        query_seconds += elapsed
        if category:
            multiplicity_query_calls += 1
            multiplicity_query_seconds += elapsed
            plan = batch[0][1]
            if plan.proof_roots:
                multiplicity_positive_path_query_calls += 1
                multiplicity_positive_path_query_seconds += elapsed
            else:
                multiplicity_no_path_query_calls += 1
                multiplicity_no_path_query_seconds += elapsed
        else:
            exact_query_calls += 1
            exact_query_seconds += elapsed
        if not isinstance(results, Sequence) or len(results) != len(batch):
            raise RuntimeError("PeTTaChainer returned an invalid query batch")
        for (root_index, _plan, _query, _kind, diagnostic), proofs in zip(
            batch, results,
        ):
            if not isinstance(proofs, Sequence) or isinstance(proofs, (str, bytes)):
                raise RuntimeError("PeTTaChainer returned an invalid proof result")
            if not proofs:
                raise RuntimeError(
                    "a required relational proof root returned no PeTTaChainer proof: "
                    f"{diagnostic}"
                )
            root_results.append((root_index, list(proofs)))
            proof_rows_returned += len(proofs)
    return {
        "root_results": root_results,
        "plan_count": len({plan.case_id for _index, plan, *_rest in roots}),
        "proof_roots": len(roots),
        "query_batches": query_batches,
        "exact_query_calls": exact_query_calls,
        "entity_path_multiplicity_query_calls": multiplicity_query_calls,
        "entity_path_multiplicity_no_path_query_calls": (
            multiplicity_no_path_query_calls
        ),
        "entity_path_multiplicity_positive_path_query_calls": (
            multiplicity_positive_path_query_calls
        ),
        "proof_rows_returned": proof_rows_returned,
        "source_statements": len(ordered_source_statements),
        "atomspace_statements": len(ordered_statements),
        "source_statements_sha256": hashlib.sha256(json.dumps(
            ordered_source_statements, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "atomspace_statements_sha256": hashlib.sha256(json.dumps(
            ordered_statements, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "insertion_seconds": insertion_seconds,
        "query_seconds": query_seconds,
        "exact_query_seconds": exact_query_seconds,
        "entity_path_multiplicity_query_seconds": multiplicity_query_seconds,
        "entity_path_multiplicity_no_path_query_seconds": (
            multiplicity_no_path_query_seconds
        ),
        "entity_path_multiplicity_positive_path_query_seconds": (
            multiplicity_positive_path_query_seconds
        ),
        "reasoner": f"{type(engine).__module__}.{type(engine).__qualname__}",
    }


def _reasoner_shard_process(
    connection, indexed_roots, add_batch_size, query_batch_size, query_steps,
):
    """Run one PeTTa workspace in a process whose exit releases all state."""

    try:
        result = _run_reasoner_shard(
            indexed_roots, _new_reasoner(), add_batch_size=add_batch_size,
            query_batch_size=query_batch_size, query_steps=query_steps,
        )
        connection.send(("ok", result))
    except BaseException as exc:  # pragma: no cover - exercised via parent
        connection.send((
            "error", exc.__class__.__name__, str(exc), traceback.format_exc(),
        ))
    finally:
        connection.close()


def _run_isolated_reasoner_shard(
    indexed_roots, *, add_batch_size, query_batch_size, query_steps,
    checkpoint_dir=None, checkpoint_identity=None,
):
    """Spawn, collect, and fully retire one PeTTa shard."""

    return _run_isolated_reasoner_shards(
        (indexed_roots,), workers=1, add_batch_size=add_batch_size,
        query_batch_size=query_batch_size, query_steps=query_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_identity=checkpoint_identity,
    )[0]


def _shard_checkpoint_path(checkpoint_dir, checkpoint_identity, shard_index, shard):
    roots = [
        {
            "root_index": root_index,
            "case_id": plan.case_id,
            "query": query,
            "kind": kind,
            "diagnostic": diagnostic,
        }
        for root_index, plan, query, kind, diagnostic in shard
    ]
    payload = {
        "schema": "recommendation-relational-shard-checkpoint-v1",
        "identity": checkpoint_identity,
        "shard_index": shard_index,
        "roots": roots,
    }
    digest = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return checkpoint_dir / f"shard_{shard_index:06d}_{digest}.json", digest


def _write_shard_checkpoint(path, digest, shard_index, result):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump({
                "schema": "recommendation-relational-shard-checkpoint-v1",
                "digest": digest,
                "shard_index": shard_index,
                "result": result,
            }, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_shard_checkpoint(path, digest, shard_index):
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("schema") != "recommendation-relational-shard-checkpoint-v1"
            or value.get("digest") != digest
            or value.get("shard_index") != shard_index
            or not isinstance(value.get("result"), Mapping)):
        raise ValueError(f"invalid relational shard checkpoint: {path}")
    return value["result"]


def _run_isolated_reasoner_shards(
    shards, *, workers, add_batch_size, query_batch_size, query_steps,
    checkpoint_dir=None, checkpoint_identity=None,
):
    """Execute one-process-per-shard in bounded concurrent waves."""

    context = mp.get_context("spawn")
    results = [None] * len(shards)
    pending = []
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_identity is None:
            raise ValueError("checkpoint_identity is required with checkpoint_dir")
    for shard_index, shard in enumerate(shards):
        if checkpoint_dir is None:
            pending.append(shard_index)
            continue
        path, digest = _shard_checkpoint_path(
            checkpoint_dir, checkpoint_identity, shard_index, shard,
        )
        if path.exists():
            results[shard_index] = _read_shard_checkpoint(
                path, digest, shard_index,
            )
        else:
            pending.append(shard_index)
    if checkpoint_dir is not None:
        print(json.dumps({
            "stage": "relational_checkpoint_resume",
            "shards": len(shards),
            "completed": len(shards) - len(pending),
            "pending": len(pending),
        }), flush=True)

    for wave_start in range(0, len(pending), workers):
        active = []
        try:
            for shard_index in pending[wave_start:wave_start + workers]:
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_reasoner_shard_process,
                    args=(
                        child, shards[shard_index], add_batch_size,
                        query_batch_size, query_steps,
                    ),
                    name=f"relational-projection-petta-shard-{shard_index}",
                    daemon=False,
                )
                process.start()
                child.close()
                active.append((shard_index, parent, process))
            for shard_index, parent, process in active:
                try:
                    try:
                        payload = parent.recv()
                    except EOFError as exc:
                        raise RuntimeError(
                            "PeTTa relational shard exited without returning "
                            f"a result: {shard_index}"
                        ) from exc
                finally:
                    parent.close()
                    process.join()
                if process.exitcode != 0:
                    raise RuntimeError(
                        "PeTTa relational shard exited unexpectedly "
                        f"({shard_index}: {process.exitcode})"
                    )
                if not payload or payload[0] != "ok":
                    _status, error_type, message, error_traceback = payload
                    raise RuntimeError(
                        f"PeTTa relational shard {shard_index} failed "
                        f"[{error_type}]: {message}\n{error_traceback}"
                    )
                result = payload[1]
                if checkpoint_dir is not None:
                    path, digest = _shard_checkpoint_path(
                        checkpoint_dir, checkpoint_identity,
                        shard_index, shards[shard_index],
                    )
                    _write_shard_checkpoint(path, digest, shard_index, result)
                results[shard_index] = result
                print(json.dumps({
                    "stage": "relational_shard_complete",
                    "shard": shard_index + 1,
                    "shards": len(shards),
                    "proof_roots": result.get("proof_roots"),
                }), flush=True)
        except BaseException:
            for _shard_index, parent, process in active:
                parent.close()
                if process.is_alive():
                    process.terminate()
                process.join()
            raise
    if any(result is None for result in results):
        raise RuntimeError("PeTTa relational shard result is missing")
    return results


def build_relational_projection(
    source,
    *,
    reasoner=None,
    add_batch_size=10_000,
    query_batch_size=500,
    query_steps=2_000,
    shard_root_size=5_000,
    shard_workers=1,
    checkpoint_dir=None,
):
    """Return a preserved replay plus proof-derived relational observations."""

    add_batch_size = _positive_integer(add_batch_size, "add_batch_size")
    query_batch_size = _positive_integer(query_batch_size, "query_batch_size")
    query_steps = _positive_integer(query_steps, "query_steps")
    shard_root_size = _positive_integer(shard_root_size, "shard_root_size")
    shard_workers = _positive_integer(shard_workers, "shard_workers")
    total_started = time.perf_counter()
    original, source_hash, source_kind = _read(source)
    checkpoint_identity = {
        "projection_schema": RELATIONAL_PROJECTION_SCHEMA,
        "source_sha256": source_hash,
        "source_kind": source_kind,
        "structural_rules_sha256": hashlib.sha256(json.dumps(
            sorted(RELATIONAL_STRUCTURAL_RULES), ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "add_batch_size": add_batch_size,
        "query_batch_size": query_batch_size,
        "query_steps": query_steps,
        "shard_root_size": shard_root_size,
    }
    metadata = original.get("metadata")
    if original.get("relational_proof_ledger") is not None or (
        isinstance(metadata, Mapping) and metadata.get("relational_workspace") is not None
    ):
        raise ValueError("source already contains a relational workspace")
    evaluation_key, _ = _evaluation(original)
    articles = _article_mapping(original)
    annotations = original.get("llm_article_annotations")
    if annotations is None:
        annotations = {}
    if not isinstance(annotations, Mapping):
        raise ValueError("llm_article_annotations must be a mapping when present")
    data = copy.deepcopy(original)

    planning_started = time.perf_counter()
    planned = {}
    entity_observation_cache = {}
    concept_observation_cache = {}
    bindings = []
    coverage = {"training": Counter(), "evaluation": Counter()}
    for context, candidate, history, user, split in _contexts(data, evaluation_key):
        if any(feature in context for feature in (
            *RELATIONAL_WORKSPACE_FEATURES,
            REL_ENTITY_CONTINUITY_PROOF_IDS,
            REL_ENTITY_PATH_MULTIPLICITY_PROOF_IDS,
            REL_CONCEPT_CONTINUITY_PROOF_IDS,
        )):
            raise ValueError("source already contains relational observation fields")
        cache_key = (user, candidate, tuple(history))
        if cache_key not in planned:
            planned[cache_key] = build_relational_plans(
                candidate, history, articles, annotations, user_id=user,
                entity_observation_cache=entity_observation_cache,
                concept_observation_cache=concept_observation_cache,
            )
        entity_plan, concept_plan = planned[cache_key]
        bindings.append((context, entity_plan, concept_plan, split))
        coverage[split]["contexts"] += 1
        coverage[split]["complete_entity_evidence_contexts"] += int(entity_plan.complete_entity_evidence)
        coverage[split]["incomplete_entity_evidence_contexts"] += int(not entity_plan.complete_entity_evidence)
        coverage[split]["queryable_entity_contexts"] += int(entity_plan.requires_query)
        coverage[split]["entity_path_multiplicity_query_contexts"] += int(
            entity_plan.requires_entity_path_multiplicity_query
        )
        coverage[split]["complete_concept_evidence_contexts"] += int(concept_plan.complete_concept_evidence)
        coverage[split]["incomplete_concept_evidence_contexts"] += int(not concept_plan.complete_concept_evidence)
        coverage[split]["queryable_concept_contexts"] += int(concept_plan.requires_query)
    planning_seconds = time.perf_counter() - planning_started

    # Retain the logical union for deterministic audit hashes.  Execution uses
    # bounded workspaces below; no production reasoner receives this union.
    statement_sources = list(RELATIONAL_STRUCTURAL_RULES)
    for entity_plan, concept_plan in planned.values():
        # Exact continuity only loads plans whose label-free overlap prefilter
        # says a path can exist. Multiplicity is different: every complete
        # entity snapshot is loaded so PeTTa can prove ``none`` from the case
        # anchor as well as ``one``/``two_plus`` from origin paths.
        if (entity_plan.requires_query
                or entity_plan.requires_entity_path_multiplicity_query):
            statement_sources.extend(entity_plan.statements)
        if concept_plan.requires_query:
            statement_sources.extend(concept_plan.statements)
    _require_unique_statement_names(statement_sources)
    statements = set(statement_sources)
    ordered_statements = sorted(statements)
    ordered_source_statements = sorted(statements - set(RELATIONAL_STRUCTURAL_RULES))

    queryable = [
        plan for pair in planned.values() for plan in pair if plan.requires_query
    ]
    exact_query_roots = [
        (
            plan,root.query,"exact",
            f"{root.origin_id}/{root.matched_value_id}",
        )
        for plan in queryable
        for root in plan.proof_roots
    ]
    if any(not plan.proof_roots for plan in queryable):
        raise RuntimeError("a queryable relational plan has no specific proof roots")

    multiplicity_queries = [
        (
            entity_plan,entity_plan.entity_path_multiplicity_query,
            "entity_path_multiplicity",f"{entity_plan.case_id}/category",
        )
        for entity_plan, _concept_plan in planned.values()
        if entity_plan.requires_entity_path_multiplicity_query
    ]
    multiplicity_no_path_queries = sum(
        not plan.proof_roots
        for plan, _query, _kind, _diagnostic in multiplicity_queries
    )
    multiplicity_positive_path_queries = (
        len(multiplicity_queries) - multiplicity_no_path_queries
    )
    query_roots = [*exact_query_roots,*multiplicity_queries]
    indexed_roots = tuple(
        (index,plan,query,kind,diagnostic)
        for index,(plan,query,kind,diagnostic) in enumerate(query_roots)
    )
    # ``reasoner=`` intentionally preserves the original one-workspace seam
    # for deterministic fake reasoners and semantic equivalence tests. Normal
    # execution preserves each original query batch and runs bounded groups of
    # those batches in spawned processes. Process exit is the only complete
    # cleanup boundary supported by the PeTTa/Janus runtime.
    if reasoner is not None:
        shards = (indexed_roots,) if indexed_roots else ()
        shard_policy = "injected-single-workspace-reference-v1"
    else:
        shards = _root_batch_shards(
            indexed_roots, query_batch_size, shard_root_size,
        )
        shard_policy = "spawned-root-batch-preserving-workspaces-v1"

    proof_results = {plan.case_id: [] for plan in queryable}
    multiplicity_proof_results = {
        plan.case_id: [] for plan,_query,_kind,_diagnostic in multiplicity_queries
    }
    indexed_proof_results = {}
    query_batches = 0
    exact_query_calls = 0
    multiplicity_query_calls = 0
    multiplicity_no_path_query_calls = 0
    multiplicity_positive_path_query_calls = 0
    proof_rows_returned = 0
    insertion_seconds = 0.0
    query_seconds = 0.0
    exact_query_seconds = 0.0
    multiplicity_query_seconds = 0.0
    multiplicity_no_path_query_seconds = 0.0
    multiplicity_positive_path_query_seconds = 0.0
    shard_audit = []
    shard_execution_started = time.perf_counter()
    if reasoner is not None:
        shard_results = [
            _run_reasoner_shard(
                shard, reasoner, add_batch_size=add_batch_size,
                query_batch_size=query_batch_size, query_steps=query_steps,
            )
            for shard in shards
        ]
        effective_shard_workers = 1
    else:
        effective_shard_workers = min(shard_workers, len(shards)) if shards else 0
        if effective_shard_workers <= 1 and checkpoint_dir is None:
            shard_results = [
                _run_isolated_reasoner_shard(
                    shard, add_batch_size=add_batch_size,
                    query_batch_size=query_batch_size, query_steps=query_steps,
                )
                for shard in shards
            ]
        else:
            shard_results = _run_isolated_reasoner_shards(
                shards, workers=max(1, effective_shard_workers),
                add_batch_size=add_batch_size,
                query_batch_size=query_batch_size, query_steps=query_steps,
                checkpoint_dir=checkpoint_dir,
                checkpoint_identity=checkpoint_identity,
            )
    for index, result in enumerate(shard_results):
        for root_index, proofs in result.pop("root_results"):
            if (isinstance(root_index, bool) or not isinstance(root_index, int)
                    or not 0 <= root_index < len(query_roots)):
                raise RuntimeError("PeTTa shard returned a foreign proof-root index")
            if root_index in indexed_proof_results:
                raise RuntimeError("PeTTa shard returned a duplicate proof-root index")
            indexed_proof_results[root_index] = proofs
        insertion_seconds += result.pop("insertion_seconds")
        query_seconds += result.pop("query_seconds")
        query_batches += result["query_batches"]
        exact_query_calls += result["exact_query_calls"]
        multiplicity_query_calls += result[
            "entity_path_multiplicity_query_calls"
        ]
        multiplicity_no_path_query_calls += result[
            "entity_path_multiplicity_no_path_query_calls"
        ]
        multiplicity_positive_path_query_calls += result[
            "entity_path_multiplicity_positive_path_query_calls"
        ]
        exact_query_seconds += result.pop("exact_query_seconds")
        multiplicity_query_seconds += result.pop(
            "entity_path_multiplicity_query_seconds"
        )
        multiplicity_no_path_query_seconds += result.pop(
            "entity_path_multiplicity_no_path_query_seconds"
        )
        multiplicity_positive_path_query_seconds += result.pop(
            "entity_path_multiplicity_positive_path_query_seconds"
        )
        proof_rows_returned += result["proof_rows_returned"]
        shard_audit.append({"index": index, **result})
    if set(indexed_proof_results) != set(range(len(query_roots))):
        raise RuntimeError("PeTTa shards did not return every proof-root index")
    for root_index, (plan,_query,kind,_diagnostic) in enumerate(query_roots):
        destination=(multiplicity_proof_results
                     if kind=="entity_path_multiplicity" else proof_results)
        destination[plan.case_id].extend(indexed_proof_results[root_index])
    shard_execution_wall_seconds = time.perf_counter() - shard_execution_started
    reasoner_names = sorted({item["reasoner"] for item in shard_audit})
    if not reasoner_names and reasoner is not None:
        reasoner_names = [f"{type(reasoner).__module__}.{type(reasoner).__qualname__}"]
    reasoner_name = (
        reasoner_names[0] if len(reasoner_names) == 1
        else "mixed[" + ",".join(reasoner_names) + "]"
        if reasoner_names else "uninstantiated:no-proof-roots"
    )

    reduction_started = time.perf_counter()
    reductions = {}
    ledger = {}
    for entity_plan, concept_plan in planned.values():
        for plan, reducer in (
            (entity_plan, reduce_relational_proofs),
            (concept_plan, reduce_concept_relational_proofs),
        ):
            facts, records = reducer(plan, proof_results.get(plan.case_id, ()))
            reductions[plan.case_id] = facts
            for proof_id, record in records.items():
                if proof_id in ledger and ledger[proof_id] != record:
                    raise RuntimeError("stable relational proof ID collision")
                ledger[proof_id] = record
        multiplicity_facts,multiplicity_records=(
            reduce_entity_path_multiplicity_proofs(
                entity_plan,
                multiplicity_proof_results.get(entity_plan.case_id,()),
            )
        )
        entity_facts=reductions[entity_plan.case_id]
        if set(entity_facts).intersection(multiplicity_facts):
            raise RuntimeError("relational reductions produced duplicate fields")
        entity_facts.update(multiplicity_facts)
        for proof_id,record in multiplicity_records.items():
            if proof_id in ledger and ledger[proof_id]!=record:
                raise RuntimeError("stable relational proof ID collision")
            ledger[proof_id]=record
    for context, entity_plan, concept_plan, split in bindings:
        entity_facts = reductions[entity_plan.case_id]
        concept_facts = reductions[concept_plan.case_id]
        context.update(copy.deepcopy(entity_facts))
        context.update(copy.deepcopy(concept_facts))
        coverage[split][f"entity_scope_{entity_facts[REL_ENTITY_CONTINUITY_SCOPE]}"] += 1
        multiplicity=entity_facts.get(REL_ENTITY_PATH_MULTIPLICITY)
        coverage[split][
            f"entity_path_multiplicity_{multiplicity or 'unknown'}"
        ] += 1
        coverage[split][f"concept_scope_{concept_facts[REL_CONCEPT_CONTINUITY_SCOPE]}"] += 1
        coverage[split]["entity_proof_origin_references"] += len(
            entity_facts[REL_ENTITY_CONTINUITY_PROOF_IDS]
        )
        coverage[split]["entity_path_multiplicity_proof_references"] += len(
            entity_facts.get(REL_ENTITY_PATH_MULTIPLICITY_PROOF_IDS,())
        )
        coverage[split]["concept_proof_origin_references"] += len(
            concept_facts[REL_CONCEPT_CONTINUITY_PROOF_IDS]
        )

    ordered_ledger = {key: ledger[key] for key in sorted(ledger)}
    data["relational_proof_ledger"] = ordered_ledger
    ledger_json = json.dumps(
        ordered_ledger, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    reduction_and_serialization_seconds = time.perf_counter() - reduction_started
    audit = {
        "schema": RELATIONAL_WORKSPACE_SCHEMA,
        "projection_schema": RELATIONAL_PROJECTION_SCHEMA,
        "source_dataset_sha256": source_hash,
        "source_dataset_sha256_kind": source_kind,
        "history_policy": "explicit row-local preceding click histories only; oldest to newest; no latest-profile reconstruction",
        "relations": {
            "wikidata_entity_continuity": "exact shared MIND Wikidata ID between candidate and a preceding clicked article",
            "wikidata_entity_path_multiplicity": (
                "PeTTa-derived degree bucket over distinct preceding-click "
                "origins connected to the candidate by exact Wikidata entities"
            ),
            "canonical_concept_continuity": "exact shared provenance-anchored canonical concept ID between candidate and a preceding clicked article",
        },
        "inference": (
            "PeTTaChainer proves exact-entity continuity, folds distinct "
            "interaction-origin path nodes into a graph degree, and classifies "
            "that degree as none/one/two_plus; "
            "canonical-concept continuity first grounds each named source "
            "annotation through a canonical mapping, then applies the same "
            "origin and continuity implications"
        ),
        "recency": "recent=matching origin in final five original history positions; older=only earlier matches; none=complete known evidence and no proof; unknown=incomplete evidence and no proof",
        "origin_policy": "one evidence record per history position and relation; alternate descriptor paths are deduplicated and entity/concept records from the same click share one dependency key",
        "label_policy": "outcomes are preserved but never passed to relational planning or proof execution",
        "structural_rule_ids": [
            "rel_v1_derive_engaged_entity_origin",
            "rel_v1_derive_entity_continuity",
            ENTITY_PATH_CASE_NODE_RULE_ID,
            ENTITY_PATH_ORIGIN_NODE_RULE_ID,
            ENTITY_PATH_COUNT_RULE_ID,
            ENTITY_PATH_NONE_RULE_ID,
            ENTITY_PATH_ONE_RULE_ID,
            ENTITY_PATH_TWO_PLUS_RULE_ID,
            CANONICAL_CONCEPT_BRIDGE_RULE_ID,
            "rel_v1_derive_engaged_concept_origin",
            "rel_v1_derive_concept_continuity",
        ],
        "source_fact_stv": [1.0, 1.0],
        "structural_rule_positive_stv": [1.0, 1.0],
        "structural_rule_negative_stv": [0.0, 1.0],
        "reasoner": reasoner_name,
        "query_steps": query_steps,
        "query_steps_per_root": query_steps,
        "entity_path_multiplicity_query_steps_per_root": (
            ENTITY_PATH_MULTIPLICITY_QUERY_STEPS
        ),
        "query_step_budget_policy": (
            "exact roots share a per-root-multiplied batch budget; every "
            "entity-path multiplicity FoldAll root has an independent fixed budget"
        ),
        "total_query_step_budget": (
            query_steps * len(exact_query_roots)
            + ENTITY_PATH_MULTIPLICITY_QUERY_STEPS * len(multiplicity_queries)
        ),
        "maximum_batch_query_step_budget": max(
            query_steps * min(query_batch_size, len(exact_query_roots))
            if exact_query_roots else 0,
            ENTITY_PATH_MULTIPLICITY_QUERY_STEPS
            if multiplicity_queries else 0,
        ),
        "add_batch_size": add_batch_size,
        "query_batch_size": query_batch_size,
        "workspace_shard_policy": shard_policy,
        "workspace_shard_root_budget": shard_root_size,
        "workspace_shard_effective_root_budget": max(
            query_batch_size,
            (shard_root_size // query_batch_size) * query_batch_size,
        ),
        "workspace_shard_count": len(shard_audit),
        "workspace_shard_workers_requested": shard_workers,
        "workspace_shard_workers_used": effective_shard_workers,
        "workspace_shards": shard_audit,
        "maximum_workspace_proof_roots": max(
            (item["proof_roots"] for item in shard_audit), default=0,
        ),
        "maximum_workspace_source_statements": max(
            (item["source_statements"] for item in shard_audit), default=0,
        ),
        "maximum_workspace_atomspace_statements": max(
            (item["atomspace_statements"] for item in shard_audit), default=0,
        ),
        "total_workspace_source_statement_insertions": sum(
            item["source_statements"] for item in shard_audit
        ),
        "unique_candidate_history_plans": len(planned),
        "cached_entity_article_observations": len(entity_observation_cache),
        "cached_concept_article_observations": len(concept_observation_cache),
        "queryable_plans": len(queryable),
        "expected_proof_roots": len(query_roots),
        "expected_entity_proof_roots": sum(
            len(entity_plan.proof_roots)
            for entity_plan, _concept_plan in planned.values()
        ),
        "expected_concept_proof_roots": sum(
            len(concept_plan.proof_roots)
            for _entity_plan, concept_plan in planned.values()
        ),
        "expected_entity_path_multiplicity_queries": len(multiplicity_queries),
        "expected_entity_path_multiplicity_no_path_queries": (
            multiplicity_no_path_queries
        ),
        "expected_entity_path_multiplicity_positive_path_queries": (
            multiplicity_positive_path_queries
        ),
        "queries_submitted": len(query_roots),
        "wildcard_queries_submitted": len(multiplicity_queries),
        "complete_proof_roots": len(exact_query_roots),
        "complete_entity_path_multiplicity_queries": len(multiplicity_queries),
        "query_batches": query_batches,
        "exact_query_calls": exact_query_calls,
        "entity_path_multiplicity_query_calls": multiplicity_query_calls,
        "entity_path_multiplicity_no_path_query_calls": (
            multiplicity_no_path_query_calls
        ),
        "entity_path_multiplicity_positive_path_query_calls": (
            multiplicity_positive_path_query_calls
        ),
        "proof_rows_returned": proof_rows_returned,
        "source_statements": len(ordered_source_statements),
        "source_statements_sha256": hashlib.sha256(json.dumps(
            ordered_source_statements, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "atomspace_statements_sha256": hashlib.sha256(json.dumps(
            ordered_statements, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "proof_origins": len(ordered_ledger),
        "coverage": {split: dict(values) for split, values in coverage.items()},
        "proof_ledger_sha256": hashlib.sha256(ledger_json.encode("utf-8")).hexdigest(),
        "context_observations_sha256": relational_context_observations_sha256(data),
        "structural_rules_sha256": hashlib.sha256(json.dumps(
            RELATIONAL_STRUCTURAL_RULES, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "safety": relational_safety_audit(data),
        "timings_seconds": {
            "planning": planning_seconds,
            "atomspace_insertion": insertion_seconds,
            "petta_queries": query_seconds,
            "petta_exact_queries": exact_query_seconds,
            "petta_entity_path_multiplicity_queries": (
                multiplicity_query_seconds
            ),
            "petta_entity_path_multiplicity_no_path_queries": (
                multiplicity_no_path_query_seconds
            ),
            "petta_entity_path_multiplicity_positive_path_queries": (
                multiplicity_positive_path_query_seconds
            ),
            "shard_execution_wall": shard_execution_wall_seconds,
            "reduction_and_ledger_serialization": reduction_and_serialization_seconds,
            "total_projection": time.perf_counter() - total_started,
        },
        "preserved_original_evidence": True,
    }
    data.setdefault("metadata", {})["relational_workspace"] = audit
    validate_relational_projection(data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--add-batch-size", type=int, default=10_000)
    parser.add_argument("--query-batch-size", type=int, default=500)
    parser.add_argument(
        "--query-steps", type=int, default=2_000,
        help="PeTTa expansion budget per specific proof root (multiplied per batch)",
    )
    parser.add_argument(
        "--shard-root-size", type=int, default=5_000,
        help=(
            "Approximate exact-root budget per spawned PeTTa workspace; "
            "whole query batches are never split"
        ),
    )
    parser.add_argument(
        "--shard-workers", type=int, default=1,
        help="Maximum concurrently running one-shot PeTTa shard processes",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        help=(
            "Optional directory for deterministic completed-shard checkpoints; "
            "matching checkpoints are resumed on a later invocation"
        ),
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or args.output.is_symlink():
        parser.error("output must be a new path; existing files are never overwritten")
    data = build_relational_projection(
        args.data,
        add_batch_size=args.add_batch_size,
        query_batch_size=args.query_batch_size,
        query_steps=args.query_steps,
        shard_root_size=args.shard_root_size,
        shard_workers=args.shard_workers,
        checkpoint_dir=args.checkpoint_dir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", dir=output.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as binary:
            if output.suffix == ".gz":
                with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                        json.dump(
                            data, text, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False,
                        )
            else:
                binary.write(json.dumps(
                    data, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8"))
            binary.flush()
            os.fsync(binary.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(output),
        "bytes": output.stat().st_size,
        "relational_workspace": data["metadata"]["relational_workspace"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RELATIONAL_PROJECTION_SCHEMA",
    "build_relational_projection",
    "main",
]
