"""Measure concurrent live-feed latency through the real HTTP scoring path.

The probe deliberately stays outside the recommendation process. Every
measured request uses the paginated live-feed route with a user not previously
used by this probe. Server-side cache counters must confirm one ranked-feed
cache miss per request. Requests in each round are released from a barrier
together so the reported tail latency includes queueing inside a scorer
replica. No ``Lab`` or reasoner method is imported.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import threading
import time
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit
import uuid

from ..paths import RESULTS_DIR
from .realtime_probe import (
    ProcessTreeSampler,
    ProbeError,
    _json_request,
    _latency_summary,
)


SCHEMA = "mindplex-recommendation-concurrent-feed-probe-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    """Return the greatest number of overlapping half-open intervals."""
    events: list[tuple[float, int]] = []
    for started, finished in intervals:
        if finished < started:
            raise ValueError("request interval finishes before it starts")
        # End events sort before starts at an identical timestamp, matching
        # half-open [start, end) request intervals.
        events.append((float(started), 1))
        events.append((float(finished), -1))
    active = peak = 0
    for _, change in sorted(events, key=lambda item: (item[0], item[1])):
        active += change
        peak = max(peak, active)
    return peak


def _memory_total_mib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024.0, 3)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _state_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint every server property that defines the measured scorer."""
    engine=snapshot.get("engine") or {}
    config=snapshot.get("config") or {}
    dataset=snapshot.get("dataset") or {}
    users=sorted(str(row.get("id")) for row in snapshot.get("users") or [])
    canonical=lambda value:json.dumps(
        value,sort_keys=True,separators=(",",":"),ensure_ascii=False
    ).encode("utf-8")
    return {
        "instance_id":str(snapshot.get("instance_id") or ""),
        "rule_version":int(snapshot.get("version",-1)),
        "worker_pid":engine.get("worker_pid"),
        "config_sha256":hashlib.sha256(canonical(config)).hexdigest(),
        "dataset_sha256":hashlib.sha256(canonical(dataset)).hexdigest(),
        "user_ids_sha256":hashlib.sha256(canonical(users)).hexdigest(),
    }


RequestFunction = Callable[[str, str, float], tuple[dict[str, Any], dict[str, Any]]]


def _default_request(base_url: str, path: str, timeout: float):
    return _json_request(base_url, path, timeout=timeout)


def run_level(
    base_url: str,
    *,
    users: list[str],
    concurrency: int,
    rounds: int,
    page_size: int,
    expected_rule_version: int,
    request_timeout: float,
    request_fn: RequestFunction = _default_request,
) -> dict[str, Any]:
    """Run synchronized request rounds for one concurrency level."""
    if concurrency < 1 or rounds < 1 or page_size < 1:
        raise ValueError("concurrency, rounds, and page_size must be positive")
    attempted = concurrency * rounds
    if len(users) != attempted or len(set(users)) != attempted:
        raise ValueError(
            "fresh-user live-feed mode requires one distinct user per request"
        )

    observations: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    round_seconds: list[float] = []

    def request_one(
        barrier: threading.Barrier,
        round_index: int,
        worker_index: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        user = users[round_index * concurrency + worker_index]
        path = "/api/feed?" + urlencode({
            "user": user,
            # ``limit`` deliberately selects Handler's live ``feed_page`` route
            # instead of its legacy direct-score route.
            "limit": page_size,
            "probe": uuid.uuid4().hex,
        })
        try:
            barrier.wait(timeout=request_timeout)
            payload, observation = request_fn(base_url, path, request_timeout)
            feed = payload.get("feed")
            if not isinstance(feed, list) or not feed:
                raise ProbeError("concurrent /api/feed returned an empty feed")
            if not payload.get("session"):
                raise ProbeError("live /api/feed returned no session")
            if int(payload.get("rule_version", -1)) != expected_rule_version:
                raise ProbeError("rule version changed during concurrent request")
            if any(row.get("engine") != "PeTTaChainer" for row in feed):
                raise ProbeError("live feed contains a non-PeTTa-ranked row")
            return {
                **observation,
                "round": round_index + 1,
                "worker": worker_index + 1,
                "user": user,
                "feed_count": len(feed),
                "session": str(payload["session"]),
                "rule_version": int(payload["rule_version"]),
                "pettachainer_rows": sum(
                    row.get("engine") == "PeTTaChainer" for row in feed
                ),
                "pairwise_rows": sum(
                    row.get("pairwise_score") is not None for row in feed
                ),
            }, None
        except BaseException as exc:
            return None, {
                "round": round_index + 1,
                "worker": worker_index + 1,
                "user": user,
                "error": f"{type(exc).__name__}: {exc}",
            }

    level_started = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="feed-load-probe",
    ) as executor:
        for round_index in range(rounds):
            barrier = threading.Barrier(concurrency)
            started = time.monotonic()
            futures = [
                executor.submit(request_one, barrier, round_index, worker_index)
                for worker_index in range(concurrency)
            ]
            for future in futures:
                observation, error = future.result()
                if observation is not None:
                    observations.append(observation)
                if error is not None:
                    errors.append(error)
            round_seconds.append(time.monotonic() - started)
    elapsed = time.monotonic() - level_started

    latencies = [float(row["elapsed_ms"]) for row in observations]
    intervals = [
        (float(row["started_at_epoch"]), float(row["finished_at_epoch"]))
        for row in observations
        if row.get("started_at_epoch") is not None
        and row.get("finished_at_epoch") is not None
    ]
    successful = len(observations)
    return {
        "concurrency": concurrency,
        "rounds": rounds,
        "requests_attempted": attempted,
        "requests_succeeded": successful,
        "requests_failed": len(errors),
        "elapsed_seconds": round(elapsed, 6),
        "throughput_requests_per_second": (
            round(successful / elapsed, 6) if elapsed > 0 else None
        ),
        "round_wall_latency": _latency_summary(
            [seconds * 1000.0 for seconds in round_seconds]
        ),
        "request_latency": _latency_summary(latencies),
        "client_observed_peak_in_flight": _peak_overlap(intervals),
        "errors": errors,
        "observations": observations,
        "passed": successful == attempted and not errors,
    }


def probe(
    base_url: str,
    *,
    concurrency_levels: list[int],
    rounds: int,
    warmup: int,
    page_size: int,
    random_seed: int,
    request_timeout: float,
    server_pid: int | None,
    resource_interval: float,
) -> dict[str, Any]:
    if (not concurrency_levels or any(level<1 for level in concurrency_levels)
            or len(set(concurrency_levels))!=len(concurrency_levels)):
        raise ValueError("concurrency levels must be nonempty, positive, and unique")
    if concurrency_levels.count(1)!=1:
        raise ValueError("concurrency levels must include 1 exactly once")
    if rounds<1 or warmup<0 or page_size<1:
        raise ValueError("rounds/page_size must be positive and warmup non-negative")
    if request_timeout<=0 or resource_interval<=0:
        raise ValueError("timeouts and sampling intervals must be positive")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProbeError(f"invalid base URL: {base_url!r}")
    base_url = base_url.rstrip("/")

    state, state_observation = _json_request(
        base_url, "/api/state", timeout=request_timeout
    )
    users = [str(row["id"]) for row in state.get("users") or []]
    if not users:
        raise ProbeError("/api/state returned no users")
    random.Random(random_seed).shuffle(users)
    measured_requests=sum(concurrency_levels) * rounds
    required_users=warmup + measured_requests
    if len(users) < required_users:
        raise ProbeError(
            f"fresh-user workload needs {required_users} distinct users but "
            f"the server exposes {len(users)}"
        )
    rule_version=int(state.get("version",-1))

    def live_path(user: str) -> str:
        return "/api/feed?" + urlencode({
            "user":user,"limit":page_size,"probe":uuid.uuid4().hex,
        })

    def counters(snapshot: dict[str, Any]) -> dict[str, int]:
        engine=snapshot.get("engine") or {}
        cache=engine.get("feed_rank_cache") or {}
        required={
            "feed_rank_cache_hits":cache.get("hits"),
            "feed_rank_cache_misses":cache.get("misses"),
            "feed_rank_cache_entries":cache.get("entries"),
            "point_case_cache_entries":engine.get("point_case_cache_entries"),
            "pair_case_cache_entries":engine.get("pair_case_cache_entries"),
            "point_channel_cache_entries":engine.get(
                "point_channel_cache_entries"
            ),
            "pair_channel_cache_entries":engine.get(
                "pair_channel_cache_entries"
            ),
            "point_reasoner_query_calls":engine.get("point_reasoner_query_calls"),
            "pair_reasoner_query_calls":engine.get("pair_reasoner_query_calls"),
        }
        if any(type(value) is not int for value in required.values()):
            raise ProbeError(
                "server lacks feed/proof cache observability required by this probe"
            )
        return required

    def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        return {key:after[key]-before[key] for key in before}

    warmup_observations: list[dict[str, Any]] = []
    for index in range(warmup):
        user = users[index % len(users)]
        payload, observation = _json_request(
            base_url,live_path(user),
            timeout=request_timeout,
        )
        if (not payload.get("feed") or not payload.get("session")
                or int(payload.get("rule_version",-1))!=rule_version
                or any(row.get("engine")!="PeTTaChainer"
                       for row in payload.get("feed") or [])):
            raise ProbeError("warm-up did not return a stable PeTTa live-feed page")
        warmup_observations.append(observation)

    measurement_state,measurement_state_observation=_json_request(
        base_url,"/api/state",timeout=request_timeout
    )
    measurement_identity=_state_identity(measurement_state)
    if _state_identity(state)!=measurement_identity:
        raise ProbeError(
            "scorer identity/config/dataset/users changed during warm-up; "
            "restart the probe against a stable server"
        )
    rule_version=int(measurement_identity["rule_version"])
    scorer_processes=int(
        (measurement_state.get("engine") or {}).get("pool_size",1) or 1
    )

    sampler = ProcessTreeSampler(server_pid, resource_interval) if server_pid else None
    if sampler:
        sampler.start()
    started_at=_utc_now(); started = time.monotonic()
    levels: list[dict[str, Any]] = []
    user_offset=warmup
    try:
        for concurrency in concurrency_levels:
            request_count=concurrency*rounds
            level_users=users[user_offset:user_offset+request_count]
            user_offset+=request_count
            before_state,_before_observation=_json_request(
                base_url,"/api/state",timeout=request_timeout
            )
            before_counters=counters(before_state)
            before_identity=_state_identity(before_state)
            level=run_level(
                base_url,
                users=level_users,
                concurrency=concurrency,
                rounds=rounds,
                page_size=page_size,
                expected_rule_version=rule_version,
                request_timeout=request_timeout,
            )
            after_state,_after_observation=_json_request(
                base_url,"/api/state",timeout=request_timeout
            )
            after_counters=counters(after_state)
            after_identity=_state_identity(after_state)
            counter_delta=delta(before_counters,after_counters)
            stable_identity=(before_identity==measurement_identity
                             and after_identity==measurement_identity)
            cache_misses_verified=(
                counter_delta["feed_rank_cache_misses"]==request_count
                and counter_delta["feed_rank_cache_hits"]==0
            )
            no_reasoner_query_calls=(
                counter_delta["point_reasoner_query_calls"]==0
                and counter_delta["pair_reasoner_query_calls"]==0
            )
            level.update(
                users_sha256=hashlib.sha256(
                    "\n".join(level_users).encode("utf-8")
                ).hexdigest(),
                server_counters_before=before_counters,
                server_counters_after=after_counters,
                server_counter_delta=counter_delta,
                scorer_identity_before=before_identity,
                scorer_identity_after=after_identity,
                scorer_identity_stable=stable_identity,
                ranked_feed_cache_misses_verified=cache_misses_verified,
                no_reasoner_query_calls_during_level=(
                    no_reasoner_query_calls
                ),
            )
            level["passed"]=(bool(level["passed"])
                             and stable_identity and cache_misses_verified)
            levels.append(level)
    finally:
        resources = sampler.stop() if sampler else None

    baseline = next(
        (row for row in levels if row["concurrency"] == 1 and row["passed"]),
        None,
    )
    baseline_p50 = (
        float(baseline["request_latency"]["p50_ms"])
        if baseline and baseline["request_latency"]["p50_ms"] is not None
        else None
    )
    for level in levels:
        p50 = level["request_latency"]["p50_ms"]
        level["descriptive_p50_ratio_vs_concurrency_1"] = (
            round(float(p50) / baseline_p50, 6)
            if p50 is not None and baseline_p50 not in {None, 0.0}
            else None
        )

    return {
        "base_url": base_url,
        "server_pid": server_pid,
        "started_at": started_at,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "workload": {
            "endpoint": "GET /api/feed?user=<id>&limit=<page_size>",
            "request_path":(
                "new-session first-page request through the real feed_page "
                "HTTP route and configured PeTTa-backed ranker"
            ),
            "cache_policy":(
                "one shuffled fresh user per measured request; server counters "
                "must show one final-ranked-feed cache miss and zero hits; run "
                "against a fresh server or choose a seed whose users were not "
                "already scored"
            ),
            "proof_cache_policy":(
                "steady-state proof/template caches are allowed and their "
                "entry/query-call deltas are reported"
            ),
            "level_comparability":(
                "absolute latency at each level is directly observed; levels "
                "use different randomly allocated users and run sequentially, "
                "so the descriptive p50 ratio is not a causal estimate of the "
                "effect of concurrency and may include user/slate difficulty "
                "or evolving proof-cache state"
            ),
            "causal_concurrency_effect_claimed":False,
            "infinite_scroll_latency_claimed":False,
            "client_overlap_definition":(
                "client-observed simultaneous outstanding HTTP requests; the "
                f"{scorer_processes} isolated scorer process(es) can execute "
                "in parallel, while each scorer serializes mutable workspace "
                "access"
            ),
            "mutates_feedback": False,
            "warmup_requests": warmup,
            "rounds_per_level": rounds,
            "available_users": len(users),
            "measured_distinct_users":measured_requests,
            "page_size":page_size,
            "random_seed":random_seed,
            "concurrency_levels": concurrency_levels,
            "scorer_processes":scorer_processes,
            "serving_mode":(
                (measurement_state.get("pool") or {}).get("mode","single")
            ),
            "feed_window":(
                (measurement_state.get("config") or {}).get("feed_window")
            ),
            "pairwise_opponents":(
                (measurement_state.get("config") or {}).get("pairwise_opponents")
            ),
            "max_pair_comparisons": (
                (measurement_state.get("config") or {}).get(
                    "max_pair_comparisons"
                )
            ),
            "dataset":measurement_state.get("dataset"),
            "rule_version":measurement_state.get("version"),
            "measurement_scorer_identity":measurement_identity,
        },
        "state_request": state_observation,
        "post_warmup_state_request":measurement_state_observation,
        "warmup_latency": _latency_summary([
            float(row["elapsed_ms"]) for row in warmup_observations
        ]),
        "levels": levels,
        "process_tree_resources": resources,
        "passed": all(level["passed"] for level in levels),
        "finished_at": _utc_now(),
    }


def _default_output() -> Path:
    stamp = datetime.now().astimezone().date().isoformat()
    return RESULTS_DIR / f"performance_{stamp}" / "concurrent_feed_probe.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure synchronized fresh-user first-page latency on a live "
            "scorer. Requires a fresh server and enough distinct users (the "
            "default workload targets MIND, not the tiny fixture)."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--concurrency", action="append", type=int, dest="concurrency_levels",
        help="Concurrent requests per synchronized round; repeatable (default: 1,2,4,8).",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--page-size",type=int,default=5)
    parser.add_argument("--seed",type=int,default=271828)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--resource-interval", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=_default_output())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    levels = args.concurrency_levels or [1, 2, 4, 8]
    if any(level < 1 for level in levels):
        raise SystemExit("--concurrency values must be positive")
    if len(set(levels)) != len(levels):
        raise SystemExit("--concurrency values must be unique")
    if levels.count(1)!=1:
        raise SystemExit("--concurrency must include level 1 exactly once")
    if args.rounds < 1 or args.warmup < 0 or args.page_size < 1:
        raise SystemExit("--rounds must be positive and --warmup non-negative")
    if args.request_timeout <= 0 or args.resource_interval <= 0:
        raise SystemExit("timeouts and sampling intervals must be positive")

    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": _utc_now(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_mib": _memory_total_mib(),
            "python": platform.python_version(),
        },
        "method": {
            "client": "synchronized stdlib urllib threads",
            "engine_mocks": False,
            "latency_scope": "client-observed HTTP wall time including scorer queueing",
            "throughput_scope": "configured live scorer deployment on this host",
            "request_scope":"new-session first-page live-feed ranking",
            "causal_concurrency_scaling_claim":False,
            "accuracy_claim": False,
        },
    }
    try:
        artifact["target"] = probe(
            args.base_url,
            concurrency_levels=levels,
            rounds=args.rounds,
            warmup=args.warmup,
            page_size=args.page_size,
            random_seed=args.seed,
            request_timeout=args.request_timeout,
            server_pid=args.server_pid,
            resource_interval=args.resource_interval,
        )
        scorer_processes=artifact["target"]["workload"]["scorer_processes"]
        artifact["method"]["throughput_scope"]=(
            f"{scorer_processes} live scorer process(es) behind the configured "
            "HTTP endpoint on this host"
        )
        artifact["passed"] = bool(artifact["target"]["passed"])
    except BaseException as exc:
        artifact["target"] = {
            "base_url": args.base_url,
            "server_pid": args.server_pid,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        artifact["passed"] = False
    artifact["finished_at"] = _utc_now()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "passed": artifact["passed"],
        "output": str(args.output.resolve()),
        "levels": (artifact.get("target") or {}).get("levels"),
        "error": (artifact.get("target") or {}).get("error"),
    }, indent=2, sort_keys=True))
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
