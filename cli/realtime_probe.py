"""Probe the live recommendation HTTP path and background miner.

The probe is intentionally an external client.  It does not import ``Lab`` or
replace any miner/reasoner function, so every measured feed and event crosses
the same HTTP, PeTTaChainer, and background-mining boundaries as the browser.

Example::

    python -m recommendation.cli.realtime_probe \
      --base-url http://127.0.0.1:7070 \
      --server-pid 12345 \
      --mine-interval 1

Repeat ``--base-url`` and ``--server-pid`` in the same order to probe several
replicas.  Process metrics use Linux ``/proc`` and are omitted when no PID is
provided.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from ..paths import RESULTS_DIR


SCHEMA = "mindplex-recommendation-realtime-probe-v1"


class ProbeError(RuntimeError):
    """A target failed a required live-probe operation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_ms(seconds: float) -> float:
    return round(seconds * 1000.0, 3)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return round(ordered[index], 3)


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min_ms": None, "p50_ms": None,
                "p95_ms": None, "max_ms": None, "mean_ms": None}
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 3),
        "mean_ms": round(statistics.fmean(values), 3),
    }


def _negative_demotion_audit(revision: dict[str, Any]) -> dict[str, Any]:
    """Validate the concrete PeTTa proof that caused one queue demotion."""
    demotion = revision.get("causal_demotion") or {}
    rule_ids = [str(value) for value in demotion.get("rule_ids") or []]
    raw_proofs = demotion.get("proofs") or []
    if isinstance(raw_proofs, str):
        raw_proofs = [raw_proofs]
    proof_text = "\n".join(
        proof if isinstance(proof, str) else json.dumps(proof, sort_keys=True)
        for proof in raw_proofs
    )
    before_rank = demotion.get("before_rank")
    after_rank = demotion.get("after_rank")
    before_score = demotion.get("before_score")
    after_score = demotion.get("after_score")
    before_ranking_score = demotion.get("before_ranking_score")
    after_ranking_score = demotion.get("after_ranking_score")
    article = str(demotion.get("article") or "")
    affected = {
        str(row.get("article"))
        for row in revision.get("negative_proof_candidates") or []
    }
    checks = {
        "candidate_identified": bool(article),
        "candidate_listed_as_negative_proof": bool(article and article in affected),
        "rank_demoted": (
            isinstance(before_rank, (int, float))
            and isinstance(after_rank, (int, float))
            and after_rank > before_rank
        ),
        "score_decreased": (
            isinstance(before_score, (int, float))
            and isinstance(after_score, (int, float))
            and after_score < before_score
        ),
        "ranking_score_decreased": (
            isinstance(before_ranking_score, (int, float))
            and isinstance(after_ranking_score, (int, float))
            and after_ranking_score < before_ranking_score
        ),
        "feedback_evidence_changed": (
            demotion.get("feedback_evidence_changed") is True
            and demotion.get("before_feedback_signature")
                != demotion.get("after_feedback_signature")
            and bool(demotion.get("after_feedback_signature"))
        ),
        "pettachainer_score_method": (
            demotion.get("score_method")
            == "pettachainer_live_feedback_revision"
        ),
        "feedback_rule_reported": any(
            rule.startswith("feedback_skip_") for rule in rule_ids
        ),
        "feedback_rule_present_in_proof": any(
            rule in proof_text for rule in rule_ids
        ),
    }
    return {
        **demotion,
        "rule_ids": rule_ids,
        "proof_text": proof_text,
        "checks": checks,
        # Rank movement alone is not causal: another row can move around this
        # candidate. Require changed proof-backed evidence and a decrease in
        # this candidate's own point score or final ranking score.
        "passed": (
            all(value for key, value in checks.items()
                if key not in {
                    "rank_demoted", "score_decreased",
                    "ranking_score_decreased",
                })
            and (checks["score_decreased"]
                 or checks["ranking_score_decreased"])
        ),
    }


def _json_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    encoded = None
    headers = {"Accept": "application/json"}
    if body is not None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=encoded, headers=headers, method=method)
    started_wall = time.time()
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = int(response.status)
    except HTTPError as exc:
        raw = exc.read()
        elapsed = time.monotonic() - started
        try:
            detail = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = raw.decode("utf-8", errors="replace")[:1000]
        raise ProbeError(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ProbeError(f"{method} {url} failed: {exc}") from exc
    elapsed = time.monotonic() - started
    finished_wall = time.time()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{method} {url} did not return JSON") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"{method} {url} returned a non-object JSON value")
    return payload, {
        "method": method,
        "path": path,
        "status": status,
        "elapsed_ms": _round_ms(elapsed),
        "started_at_epoch": round(started_wall, 6),
        "finished_at_epoch": round(finished_wall, 6),
        "response_bytes": len(raw),
    }


@dataclass(frozen=True)
class _Process:
    pid: int
    ppid: int
    cpu_ticks: int
    rss_kib: int
    command: str


def _read_process(pid: int) -> _Process | None:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text(encoding="utf-8")
        close = stat.rfind(")")
        if close < 0:
            return None
        fields = stat[close + 2:].split()
        # Fields begin at process-state (proc(5) field 3).
        ppid = int(fields[1])
        cpu_ticks = int(fields[11]) + int(fields[12])
        rss_kib = 0
        for line in (root / "status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
        try:
            command = (root / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except OSError:
            command = ""
        return _Process(pid, ppid, cpu_ticks, rss_kib, command[:500])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
        return None


def _process_tree(root_pid: int) -> list[_Process]:
    processes: dict[int, _Process] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        record = _read_process(int(entry.name))
        if record is not None:
            processes[record.pid] = record
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for record in processes.values():
            if record.ppid in selected and record.pid not in selected:
                selected.add(record.pid)
                changed = True
    return [processes[pid] for pid in sorted(selected) if pid in processes]


class ProcessTreeSampler:
    """Sample aggregate CPU and RSS for a server and every descendant."""

    def __init__(self, root_pid: int, interval: float) -> None:
        self.root_pid = int(root_pid)
        self.interval = float(interval)
        self.samples: list[dict[str, Any]] = []
        self._commands: dict[str, str] = {}
        self._previous_ticks: dict[int, int] = {}
        self._previous_time: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            self._ticks_per_second = int(os.sysconf("SC_CLK_TCK"))
        except (ValueError, OSError):
            self._ticks_per_second = 100

    def _sample(self) -> None:
        now_mono = time.monotonic()
        records = _process_tree(self.root_pid)
        ticks = {record.pid: record.cpu_ticks for record in records}
        cpu_percent = None
        if self._previous_time is not None:
            elapsed = now_mono - self._previous_time
            delta = sum(
                max(0, value - self._previous_ticks[pid])
                for pid, value in ticks.items()
                if pid in self._previous_ticks
            )
            if elapsed > 0:
                cpu_percent = 100.0 * delta / self._ticks_per_second / elapsed
        for record in records:
            self._commands[str(record.pid)] = record.command
        self.samples.append({
            "at_epoch": round(time.time(), 6),
            "elapsed_seconds": (
                0.0 if not self.samples
                else round(now_mono - self._started_monotonic, 6)
            ),
            "process_count": len(records),
            "pids": [record.pid for record in records],
            "rss_kib": sum(record.rss_kib for record in records),
            "cpu_percent_one_core_100": (
                None if cpu_percent is None else round(cpu_percent, 3)
            ),
        })
        self._previous_ticks = ticks
        self._previous_time = now_mono

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> None:
        self._started_monotonic = time.monotonic()
        self._sample()
        self._thread = threading.Thread(
            target=self._run, name="recommendation-resource-probe", daemon=True
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2.0))
        self._sample()
        rss = [float(sample["rss_kib"]) for sample in self.samples]
        cpu = [float(sample["cpu_percent_one_core_100"])
               for sample in self.samples
               if sample["cpu_percent_one_core_100"] is not None]
        counts = [int(sample["process_count"]) for sample in self.samples]
        return {
            "root_pid": self.root_pid,
            "sampling_interval_seconds": self.interval,
            "clock_ticks_per_second": self._ticks_per_second,
            "sample_count": len(self.samples),
            "summary": {
                "rss_initial_mib": round(rss[0] / 1024.0, 3) if rss else None,
                "rss_peak_mib": round(max(rss) / 1024.0, 3) if rss else None,
                "rss_final_mib": round(rss[-1] / 1024.0, 3) if rss else None,
                "cpu_peak_percent_one_core_100": round(max(cpu), 3) if cpu else None,
                "cpu_mean_percent_one_core_100": round(statistics.fmean(cpu), 3) if cpu else None,
                "process_count_peak": max(counts) if counts else 0,
            },
            "process_commands": self._commands,
            "samples": self.samples,
        }


def _feed_path(user: str, *, limit: int, session: str | None = None) -> str:
    query: dict[str, str | int] = {"user": user, "limit": limit}
    if session:
        query["session"] = session
    return f"/api/feed?{urlencode(query)}"


def _compact_status(state: dict[str, Any], observation: dict[str, Any], started: float) -> dict[str, Any]:
    background = state.get("background_mining") or {}
    return {
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "request_latency_ms": observation["elapsed_ms"],
        "rule_version": state.get("version"),
        "pending_events": state.get("pending_events"),
        "state": background.get("state"),
        "running": bool(background.get("running")),
        "attempts": background.get("attempts"),
        "successes": background.get("successes"),
        "failures": background.get("failures"),
        "active_event_sequence": background.get("active_event_sequence"),
        "last_promoted_version": background.get("last_promoted_version"),
        "last_duration_seconds": background.get("last_duration_seconds"),
        "last_error": background.get("last_error"),
    }


def probe_target(
    base_url: str,
    *,
    server_pid: int | None,
    user: str | None,
    action: str,
    mine_interval: int | None,
    events: int | None,
    page_size: int | None,
    timeout: float,
    request_timeout: float,
    poll_interval: float,
    resource_interval: float,
) -> dict[str, Any]:
    """Exercise one live server and return a JSON-safe measurement record."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProbeError(f"invalid base URL: {base_url!r}")
    base_url = base_url.rstrip("/")
    sampler = ProcessTreeSampler(server_pid, resource_interval) if server_pid else None
    if sampler:
        sampler.start()
    started = time.monotonic()
    report: dict[str, Any] = {
        "base_url": base_url,
        "server_pid": server_pid,
        "started_at": _utc_now(),
        "http_observations": [],
        "event_requests": [],
        "background_timeline": [],
    }
    feed_latencies: list[float] = []
    event_latencies: list[float] = []
    try:
        initial, observation = _json_request(
            base_url, "/api/state", timeout=request_timeout
        )
        report["http_observations"].append(observation)
        report["initial_state"] = initial
        available_users = [str(row["id"]) for row in initial.get("users", [])]
        if not available_users:
            raise ProbeError("/api/state returned no users")
        selected_user = str(user) if user is not None else available_users[0]
        if selected_user not in available_users:
            raise ProbeError(f"unknown requested user {selected_user!r}")
        report["user"] = selected_user

        if mine_interval is not None:
            configured, observation = _json_request(
                base_url, "/api/config", method="POST",
                body={"mine_interval": int(mine_interval)},
                timeout=request_timeout,
            )
            report["http_observations"].append(observation)
            report["configuration_request"] = {
                "requested_mine_interval": int(mine_interval),
                "latency_ms": observation["elapsed_ms"],
                "effective_mine_interval": configured.get("config", {}).get("mine_interval"),
                "triggered_synchronous_mining": configured.get("mined") is not None,
            }

        before, observation = _json_request(
            base_url, "/api/state", timeout=request_timeout
        )
        report["http_observations"].append(observation)
        config = before.get("config") or {}
        effective_interval = int(config.get("mine_interval", 1))
        effective_page_size = int(page_size or config.get("top_k", 5))
        if effective_page_size < 1:
            raise ProbeError("page size must be positive")
        background_before = before.get("background_mining") or {}
        baseline_attempts = int(background_before.get("attempts") or 0)
        baseline_successes = int(background_before.get("successes") or 0)
        baseline_failures = int(background_before.get("failures") or 0)
        baseline_version = int(before.get("version") or 0)
        pending_before = int(before.get("pending_events") or 0)
        required_events = (
            int(events) if events is not None
            else max(1, effective_interval - pending_before)
        )
        if required_events < 1:
            raise ProbeError("event count must be positive")
        report["pre_interaction_state"] = before

        page, observation = _json_request(
            base_url,
            _feed_path(selected_user, limit=effective_page_size),
            timeout=request_timeout,
        )
        report["http_observations"].append(observation)
        feed_latencies.append(float(observation["elapsed_ms"]))
        rows = list(page.get("feed") or [])
        if not rows:
            raise ProbeError("initial /api/feed page was empty")
        session = str(page.get("session") or "")
        if not session:
            raise ProbeError("paginated /api/feed returned no session")
        report["initial_feed"] = {
            "latency_ms": observation["elapsed_ms"],
            "session": session,
            "rule_version": page.get("rule_version"),
            "queue_revision": page.get("queue_revision"),
            "article_ids": [str(row.get("article", {}).get("id")) for row in rows],
            "count": len(rows),
        }

        # Keep the normal configured-page latency above, but use a one-item
        # page for the negative-feedback causal check.  That leaves the rest
        # of the same arrival window unserved, giving PeTTa a real queue on
        # which a generalized exact/subcategory/topic skip proof can act.
        interaction_page_size = 1 if action == "skip" else effective_page_size
        if interaction_page_size != effective_page_size:
            interaction_page, interaction_observation = _json_request(
                base_url,
                _feed_path(selected_user, limit=interaction_page_size),
                timeout=request_timeout,
            )
            report["http_observations"].append(interaction_observation)
            feed_latencies.append(float(interaction_observation["elapsed_ms"]))
            interaction_rows = list(interaction_page.get("feed") or [])
            session = str(interaction_page.get("session") or "")
            report["causal_interaction_feed"] = {
                "latency_ms": interaction_observation["elapsed_ms"],
                "session": session,
                "article_ids": [
                    str(row.get("article", {}).get("id"))
                    for row in interaction_rows
                ],
                "count": len(interaction_rows),
            }
        else:
            interaction_rows = rows
        trigger_response: dict[str, Any] | None = None
        trigger_observation: dict[str, Any] | None = None
        causal_response: dict[str, Any] | None = None
        for event_index in range(required_events):
            if not interaction_rows:
                page, observation = _json_request(
                    base_url,
                    _feed_path(selected_user, limit=interaction_page_size, session=session),
                    timeout=request_timeout,
                )
                report["http_observations"].append(observation)
                feed_latencies.append(float(observation["elapsed_ms"]))
                session = str(page.get("session") or session)
                interaction_rows.extend(page.get("feed") or [])
            if not interaction_rows:
                raise ProbeError(
                    f"feed exhausted after {event_index} of {required_events} interactions"
                )
            row = interaction_rows.pop(0)
            article = str((row.get("article") or {}).get("id"))
            impression = row.get("impression")
            if not article or not impression:
                raise ProbeError("feed row lacks article id or live impression token")
            response, event_observation = _json_request(
                base_url, "/api/event", method="POST",
                body={
                    "user": selected_user,
                    "article": article,
                    "action": action,
                    "impression": impression,
                },
                timeout=request_timeout,
            )
            report["http_observations"].append(event_observation)
            event_latencies.append(float(event_observation["elapsed_ms"]))
            event_record = {
                "index": event_index + 1,
                "article": article,
                "action": action,
                "latency_ms": event_observation["elapsed_ms"],
                "pending_events": response.get("pending_events"),
                "mining_scheduled": response.get("mining_scheduled"),
                "background_state": (response.get("background_mining") or {}).get("state"),
                "background_running": bool((response.get("background_mining") or {}).get("running")),
                "queue_revision": response.get("queue_revision"),
                "mined_inline": response.get("mined") is not None,
            }
            report["event_requests"].append(event_record)
            if (response.get("queue_revision") or {}).get("causal_demotion"):
                causal_response = response
            if response.get("mining_scheduled") or (
                response.get("background_mining") or {}
            ).get("running"):
                trigger_response = response
                trigger_observation = event_observation
                # Automatic mode stops as soon as the configured threshold
                # starts a build. An explicit --events count is respected.
                if events is None:
                    break

        if trigger_response is None:
            raise ProbeError(
                "submitted interactions did not schedule or observe background mining"
            )

        event_background = trigger_response.get("background_mining") or {}
        running_observed = bool(event_background.get("running"))
        feed_during: dict[str, Any] | None = None
        # Ask for a fresh proof-ranked page, rather than draining only the old
        # queue. This tests scorer availability while fpMiner works in the
        # staged workspace.
        if running_observed:
            probe_user = available_users[1] if len(available_users) > 1 else selected_user
            during_page, during_observation = _json_request(
                base_url,
                _feed_path(probe_user, limit=effective_page_size),
                timeout=request_timeout,
            )
            report["http_observations"].append(during_observation)
            feed_latencies.append(float(during_observation["elapsed_ms"]))
            feed_during = {
                "user": probe_user,
                "latency_ms": during_observation["elapsed_ms"],
                "count": len(during_page.get("feed") or []),
                "rule_version": during_page.get("rule_version"),
                "request_started_at_epoch": during_observation["started_at_epoch"],
                "request_finished_at_epoch": during_observation["finished_at_epoch"],
            }

        deadline = time.monotonic() + timeout
        completed_state: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            state, state_observation = _json_request(
                base_url, "/api/state", timeout=request_timeout
            )
            report["http_observations"].append(state_observation)
            status = _compact_status(state, state_observation, started)
            report["background_timeline"].append(status)
            running_observed = running_observed or status["running"]
            attempts = int(status.get("attempts") or 0)
            failures = int(status.get("failures") or 0)
            if attempts > baseline_attempts and not status["running"]:
                completed_state = state
                break
            if failures > baseline_failures and not status["running"]:
                completed_state = state
                break
            time.sleep(poll_interval)
        if completed_state is None:
            raise ProbeError(f"background mining did not finish within {timeout:.1f}s")

        final_background = completed_state.get("background_mining") or {}
        background_duration = final_background.get("last_duration_seconds")
        last_completed_at = final_background.get("last_completed_at")
        if feed_during is not None and isinstance(last_completed_at, (int, float)):
            feed_during["overlapped_background_build"] = bool(
                feed_during["request_started_at_epoch"] <= float(last_completed_at)
            )
        elif feed_during is not None:
            feed_during["overlapped_background_build"] = running_observed
        final_version = int(completed_state.get("version") or 0)
        final_successes = int(final_background.get("successes") or 0)
        final_failures = int(final_background.get("failures") or 0)
        event_latency = float(trigger_observation["elapsed_ms"])
        report["feed_during_background_mining"] = feed_during
        report["threshold_behavior"] = {
            "configured_mine_interval": effective_interval,
            "pending_events_before": pending_before,
            "events_submitted": len(report["event_requests"]),
            "trigger_event_index": report["event_requests"][-1]["index"],
            "trigger_event_latency_ms": event_latency,
            "trigger_returned_inline_model": trigger_response.get("mined") is not None,
            "mining_scheduled": bool(trigger_response.get("mining_scheduled")),
            "running_observed": running_observed,
            "background_attempt_completed": int(final_background.get("attempts") or 0) > baseline_attempts,
            "background_success": final_successes > baseline_successes,
            "background_failure": final_failures > baseline_failures,
            "background_duration_seconds": background_duration,
            "event_returned_before_background_completed": (
                isinstance(background_duration, (int, float))
                and event_latency < float(background_duration) * 1000.0
            ),
            "active_rule_version_before": baseline_version,
            "active_rule_version_after": final_version,
            "active_rule_version_advanced": final_version > baseline_version,
            "last_error": final_background.get("last_error"),
        }
        revision = (
            (causal_response or trigger_response).get("queue_revision") or {}
        )
        negative_demotion = _negative_demotion_audit(revision)
        report["live_feedback"] = {
            "queue_revision": revision,
            "petta_reasoner_reported": revision.get("reasoner") == "PeTTaChainer",
            "changed_positions": revision.get("changed_positions"),
            "negative_proof_candidate_count": len(
                revision.get("negative_proof_candidates") or []
            ),
            "causal_negative_demotion": negative_demotion,
        }
        report["final_state"] = completed_state
        report["latency"] = {
            "feed": _latency_summary(feed_latencies),
            "event": _latency_summary(event_latencies),
        }
        report["checks"] = {
            "initial_feed_nonempty": report["initial_feed"]["count"] > 0,
            "all_events_returned_without_inline_mining": all(
                not row["mined_inline"] for row in report["event_requests"]
            ),
            "background_was_scheduled": bool(trigger_response.get("mining_scheduled")),
            "background_attempt_completed": int(final_background.get("attempts") or 0) > baseline_attempts,
            "rule_version_advanced": final_version > baseline_version,
            "feed_served_during_background": bool(
                feed_during and feed_during["count"] > 0
            ),
            "live_queue_reranked_by_petta": revision.get("reasoner") == "PeTTaChainer",
            "negative_feedback_has_causal_petta_demotion": (
                action != "skip" or bool(negative_demotion["passed"])
            ),
        }
        report["passed"] = all(report["checks"].values())
        report["finished_at"] = _utc_now()
        report["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return report
    finally:
        if sampler:
            report["process_tree_resources"] = sampler.stop()


def _default_output() -> Path:
    stamp = datetime.now().astimezone().date().isoformat()
    return RESULTS_DIR / f"realtime_{stamp}" / "http_probe.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure live feed/event latency, asynchronous mining promotion, "
            "and optional Linux process-tree resources without mocking the engine."
        )
    )
    parser.add_argument(
        "--base-url", action="append", required=True,
        help="Server URL; repeat for future multi-replica probes.",
    )
    parser.add_argument(
        "--server-pid", action="append", type=int, default=[],
        help="Server PID paired by order with --base-url; repeat as needed.",
    )
    parser.add_argument("--user", help="User id; defaults to the first live user.")
    parser.add_argument("--action", choices=("click", "skip", "like", "complete"), default="skip")
    parser.add_argument("--mine-interval", type=int, help="Set mine_interval through /api/config first.")
    parser.add_argument("--events", type=int, help="Explicit events; default reaches the next mining threshold.")
    parser.add_argument("--page-size", type=int, help="Feed page size; defaults to server top_k.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds to await background completion.")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=0.10)
    parser.add_argument("--resource-interval", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=_default_output())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.server_pid and len(args.server_pid) != len(args.base_url):
        raise SystemExit(
            "--server-pid must be omitted or repeated once per --base-url"
        )
    if args.mine_interval is not None and args.mine_interval < 1:
        raise SystemExit("--mine-interval must be positive")
    if args.events is not None and args.events < 1:
        raise SystemExit("--events must be positive")
    if min(args.timeout, args.request_timeout, args.poll_interval, args.resource_interval) <= 0:
        raise SystemExit("timeouts and sampling intervals must be positive")

    pids: list[int | None] = list(args.server_pid) or [None] * len(args.base_url)
    artifact: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": _utc_now(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "method": {
            "client": "stdlib urllib external HTTP client",
            "engine_mocks": False,
            "process_metrics": "Linux /proc aggregate of server PID and descendants",
            "cpu_unit": "100 percent equals one fully utilized logical CPU",
        },
        "targets": [],
    }
    failures = 0
    for base_url, server_pid in zip(args.base_url, pids):
        try:
            target = probe_target(
                base_url,
                server_pid=server_pid,
                user=args.user,
                action=args.action,
                mine_interval=args.mine_interval,
                events=args.events,
                page_size=args.page_size,
                timeout=args.timeout,
                request_timeout=args.request_timeout,
                poll_interval=args.poll_interval,
                resource_interval=args.resource_interval,
            )
        except BaseException as exc:
            failures += 1
            target = {
                "base_url": base_url,
                "server_pid": server_pid,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        artifact["targets"].append(target)
    artifact["passed"] = failures == 0 and all(
        bool(target.get("passed")) for target in artifact["targets"]
    )
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
        "targets": [
            {
                "base_url": target.get("base_url"),
                "passed": target.get("passed"),
                "latency": target.get("latency"),
                "threshold_behavior": target.get("threshold_behavior"),
                "resources": (target.get("process_tree_resources") or {}).get("summary"),
                "error": target.get("error"),
            }
            for target in artifact["targets"]
        ],
    }, indent=2, sort_keys=True))
    return 1 if failures or not artifact["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
