"""Single-flight background mining orchestration.

The coordinator deliberately knows nothing about recommendation data or PeTTa.
Callers provide four small lifecycle functions: capture an immutable job input,
build a replacement, atomically promote it, and decide whether events that
arrived during the build justify another pass.  This keeps scheduling state out
of the symbolic engine and makes failure/retrigger behaviour deterministic.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass(frozen=True)
class MiningSnapshot:
    """Immutable identity plus caller-owned payload for one mining attempt."""

    event_sequence: int
    base_version: int
    payload: Any


class AsyncMiningCoordinator:
    """Run at most one background model build and safely coalesce triggers."""

    def __init__(
        self,
        *,
        capture: Callable[[], MiningSnapshot],
        build: Callable[[MiningSnapshot], Any],
        promote: Callable[[MiningSnapshot, Any], Mapping[str, Any]],
        should_retrigger: Callable[[], bool],
        dispose: Callable[[Any], None],
        thread_name: str = "recommendation-background-miner",
    ) -> None:
        self._capture = capture
        self._build = build
        self._promote = promote
        self._should_retrigger = should_retrigger
        self._dispose = dispose
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._running = False
        self._request_epoch = 0
        self._attempts = 0
        self._successes = 0
        self._failures = 0
        self._stale_discards = 0
        self._retriggers = 0
        self._active_event_sequence: int | None = None
        self._last_completed_event_sequence: int | None = None
        self._last_promoted_version: int | None = None
        self._last_started_at: float | None = None
        self._last_completed_at: float | None = None
        self._last_duration_seconds: float | None = None
        self._last_error: str | None = None

    def schedule(self) -> bool:
        """Request mining and return whether this call started the worker."""
        with self._condition:
            if self._closed:
                return False
            self._request_epoch += 1
            if self._running:
                return False
            self._running = True
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()
            return True

    def status(self) -> dict[str, Any]:
        """Return a JSON-safe scheduling snapshot."""
        with self._lock:
            if self._closed:
                state = "closed"
            elif self._running:
                state = "running"
            elif self._last_error is not None:
                state = "failed"
            elif self._successes or self._stale_discards:
                state = "idle"
            else:
                state = "idle"
            return {
                "state": state,
                "running": self._running,
                "attempts": self._attempts,
                "successes": self._successes,
                "failures": self._failures,
                "stale_discards": self._stale_discards,
                "retriggers": self._retriggers,
                "active_event_sequence": self._active_event_sequence,
                "last_completed_event_sequence": self._last_completed_event_sequence,
                "last_promoted_version": self._last_promoted_version,
                "last_started_at": self._last_started_at,
                "last_completed_at": self._last_completed_at,
                "last_duration_seconds": self._last_duration_seconds,
                "last_error": self._last_error,
            }

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current trigger chain; intended for shutdown and tests."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._running:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, *, wait: bool = False, timeout: float | None = None) -> bool:
        """Prevent promotion/new work; an in-flight foreign call is not killed."""
        with self._condition:
            self._closed = True
            running = self._running
            self._condition.notify_all()
        if wait and running:
            return self.wait(timeout)
        return not running

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        text = f"{type(exc).__name__}: {exc}"
        return text[:2000]

    def _stable_retrigger_decision(
        self,
        *,
        failed: bool,
        attempt_epoch: int,
    ) -> tuple[bool, int]:
        """Return a threshold decision and the request epoch it observed.

        The caller validates the returned epoch while publishing either the
        next attempt or the idle state.  A request arriving during a build is
        therefore a prompt to re-check ``should_retrigger``; it is not, by
        itself, sufficient reason to rebuild the same almost-identical
        snapshot.
        """
        while True:
            with self._lock:
                if self._closed:
                    return False, self._request_epoch
                observed_epoch = self._request_epoch
            needed = bool(self._should_retrigger())
            with self._lock:
                if self._closed:
                    return False, self._request_epoch
                if observed_epoch != self._request_epoch:
                    continue
                # A failed batch is retried only after a genuinely newer request.
                # This prevents a permanent miner error from becoming a hot loop.
                return (
                    needed and (not failed or observed_epoch > attempt_epoch),
                    observed_epoch,
                )

    def _run(self) -> None:
        while True:
            snapshot: MiningSnapshot | None = None
            built: Any = None
            promoted = False
            failed = False
            error: str | None = None
            outcome: Mapping[str, Any] = {}
            started_monotonic = time.monotonic()
            with self._condition:
                self._attempts += 1
                self._last_started_at = time.time()
                self._active_event_sequence = None
                # This epoch belongs to the attempt that is about to capture
                # its immutable input.  A schedule() call racing with capture
                # is a newer request even when capture itself raises.  Taking
                # the epoch after capture would incorrectly assign that newer
                # request to the failed attempt and could publish idle without
                # retrying it.
                attempt_epoch = self._request_epoch
            try:
                snapshot = self._capture()
                with self._lock:
                    self._active_event_sequence = snapshot.event_sequence
                built = self._build(snapshot)
                with self._lock:
                    closed = self._closed
                if not closed:
                    outcome = self._promote(snapshot, built)
                    promoted = bool(outcome.get("promoted"))
                if not promoted:
                    self._dispose(built)
                    built = None
            except BaseException as exc:
                failed = True
                error = self._error_text(exc)
                if built is not None:
                    try:
                        self._dispose(built)
                    except BaseException:
                        pass
            duration = time.monotonic() - started_monotonic
            with self._condition:
                self._last_completed_at = time.time()
                self._last_duration_seconds = round(duration, 6)
                self._active_event_sequence = None
                if failed:
                    self._failures += 1
                    self._last_error = error
                elif promoted:
                    self._successes += 1
                    self._last_error = None
                    if snapshot is not None:
                        self._last_completed_event_sequence = snapshot.event_sequence
                    version = outcome.get("version")
                    if isinstance(version, int):
                        self._last_promoted_version = version
                else:
                    self._stale_discards += 1
                    self._last_error = None

            while True:
                retry, decision_epoch = self._stable_retrigger_decision(
                    failed=failed,
                    attempt_epoch=attempt_epoch,
                )
                with self._condition:
                    if self._closed:
                        self._running = False
                        self._condition.notify_all()
                        return
                    # ``schedule`` can race the small interval after the
                    # callback. Re-evaluate its threshold under the new epoch;
                    # never treat the request itself as a reason to mine.
                    if decision_epoch != self._request_epoch:
                        continue
                    if retry:
                        self._retriggers += 1
                        break
                    self._running = False
                    self._condition.notify_all()
                    return


__all__ = ["AsyncMiningCoordinator", "MiningSnapshot"]
