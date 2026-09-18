import threading
import time

from recommendation.app.async_mining import AsyncMiningCoordinator, MiningSnapshot


def test_schedule_is_nonblocking_and_promotes_after_build_finishes():
    release = threading.Event()
    build_started = threading.Event()
    active = {"version": 4, "value": "old"}

    def build(snapshot):
        build_started.set()
        assert release.wait(2)
        return "new"

    def promote(snapshot, value):
        active.update(version=active["version"] + 1, value=value)
        return {"promoted": True, "version": active["version"]}

    coordinator = AsyncMiningCoordinator(
        capture=lambda: MiningSnapshot(8, active["version"], None),
        build=build,
        promote=promote,
        should_retrigger=lambda: False,
        dispose=lambda _value: None,
    )
    started = time.monotonic()
    assert coordinator.schedule()
    assert time.monotonic() - started < 0.25
    assert build_started.wait(1)
    assert active == {"version": 4, "value": "old"}
    assert coordinator.status()["running"] is True

    release.set()
    assert coordinator.wait(2)
    assert active == {"version": 5, "value": "new"}
    status = coordinator.status()
    assert status["state"] == "idle"
    assert status["running"] is False
    assert status["attempts"] == 1
    assert status["successes"] == 1
    assert status["failures"] == 0
    assert status["stale_discards"] == 0
    assert status["retriggers"] == 0
    assert status["last_completed_event_sequence"] == 8
    assert status["last_promoted_version"] == 5
    assert status["last_error"] is None


def test_failure_is_visible_and_requires_a_new_trigger_before_retry():
    attempts = []
    fail = {"enabled": True}
    pending = {"needed": True}

    def build(snapshot):
        attempts.append(snapshot.event_sequence)
        if fail["enabled"]:
            raise RuntimeError("deliberate mining failure")
        return snapshot.event_sequence

    def promote(snapshot, value):
        pending["needed"] = False
        return {"promoted": True, "version": 2}

    coordinator = AsyncMiningCoordinator(
        capture=lambda: MiningSnapshot(9, 1, None),
        build=build,
        promote=promote,
        should_retrigger=lambda: pending["needed"],
        dispose=lambda _value: None,
    )
    assert coordinator.schedule()
    assert coordinator.wait(2)
    status = coordinator.status()
    assert attempts == [9]
    assert status["state"] == "failed"
    assert status["failures"] == 1
    assert status["successes"] == 0
    assert status["last_error"] == "RuntimeError: deliberate mining failure"

    fail["enabled"] = False
    assert coordinator.schedule()
    assert coordinator.wait(2)
    status = coordinator.status()
    assert attempts == [9, 9]
    assert status["state"] == "idle"
    assert status["failures"] == 1
    assert status["successes"] == 1
    assert status["last_error"] is None


def test_request_racing_failed_snapshot_capture_is_retried():
    capture_started = threading.Event()
    release_capture = threading.Event()
    state = {"captures": 0, "pending": True}

    def capture():
        state["captures"] += 1
        if state["captures"] == 1:
            capture_started.set()
            assert release_capture.wait(2)
            raise RuntimeError("snapshot capture failed")
        return MiningSnapshot(2, 1, None)

    def promote(_snapshot, _value):
        state["pending"] = False
        return {"promoted": True, "version": 2}

    coordinator = AsyncMiningCoordinator(
        capture=capture,
        build=lambda snapshot: snapshot.event_sequence,
        promote=promote,
        should_retrigger=lambda: state["pending"],
        dispose=lambda _value: None,
    )
    assert coordinator.schedule() is True
    assert capture_started.wait(1)
    # This request belongs to a later epoch than the capture already in
    # progress.  The first capture then fails, so the worker must not swallow
    # this request while transitioning to idle.
    assert coordinator.schedule() is False
    release_capture.set()

    assert coordinator.wait(2)
    status = coordinator.status()
    assert state == {"captures": 2, "pending": False}
    assert status["attempts"] == 2
    assert status["failures"] == 1
    assert status["successes"] == 1
    assert status["retriggers"] == 1
    assert status["last_completed_event_sequence"] == 2


def test_events_during_successful_build_retrigger_exactly_one_followup():
    lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()
    state = {"sequence": 8, "mined": 0, "version": 1, "builds": 0}

    def capture():
        with lock:
            return MiningSnapshot(state["sequence"], state["version"], None)

    def build(snapshot):
        with lock:
            state["builds"] += 1
            attempt = state["builds"]
        if attempt == 1:
            first_started.set()
            assert release_first.wait(2)
        return snapshot.event_sequence

    def promote(snapshot, value):
        with lock:
            state["mined"] = value
            state["version"] += 1
            return {"promoted": True, "version": state["version"]}

    def should_retrigger():
        with lock:
            return state["sequence"] - state["mined"] >= 8

    coordinator = AsyncMiningCoordinator(
        capture=capture,
        build=build,
        promote=promote,
        should_retrigger=should_retrigger,
        dispose=lambda _value: None,
    )
    assert coordinator.schedule()
    assert first_started.wait(1)
    with lock:
        state["sequence"] = 16
    assert coordinator.schedule() is False
    release_first.set()

    assert coordinator.wait(2)
    status = coordinator.status()
    assert state == {"sequence": 16, "mined": 16, "version": 3, "builds": 2}
    assert status["attempts"] == 2
    assert status["successes"] == 2
    assert status["retriggers"] == 1
    assert status["last_completed_event_sequence"] == 16


def test_one_event_during_build_does_not_retrigger_before_interval():
    lock = threading.Lock()
    first_started = threading.Event()
    release_first = threading.Event()
    state = {"sequence": 8, "mined": 0, "version": 1, "builds": 0}

    def capture():
        with lock:
            return MiningSnapshot(state["sequence"], state["version"], None)

    def build(snapshot):
        with lock:
            state["builds"] += 1
        first_started.set()
        assert release_first.wait(2)
        return snapshot.event_sequence

    def promote(snapshot, value):
        with lock:
            state["mined"] = value
            state["version"] += 1
            return {"promoted": True, "version": state["version"]}

    def should_retrigger():
        with lock:
            return state["sequence"] - state["mined"] >= 8

    coordinator = AsyncMiningCoordinator(
        capture=capture,
        build=build,
        promote=promote,
        should_retrigger=should_retrigger,
        dispose=lambda _value: None,
    )
    assert coordinator.schedule()
    assert first_started.wait(1)
    with lock:
        state["sequence"] = 9
    assert coordinator.schedule() is False
    release_first.set()

    assert coordinator.wait(2)
    status = coordinator.status()
    assert state == {"sequence": 9, "mined": 8, "version": 2, "builds": 1}
    assert status["attempts"] == 1
    assert status["successes"] == 1
    assert status["retriggers"] == 0
    assert status["last_completed_event_sequence"] == 8


def test_stale_build_is_disposed_without_becoming_a_success():
    disposed = []
    coordinator = AsyncMiningCoordinator(
        capture=lambda: MiningSnapshot(8, 1, None),
        build=lambda _snapshot: "stale-worker",
        promote=lambda _snapshot, _value: {"promoted": False},
        should_retrigger=lambda: False,
        dispose=disposed.append,
    )
    assert coordinator.schedule()
    assert coordinator.wait(2)
    assert disposed == ["stale-worker"]
    status = coordinator.status()
    assert status["successes"] == 0
    assert status["stale_discards"] == 1


def test_trigger_racing_idle_transition_is_not_lost():
    lock = threading.Lock()
    first_decision = threading.Event()
    allow_idle_transition = threading.Event()
    state = {"sequence": 8, "mined": 0}
    captures = []

    def capture():
        with lock:
            sequence = state["sequence"]
            captures.append(sequence)
            return MiningSnapshot(sequence, len(captures) - 1, object())

    def promote(snapshot, _built):
        with lock:
            state["mined"] = snapshot.event_sequence
        return {"promoted": True, "version": snapshot.base_version + 1}

    decision_calls = 0

    def should_retrigger():
        nonlocal decision_calls
        with lock:
            needed = state["sequence"] - state["mined"] >= 8
            decision_calls += 1
            call = decision_calls
        if call == 1:
            first_decision.set()
            assert allow_idle_transition.wait(1)
        return needed

    coordinator = AsyncMiningCoordinator(
        capture=capture,
        build=lambda snapshot: snapshot.payload,
        promote=promote,
        should_retrigger=should_retrigger,
        dispose=lambda _built: None,
    )
    assert coordinator.schedule() is True
    assert first_decision.wait(1)
    # This observes the first worker as running. The final epoch check must
    # re-check the threshold before letting the worker publish idle.
    with lock:
        state["sequence"] = 16
    assert coordinator.schedule() is False
    allow_idle_transition.set()
    assert coordinator.wait(1)
    assert captures == [8, 16]
    assert coordinator.status()["retriggers"] == 1
