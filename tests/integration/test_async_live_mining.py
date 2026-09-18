import threading
import time

import pytest

from recommendation.app.server import Lab, fixture


pytestmark = [pytest.mark.integration, pytest.mark.pettachainer]


def _fake_promoter(lab):
    def promote(snapshot, staged):
        # These orchestration tests keep the already compiled scorer active;
        # production promotion is covered by the existing atomic-worker test.
        staged.engine.close()
        staged.engine = None
        with lab.lock:
            if lab.version != snapshot.base_version:
                return {"promoted": False}
            lab.version += 1
            lab._last_mined_event_sequence = snapshot.event_sequence
            lab.pending_events = max(
                0, lab._event_sequence - snapshot.event_sequence
            )
            return {"promoted": True, "version": lab.version}

    return promote


def test_threshold_event_returns_while_background_build_is_running():
    lab = Lab(data=fixture())
    release = threading.Event()
    started = threading.Event()
    original_version = lab.version
    try:
        lab.configure({"mine_interval": 1})

        def build(snapshot):
            started.set()
            assert release.wait(2)
            return snapshot.payload

        lab._build_background_mining = build
        lab._promote_background_mining = _fake_promoter(lab)
        before = time.monotonic()
        result = lab.event("u1", "n1", "skip", impression="async_test_1")

        assert time.monotonic() - before < 0.25
        assert result["mined"] is None
        assert result["mining_scheduled"] is True
        assert started.wait(1)
        assert lab.version == original_version
        assert lab.score("u1", candidates=["n1"], limit=1)

        release.set()
        assert lab._background_mining.wait(2)
        assert lab.version == original_version + 1
        assert lab.state()["background_mining"]["state"] == "idle"
    finally:
        release.set()
        lab.close()


def test_background_failure_keeps_scorer_and_retries_on_next_event():
    lab = Lab(data=fixture())
    original_pid = lab.engine.pid
    fail = {"enabled": True}
    attempts = []
    try:
        lab.configure({"mine_interval": 1})

        def build(snapshot):
            attempts.append(snapshot.event_sequence)
            if fail["enabled"]:
                snapshot.payload.engine.close()
                snapshot.payload.engine = None
                raise RuntimeError("deliberate staged failure")
            return snapshot.payload

        lab._build_background_mining = build
        lab._promote_background_mining = _fake_promoter(lab)
        lab.event("u1", "n1", "skip", impression="async_failure_1")
        assert lab._background_mining.wait(2)
        status = lab.state()["background_mining"]
        assert status["state"] == "failed"
        assert status["last_error"] == "RuntimeError: deliberate staged failure"
        assert status["attempts"] == 1
        assert lab.engine.pid == original_pid
        assert lab.score("u1", candidates=["n1"], limit=1)

        fail["enabled"] = False
        retry = lab.event("u1", "n2", "skip", impression="async_failure_2")
        assert retry["mining_scheduled"] is True
        assert lab._background_mining.wait(2)
        status = lab.state()["background_mining"]
        assert attempts == [1, 2]
        assert status["failures"] == 1
        assert status["successes"] == 1
        assert status["last_error"] is None
        assert lab.engine.pid == original_pid
    finally:
        lab.close()


def test_events_arriving_during_build_trigger_one_followup_snapshot():
    lab = Lab(data=fixture())
    release = threading.Event()
    first_started = threading.Event()
    attempts = []
    original_version = lab.version
    try:
        lab.configure({"mine_interval": 1})

        def build(snapshot):
            attempts.append(snapshot.event_sequence)
            if len(attempts) == 1:
                first_started.set()
                assert release.wait(2)
            return snapshot.payload

        lab._build_background_mining = build
        lab._promote_background_mining = _fake_promoter(lab)
        first = lab.event("u1", "n1", "skip", impression="async_retrigger_1")
        assert first["mining_scheduled"] is True
        assert first_started.wait(1)
        second = lab.event("u1", "n2", "skip", impression="async_retrigger_2")
        assert second["mining_scheduled"] is False
        release.set()

        assert lab._background_mining.wait(3)
        status = lab.state()["background_mining"]
        assert attempts == [1, 2]
        assert status["attempts"] == 2
        assert status["successes"] == 2
        assert status["retriggers"] == 1
        assert lab.version == original_version + 2
        assert lab.pending_events == 0
    finally:
        release.set()
        lab.close()
