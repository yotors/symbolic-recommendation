"""Hard outer-deadline and recovery tests for the isolated scorer."""

from __future__ import annotations

import time

import pytest

from recommendation.app.server import IsolatedPeTTaChainer


def _deadline_probe_worker(connection, statements, mutations=()):
    """Small spawned protocol peer; no PeTTa dependency is exercised here."""
    state=list(statements)
    for command,payload in mutations:
        if command=="add":
            state.extend(payload)
        elif command=="remove":
            state=[item for item in state if str(payload) not in str(item)]
    connection.send(("ok",None))
    try:
        while True:
            command,payload=connection.recv()
            if command=="stop":
                connection.send(("ok",None))
                return
            if command=="add":
                state.extend(payload)
                connection.send(("ok",None))
            elif command=="remove":
                state=[item for item in state if str(payload) not in str(item)]
                connection.send(("ok",None))
            elif command=="query_many":
                queries,_steps=payload
                if any(query=="hang" for query in queries):
                    time.sleep(60)
                connection.send(("ok",[[str(len(state))] for _ in queries]))
    except (EOFError,BrokenPipeError):
        return


def test_expired_rpc_retires_worker_and_recovers_committed_snapshot():
    engine=IsolatedPeTTaChainer(
        worker_target=_deadline_probe_worker,
        default_timeout_seconds=0.1,
        startup_timeout_seconds=5.0,
    )
    try:
        engine.replace(["rule_snapshot"])
        engine.add_atoms_no_check(["grounded_fact"],timeout_sec=1.0)
        old_pid=engine.pid

        with pytest.raises(TimeoutError,match="worker snapshot recovered"):
            engine.query_many(["hang"],steps=1,timeout_sec=0.05)

        assert engine.pid is not None
        assert engine.pid!=old_pid
        assert engine.recovery_count==1
        assert engine.last_timeout["command"]=="query_many"
        assert engine.last_timeout["worker_recovered"] is True
        # The replacement contains both the compiled snapshot and the last
        # acknowledged lazy fact; the timed-out request itself was not replayed.
        assert engine.query_many(["count"],steps=1,timeout_sec=1.0)==[["2"]]
    finally:
        engine.close()


@pytest.mark.parametrize("value",[-1,float("nan"),float("inf")])
def test_nonfinite_or_negative_rpc_deadline_is_rejected(value):
    engine=IsolatedPeTTaChainer(
        worker_target=_deadline_probe_worker,
        startup_timeout_seconds=5.0,
    )
    try:
        engine.replace(["rule_snapshot"])
        with pytest.raises(ValueError,match="finite number greater than 0"):
            engine.query_many(["count"],steps=1,timeout_sec=value)
    finally:
        engine.close()


def test_mutation_journal_fails_closed_before_worker_state_changes():
    engine=IsolatedPeTTaChainer(
        worker_target=_deadline_probe_worker,
        startup_timeout_seconds=5.0,
        max_journal_statements=2,
        max_journal_mutations=2,
    )
    try:
        engine.replace(["rule_snapshot"])
        engine.add_atoms_no_check(["fact_1","fact_2"],timeout_sec=1.0)
        assert engine.journal_audit=={
            "mutations":1,"statements":2,
            "mutation_limit":2,"statement_limit":2,
        }
        with pytest.raises(RuntimeError,match="statement journal limit"):
            engine.add_atoms_no_check(["fact_3"],timeout_sec=1.0)
        # Rejection happened before the command was sent.
        assert engine.query_many(["count"],steps=1,timeout_sec=1.0)==[["3"]]
        assert engine.journal_audit["statements"]==2

        # Atomic replacement is the explicit compaction boundary.
        engine.replace(["new_rule_snapshot"])
        assert engine.journal_audit["statements"]==0
        assert engine.journal_audit["mutations"]==0
        assert engine.query_many(["count"],steps=1,timeout_sec=1.0)==[["1"]]
    finally:
        engine.close()


def test_mutation_record_limit_counts_removals_and_rejects_before_send():
    engine=IsolatedPeTTaChainer(
        worker_target=_deadline_probe_worker,
        startup_timeout_seconds=5.0,
        max_journal_statements=10,
        max_journal_mutations=1,
    )
    try:
        engine.replace(["rule_snapshot"])
        engine.remove_statement("absent",timeout_sec=1.0)
        with pytest.raises(RuntimeError,match="mutation journal limit"):
            engine.remove_statement("rule_snapshot",timeout_sec=1.0)
        assert engine.query_many(["count"],steps=1,timeout_sec=1.0)==[["1"]]
    finally:
        engine.close()
