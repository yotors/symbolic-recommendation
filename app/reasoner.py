"""Process-isolated PeTTaChainer runtime with bounded recovery."""
from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
import traceback

from pettachainer.pettachainer import PeTTaChainer


DEFAULT_REASONER_RPC_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_REASONER_JOURNAL_STATEMENTS = 250_000
DEFAULT_MAX_REASONER_JOURNAL_MUTATIONS = 50_000


def _scoring_worker(connection, statements, mutations=()):
    """Own one PeTTaChainer runtime in an isolated spawned process.

    PeTTa's compiled rule/index spaces are process-global even though each
    ``PeTTaChainer`` instance has a distinct KB name.  Keeping the live scorer
    behind this small RPC boundary prevents an old Lab/KB from consuming a
    later Lab's finite backward-search budget.
    """
    try:
        engine=PeTTaChainer()
        engine.set_backward_premise_prefilter(True)
        engine.add_atoms_no_check(list(statements))
        # Replaying only parent-acknowledged mutations reconstructs the exact
        # committed scorer state after an outer-deadline abort.  The command
        # that timed out is intentionally absent: whether it reached PeTTa is
        # unknowable, so retrying it implicitly would violate at-most-once RPC
        # semantics.
        for command,payload in mutations:
            if command=="add":
                engine.add_atoms_no_check(list(payload))
            elif command=="remove":
                engine.remove_statement(str(payload))
            else:
                raise ValueError(f"unknown scoring-worker bootstrap mutation: {command}")
        connection.send(("ok",None))
    except BaseException as exc:  # The parent must retain its previous worker.
        try: connection.send(("error",f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        finally: connection.close()
        return
    try:
        while True:
            command,payload=connection.recv()
            try:
                if command=="add":
                    result=engine.add_atoms_no_check(payload)
                elif command=="remove":
                    result=engine.remove_statement(payload)
                elif command=="query_many":
                    queries,steps=payload
                    result=engine.query_many(queries,steps=steps,timeout_sec=0)
                elif command=="stop":
                    connection.send(("ok",None)); break
                else:
                    raise ValueError(f"unknown scoring-worker command: {command}")
                connection.send(("ok",result))
            except BaseException as exc:
                connection.send(("error",f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    except (EOFError,BrokenPipeError):
        pass
    finally:
        connection.close()


class IsolatedPeTTaChainer:
    """A replaceable PeTTaChainer whose global MeTTa state cannot leak.

    A remine starts a clean worker with the complete new rule snapshot and
    swaps it in only after compilation succeeds. Candidate facts are then
    loaded lazily into that snapshot. This is deliberately a process boundary:
    constructing a second PeTTaChainer in this process would still share the
    old global rule indexes.
    """
    def __init__(self, *, worker_target=None,
                 default_timeout_seconds=DEFAULT_REASONER_RPC_TIMEOUT_SECONDS,
                 startup_timeout_seconds=120.0,
                 max_journal_statements=DEFAULT_MAX_REASONER_JOURNAL_STATEMENTS,
                 max_journal_mutations=DEFAULT_MAX_REASONER_JOURNAL_MUTATIONS):
        self._context=mp.get_context("spawn")
        self._lock=threading.RLock()
        self._connection=None; self._process=None
        self._worker_target=worker_target or _scoring_worker
        self._default_timeout_seconds=self._finite_timeout(
            default_timeout_seconds,"default reasoner RPC timeout"
        )
        self._startup_timeout_seconds=self._finite_timeout(
            startup_timeout_seconds,"reasoner worker startup timeout"
        )
        self._max_journal_statements=self._positive_integer(
            max_journal_statements,"max reasoner journal statements"
        )
        self._max_journal_mutations=self._positive_integer(
            max_journal_mutations,"max reasoner journal mutations"
        )
        self._bootstrap_statements=()
        self._has_snapshot=False
        self._mutation_journal=[]
        self._journal_statements=0
        self._recovery_count=0
        self._last_timeout=None

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def recovery_count(self):
        return self._recovery_count

    @property
    def last_timeout(self):
        return dict(self._last_timeout) if self._last_timeout is not None else None

    @property
    def journal_audit(self):
        return {
            "mutations":len(self._mutation_journal),
            "statements":self._journal_statements,
            "mutation_limit":self._max_journal_mutations,
            "statement_limit":self._max_journal_statements,
        }

    @staticmethod
    def _positive_integer(value,label):
        if isinstance(value,bool):
            raise ValueError(f"{label} must be a positive integer")
        try:
            parsed=int(value)
        except (TypeError,ValueError) as exc:
            raise ValueError(f"{label} must be a positive integer") from exc
        if parsed<=0 or parsed!=value:
            raise ValueError(f"{label} must be a positive integer")
        return parsed

    @staticmethod
    def _finite_timeout(value,label):
        try:
            timeout=float(value)
        except (TypeError,ValueError) as exc:
            raise ValueError(f"{label} must be a finite number greater than 0") from exc
        if not math.isfinite(timeout) or timeout<=0.0:
            raise ValueError(f"{label} must be a finite number greater than 0")
        return timeout

    @staticmethod
    def _shutdown(connection,process):
        if connection is not None:
            try:
                connection.send(("stop",None))
                if connection.poll(2): connection.recv()
            except (BrokenPipeError,EOFError,OSError):
                pass
            finally:
                try: connection.close()
                except OSError: pass
        if process is not None and process.pid is not None:
            process.join(5)
            if process.is_alive():
                process.terminate(); process.join(5)

    @staticmethod
    def _terminate(connection,process):
        """Abort a possibly wedged worker without waiting for its protocol."""
        if connection is not None:
            try: connection.close()
            except OSError: pass
        if process is not None and process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(5)
            if process.is_alive() and hasattr(process,"kill"):
                process.kill(); process.join(5)

    @staticmethod
    def _receive(connection,process,timeout=None,operation="request"):
        deadline=time.monotonic()+timeout if timeout is not None else None
        while True:
            wait_seconds=1.0
            if deadline is not None:
                remaining=deadline-time.monotonic()
                if remaining<=0.0:
                    raise TimeoutError(
                        f"PeTTaChainer worker {operation} exceeded its "
                        f"{timeout:g}s outer deadline"
                    )
                wait_seconds=min(wait_seconds,remaining)
            if connection.poll(wait_seconds):
                break
            if not process.is_alive():
                raise RuntimeError(
                    f"PeTTaChainer worker exited unexpectedly ({process.exitcode})"
                )
        try:
            status,payload=connection.recv()
        except EOFError as exc:
            exit_code=process.exitcode
            raise RuntimeError(f"PeTTaChainer worker exited unexpectedly ({exit_code})") from exc
        if status!="ok": raise RuntimeError(f"PeTTaChainer worker failure: {payload}")
        return payload

    def _launch_locked(self,statements,mutations):
        parent,child=self._context.Pipe()
        process=self._context.Process(
            target=self._worker_target,
            args=(child,tuple(statements),tuple(mutations)),
            name="recommendation-pettachainer",daemon=True,
        )
        try:
            process.start(); child.close()
            self._receive(
                parent,process,timeout=self._startup_timeout_seconds,
                operation="startup",
            )
        except BaseException:
            try: child.close()
            except OSError: pass
            self._terminate(parent,process)
            raise
        return parent,process

    def _recover_locked(self):
        if not self._has_snapshot:
            raise RuntimeError("PeTTaChainer worker has no recoverable snapshot")
        connection,process=self._launch_locked(
            self._bootstrap_statements,self._mutation_journal
        )
        self._connection,self._process=connection,process
        self._recovery_count+=1

    def replace(self,statements):
        """Atomically replace the scorer with a clean compiled snapshot."""
        with self._lock:
            statements=tuple(statements)
            parent,process=self._launch_locked(statements,())
            old_connection,old_process=self._connection,self._process
            self._connection,self._process=parent,process
            self._bootstrap_statements=statements
            self._has_snapshot=True
            self._mutation_journal=[]
            self._journal_statements=0
            self._shutdown(old_connection,old_process)

    def _rpc(self,command,payload,*,timeout_sec=None,journal=False):
        with self._lock:
            if self._connection is None or self._process is None:
                raise RuntimeError("PeTTaChainer worker has no compiled rule snapshot")
            if timeout_sec is None:
                timeout=self._default_timeout_seconds
            else:
                try: requested_timeout=float(timeout_sec)
                except (TypeError,ValueError) as exc:
                    raise ValueError(
                        "reasoner RPC timeout must be a finite number greater than 0"
                    ) from exc
                # Zero was the historical spelling for "no in-process
                # timeout". Preserve call compatibility without preserving an
                # unbounded outer wait: it now selects the finite default.
                timeout=(self._default_timeout_seconds
                         if requested_timeout==0.0 else self._finite_timeout(
                             requested_timeout,"reasoner RPC timeout"
                         ))
            journal_units=0
            if journal:
                journal_units=(len(payload) if command=="add" else 1)
                if len(self._mutation_journal)+1>self._max_journal_mutations:
                    raise RuntimeError(
                        "PeTTaChainer mutation journal limit exceeded before "
                        "worker mutation; promote a fresh rule snapshot"
                    )
                if (self._journal_statements+journal_units
                        > self._max_journal_statements):
                    raise RuntimeError(
                        "PeTTaChainer statement journal limit exceeded before "
                        "worker mutation; promote a fresh rule snapshot"
                    )
            connection,process=self._connection,self._process
            try:
                connection.send((command,payload))
                result=self._receive(
                    connection,process,timeout=timeout,operation=command
                )
                if journal:
                    committed=(tuple(payload) if command=="add" else str(payload))
                    self._mutation_journal.append((command,committed))
                    self._journal_statements+=journal_units
                return result
            except TimeoutError as exc:
                # A deadline makes the worker state unknowable. Kill it rather
                # than consuming a late response, then restore the last fully
                # acknowledged snapshot. The failed caller still receives a
                # timeout and decides whether its operation is safe to retry.
                self._connection=None; self._process=None
                self._terminate(connection,process)
                recovered=False; recovery_error=None
                try:
                    self._recover_locked(); recovered=True
                except BaseException as recovery_exc:
                    recovery_error=(
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
                self._last_timeout={
                    "command":str(command),"timeout_seconds":timeout,
                    "worker_recovered":recovered,
                    "recovery_error":recovery_error,
                    "occurred_at":time.time(),
                }
                suffix=("worker snapshot recovered" if recovered else
                        f"worker recovery failed ({recovery_error})")
                raise TimeoutError(f"{exc}; {suffix}") from exc
            except BaseException:
                # A failed RPC may leave unread protocol bytes or partial MeTTa
                # state. Retire it; the caller can remine a clean snapshot.
                self._connection=None; self._process=None
                self._shutdown(connection,process)
                raise

    def add_atoms_no_check(self,atoms,timeout_sec=None):
        return self._rpc(
            "add",list(atoms),timeout_sec=timeout_sec,journal=True
        )

    def remove_statement(self,atom_name,timeout_sec=None):
        return self._rpc(
            "remove",str(atom_name),timeout_sec=timeout_sec,journal=True
        )

    def query_many(self,atoms,steps=100,timeout_sec=None):
        # The worker is itself the timeout/isolation boundary; its in-process
        # PeTTa query must not fork a second runtime.
        return self._rpc(
            "query_many",(list(atoms),int(steps)),timeout_sec=timeout_sec
        )

    def close(self):
        with self._lock:
            connection,process=self._connection,self._process
            self._connection=None; self._process=None
            self._shutdown(connection,process)

    def __del__(self):
        try: self.close()
        except BaseException: pass
