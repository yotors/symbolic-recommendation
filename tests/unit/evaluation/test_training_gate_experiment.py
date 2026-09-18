import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from recommendation.evaluation.training_gate_experiment import (
    _AUDITED_SOURCE_PATHS,
    _attempt_path,
    _claim_attempt,
    _finalize_attempt,
    _preflight_new_output,
    _runtime_provenance,
)
from recommendation.paths import WORKSPACE_ROOT


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _claim_in_process(registry: str, fingerprint: str, start, results) -> None:
    start.wait()
    try:
        _claim_attempt(Path(registry), fingerprint)
    except BaseException as exc:
        results.put(("blocked", type(exc).__name__, str(exc)))
    else:
        results.put(("claimed", None, None))


class DurableTrainingConfirmationRegistryTest(unittest.TestCase):
    def test_same_cohort_is_blocked_independently_of_output_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            first_output = root / "first.json"
            different_output = root / "different.json"
            fingerprint = _fingerprint("same-cohort")

            _preflight_new_output(first_output)
            first = _claim_attempt(registry, fingerprint)
            self.assertEqual(first["status"], "running")
            self.assertEqual(first["origin"], "pre_run")
            self.assertIsNone(first["artifact_sha256"])

            # A different, still-unused artifact path does not create a new
            # statistical cohort or permit another architecture attempt.
            _preflight_new_output(different_output)
            with self.assertRaisesRegex(ValueError, "already consumed.*running"):
                _claim_attempt(registry, fingerprint)

            mode = _attempt_path(registry, fingerprint).stat().st_mode & 0o777
            self.assertEqual(mode, 0o600)

    def test_different_cohorts_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            first = _claim_attempt(registry, _fingerprint("cohort-one"))
            second = _claim_attempt(registry, _fingerprint("cohort-two"))
            self.assertEqual(first["status"], "running")
            self.assertEqual(second["status"], "running")

    def test_corrupt_existing_record_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            registry.mkdir()
            fingerprint = _fingerprint("corrupt-cohort")
            _attempt_path(registry, fingerprint).write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "corrupt.*remains consumed"):
                _claim_attempt(registry, fingerprint)
            with self.assertRaisesRegex(RuntimeError, "corrupt.*remains consumed"):
                _finalize_attempt(registry, fingerprint, "error")

    def test_running_claim_and_every_final_result_remain_consumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            for status in ("pass", "hold", "error"):
                with self.subTest(status=status):
                    fingerprint = _fingerprint(f"final-{status}")
                    _claim_attempt(registry, fingerprint)
                    artifact_sha256 = (
                        None if status == "error" else _fingerprint(f"artifact-{status}")
                    )
                    finalized = _finalize_attempt(
                        registry, fingerprint, status,
                        artifact_sha256=artifact_sha256,
                    )
                    self.assertEqual(finalized["status"], status)
                    self.assertEqual(
                        finalized["schema"],
                        "recommendation-training-confirmation-attempt-v2",
                    )
                    self.assertEqual(finalized["origin"], "pre_run")
                    self.assertEqual(finalized["artifact_sha256"], artifact_sha256)
                    self.assertIsNotNone(finalized["finalized_at"])
                    stored = json.loads(
                        _attempt_path(registry, fingerprint).read_text(encoding="utf-8")
                    )
                    self.assertEqual(stored, finalized)
                    self.assertFalse(any(registry.glob(f".{fingerprint}.*.tmp")))
                    with self.assertRaisesRegex(ValueError, f"already consumed.*{status}"):
                        _claim_attempt(registry, fingerprint)

    def test_final_status_enforces_artifact_hash_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            hold_fingerprint = _fingerprint("hold-without-artifact")
            _claim_attempt(registry, hold_fingerprint)
            with self.assertRaisesRegex(ValueError, "pass or hold.*artifact hash"):
                _finalize_attempt(registry, hold_fingerprint, "hold")
            _finalize_attempt(registry, hold_fingerprint, "error")

            error_fingerprint = _fingerprint("error-with-artifact")
            _claim_attempt(registry, error_fingerprint)
            with self.assertRaisesRegex(ValueError, "error attempt.*artifact hash"):
                _finalize_attempt(
                    registry, error_fingerprint, "error",
                    artifact_sha256=_fingerprint("should-not-bind"),
                )
            _finalize_attempt(registry, error_fingerprint, "error")

    def test_cross_process_claim_has_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            registry = str(Path(temporary) / "registry")
            fingerprint = _fingerprint("concurrent-cohort")
            context = multiprocessing.get_context("fork")
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_claim_in_process,
                    args=(registry, fingerprint, start, results),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start.set()
            outcomes = [results.get(timeout=10) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)

            self.assertEqual([row[0] for row in outcomes].count("claimed"), 1)
            self.assertEqual([row[0] for row in outcomes].count("blocked"), 1)
            blocked = next(row for row in outcomes if row[0] == "blocked")
            self.assertEqual(blocked[1], "ValueError")
            self.assertIn("already consumed", blocked[2])

    def test_existing_output_fails_before_any_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            output = root / "existing.json"
            output.write_text("immutable", encoding="utf-8")
            fingerprint = _fingerprint("unclaimed-cohort")

            with self.assertRaisesRegex(ValueError, "output must be a new artifact"):
                _preflight_new_output(output)
            self.assertFalse(_attempt_path(registry, fingerprint).exists())

    def test_runtime_provenance_hashes_all_declared_core_sources(self):
        provenance = _runtime_provenance()
        source_hashes = provenance["selected_core_source_sha256"]
        self.assertEqual(set(source_hashes), set(_AUDITED_SOURCE_PATHS))
        required = {
            "recommendation/app/server.py",
            "recommendation/evaluation/training_gate.py",
            "recommendation/miner/fpMiner.metta",
            "PeTTaChainer/pettachainer/pettachainer.py",
            "PeTTaChainer/pettachainer/metta/petta_chainer.metta",
        }
        self.assertTrue(required.issubset(source_hashes))
        for relative, digest in source_hashes.items():
            self.assertEqual(
                digest,
                hashlib.sha256((WORKSPACE_ROOT / relative).read_bytes()).hexdigest(),
            )
        runtime = provenance["runtime"]
        for key in (
            "python_implementation", "python_version", "python_full_version",
            "python_cache_tag", "operating_system", "machine",
        ):
            self.assertTrue(runtime[key])
        self.assertEqual(set(runtime["packages"]), {"janus-swi", "numpy"})


if __name__ == "__main__":
    unittest.main()
