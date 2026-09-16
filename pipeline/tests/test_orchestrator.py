"""Test Harness for Master Orchestrator, Scheduler Safety, and Circuit Breaker.

Verifies:
1. Overlapping scheduler triggers: second run refuses to start while first holds lock.
2. Stale lock recovery: dead PID detected from crashed run, pruned, and execution proceeds.
3. Circuit breaker trip: 3 consecutive failures trip the breaker, blocking the 4th run.
4. Circuit breaker durability: state file survives process exit and blocks independent child process.
5. Circuit breaker recovery: manual reset re-arms the pipeline for normal execution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import SchedulerLock, is_pid_alive
from pipeline.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError


class TestOrchestratorSafety(unittest.TestCase):
    """Verifies scheduler locking, stale lock recovery, and circuit breaker resilience."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_orch_"))
        self.lock_file = self.temp_dir / "pipeline.lock"
        self.cb_state = self.temp_dir / "circuit_breaker_state.json"
        self.alert_file = self.temp_dir / "alerts.log"
        self.breaker = CircuitBreaker(state_file=self.cb_state, alert_file=self.alert_file, max_failures=3)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # TEST 1: Overlapping Scheduler Triggers
    # -------------------------------------------------------------------------
    def test_overlapping_scheduler_triggers_blocked(self):
        """Proves that a second concurrent scheduler invocation detects the active
        PID and exits immediately without executing.
        """
        lock_1 = SchedulerLock(self.lock_file)
        # Process 1 acquires lock
        self.assertTrue(lock_1.acquire())
        self.assertTrue(self.lock_file.exists())

        # Process 2 attempts to acquire same lock
        lock_2 = SchedulerLock(self.lock_file)
        # Must return False because lock_1's PID (current test process) is alive
        self.assertFalse(lock_2.acquire())
        print("\n[CRITERIA 1 PASS] Overlapping trigger rejected: second instance refused to run.")

        # Clean up lock_1
        lock_1.release()
        self.assertFalse(self.lock_file.exists())

    # -------------------------------------------------------------------------
    # TEST 2: Stale Lock Recovery
    # -------------------------------------------------------------------------
    def test_stale_lock_recovery(self):
        """Proves that a lock file left behind by a dead/crashed PID is detected,
        cleared, and execution proceeds normally.
        """
        dead_pid = 999998
        # Ensure dead_pid is indeed dead
        self.assertFalse(is_pid_alive(dead_pid))

        # Write stale lock
        stale_data = {
            "pid": dead_pid,
            "started_at": "2026-09-16T12:00:00Z"
        }
        with open(self.lock_file, "w", encoding="utf-8") as f:
            json.dump(stale_data, f)

        self.assertTrue(self.lock_file.exists())

        # New scheduler run arrives
        fresh_lock = SchedulerLock(self.lock_file)
        acquired = fresh_lock.acquire()

        # Must detect dead PID, clear lock, and acquire cleanly
        self.assertTrue(acquired)
        self.assertTrue(self.lock_file.exists())
        with open(self.lock_file, "r", encoding="utf-8") as f:
            active_data = json.load(f)
        self.assertEqual(active_data["pid"], os.getpid())
        print("\n[CRITERIA 2 PASS] Stale lock recovery verified: dead PID detected, pruned, fresh lock acquired.")

        fresh_lock.release()
        self.assertFalse(self.lock_file.exists())

    # -------------------------------------------------------------------------
    # TEST 3: Circuit Breaker Trips After 3 Consecutive Failures
    # -------------------------------------------------------------------------
    def test_circuit_breaker_trips_after_3_failures(self):
        """Injects 3 consecutive pipeline failures; proves breaker trips and blocks 4th run."""
        self.assertTrue(self.breaker.can_execute()[0])

        # Failure 1
        self.breaker.record_failure("Stage 1 (Ideator): API quota rate limit exceeded")
        self.assertFalse(self.breaker.is_tripped())
        self.assertEqual(self.breaker.get_consecutive_failures(), 1)
        self.assertTrue(self.breaker.can_execute()[0])

        # Failure 2
        self.breaker.record_failure("Stage 2 (Scriptwriter): Validation failed on 3 attempts")
        self.assertFalse(self.breaker.is_tripped())
        self.assertEqual(self.breaker.get_consecutive_failures(), 2)
        self.assertTrue(self.breaker.can_execute()[0])

        # Failure 3 (Threshold reached!)
        self.breaker.record_failure("Stage 3 (Director): Subprocess render_worker timed out")
        self.assertTrue(self.breaker.is_tripped())
        self.assertEqual(self.breaker.get_consecutive_failures(), 3)

        # 4th run must be BLOCKED
        allowed, alert_msg = self.breaker.can_execute()
        self.assertFalse(allowed)
        self.assertIn("CIRCUIT BREAKER IS OPEN", alert_msg)
        self.assertIn("Stage 1 (Ideator)", alert_msg)
        self.assertIn("Stage 2 (Scriptwriter)", alert_msg)
        self.assertIn("Stage 3 (Director)", alert_msg)

        # Confirm alert file was written to disk
        self.assertTrue(self.alert_file.exists())
        with open(self.alert_file, "r", encoding="utf-8") as f:
            alert_content = f.read()
        self.assertIn("CRITICAL ALERT: PIPELINE CIRCUIT BREAKER TRIPPED", alert_content)
        print("\n[CRITERIA 3 PASS] Circuit breaker tripped after 3 failures: 4th run blocked, alert logged.")

    # -------------------------------------------------------------------------
    # TEST 4: Durability Across Process Restarts
    # -------------------------------------------------------------------------
    def test_circuit_breaker_persistence_across_process_restart(self):
        """Spawns an independent Python subprocess that records 3 failures, terminates,
        and verifies that a second independent process reads the tripped state from disk.
        """
        state_path_str = str(self.cb_state).replace("\\", "/")
        alert_path_str = str(self.alert_file).replace("\\", "/")

        # Subprocess 1: Trips breaker and exits
        proc1_cmd = [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, r'{ROOT_DIR}'); "
            f"from pipeline.core.circuit_breaker import CircuitBreaker; "
            f"cb = CircuitBreaker(state_file=r'{state_path_str}', alert_file=r'{alert_path_str}'); "
            f"cb.record_failure('Proc1 Fail A'); "
            f"cb.record_failure('Proc1 Fail B'); "
            f"cb.record_failure('Proc1 Fail C'); "
            f"print('PROC1_DONE')"
        ]
        res1 = subprocess.run(proc1_cmd, capture_output=True, text=True, check=True)
        self.assertIn("PROC1_DONE", res1.stdout)

        # Subprocess 1 has completely exited and terminated.
        # Subprocess 2: Starts with zero in-memory context and verifies state file was preserved.
        proc2_cmd = [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, r'{ROOT_DIR}'); "
            f"from pipeline.core.circuit_breaker import CircuitBreaker; "
            f"cb = CircuitBreaker(state_file=r'{state_path_str}', alert_file=r'{alert_path_str}'); "
            f"allowed, _ = cb.can_execute(); "
            f"print(f'PROC2_TRIPPED:{{cb.is_tripped()}}'); "
            f"print(f'PROC2_ALLOWED:{{allowed}}'); "
            f"print(f'PROC2_FAILURES:{{cb.get_consecutive_failures()}}')"
        ]
        res2 = subprocess.run(proc2_cmd, capture_output=True, text=True, check=True)
        self.assertIn("PROC2_TRIPPED:True", res2.stdout)
        self.assertIn("PROC2_ALLOWED:False", res2.stdout)
        self.assertIn("PROC2_FAILURES:3", res2.stdout)
        print("\n[CRITERIA 3 PASS] Durability confirmed: breaker state survived complete process restart.")

    # -------------------------------------------------------------------------
    # TEST 5: Recovery After Manual Reset
    # -------------------------------------------------------------------------
    def test_circuit_breaker_recovery_after_reset(self):
        """Verifies that calling reset() re-arms the breaker and permits normal execution."""
        # Trip the breaker first
        for i in range(3):
            self.breaker.record_failure(f"Error {i}")
        self.assertTrue(self.breaker.is_tripped())

        # Reset
        self.breaker.reset()
        self.assertFalse(self.breaker.is_tripped())
        self.assertEqual(self.breaker.get_consecutive_failures(), 0)

        # Execution allowed
        allowed, msg = self.breaker.can_execute()
        self.assertTrue(allowed)
        self.assertIn("closed", msg.lower())
        print("\n[CRITERIA 4 PASS] Recovery verified: manual reset re-arms pipeline for scheduled execution.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
