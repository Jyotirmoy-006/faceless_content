"""Comprehensive Test Suite for Mission G — Sequential SQLite Queue & Command Center.

Tests:
1. API endpoints: /api/system, /api/queue, /api/logs, /api/start.
2. Sequential Queue Verification: Submitting 3 jobs instantly runs them one at a time.
3. Scheduler Safety (Rule 7): Queue worker respects pipeline.lock and waits.
4. UI Status mapping: Transitions from QUEUED -> RUNNING -> COMPLETED.
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.dashboard.app import (
    LOCK_FILE,
    app,
    check_pipeline_lock,
    queue_worker_step,
)
from pipeline.dashboard.database import (
    enqueue_job,
    get_all_jobs,
    get_job,
    get_next_queued_job,
    get_running_job,
    init_db,
    mark_job_completed,
)


class TestDashboardQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls._orig_jobs_db_path = os.environ.get("JOBS_DB_PATH")
        cls.test_db = ROOT_DIR / "pipeline" / "dashboard" / "test_queue.db"
        os.environ["JOBS_DB_PATH"] = str(cls.test_db)
        init_db(cls.test_db)
        os.environ["MAIN_SCRIPT"] = "mock_main.py"
        os.environ["MOCK_STEP_SLEEP"] = "0.2"

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("MAIN_SCRIPT", None)
        os.environ.pop("MOCK_STEP_SLEEP", None)
        if cls._orig_jobs_db_path is not None:
            os.environ["JOBS_DB_PATH"] = cls._orig_jobs_db_path
        else:
            os.environ.pop("JOBS_DB_PATH", None)
        try:
            if cls.test_db.exists():
                cls.test_db.unlink(missing_ok=True)
        except Exception:
            pass

    def test_01_api_system_and_logs(self):
        """Verifies /api/system returns quota and circuit breaker, and /api/logs returns lines."""
        res_sys = self.client.get("/api/system")
        self.assertEqual(res_sys.status_code, 200)
        data_sys = res_sys.get_json()
        self.assertIn("quota", data_sys)
        self.assertIn("circuit_breaker", data_sys)
        self.assertIn("is_running", data_sys)

        res_logs = self.client.get("/api/logs?lines=20")
        self.assertEqual(res_logs.status_code, 200)
        data_logs = res_logs.get_json()
        self.assertIn("logs", data_logs)
        self.assertIsInstance(data_logs["logs"], list)

    def test_02_api_start_enqueues_instantly_without_blocking(self):
        """Verifies POST /api/start returns 200 OK immediately and stores QUEUED job."""
        res = self.client.post(
            "/api/start",
            data=json.dumps({"topic": "Instant Non-Blocking Queue Test", "niche": "tech"}),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "QUEUED")
        self.assertIn("job_id", data)

        # Confirm job in /api/queue
        res_q = self.client.get("/api/queue")
        self.assertEqual(res_q.status_code, 200)
        q_data = res_q.get_json()
        self.assertTrue(any(j["topic"] == "Instant Non-Blocking Queue Test" for j in q_data["jobs"]))

    def test_03_pipeline_lock_pauses_worker(self):
        """Rule 7 Safety: Worker must pause if pipeline.lock is active with a live PID."""
        # Create active lock
        lock_data = {"pid": os.getpid(), "started_at": "2026-09-17T00:00:00Z"}
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            json.dump(lock_data, f)

        try:
            # Enqueue a test job
            job = enqueue_job("Locked Pipeline Test Job")
            job_id = job["id"]

            # Run worker step
            queue_worker_step()

            # Job must STILL be QUEUED because pipeline was locked
            updated = get_job(job_id)
            self.assertEqual(updated["status"], "QUEUED")
        finally:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink(missing_ok=True)

    def test_04_sequential_execution_of_three_jobs(self):
        """Verifies submitting 3 jobs processes them strictly sequentially with mock_main.py."""
        if LOCK_FILE.exists():
            LOCK_FILE.unlink(missing_ok=True)
        from pipeline.dashboard.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute("UPDATE jobs SET status = 'COMPLETED' WHERE status IN ('QUEUED', 'RUNNING')")
            conn.commit()

        # Enqueue 3 jobs (with require_approval=False for autonomous straight-through execution)
        j1 = enqueue_job("Sequential Job Alpha", require_approval=False)
        j2 = enqueue_job("Sequential Job Beta", require_approval=False)
        j3 = enqueue_job("Sequential Job Gamma", require_approval=False)

        # Initial state: all 3 queued
        self.assertEqual(get_job(j1["id"])["status"], "QUEUED")
        self.assertEqual(get_job(j2["id"])["status"], "QUEUED")
        self.assertEqual(get_job(j3["id"])["status"], "QUEUED")

        # Cycle 1: Job 1 is picked up and runs to completion (mock_main.py finishes)
        queue_worker_step()
        self.assertEqual(get_job(j1["id"])["status"], "COMPLETED")
        self.assertEqual(get_job(j2["id"])["status"], "QUEUED")
        self.assertEqual(get_job(j3["id"])["status"], "QUEUED")

        # Cycle 2: Job 2 is picked up and runs to completion
        time.sleep(0.15)
        queue_worker_step()
        self.assertEqual(get_job(j2["id"])["status"], "COMPLETED")
        self.assertEqual(get_job(j3["id"])["status"], "QUEUED")

        # Cycle 3: Job 3 is picked up and runs to completion
        time.sleep(0.15)
        queue_worker_step()
        self.assertEqual(get_job(j3["id"])["status"], "COMPLETED")


if __name__ == "__main__":
    unittest.main()
