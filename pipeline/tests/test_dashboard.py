"""Automated Test Suite for Mission G — Flask Web Dashboard Control Plane.

Tests:
1. Web server initialization and Jinja2 HTML rendering.
2. /api/status metrology (lock status, quota metrics, circuit breaker).
3. /api/logs file-tailing and streaming.
4. SQLite database job lifecycle (create, update, retrieve).
5. Rule 7 (Scheduler Safety) Lock Conflict: Rejects /api/start with 409 Conflict when pipeline.lock is active.
6. Asset Inspector video streaming and static serving.
"""

import json
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.dashboard.app import app, LOCK_FILE, LOG_FILE, OUTPUT_DIR
from pipeline.dashboard.database import create_job, get_history, get_job, init_db, update_job


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.test_db_path = ROOT_DIR / "pipeline" / "dashboard" / "test_jobs.db"
        init_db(cls.test_db_path)

    @classmethod
    def tearDownClass(cls):
        import gc
        gc.collect()
        try:
            if cls.test_db_path.exists():
                cls.test_db_path.unlink(missing_ok=True)
        except Exception:
            pass

    def test_01_index_page_renders_html(self):
        """Verifies GET / returns HTTP 200 and contains core Command Center elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("COMMAND CENTER", html)
        self.assertIn("Queue Video", html)
        self.assertIn("pipeline.log", html)
        self.assertIn("Asset Inspector", html)
        self.assertIn("Execution Queue", html)

    def test_02_api_status_endpoint(self):
        """Verifies GET /api/status returns valid JSON structure with quota and circuit breaker."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("is_running", data)
        self.assertIn("quota", data)
        self.assertIn("circuit_breaker", data)
        self.assertIn("gpu", data)
        self.assertIsInstance(data["is_running"], bool)
        self.assertIn("available", data["quota"])
        self.assertIn("can_execute", data["circuit_breaker"])

    def test_03_api_logs_tailing(self):
        """Verifies GET /api/logs returns recent lines from pipeline.log."""
        response = self.client.get("/api/logs?lines=20")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("logs", data)
        self.assertIn("total_lines", data)
        self.assertIsInstance(data["logs"], list)

    def test_04_sqlite_database_lifecycle(self):
        """Verifies SQLite jobs database records creation, status updates, and history retrieval."""
        import uuid
        test_job_id = f"test_job_lifecycle_{uuid.uuid4().hex[:8]}"
        job = create_job(
            job_id=test_job_id,
            topic="Testing SQLite Persistence",
            niche="science",
            pid=99999,
            status="RUNNING",
            db_path=self.test_db_path
        )
        self.assertEqual(job["job_id"], test_job_id)
        self.assertEqual(job["status"], "RUNNING")
        self.assertEqual(job["niche"], "science")

        # Update status
        updated = update_job(
            job_id=test_job_id,
            status="SUCCESS",
            video_path="pipeline/output/sample.mp4",
            db_path=self.test_db_path
        )
        self.assertEqual(updated["status"], "SUCCESS")
        self.assertEqual(updated["video_path"], "pipeline/output/sample.mp4")

        # Verify history
        history = get_history(limit=10, db_path=self.test_db_path)
        self.assertTrue(any(j["job_id"] == test_job_id for j in history))

    def test_05_api_start_enqueues_job(self):
        """Under the new Queue architecture, /api/start enqueues the job and returns 200 OK."""
        response = self.client.post(
            "/api/start",
            json={"topic": "Queued Test Topic", "niche": "tech"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "QUEUED")
        self.assertIn("job_id", data)

    def test_06_asset_inspector_videos_and_streaming(self):
        """Verifies /api/videos lists generated assets and /videos/<path> streams them with video/mp4."""
        response = self.client.get("/api/videos")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("videos", data)
        self.assertIsInstance(data["videos"], list)

        if data["videos"]:
            first_video = data["videos"][0]
            filename = first_video["filename"]
            # Request static stream
            stream_res = self.client.get(f"/videos/{filename}")
            self.assertEqual(stream_res.status_code, 200)
            self.assertEqual(stream_res.mimetype, "video/mp4")


if __name__ == "__main__":
    unittest.main()
