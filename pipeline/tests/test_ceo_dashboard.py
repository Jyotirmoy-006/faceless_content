"""Comprehensive Test Suite for Mission: Flask CEO — Read-Only Dashboard (Rule 7).

Verifies:
1. Flask app never calls, triggers, or schedules main.py itself (Rule 7 enforcement).
2. Zero APScheduler imports or dependencies.
3. Strictly GET-only endpoints (/status, /). Zero POST routes.
4. SQLite telemetry table persistence and retrieval (record_run_telemetry, get_latest_run_telemetry).
5. /status correctly reflects the most recent real run's telemetry and system health.
6. Subprocess boundary & exit code observation pattern (proc.wait() + proc.returncode).
"""

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ceo_dashboard import app, build_status_payload, LOCK_FILE
from pipeline.core.circuit_breaker import circuit_breaker
from pipeline.core.quota_tracker import quota_tracker
from pipeline.dashboard.database import (
    init_db,
    record_run_telemetry,
    get_latest_run_telemetry,
    get_run_telemetry_history
)


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite DB path for testing."""
    db_file = tmp_path / "test_jobs.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def client(temp_db):
    """Configures Flask test client with the isolated test DB."""
    app.config["TESTING"] = True
    app.config["DB_PATH"] = str(temp_db)
    with app.test_client() as c:
        yield c


class TestCEOArchitectureAndSafety:
    """Verifies architectural guarantees ensuring Flask never acts as a scheduler."""

    def test_01_no_apscheduler_or_triggering_logic_in_ceo_dashboard(self):
        """Rule 7: Verifies ceo_dashboard.py contains zero APScheduler imports or triggering logic."""
        file_path = ROOT_DIR / "ceo_dashboard.py"
        content = file_path.read_text(encoding="utf-8")

        # Must not import or instantiate APScheduler
        assert "import apscheduler" not in content.lower(), "ceo_dashboard.py must not import apscheduler"
        assert "from apscheduler" not in content.lower(), "ceo_dashboard.py must not import from apscheduler"
        assert "backgroundscheduler" not in content.lower(), "ceo_dashboard.py must not use BackgroundScheduler"

        # Must not import main.py directly or start subprocesses in routes
        assert "import main" not in content, "ceo_dashboard.py must not import main directly"
        assert "from main import" not in content, "ceo_dashboard.py must not import from main directly"
        assert "subprocess.Popen" not in content, "ceo_dashboard.py must not spawn subprocesses"

    def test_02_strictly_read_only_routes(self):
        """Rule 7: Verifies all routes in ceo_dashboard are strictly GET-only (zero POST routes)."""
        for rule in app.url_map.iter_rules():
            # Filter out static endpoint
            if rule.endpoint == "static":
                continue
            methods = rule.methods - {"HEAD", "OPTIONS"}
            assert methods == {"GET"}, f"Route {rule.rule} has non-GET methods: {methods}"

    def test_03_windows_task_scheduler_remains_sole_trigger(self):
        """Rule 7: Verifies metadata in status payload documents Windows Task Scheduler as sole trigger."""
        payload = build_status_payload()
        arch = payload.get("architecture", {})
        assert "Windows Task Scheduler" in arch.get("execution_trigger", "")
        assert arch.get("scheduling_allowed") is False


class TestTelemetryDatabasePersistence:
    """Verifies SQLite telemetry table operations."""

    def test_04_record_and_retrieve_successful_run(self, temp_db):
        """Verifies full telemetry payload persists and unpacks correctly."""
        telemetry_data = {
            "concept": {"topic": "Quantum Computing Reality", "niche": "science"},
            "total_wall_clock_time": 47.85,
            "video_path": "pipeline/output/20260917_quantum.mp4",
            "video_size_mb": 14.2,
            "quota_spent": 50,
            "stages": {
                "ideation": 2.1,
                "scriptwriting": 3.4,
                "production": 40.2,
                "youtube_publish": 2.15
            },
            "results": {
                "youtube": {"status": "SUCCESS", "post_id": "yt_quantum_123"}
            },
            "vram": {
                "pre_render_mb": 110.0,
                "post_render_mb": 135.5
            }
        }

        row_id = record_run_telemetry(telemetry_data, status="SUCCESS", db_path=temp_db)
        assert row_id > 0

        latest = get_latest_run_telemetry(temp_db)
        assert latest is not None
        assert latest["topic"] == "Quantum Computing Reality"
        assert latest["niche"] == "science"
        assert latest["status"] == "SUCCESS"
        assert latest["duration_seconds"] == pytest.approx(47.85, rel=1e-2)
        assert latest["video_path"] == "pipeline/output/20260917_quantum.mp4"
        assert latest["video_size_mb"] == 14.2
        assert latest["quota_spent"] == 50
        assert latest["stages"]["production"] == 40.2
        assert latest["results"]["youtube"]["post_id"] == "yt_quantum_123"
        assert latest["vram"]["pre_render_mb"] == 110.0

    def test_05_record_failed_run_telemetry(self, temp_db):
        """Verifies failure telemetry captures error message."""
        failed_telem = {
            "concept": {"topic": "Failed Run Video", "niche": "tech"},
            "total_wall_clock_time": 5.2,
            "stages": {"ideation": 1.2, "scriptwriting": 4.0},
            "quota_spent": 0
        }

        record_run_telemetry(
            failed_telem,
            status="FAILED",
            error_message="Simulated ComfyUI VRAM Out of Memory",
            db_path=temp_db
        )

        latest = get_latest_run_telemetry(temp_db)
        assert latest is not None
        assert latest["status"] == "FAILED"
        assert latest["error_message"] == "Simulated ComfyUI VRAM Out of Memory"
        assert latest["topic"] == "Failed Run Video"

    def test_06_telemetry_history_order(self, temp_db):
        """Verifies get_run_telemetry_history returns runs in reverse chronological order."""
        for i in range(3):
            record_run_telemetry(
                {"concept": {"topic": f"Run #{i+1}"}, "total_wall_clock_time": float(i + 10)},
                status="SUCCESS",
                db_path=temp_db
            )

        history = get_run_telemetry_history(limit=10, db_path=temp_db)
        assert len(history) == 3
        assert history[0]["topic"] == "Run #3"
        assert history[1]["topic"] == "Run #2"
        assert history[2]["topic"] == "Run #1"


class TestCEOStatusEndpoints:
    """Verifies /status and / endpoints on the Flask app."""

    def test_07_status_endpoint_empty_database(self, client):
        """Verifies /status returns clean idle state when no runs exist."""
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "idle"
        assert data["latest_run"] is None
        assert "circuit_breaker" in data
        assert "quota" in data
        assert "scheduler_lock" in data

    def test_08_status_endpoint_reflects_recorded_run(self, client, temp_db):
        """Verifies /status accurately reflects newly persisted telemetry."""
        record_run_telemetry(
            {
                "concept": {"topic": "Autonomous AI Evolution", "niche": "tech"},
                "total_wall_clock_time": 33.2,
                "video_path": "pipeline/output/test_video.mp4",
                "video_size_mb": 9.8,
                "quota_spent": 0,
                "stages": {"ideation": 1.0, "scriptwriting": 2.0, "production": 30.2}
            },
            status="SUCCESS",
            db_path=temp_db
        )

        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        latest = data["latest_run"]
        assert latest["topic"] == "Autonomous AI Evolution"
        assert latest["status"] == "SUCCESS"
        assert latest["duration_seconds"] == pytest.approx(33.2, rel=1e-2)

    def test_09_status_reflects_circuit_breaker_trip(self, client, monkeypatch):
        """Verifies /status reflects degraded health when circuit breaker is tripped."""
        monkeypatch.setattr(circuit_breaker, "is_tripped", lambda: True)
        resp = client.get("/status")
        data = resp.get_json()
        assert data["status"] == "degraded"
        assert data["circuit_breaker"]["is_tripped"] is True
        assert data["circuit_breaker"]["state"] == "OPEN (TRIPPED)"

    def test_10_index_html_renders_read_only_dashboard(self, client):
        """Verifies root route returns modern HTML dashboard with Rule 7 banner."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Executive Control Node" in html
        assert "Rule 7 Compliance (Scheduler Safety)" in html
        assert "Windows Task Scheduler" in html
        assert "STRICTLY READ-ONLY" in html


class TestSubprocessExitCodeObservation:
    """Verifies launching main.py across process boundaries and observing proc.returncode."""

    def test_11_subprocess_launch_and_returncode_observation(self, temp_db, monkeypatch):
        """Requirement 3: External runners must use subprocess.Popen + proc.wait() + check proc.returncode."""
        env = os.environ.copy()
        # Direct main.py to write to our test DB
        env["TELEMETRY_DB_PATH"] = str(temp_db)

        # Launch main.py with --status flag (dry check that completes in <1s)
        proc = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "main.py"), "--status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT_DIR),
            env=env
        )

        stdout, stderr = proc.communicate(timeout=15)
        # Cannot catch sys.exit across process boundaries; must check proc.returncode
        assert proc.returncode == 0, f"Process failed with code {proc.returncode}: {stderr}"
        assert "Pipeline Operational Status" in stdout

    def test_12_mock_pipeline_subprocess_run_writes_telemetry_to_sqlite(self, client, temp_db):
        """Spawns pipeline as subprocess, checks exit code, and verifies /status reflects telemetry."""
        env = os.environ.copy()
        env["TELEMETRY_DB_PATH"] = str(temp_db)
        env["MOCK_STEP_SLEEP"] = "0.05"

        proc = subprocess.Popen(
            [
                sys.executable,
                str(ROOT_DIR / "mock_main.py"),
                "--topic",
                "Autonomous Drone Swarms",
                "--skip-publish"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT_DIR),
            env=env
        )

        proc.wait(timeout=15)
        assert proc.returncode == 0, "mock_main.py failed to exit with code 0"

        # Query /status through Flask app
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["latest_run"] is not None
        assert data["latest_run"]["topic"] == "Autonomous Drone Swarms"
        assert data["latest_run"]["status"] == "SUCCESS"
        assert data["latest_run"]["video_size_mb"] == 18.4

    def test_13_pipeline_circuit_breaker_exit_code_observation(self, temp_db, monkeypatch):
        """Verifies process boundary observation when circuit breaker trips (exit code 2)."""
        env = os.environ.copy()
        env["TELEMETRY_DB_PATH"] = str(temp_db)
        # Create a temporary circuit breaker state file with tripped state
        cb_state_path = ROOT_DIR / "pipeline" / "core" / "test_cb_state.json"
        cb_state_path.write_text(json.dumps({
            "consecutive_failures": 3,
            "max_failures": 3,
            "is_open": True,
            "tripped_at": "2026-09-17T11:00:00Z",
            "history": ["Fatal test error 1", "Fatal test error 2", "Fatal test error 3"]
        }), encoding="utf-8")

        env["CIRCUIT_BREAKER_STATE_FILE"] = str(cb_state_path)

        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT_DIR / "main.py"),
                    "--skip-publish",
                    "--dry-run"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(ROOT_DIR),
                env=env
            )
            stdout, stderr = proc.communicate(timeout=15)
            # Circuit breaker trip must cause exit code 2
            assert proc.returncode == 2, f"Expected exit code 2 (CircuitBreakerOpen), got {proc.returncode}. STDOUT: {stdout} STDERR: {stderr}"
        finally:
            if cb_state_path.exists():
                cb_state_path.unlink()


