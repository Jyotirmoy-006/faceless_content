import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.dashboard.database import (
    init_db,
    enqueue_job,
    get_job,
    get_next_queued_job,
    get_running_job,
    get_pending_review_jobs,
    update_job_stage,
    pause_job_for_review,
    approve_job,
    mark_job_completed,
)
from pipeline.dashboard.app import app, queue_worker_step


@pytest.fixture
def temp_env(monkeypatch):
    """Creates an isolated temporary SQLite DB and outputs directory for test runs."""
    tmp_dir = Path(tempfile.mkdtemp())
    db_file = tmp_dir / "test_jobs.db"
    init_db(db_file)

    import pipeline.dashboard.database as db_mod
    monkeypatch.setattr(db_mod, "DEFAULT_DB_PATH", db_file)
    monkeypatch.setenv("JOBS_DB_PATH", str(db_file))

    # Patch app.py database paths as well
    monkeypatch.setattr("pipeline.dashboard.app.init_db", lambda: init_db(db_file))

    # Clear stale pipeline lock if left over by prior tests
    lock_file = ROOT_DIR / "pipeline.lock"
    if lock_file.exists():
        lock_file.unlink(missing_ok=True)

    yield tmp_dir, db_file
    if lock_file.exists():
        lock_file.unlink(missing_ok=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_database_stage_and_approval_fields(temp_env):
    """Verifies that enqueue_job stores approval and publish target flags accurately."""
    _, db_file = temp_env

    job = enqueue_job(
        topic="Neural Quantum Computing",
        niche="science",
        require_approval=True,
        publish_target="disk_only",
        db_path=db_file
    )

    assert job["topic"] == "Neural Quantum Computing"
    assert job["niche"] == "science"
    assert job["status"] == "QUEUED"
    assert job["current_stage"] == "QUEUED"
    assert job["require_approval"] == 1
    assert job["publish_target"] == "disk_only"
    assert job["script_json"] is None

    # Update stage to IDEATION
    updated = update_job_stage(job["id"], "IDEATION", db_path=db_file)
    assert updated["current_stage"] == "IDEATION"

    # Update stage to SCRIPTWRITING
    updated = update_job_stage(job["id"], "SCRIPTWRITING", db_path=db_file)
    assert updated["current_stage"] == "SCRIPTWRITING"

    # Pause for review
    mock_script = json.dumps({"hook": "Did you know?", "segments": []})
    paused = pause_job_for_review(job["id"], mock_script, db_path=db_file)
    assert paused["status"] == "PENDING_REVIEW"
    assert paused["current_stage"] == "PENDING_REVIEW"
    assert paused["script_json"] == mock_script

    # Review list
    pending = get_pending_review_jobs(db_path=db_file)
    assert len(pending) == 1
    assert pending[0]["id"] == job["id"]

    # Approve script
    edited_script = json.dumps({"hook": "Edited Hook", "segments": []})
    approved = approve_job(job["id"], script_json=edited_script, db_path=db_file)
    assert approved["status"] == "QUEUED"
    assert approved["current_stage"] == "APPROVED_FOR_RENDER"
    assert approved["require_approval"] == 0
    assert approved["script_json"] == edited_script

    # Review list now empty
    pending_after = get_pending_review_jobs(db_path=db_file)
    assert len(pending_after) == 0


def test_api_review_and_approval_endpoints(temp_env):
    """Verifies REST endpoints /api/start, /api/review, /api/jobs/<id>/approve, /api/jobs/<id>/reject."""
    _, db_file = temp_env
    client = app.test_client()

    # 1. Queue job with require_approval
    res = client.post("/api/start", json={
        "topic": "Future of AI Agents",
        "niche": "tech",
        "require_approval": True,
        "publish_target": "youtube"
    })
    assert res.status_code == 200
    data = res.get_json()
    job_id = data["job_id"]
    assert data["require_approval"] == 1
    assert data["publish_target"] == "youtube"

    # 2. Simulate worker pausing for review
    pause_job_for_review(job_id, json.dumps({"hook": "Initial script"}), db_path=db_file)

    # 3. GET /api/review
    res = client.get("/api/review")
    assert res.status_code == 200
    review_data = res.get_json()
    assert review_data["count"] >= 1
    matching = [j for j in review_data["jobs"] if j["id"] == job_id]
    assert len(matching) == 1
    assert matching[0]["status"] == "PENDING_REVIEW"

    # 4. POST /api/jobs/<id>/approve
    res = client.post(f"/api/jobs/{job_id}/approve", json={
        "script_json": json.dumps({"hook": "Polished hook", "segments": []})
    })
    assert res.status_code == 200
    appr_data = res.get_json()
    assert appr_data["job"]["status"] == "QUEUED"
    assert appr_data["job"]["require_approval"] == 0

    # 5. Queue another and reject
    res2 = client.post("/api/start", json={"topic": "Rejectable", "require_approval": True})
    job_id2 = res2.get_json()["job_id"]
    pause_job_for_review(job_id2, "{}", db_path=db_file)

    res_reject = client.post(f"/api/jobs/{job_id2}/reject")
    assert res_reject.status_code == 200
    rej_data = res_reject.get_json()
    assert rej_data["job"]["status"] == "CANCELLED"


def test_queue_worker_halts_and_resumes_approval(temp_env, monkeypatch):
    """Verifies that the background worker halts at PENDING_REVIEW when approval is required,
    and resumes to COMPLETED once the user approves the script.
    """
    _, db_file = temp_env
    monkeypatch.setenv("MAIN_SCRIPT", "mock_main.py")
    monkeypatch.setenv("MOCK_STEP_SLEEP", "0.05")

    # Queue job with approval required
    job = enqueue_job(
        topic="Simulation Hypothesis",
        niche="science",
        require_approval=True,
        publish_target="disk_only",
        db_path=db_file
    )
    job_id = job["id"]

    # First worker cycle: executes ideation and scriptwriting, then halts for review
    queue_worker_step()

    job_after_halt = get_job(str(job_id), db_path=db_file)
    assert job_after_halt["status"] == "PENDING_REVIEW"
    assert job_after_halt["current_stage"] == "PENDING_REVIEW"
    assert job_after_halt["script_json"] is not None
    assert "Did you know" in job_after_halt["script_json"]

    # User approves script
    approve_job(job_id, script_json=job_after_halt["script_json"], db_path=db_file)
    job_queued_again = get_job(str(job_id), db_path=db_file)
    assert job_queued_again["status"] == "QUEUED"
    assert job_queued_again["require_approval"] == 0

    # Second worker cycle: picks up approved job and completes rendering and publishing
    import time
    time.sleep(0.2)
    queue_worker_step()

    job_completed = get_job(str(job_id), db_path=db_file)
    assert job_completed["status"] == "COMPLETED"
    assert job_completed["current_stage"] == "COMPLETED"
