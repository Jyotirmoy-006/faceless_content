"""Flask Web Dashboard Control Plane & SQLite Sequential Job Queue (Mission G).

Architecture & Constraints:
- Zero frontend build tooling (Pure HTML5 & Vanilla JS + Tailwind CSS CDN).
- Non-blocking execution with SQLite Sequential Job Queue.
- Background Daemon Thread checks queue every 5 seconds.
- Strictly sequential execution guaranteeing Rule 1 (GPU Isolation) and Rule 7 (Scheduler Safety).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, send_from_directory, Response

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.circuit_breaker import circuit_breaker
from pipeline.core.quota_tracker import quota_tracker
from pipeline.core.logger import get_logger
from pipeline.dashboard.database import (
    approve_job,
    cleanup_stale_jobs,
    enqueue_job,
    get_all_jobs,
    get_history,
    get_job,
    get_next_queued_job,
    get_pending_review_jobs,
    get_running_job,
    init_db,
    mark_job_completed,
    mark_job_running,
    pause_job_for_review,
    update_job_stage,
    update_video_gate_status,
    get_job_simulation_trace,
    get_categorized_assets,
    get_video_simulation_trace,
    _enrich_job_dict,
    is_valid_youtube_url,
    normalize_youtube_url,
    set_job_youtube_url,
    set_video_youtube_url,
)
from main import is_pid_alive

LOCK_FILE = ROOT_DIR / "pipeline.lock"
LOG_FILE_ROOT = ROOT_DIR / "pipeline.log"
LOG_FILE_SUB = ROOT_DIR / "pipeline" / "logs" / "pipeline.log"
LOG_FILE = LOG_FILE_SUB  # Backwards compatibility
OUTPUT_DIR = ROOT_DIR / "pipeline" / "output"
DEFAULT_ASSETS_CACHE = ROOT_DIR / "pipeline" / "assets_cache"
QUOTA_FILE = ROOT_DIR / "pipeline" / "core" / "quota_state.json"
BREAKER_FILE = ROOT_DIR / "pipeline" / "core" / "circuit_breaker_state.json"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static")
)

logger = get_logger("dashboard")

_worker_thread: Optional[threading.Thread] = None
_worker_stop_event = threading.Event()
_worker_lock = threading.Lock()


def check_pipeline_lock() -> Tuple[bool, Optional[int], Dict[str, Any]]:
    """Inspects pipeline.lock and checks if recorded PID is actively alive."""
    if not LOCK_FILE.exists():
        return False, None, {}

    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        recorded_pid = int(data.get("pid", -1))
        started_at = data.get("started_at", "unknown")
    except Exception as err:
        logger.warning(f"Failed to parse lock file {LOCK_FILE}: {err}")
        recorded_pid = -1
        started_at = "corrupt"
        data = {}

    if is_pid_alive(recorded_pid):
        return True, recorded_pid, data
    else:
        logger.info(f"Removing stale lock file from dead PID {recorded_pid}")
        try:
            LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return False, None, {}


def read_tail_lines(file_path: Path, max_lines: int = 50) -> List[str]:
    """Reads the last N lines of a text file efficiently without memory exhaustion."""
    if not file_path.exists():
        if LOG_FILE_ROOT.exists():
            file_path = LOG_FILE_ROOT
        else:
            return ["[SYSTEM] Log file pipeline.log does not exist yet. Launch a generation job to begin streaming."]

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return [line.rstrip("\r\n") for line in lines[-max_lines:]]
    except Exception as e:
        return [f"[ERROR] Could not read log file: {e}"]


def find_latest_video(after_timestamp: float) -> Optional[Path]:
    """Finds the newest rendered video created after a given timestamp."""
    if not OUTPUT_DIR.exists():
        return None
    videos = sorted(
        [
            p for p in OUTPUT_DIR.glob("*.mp4")
            if not p.name.endswith("_original_unfixed.mp4")
            and not p.name.endswith("_qa_proxy.mp4")
            and "_proxy" not in p.name
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if videos and videos[0].stat().st_mtime >= after_timestamp - 5.0:
        return videos[0]
    return None


# ==============================================================================
# BACKGROUND SEQUENTIAL QUEUE WORKER THREAD
# ==============================================================================

def queue_worker_step():
    """Executes a single evaluation cycle of the sequential queue worker."""
    # 1. Check if pipeline.lock exists (e.g. manual CLI run or active job)
    is_locked, lock_pid, _ = check_pipeline_lock()
    if is_locked:
        logger.debug(f"[QUEUE_WORKER] Pipeline lock active (PID {lock_pid}). Pausing worker.")
        return

    # 2. Check if a job is already RUNNING in SQLite
    running_job = get_running_job()
    if running_job:
        running_pid = running_job.get("pid")
        if running_pid and is_pid_alive(int(running_pid)):
            # Active job is still progressing normally
            return
        else:
            # Stale / dead process cleanup
            logger.info(f"[QUEUE_WORKER] Detected dead PID for running job {running_job['id']}. Marking FAILED.")
            mark_job_completed(
                running_job["id"],
                status="FAILED",
                stage="FAILED",
                error_message="Process terminated unexpectedly (dead PID)."
            )

    # 3. Fetch oldest QUEUED job
    next_job = get_next_queued_job()
    if not next_job:
        return

    job_id = next_job["id"]
    topic = next_job.get("topic")
    niche = next_job.get("niche") or "tech"
    require_approval = bool(next_job.get("require_approval", 1) == 1)
    publish_target = next_job.get("publish_target") or "youtube"
    script_json = next_job.get("script_json")

    logger.info(
        f"[QUEUE_WORKER] Acquired QUEUED job #{job_id} (topic='{topic or 'Auto-Ideated'}', "
        f"approval={require_approval}, target={publish_target}, has_script={bool(script_json)})"
    )

    # 4. Mark job as RUNNING
    initial_stage = "AUDIO_TTS" if script_json else "IDEATION"
    mark_job_running(job_id, stage=initial_stage)

    # 5. Build command line (respecting MAIN_SCRIPT override for mock testing)
    script_name = os.environ.get("MAIN_SCRIPT", "main.py")
    script_path = ROOT_DIR / script_name

    cmd = [sys.executable, str(script_path)]
    if topic:
        cmd.extend(["--topic", topic])
    if niche:
        cmd.extend(["--niche", niche])

    # If script_json is provided (approved script), save to file and pass --approved-script
    if script_json:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        approved_file = OUTPUT_DIR / f"approved_script_job_{job_id}.json"
        try:
            approved_file.write_text(script_json, encoding="utf-8")
            cmd.extend(["--approved-script", str(approved_file)])
        except Exception as e:
            logger.warning(f"[QUEUE_WORKER] Could not write approved script file: {e}")
    else:
        if require_approval:
            cmd.append("--require-approval")

    if publish_target == "disk_only":
        cmd.extend(["--publish-target", "disk_only", "--skip-publish"])
    else:
        cmd.extend(["--publish-target", "youtube"])

    cmd.extend(["--job-id", str(job_id)])

    # 6. Ensure logs directory exists and write initiation header
    LOG_FILE_SUB.parent.mkdir(parents=True, exist_ok=True)
    jobs_log_dir = ROOT_DIR / "pipeline" / "logs" / "jobs"
    jobs_log_dir.mkdir(parents=True, exist_ok=True)
    job_log_file = jobs_log_dir / f"job_{job_id}.log"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"\n{'='*70}\n"
        f"[{now_str}] [QUEUE_WORKER] EXECUTING JOB #{job_id}\n"
        f"Topic: {topic or 'Auto-Ideated Concept'} | Script: {script_name} | Stage: {initial_stage}\n"
        f"{'='*70}\n\n"
    )
    try:
        with open(LOG_FILE_SUB, "a", encoding="utf-8") as f:
            f.write(header)
        with open(LOG_FILE_ROOT, "a", encoding="utf-8") as f:
            f.write(header)
        with open(job_log_file, "a", encoding="utf-8") as f:
            f.write(header)
    except Exception:
        pass

    start_ts = time.time()
    log_fp_sub = None
    log_fp_root = None
    log_fp_job = None
    proc = None
    extracted_script_json = script_json
    extracted_youtube_url = None
    is_pending_review = False

    try:
        log_fp_sub = open(LOG_FILE_SUB, "a", encoding="utf-8", buffering=1)
        try:
            log_fp_root = open(LOG_FILE_ROOT, "a", encoding="utf-8", buffering=1)
        except Exception:
            log_fp_root = None
        try:
            log_fp_job = open(job_log_file, "a", encoding="utf-8", buffering=1)
        except Exception:
            log_fp_job = None

        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(ROOT_DIR),
            creationflags=flags
        )
        mark_job_running(job_id, pid=proc.pid, stage=initial_stage)
        logger.info(f"[QUEUE_WORKER] Spawned PID {proc.pid} for job #{job_id}")

        is_pending_review = False
        is_publish_failed = False
        is_unverified_restricted = False
        is_publish_deferred = False
        publish_error_msg = None

        for line in proc.stdout:
            # Mirror real-time process logs into job log file
            if log_fp_sub:
                log_fp_sub.write(line)
                log_fp_sub.flush()
            if log_fp_root:
                log_fp_root.write(line)
                log_fp_root.flush()
            if log_fp_job:
                log_fp_job.write(line)
                log_fp_job.flush()

            clean_line = line.strip()

            # Parse stage transitions
            if "[STAGE:IDEATION]" in clean_line or ">>> [STAGE 1]" in clean_line:
                update_job_stage(job_id, "IDEATION")
            elif "[STAGE:SCRIPTWRITING]" in clean_line or ">>> [STAGE 2]" in clean_line:
                update_job_stage(job_id, "SCRIPTWRITING")
            elif "[STAGE:AUDIO_TTS]" in clean_line:
                update_job_stage(job_id, "AUDIO_TTS")
            elif "[STAGE:RENDER]" in clean_line or ">>> [STAGE 3]" in clean_line:
                update_job_stage(job_id, "RENDER")
            elif "[STAGE:COMPLIANCE]" in clean_line or ">>> [STAGE 3.5]" in clean_line:
                update_job_stage(job_id, "COMPLIANCE")
            elif "[STAGE:COMPLIANCE_HELD]" in clean_line:
                is_pending_review = True
                update_job_stage(job_id, "COMPLIANCE_HELD")
            elif "[STAGE:PUBLISH]" in clean_line or ">>> [STAGE 4]" in clean_line:
                update_job_stage(job_id, "PUBLISH")
            elif "[STAGE:PENDING_REVIEW]" in clean_line:
                is_pending_review = True
                update_job_stage(job_id, "PENDING_REVIEW")
            elif "[STAGE:PUBLISH_FAILED]" in clean_line:
                is_publish_failed = True
                publish_error_msg = clean_line.split("[STAGE:PUBLISH_FAILED]", 1)[1].strip()
                update_job_stage(job_id, "PUBLISH_FAILED")
            elif "[STAGE:PUBLISH_DEFERRED]" in clean_line:
                is_publish_deferred = True
                update_job_stage(job_id, "QUOTA_DEFERRED")

            # Parse script JSON payload
            if "[SCRIPT_JSON]" in clean_line:
                try:
                    json_part = clean_line.split("[SCRIPT_JSON]", 1)[1].strip()
                    extracted_script_json = json_part
                except Exception:
                    pass

            # Parse unverified project restriction
            if "[UNVERIFIED_PROJECT_RESTRICTION]" in clean_line:
                is_unverified_restricted = True

            # Parse YouTube publish link
            if "[YOUTUBE_URL]" in clean_line:
                try:
                    extracted_youtube_url = clean_line.split("[YOUTUBE_URL]", 1)[1].strip()
                except Exception:
                    pass
            elif "YouTube Shorts upload verified (Video ID: " in clean_line:
                try:
                    yt_id = clean_line.split("(Video ID:", 1)[1].split(")")[0].strip()
                    extracted_youtube_url = f"https://youtube.com/shorts/{yt_id}"
                except Exception:
                    pass
            elif "https://youtube.com/shorts/" in clean_line:
                try:
                    for token in clean_line.split():
                        if "https://youtube.com/shorts/" in token:
                            extracted_youtube_url = token.strip("()[]'\",.")
                            break
                except Exception:
                    pass

        proc.wait()
        exit_code = proc.returncode

        latest_vid = find_latest_video(start_ts)
        vid_str = str(latest_vid) if latest_vid else None

        if latest_vid and job_log_file.exists():
            try:
                videos_log_dir = ROOT_DIR / "pipeline" / "logs" / "videos"
                videos_log_dir.mkdir(parents=True, exist_ok=True)
                target_vlog = videos_log_dir / f"{latest_vid.stem}.log"
                target_vlog.write_text(job_log_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            except Exception as log_err:
                logger.warning(f"[QUEUE_WORKER] Could not copy video log file: {log_err}")

        if exit_code == 10 or is_pending_review:
            logger.info(f"[QUEUE_WORKER] Job #{job_id} halted for human script review.")
            pause_job_for_review(job_id, script_json=extracted_script_json or "{}")
        elif exit_code == 3 or is_publish_failed:
            err_msg = publish_error_msg or f"Publishing failed (exit code {exit_code})"
            logger.warning(f"[QUEUE_WORKER] Job #{job_id} FAILED publishing: {err_msg}")
            mark_job_completed(
                job_id,
                status="FAILED_PUBLISH",
                stage="PUBLISH_FAILED",
                video_path=vid_str,
                error_message=err_msg
            )
        elif is_unverified_restricted:
            logger.info(f"[QUEUE_WORKER] Job #{job_id} COMPLETED but held for review due to unverified project restriction.")
            verified_yt = normalize_youtube_url(extracted_youtube_url)
            mark_job_completed(
                job_id,
                status="HELD_FOR_REVIEW",
                stage="RESTRICTED_PRIVATE",
                video_path=vid_str,
                youtube_url=verified_yt,
                error_message="[UNVERIFIED_PROJECT_RESTRICTION] Video is locked to private by Google OAuth policy."
            )
            update_video_gate_status(job_id, chief_critic_passed=True, compliance_passed=True)
        elif is_publish_deferred:
            logger.info(f"[QUEUE_WORKER] Job #{job_id} deferred due to YouTube quota budget.")
            mark_job_completed(
                job_id,
                status="DEFERRED",
                stage="QUOTA_DEFERRED",
                video_path=vid_str,
                error_message="YouTube quota budget exhausted for today; deferred."
            )
            update_video_gate_status(job_id, chief_critic_passed=True, compliance_passed=True)
        elif exit_code == 0:
            logger.info(f"[QUEUE_WORKER] Job #{job_id} COMPLETED successfully.")
            verified_yt = normalize_youtube_url(extracted_youtube_url)
            mark_job_completed(
                job_id,
                status="COMPLETED",
                stage="COMPLETED",
                video_path=vid_str,
                youtube_url=verified_yt
            )
            update_video_gate_status(job_id, chief_critic_passed=True, compliance_passed=True)
            if publish_target == "disk_only":
                try:
                    from pipeline.agents.strategist import strategist
                    strategist.schedule_video(job_id, topic=topic or "Video", niche=niche)
                except Exception as sch_err:
                    logger.warning(f"Could not schedule video for job {job_id}: {sch_err}")
        else:
            logger.warning(f"[QUEUE_WORKER] Job #{job_id} FAILED with exit code {exit_code}")
            mark_job_completed(
                job_id,
                status="FAILED",
                stage="FAILED",
                video_path=vid_str,
                error_message=f"Subprocess exited with code {exit_code}"
            )
    except Exception as exc:
        logger.error(f"[QUEUE_WORKER] Exception executing job #{job_id}: {exc}", exc_info=True)
        mark_job_completed(
            job_id,
            status="FAILED",
            stage="FAILED",
            error_message=f"Subprocess exception: {exc}"
        )
    finally:
        if log_fp_sub:
            try:
                log_fp_sub.close()
            except Exception:
                pass
        if log_fp_root:
            try:
                log_fp_root.close()
            except Exception:
                pass
        if log_fp_job:
            try:
                log_fp_job.close()
            except Exception:
                pass


def queue_worker_loop(poll_interval: float = 5.0):
    """Background daemon loop checking the queue sequentially every 5 seconds."""
    logger.info(f"[QUEUE_WORKER] Worker thread active. Polling every {poll_interval}s.")
    while not _worker_stop_event.is_set():
        try:
            queue_worker_step()
        except Exception as err:
            logger.error(f"[QUEUE_WORKER] Loop caught unexpected exception: {err}", exc_info=True)
        _worker_stop_event.wait(poll_interval)


def start_queue_worker(poll_interval: float = 5.0):
    """Starts the background queue worker thread if not already running."""
    global _worker_thread
    with _worker_lock:
        if _worker_thread is None or not _worker_thread.is_alive():
            _worker_stop_event.clear()
            _worker_thread = threading.Thread(
                target=queue_worker_loop,
                kwargs={"poll_interval": poll_interval},
                daemon=True,
                name="QueueWorkerDaemon"
            )
            _worker_thread.start()
            logger.info("[QUEUE_WORKER] Daemon thread started.")


def stop_queue_worker():
    """Signals the queue worker daemon thread to stop."""
    global _worker_thread
    with _worker_lock:
        _worker_stop_event.set()
        if _worker_thread and _worker_thread.is_alive():
            _worker_thread = None


# ==============================================================================
# UI ROUTE
# ==============================================================================

@app.route("/")
def index():
    """Serves the Single Page Application dashboard."""
    return render_template("index.html")


# ==============================================================================
# API ROUTES
# ==============================================================================

@app.errorhandler(500)
def internal_server_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error occurred.", "details": str(e)}), 500
    return str(e), 500


@app.errorhandler(404)
def not_found_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "API resource not found.", "path": request.path}), 404
    return str(e), 404


@app.errorhandler(400)
def bad_request_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Bad request.", "details": str(e)}), 400
    return str(e), 400


@app.route("/api/start", methods=["POST"])
def start_pipeline():
    """Inserts a new job into SQLite in QUEUED status.
    
    Accepts:
    - topic: Optional[str]
    - niche: Optional[str] (default 'tech')
    - require_approval: bool (default True)
    - publish_target: 'youtube' | 'disk_only' (default 'youtube')
    """
    payload = request.get_json(silent=True) or {}
    raw_topic = payload.get("topic")
    topic = raw_topic.strip() if isinstance(raw_topic, str) and raw_topic.strip() else None
    
    raw_niche = payload.get("niche")
    niche = raw_niche.strip() if isinstance(raw_niche, str) and raw_niche.strip() else "tech"
    
    require_approval = payload.get("require_approval", True)
    if isinstance(require_approval, str):
        require_approval = require_approval.lower() in ("true", "1", "yes")
    publish_target = payload.get("publish_target", "youtube")
    if publish_target not in ("youtube", "disk_only"):
        publish_target = "youtube"

    job = enqueue_job(
        topic=topic,
        niche=niche,
        require_approval=bool(require_approval),
        publish_target=publish_target
    )
    logger.info(
        f"[API] Enqueued job #{job['id']} ('{job.get('topic') or 'Auto-Ideated'}') "
        f"[approval={require_approval}, target={publish_target}]"
    )

    return jsonify({
        "message": "Job queued successfully.",
        "job": job,
        "job_id": job.get("id"),
        "status": "QUEUED",
        "current_stage": job.get("current_stage", "QUEUED"),
        "topic": job.get("topic"),
        "require_approval": job.get("require_approval", 1),
        "publish_target": job.get("publish_target", "youtube")
    }), 200


@app.route("/api/review", methods=["GET"])
def get_pending_review():
    """Returns all jobs currently in PENDING_REVIEW status awaiting script approval."""
    jobs = get_pending_review_jobs()
    return jsonify({
        "jobs": jobs,
        "count": len(jobs)
    }), 200


@app.route("/api/jobs/<job_id_param>/approve", methods=["POST"])
def approve_job_endpoint(job_id_param: str):
    """Approves a PENDING_REVIEW job, updates script JSON, and sets status back to QUEUED."""
    payload = request.get_json(silent=True) or {}
    script_json = payload.get("script_json")
    if script_json and isinstance(script_json, dict):
        script_json = json.dumps(script_json)

    job = approve_job(job_id_param, script_json=script_json)
    if not job:
        return jsonify({"error": f"Job #{job_id_param} not found or could not be approved."}), 404

    logger.info(f"[API] Job #{job_id_param} approved by user. Re-queued for video render.")
    return jsonify({
        "message": "Job approved and queued for video rendering.",
        "job": job
    }), 200


@app.route("/api/jobs/<job_id_param>/reject", methods=["POST"])
def reject_job_endpoint(job_id_param: str):
    """Rejects a PENDING_REVIEW job, marking it CANCELLED."""
    job = mark_job_completed(
        job_id_param,
        status="CANCELLED",
        stage="REJECTED",
        error_message="Rejected by user during script review."
    )
    logger.info(f"[API] Job #{job_id_param} rejected by user.")
    return jsonify({
        "message": "Job rejected.",
        "job": job
    }), 200


@app.route("/api/queue", methods=["GET"])
def get_queue():
    """Returns the full list of jobs (History + Active + Queued)."""
    limit = request.args.get("limit", default=100, type=int)
    jobs = get_all_jobs(limit=min(limit, 200))
    return jsonify({
        "jobs": jobs,
        "count": len(jobs)
    }), 200


@app.route("/api/system", methods=["GET"])
def get_system():
    """Returns quota_state.json and circuit_breaker_state.json telemetry."""
    is_locked, lock_pid, lock_data = check_pipeline_lock()
    running_job = get_running_job()

    quota_info = {
        "available": quota_tracker.get_available_quota(),
        "limit": quota_tracker.daily_limit,
        "used": quota_tracker.get_used_units(),
        "reset_timezone": "America/Los_Angeles",
        "analytics": {
            "available": quota_tracker.get_analytics_available_quota(),
            "limit": quota_tracker.analytics_daily_limit,
            "used": quota_tracker.get_analytics_used_queries()
        }
    }

    allowed, breaker_msg = circuit_breaker.can_execute()
    cb_state = circuit_breaker.get_state()
    cb_info = {
        "is_open": cb_state.get("is_open", False),
        "consecutive_failures": cb_state.get("consecutive_failures", 0),
        "max_failures": cb_state.get("max_failures", 3),
        "can_execute": allowed,
        "message": breaker_msg
    }

    free_vram = 0
    try:
        from pipeline.core.gpu_lock import check_free_vram_mb
        free_vram = check_free_vram_mb()
    except Exception:
        pass

    llm_quota_info = {}
    try:
        from pipeline.core.llm_manager import model_quota_tracker
        llm_quota_info = model_quota_tracker.get_status()
    except Exception:
        pass

    return jsonify({
        "status": "online",
        "quota": quota_info,
        "llm_quota": llm_quota_info,
        "circuit_breaker": cb_info,
        "is_running": is_locked or (running_job is not None),
        "active_pid": lock_pid or (running_job.get("pid") if running_job else None),
        "running_job": running_job,
        "pipeline_locked": is_locked,
        "free_vram_mb": free_vram,
        "gpu": {"free_vram_mb": free_vram},
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200


@app.route("/api/llm/quota", methods=["GET"])
def get_llm_quota_endpoint():
    """Returns real-time Gemini model daily quota tracking and exhaustion status."""
    try:
        from pipeline.core.llm_manager import model_quota_tracker
        return jsonify(model_quota_tracker.get_status()), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to retrieve LLM quota status: {exc}"}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """Compatibility endpoint returning combined system state."""
    return get_system()


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Returns the last 50 lines of pipeline/logs/pipeline.log."""
    count = request.args.get("lines", default=50, type=int)
    target = LOG_FILE_SUB if LOG_FILE_SUB.exists() else LOG_FILE_ROOT
    lines = read_tail_lines(target, max_lines=min(count, 200))
    return jsonify({
        "logs": lines,
        "total_lines": len(lines)
    }), 200


@app.route("/api/history", methods=["GET"])
def get_job_history():
    """Compatibility endpoint returning all jobs."""
    limit = request.args.get("limit", default=50, type=int)
    jobs = get_history(limit=min(limit, 100))
    return jsonify({
        "history": jobs,
        "count": len(jobs)
    }), 200


@app.route("/api/jobs/<job_id_param>/simulation", methods=["GET"])
def get_job_simulation_endpoint(job_id_param: str):
    """Returns chronological agent simulation execution trace for the requested job (Mission M)."""
    job = get_job(job_id_param)
    trace = get_job_simulation_trace(job_id_param)
    topic = job.get("topic") if job else "Autonomous Video"
    status = job.get("status") if job else "UNKNOWN"
    youtube_url = job.get("youtube_url") if job else None
    return jsonify({
        "job_id": job_id_param,
        "topic": topic,
        "status": status,
        "youtube_url": youtube_url,
        "trace": trace,
        "count": len(trace)
    }), 200


@app.route("/api/jobs/<job_id_param>/simulation/download", methods=["GET"])
def download_job_simulation_endpoint(job_id_param: str):
    """Exports structured cognitive telemetry for all agents behind a job as downloadable JSON."""
    job = get_job(job_id_param)
    if not job:
        return jsonify({"error": f"Job #{job_id_param} not found"}), 404

    job = _enrich_job_dict(job)
    trace = get_job_simulation_trace(job_id_param)
    topic = job.get("topic") or "Auto-Ideated Concept"

    payload = {
        "export_title": f"Autonomous Pipeline Agent Telemetry — Job #{job_id_param}",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "job_metadata": {
            "id": job.get("id"),
            "job_id": job.get("job_id"),
            "topic": topic,
            "niche": job.get("niche", "tech"),
            "status": job.get("status"),
            "current_stage": job.get("current_stage"),
            "youtube_url": job.get("youtube_url"),
            "video_path": job.get("video_path"),
            "target_publish_datetime": job.get("target_publish_datetime"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at")
        },
        "agents_executed": trace,
        "summary": {
            "total_agents": len(trace),
            "total_duration_seconds": round(sum(s.get("duration_seconds") or 0.0 for s in trace), 2),
            "quality_gate_passed": bool(job.get("chief_critic_passed", 1)),
            "compliance_gate_passed": bool(job.get("compliance_passed", 1))
        }
    }

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:30].strip("_")
    filename = f"job_{job_id_param}_{safe_topic}_agents_telemetry.json"
    response = Response(json_str, mimetype="application/json")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@app.route("/api/videos/<path:filename>/simulation", methods=["GET"])
def get_video_simulation_endpoint(filename: str):
    """Returns agent simulation trace for a specific video file."""
    data = get_video_simulation_trace(filename)
    return jsonify(data), 200


@app.route("/api/videos/<path:filename>/simulation/download", methods=["GET"])
def download_video_simulation_endpoint(filename: str):
    """Exports structured cognitive telemetry for all agents behind a specific video as downloadable JSON."""
    data = get_video_simulation_trace(filename)
    stem = Path(filename).stem
    topic = data.get("topic") or stem

    payload = {
        "export_title": f"Autonomous Pipeline Agent Telemetry — Video: {Path(filename).name}",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "video_metadata": {
            "filename": Path(filename).name,
            "topic": topic,
            "niche": data.get("niche", "tech"),
            "status": data.get("status", "COMPLETED"),
            "youtube_url": data.get("youtube_url"),
            "job_id": data.get("job_id"),
            "created_at": data.get("created_at")
        },
        "agents_executed": data.get("trace", []),
        "summary": {
            "total_agents": len(data.get("trace", [])),
            "total_duration_seconds": round(sum(s.get("duration_seconds") or 0.0 for s in data.get("trace", [])), 2),
            "quality_gate_passed": True,
            "compliance_gate_passed": True
        }
    }

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    out_filename = f"{stem}_agents_telemetry.json"
    response = Response(json_str, mimetype="application/json")
    response.headers["Content-Disposition"] = f'attachment; filename="{out_filename}"'
    return response


@app.route("/api/videos/<path:filename>/logs/download", methods=["GET"])
def download_video_logs_endpoint(filename: str):
    """Exports raw execution log file for a specific video as a downloadable text file."""
    stem = Path(filename).stem
    videos_log_dir = ROOT_DIR / "pipeline" / "logs" / "videos"
    jobs_log_dir = ROOT_DIR / "pipeline" / "logs" / "jobs"
    video_log_path = videos_log_dir / f"{stem}.log"

    log_content = ""
    if video_log_path.exists() and video_log_path.stat().st_size > 0:
        log_content = video_log_path.read_text(encoding="utf-8", errors="replace")
    else:
        trace_data = get_video_simulation_trace(filename)
        job_id = trace_data.get("job_id")
        if job_id:
            job_log_path = jobs_log_dir / f"job_{job_id}.log"
            if job_log_path.exists() and job_log_path.stat().st_size > 0:
                log_content = job_log_path.read_text(encoding="utf-8", errors="replace")

        if not log_content:
            target = LOG_FILE_SUB if LOG_FILE_SUB.exists() else LOG_FILE_ROOT
            if target.exists():
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                matched_slice = []
                capturing = False
                for line in lines:
                    if stem in line or (trace_data.get("topic") and trace_data["topic"] in line):
                        capturing = True
                    if capturing:
                        matched_slice.append(line)
                        if len(matched_slice) > 800:
                            break
                if matched_slice:
                    log_content = "\n".join(matched_slice)
                else:
                    log_content = "\n".join(lines[-500:])

    if not log_content:
        log_content = f"[SYSTEM] No log telemetry found for video: {Path(filename).name}\n"

    out_name = f"{stem}_execution.log"
    response = Response(log_content, mimetype="text/plain; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="{out_name}"'
    return response


@app.route("/api/jobs/<job_id_param>/logs/download", methods=["GET"])
def download_job_logs_endpoint(job_id_param: str):
    """Exports raw execution log file for a job as a downloadable text file."""
    jobs_log_dir = ROOT_DIR / "pipeline" / "logs" / "jobs"
    job_log_path = jobs_log_dir / f"job_{job_id_param}.log"

    log_content = ""
    if job_log_path.exists() and job_log_path.stat().st_size > 0:
        log_content = job_log_path.read_text(encoding="utf-8", errors="replace")
    else:
        target = LOG_FILE_SUB if LOG_FILE_SUB.exists() else LOG_FILE_ROOT
        if target.exists():
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            matched_slice = []
            capturing = False
            start_marker = f"EXECUTING JOB #{job_id_param}"
            for line in lines:
                if start_marker in line:
                    capturing = True
                elif capturing and "EXECUTING JOB #" in line and start_marker not in line:
                    break
                if capturing:
                    matched_slice.append(line)
            if matched_slice:
                log_content = "\n".join(matched_slice)
            else:
                log_content = "\n".join(lines[-500:])

    if not log_content:
        log_content = f"[SYSTEM] No log telemetry found for job #{job_id_param}\n"

    out_name = f"job_{job_id_param}_execution.log"
    response = Response(log_content, mimetype="text/plain; charset=utf-8")
    response.headers["Content-Disposition"] = f'attachment; filename="{out_name}"'
    return response


@app.route("/api/jobs/<job_id_param>/youtube_url", methods=["POST"])
def set_job_youtube_url_endpoint(job_id_param: str):
    """Attaches or updates a verified, real YouTube URL for a job."""
    body = request.get_json(silent=True) or {}
    url = body.get("youtube_url")
    if not url or not is_valid_youtube_url(url):
        return jsonify({
            "error": "Invalid YouTube URL. Please provide a valid YouTube Shorts or Video URL with an 11-character video ID (e.g. https://youtube.com/shorts/dQw4w9WgXcQ)."
        }), 400
    saved = set_job_youtube_url(job_id_param, url)
    if not saved:
        return jsonify({"error": "Failed to save YouTube URL"}), 400
    return jsonify({"success": True, "job_id": job_id_param, "youtube_url": saved}), 200


@app.route("/api/videos/<path:filename>/youtube_url", methods=["POST"])
def set_video_youtube_url_endpoint(filename: str):
    """Attaches or updates a verified, real YouTube URL for a master video."""
    body = request.get_json(silent=True) or {}
    url = body.get("youtube_url")
    if not url or not is_valid_youtube_url(url):
        return jsonify({
            "error": "Invalid YouTube URL. Please provide a valid YouTube Shorts or Video URL with an 11-character video ID (e.g. https://youtube.com/shorts/dQw4w9WgXcQ)."
        }), 400
    saved = set_video_youtube_url(filename, url)
    if not saved:
        return jsonify({"error": "Failed to save YouTube URL"}), 400
    return jsonify({"success": True, "filename": filename, "youtube_url": saved}), 200


@app.route("/api/assets", methods=["GET"])
def get_assets_endpoint():
    """Strictly segregates Final Published Artifacts from Raw Intermediate Materials (Mission M)."""
    categorized = get_categorized_assets()
    return jsonify(categorized), 200


@app.route("/api/videos", methods=["GET"])
def get_videos():
    """Lists all rendered video assets in pipeline/output for Asset Inspector.
    
    Includes backward-compatible 'videos' array and Mission M segregated asset data.
    """
    categorized = get_categorized_assets()
    videos = []
    for item in categorized.get("final_artifacts", []):
        if item.get("type") == "video":
            videos.append({
                "filename": item["name"],
                "size_mb": item["size_mb"],
                "modified": item["modified"],
                "url": item["url"],
                "badge": item.get("badge", "FINAL MASTER")
            })

    return jsonify({
        "videos": videos,
        "count": len(videos),
        "final_artifacts": categorized.get("final_artifacts", []),
        "raw_materials": categorized.get("raw_materials", []),
        "final_count": categorized.get("final_count", 0),
        "raw_count": categorized.get("raw_count", 0)
    })


@app.route("/api/circuit-breaker/reset", methods=["POST"])
def reset_circuit_breaker():
    """Manually resets the durable circuit breaker."""
    circuit_breaker.reset()
    return jsonify({
        "message": "Circuit breaker reset successfully.",
        "state": circuit_breaker.get_state()
    })


@app.route("/api/quota/reset", methods=["POST"])
def reset_quota():
    """Manually resets today's YouTube API quota tracking."""
    quota_tracker.reset()
    return jsonify({
        "message": "YouTube quota tracker reset successfully.",
        "available": quota_tracker.get_available_quota()
    })


# ==============================================================================
# ASSET INSPECTOR: STATIC VIDEO & THUMBNAIL STREAMING
# ==============================================================================

THUMBNAILS_DIR = DEFAULT_ASSETS_CACHE / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def generate_video_thumbnail(source_video: Path, target_jpg: Path) -> bool:
    """Extracts a vibrant poster frame from a video using FFmpeg and caches it as JPEG.
    
    Tries 2.0s first to bypass initial black fade-ins, falling back to 1.0s or 0.1s.
    """
    try:
        target_jpg.parent.mkdir(parents=True, exist_ok=True)
        for timestamp in ["00:00:02", "00:00:01", "00:00:00.1"]:
            cmd = [
                "ffmpeg", "-y", "-ss", timestamp,
                "-i", str(source_video),
                "-vframes", "1",
                "-q:v", "3",
                "-update", "1",
                str(target_jpg)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if res.returncode == 0 and target_jpg.exists() and target_jpg.stat().st_size > 3000:
                return True
        return bool(target_jpg.exists() and target_jpg.stat().st_size > 0)
    except Exception:
        return False


@app.route("/videos/<path:filename>")
def serve_video(filename: str):
    """Serves rendered MP4 videos with HTTP byte-range support for HTML5 video player."""
    return send_from_directory(str(OUTPUT_DIR), filename, mimetype="video/mp4")


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename: str):
    """Serves or dynamically extracts a crisp poster thumbnail for a video."""
    stem = Path(filename).stem
    target_jpg = THUMBNAILS_DIR / f"{stem}.jpg"

    if target_jpg.exists() and target_jpg.stat().st_size > 0:
        return send_from_directory(str(THUMBNAILS_DIR), f"{stem}.jpg", mimetype="image/jpeg")

    # Locate source video in output directory or assets cache
    source_video = None
    direct_mp4 = OUTPUT_DIR / f"{stem}.mp4"
    if direct_mp4.exists() and direct_mp4.stat().st_size > 1000:
        source_video = direct_mp4
    else:
        for p in DEFAULT_ASSETS_CACHE.rglob(f"{stem}.mp4"):
            if p.is_file() and p.stat().st_size > 1000:
                source_video = p
                break

    if source_video:
        ok = generate_video_thumbnail(source_video, target_jpg)
        if ok:
            return send_from_directory(str(THUMBNAILS_DIR), f"{stem}.jpg", mimetype="image/jpeg")

    return ("Thumbnail Not Found", 404)


@app.route("/raw_materials/<path:filename>")
def serve_raw_material(filename: str):
    """Serves raw intermediate materials (Pexels clips, TTS chunks, QA proxies) from assets_cache."""
    return send_from_directory(str(DEFAULT_ASSETS_CACHE), filename)


# Auto-initialize database schema
init_db()


def main():
    parser = argparse.ArgumentParser(description="Autonomous Faceless Command Center Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", default=5000, type=int, help="Port number (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--interval", default=5.0, type=float, help="Queue worker poll interval in seconds (default: 5.0)")
    args = parser.parse_args()

    init_db()
    start_queue_worker(poll_interval=args.interval)

    print("=" * 70)
    print(f"COMMAND CENTER ACTIVE -> http://{args.host}:{args.port}")
    print(f"Queue Worker Daemon Polling Every {args.interval}s")
    print("=" * 70)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
