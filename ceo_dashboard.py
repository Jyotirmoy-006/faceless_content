"""Flask CEO Dashboard — Read-Only Observability Node (Rule 7 Compliance).

Architectural Guarantees:
1. STRICTLY READ-ONLY: Never triggers, spawns, or schedules main.py.
2. NO SCHEDULER: Zero APScheduler or queue worker dependencies. Windows Task
   Scheduler remains the sole automated trigger for pipeline runs (Rule 7).
3. DIRECT SQLITE READ: Exposes /status endpoint reflecting the latest run's
   telemetry from SQLite (jobs.db -> telemetry table) alongside circuit breaker
   and quota tracker state.
4. PROCESS ISOLATION: Observes external execution state solely via the OS-level
   pipeline.lock file and persistent SQLite telemetry.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, render_template_string, send_from_directory, Response

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.circuit_breaker import circuit_breaker
from pipeline.core.quota_tracker import quota_tracker
from pipeline.dashboard.database import (
    get_latest_run_telemetry,
    get_run_telemetry_history,
    get_job_simulation_trace,
    get_video_simulation_trace,
    get_categorized_assets,
    get_job,
    _enrich_job_dict,
    init_db
)

LOCK_FILE = ROOT_DIR / "pipeline.lock"
OUTPUT_DIR = ROOT_DIR / "pipeline" / "output"
DEFAULT_ASSETS_CACHE = ROOT_DIR / "pipeline" / "assets_cache"
THUMBNAILS_DIR = DEFAULT_ASSETS_CACHE / "thumbnails"

app = Flask(__name__)


def is_pid_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is actively running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            exit_code = ctypes.wintypes.DWORD()
            success = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            # STILL_ACTIVE = 259
            return bool(success and exit_code.value == 259)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def get_active_scheduler_lock() -> Dict[str, Any]:
    """Inspects pipeline.lock to check if Windows Task Scheduler is currently running."""
    if not LOCK_FILE.exists():
        return {"active": False, "pid": None, "started_at": None}
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        pid = int(data.get("pid", -1))
        alive = is_pid_alive(pid)
        return {
            "active": alive,
            "pid": pid if alive else None,
            "started_at": data.get("started_at"),
            "stale": not alive
        }
    except Exception:
        return {"active": False, "pid": None, "started_at": None, "corrupt": True}


def build_status_payload(db_path: Path | str | None = None) -> Dict[str, Any]:
    """Builds the comprehensive read-only executive status payload."""
    init_db(db_path)
    latest_run = get_latest_run_telemetry(db_path)
    lock_info = get_active_scheduler_lock()
    cb_tripped = circuit_breaker.is_tripped()

    # Determine executive status indicator
    if cb_tripped:
        overall_status = "degraded"
    elif lock_info.get("active"):
        overall_status = "executing"
    elif latest_run and latest_run.get("status") == "FAILED":
        overall_status = "warning"
    elif latest_run:
        overall_status = "healthy"
    else:
        overall_status = "idle"

    return {
        "status": overall_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "execution_trigger": "Windows Task Scheduler (Sole Automated Trigger)",
            "node_type": "Read-Only Observability Node (Rule 7 Compliant)",
            "scheduling_allowed": False
        },
        "latest_run": latest_run,
        "circuit_breaker": {
            "state": "OPEN (TRIPPED)" if cb_tripped else "CLOSED (HEALTHY)",
            "is_tripped": cb_tripped,
            "consecutive_failures": circuit_breaker.get_consecutive_failures(),
            "max_failures": circuit_breaker.max_failures
        },
        "quota": {
            "youtube_data_units_used": quota_tracker.get_used_units(),
            "youtube_data_daily_limit": quota_tracker.daily_limit,
            "youtube_data_available_units": quota_tracker.get_available_quota(),
            "youtube_analytics_queries_used": quota_tracker.get_analytics_used_queries(),
            "youtube_analytics_daily_limit": quota_tracker.analytics_daily_limit
        },
        "scheduler_lock": lock_info
    }


@app.route("/status", methods=["GET"])
def status():
    """Returns the most recent pipeline telemetry and system status as JSON.
    
    Strictly read-only: does not trigger or schedule any execution.
    """
    db_path = app.config.get("DB_PATH")
    payload = build_status_payload(db_path)
    return jsonify(payload), 200


@app.route("/api/jobs/<job_id_param>/simulation", methods=["GET"])
def get_job_simulation_endpoint(job_id_param: str):
    """Returns chronological agent simulation execution trace for the requested job (Mission M)."""
    db_path = app.config.get("DB_PATH")
    job = get_job(job_id_param, db_path=db_path)
    trace = get_job_simulation_trace(job_id_param, db_path=db_path)
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
    db_path = app.config.get("DB_PATH")
    job = get_job(job_id_param, db_path=db_path)
    if not job:
        return jsonify({"error": f"Job #{job_id_param} not found"}), 404

    job = _enrich_job_dict(job)
    trace = get_job_simulation_trace(job_id_param, db_path=db_path)
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
            "youtube_url": job.get("youtube_url"),
            "video_path": job.get("video_path")
        },
        "agents_executed": trace,
        "summary": {
            "total_agents": len(trace),
            "total_duration_seconds": round(sum(s.get("duration_seconds") or 0.0 for s in trace), 2)
        }
    }

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    filename = f"job_{job_id_param}_agents_telemetry.json"
    response = Response(json_str, mimetype="application/json")
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@app.route("/api/videos/<path:filename>/simulation", methods=["GET"])
def get_video_simulation_endpoint(filename: str):
    """Returns chronological agent simulation execution trace for a specific video."""
    db_path = app.config.get("DB_PATH")
    trace_info = get_video_simulation_trace(filename, db_path=db_path)
    return jsonify(trace_info), 200


@app.route("/api/videos/<path:filename>/simulation/download", methods=["GET"])
def download_video_simulation_endpoint(filename: str):
    """Exports structured cognitive telemetry for all agents behind a specific video as downloadable JSON."""
    db_path = app.config.get("DB_PATH")
    trace_info = get_video_simulation_trace(filename, db_path=db_path)
    stem = Path(filename).stem
    
    payload = {
        "export_title": f"Autonomous Pipeline Agent Telemetry — {filename}",
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "video_metadata": {
            "filename": filename,
            "topic": trace_info.get("topic"),
            "status": trace_info.get("status"),
            "youtube_url": trace_info.get("youtube_url"),
            "assets_associated": trace_info.get("assets", [])
        },
        "agents_executed": trace_info.get("trace", []),
        "summary": {
            "total_agents": len(trace_info.get("trace", [])),
            "total_duration_seconds": round(sum(s.get("duration_seconds") or 0.0 for s in trace_info.get("trace", [])), 2)
        }
    }

    json_str = json.dumps(payload, indent=2, ensure_ascii=False)
    download_filename = f"{stem}_agents_telemetry.json"
    response = Response(json_str, mimetype="application/json")
    response.headers["Content-Disposition"] = f'attachment; filename="{download_filename}"'
    return response


@app.route("/api/assets", methods=["GET"])
def get_assets_endpoint():
    """Strictly segregates Final Published Artifacts from Raw Intermediate Materials (Mission M)."""
    db_path = app.config.get("DB_PATH")
    categorized = get_categorized_assets(db_path=db_path)
    return jsonify(categorized), 200


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Executive Control Node — Read-Only Telemetry</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['"JetBrains Mono"', 'monospace']
          },
          colors: {
            zinc: {
              950: '#09090b',
              900: '#121215',
              850: '#18181b',
              800: '#27272a',
              700: '#3f3f46'
            }
          }
        }
      }
    }
  </script>
</head>
<body class="bg-zinc-950 text-zinc-100 min-h-screen font-sans flex flex-col antialiased selection:bg-indigo-500/30 selection:text-indigo-300">
  <!-- TOP NAV -->
  <header class="border-b border-zinc-800/80 bg-zinc-900/60 backdrop-blur-md sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <div class="h-9 w-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-emerald-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/10">
          <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <div>
          <div class="flex items-center space-x-3">
            <h1 class="text-base font-bold tracking-tight text-white">Executive Control Node</h1>
            <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" id="headerStatusBadge">
              ONLINE
            </span>
          </div>
          <p class="text-xs text-zinc-400">Autonomous Video Pipeline — Read-Only Observability</p>
        </div>
      </div>

      <!-- Live Clock & Architecture Tag -->
      <div class="flex items-center space-x-6">
        <div class="hidden sm:flex items-center space-x-2 text-xs font-mono text-zinc-400 bg-zinc-900 px-3 py-1.5 rounded-lg border border-zinc-800">
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span id="liveClock">--:--:-- UTC</span>
        </div>
        <a href="/status" target="_blank" class="text-xs font-mono text-indigo-400 hover:text-indigo-300 underline underline-offset-4">
          JSON API (/status) &rarr;
        </a>
      </div>
    </div>
  </header>

  <!-- MAIN CONTAINER -->
  <main class="max-w-7xl mx-auto px-6 py-8 flex-1 space-y-8 w-full">
    <!-- RULE 7 ARCHITECTURAL NOTICE BANNER -->
    <div class="rounded-xl border border-zinc-800 bg-gradient-to-r from-zinc-900 to-zinc-900/60 p-5 shadow-sm">
      <div class="flex items-start space-x-4">
        <div class="p-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 mt-0.5">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <div class="space-y-1">
          <div class="flex items-center space-x-2">
            <h2 class="text-sm font-semibold text-white">Rule 7 Compliance (Scheduler Safety)</h2>
            <span class="px-2 py-0.5 text-[10px] font-mono rounded bg-zinc-800 text-zinc-300 border border-zinc-700">STRICTLY READ-ONLY</span>
          </div>
          <p class="text-xs text-zinc-400 leading-relaxed">
            Windows Task Scheduler remains the <strong class="text-zinc-200">sole execution trigger</strong> for the autonomous video production pipeline. 
            This dashboard operates purely as a telemetry observer reading from the persistent SQLite record table. 
            No APScheduler, triggering buttons, or background worker threads exist on this node.
          </p>
        </div>
      </div>
    </div>

    <!-- METRIC CARDS GRID -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
      <!-- Card 1: Pipeline Execution Status -->
      <div class="rounded-xl border border-zinc-800 bg-zinc-900/90 p-5 space-y-3">
        <div class="flex items-center justify-between text-xs text-zinc-400 font-medium">
          <span>Active Task Status</span>
          <span class="w-2 h-2 rounded-full" id="lockDot"></span>
        </div>
        <div class="flex items-baseline space-x-2">
          <span class="text-2xl font-bold font-mono tracking-tight text-white" id="lockStatusText">IDLE</span>
        </div>
        <div class="text-[11px] text-zinc-500 font-mono" id="lockSubText">
          Lock File: Clean
        </div>
      </div>

      <!-- Card 2: Circuit Breaker -->
      <div class="rounded-xl border border-zinc-800 bg-zinc-900/90 p-5 space-y-3">
        <div class="flex items-center justify-between text-xs text-zinc-400 font-medium">
          <span>Circuit Breaker (Rule 8)</span>
          <span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold" id="cbBadge">CLOSED</span>
        </div>
        <div class="flex items-baseline space-x-2">
          <span class="text-2xl font-bold font-mono tracking-tight text-emerald-400" id="cbFailures">0 / 3</span>
          <span class="text-xs text-zinc-500 font-mono">Failures</span>
        </div>
        <div class="text-[11px] text-zinc-500 font-mono" id="cbSubText">
          Auto-trips at 3 failures
        </div>
      </div>

      <!-- Card 3: YouTube Data API Quota -->
      <div class="rounded-xl border border-zinc-800 bg-zinc-900/90 p-5 space-y-3">
        <div class="flex items-center justify-between text-xs text-zinc-400 font-medium">
          <span>YouTube Data Quota</span>
          <span class="text-[10px] font-mono text-zinc-400">10k Units/Day</span>
        </div>
        <div class="flex items-baseline space-x-2">
          <span class="text-2xl font-bold font-mono tracking-tight text-indigo-400" id="quotaUsed">--</span>
          <span class="text-xs text-zinc-500 font-mono">units</span>
        </div>
        <div class="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
          <div class="bg-indigo-500 h-full rounded-full transition-all duration-500" id="quotaBar" style="width: 0%"></div>
        </div>
      </div>

      <!-- Card 4: YouTube Analytics API Quota -->
      <div class="rounded-xl border border-zinc-800 bg-zinc-900/90 p-5 space-y-3">
        <div class="flex items-center justify-between text-xs text-zinc-400 font-medium">
          <span>YT Analytics API Quota</span>
          <span class="text-[10px] font-mono text-zinc-400">100k Queries/Day</span>
        </div>
        <div class="flex items-baseline space-x-2">
          <span class="text-2xl font-bold font-mono tracking-tight text-emerald-400" id="analyticsUsed">--</span>
          <span class="text-xs text-zinc-500 font-mono">queries</span>
        </div>
        <div class="text-[11px] text-zinc-500 font-mono">
          Independent quota pool
        </div>
      </div>
    </div>

    <!-- MOST RECENT REAL RUN TELEMETRY -->
    <div class="rounded-xl border border-zinc-800 bg-zinc-900/90 overflow-hidden shadow-lg">
      <div class="border-b border-zinc-800/80 px-6 py-4 flex items-center justify-between bg-zinc-900 gap-4 flex-wrap">
        <div class="flex items-center space-x-3">
          <div class="w-2.5 h-2.5 rounded-full bg-indigo-500"></div>
          <h2 class="text-sm font-bold text-white tracking-wide">Latest Real Run Telemetry</h2>
          <span class="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase" id="latestRunBadge">--</span>
        </div>
        <div class="flex items-center gap-3">
          <a id="latestRunYoutubeLink" href="#" target="_blank" rel="noopener noreferrer" class="hidden items-center gap-1.5 text-xs text-red-400 hover:text-red-300 font-mono px-2.5 py-1 rounded bg-red-600/15 hover:bg-red-600/25 border border-red-500/30 transition-all font-semibold" title="Watch on YouTube">
            <svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24"><path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>
            <span>Watch on YouTube</span>
          </a>
          <a id="latestRunDownloadBtn" href="#" class="hidden items-center gap-1.5 text-xs text-zinc-300 hover:text-white font-mono px-2.5 py-1 rounded bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 transition-all font-semibold" title="Download cognitive traces for every agent behind this run">
            <span>📥 Download Agent Info</span>
          </a>
          <span class="text-xs font-mono text-zinc-400" id="latestRunTimestamp">--</span>
        </div>
      </div>

      <div class="p-6 space-y-6">
        <!-- Topic & Video Info Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 pb-6 border-b border-zinc-800/80">
          <div class="space-y-1">
            <span class="text-[11px] uppercase tracking-wider text-zinc-500 font-semibold">Video Topic</span>
            <div class="text-base font-semibold text-white tracking-tight" id="latestTopic">No run telemetry recorded yet</div>
            <div class="text-xs font-mono text-zinc-400">Niche: <span id="latestNiche" class="text-zinc-200">--</span></div>
          </div>
          <div class="space-y-1">
            <span class="text-[11px] uppercase tracking-wider text-zinc-500 font-semibold">Rendered Asset</span>
            <div class="text-xs font-mono text-zinc-300 truncate" id="latestVideoPath">--</div>
            <div class="text-xs font-mono text-zinc-400">Size: <span id="latestVideoSize" class="text-zinc-200">-- MB</span></div>
          </div>
          <div class="space-y-1">
            <span class="text-[11px] uppercase tracking-wider text-zinc-500 font-semibold">Performance & Spend</span>
            <div class="text-base font-bold font-mono text-emerald-400" id="latestDuration">0.0s</div>
            <div class="text-xs font-mono text-zinc-400">Quota Spent: <span id="latestQuotaSpent" class="text-zinc-200">0 units</span></div>
          </div>
        </div>

        <!-- STAGE WALL-CLOCK DURATIONS -->
        <div class="space-y-3">
          <h3 class="text-xs font-bold uppercase tracking-wider text-zinc-400">Stage Wall-Clock Breakdown</h3>
          <div class="grid grid-cols-2 sm:grid-cols-5 gap-3" id="stageGrid">
            <div class="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <div class="text-[10px] text-zinc-500 font-mono">Stage 1: Ideation</div>
              <div class="text-sm font-bold font-mono text-zinc-200" id="stageIdeation">--</div>
            </div>
            <div class="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <div class="text-[10px] text-zinc-500 font-mono">Stage 2: Scriptwriting</div>
              <div class="text-sm font-bold font-mono text-zinc-200" id="stageScript">--</div>
            </div>
            <div class="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <div class="text-[10px] text-zinc-500 font-mono">Stage 3: Production</div>
              <div class="text-sm font-bold font-mono text-zinc-200" id="stageProd">--</div>
            </div>
            <div class="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <div class="text-[10px] text-zinc-500 font-mono">Stage 4: YouTube Pub</div>
              <div class="text-sm font-bold font-mono text-zinc-200" id="stageYt">--</div>
            </div>
            <div class="p-3 rounded-lg bg-zinc-950 border border-zinc-800 space-y-1">
              <div class="text-[10px] text-zinc-500 font-mono">Stage 4: Instagram Pub</div>
              <div class="text-sm font-bold font-mono text-zinc-200" id="stageIg">--</div>
            </div>
          </div>
        </div>

        <!-- ERROR MESSAGE (IF FAILED) -->
        <div id="errorBox" class="hidden p-4 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 font-mono">
          <div class="font-bold mb-1">Execution Failure:</div>
          <div id="errorText"></div>
        </div>
      </div>
    </div>
  </main>

  <footer class="border-t border-zinc-800/80 bg-zinc-900/40 py-4 text-center text-xs text-zinc-500 font-mono">
    Rule 7 Compliant Executive Observability Node &bull; Windows Task Scheduler Sole Trigger &bull; SQLite Telemetry Persistence
  </footer>

  <script>
    function updateClock() {
      const now = new Date();
      document.getElementById('liveClock').textContent = now.toISOString().replace('T', ' ').slice(0, 19) + ' UTC';
    }
    setInterval(updateClock, 1000);
    updateClock();

    async function fetchStatus() {
      try {
        const res = await fetch('/status');
        if (!res.ok) return;
        const data = await res.json();

        // 1. Header status badge
        const hBadge = document.getElementById('headerStatusBadge');
        if (data.status === 'healthy') {
          hBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
          hBadge.textContent = 'HEALTHY';
        } else if (data.status === 'executing') {
          hBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 animate-pulse';
          hBadge.textContent = 'EXECUTING';
        } else if (data.status === 'degraded' || data.status === 'warning') {
          hBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/20';
          hBadge.textContent = data.status.toUpperCase();
        } else {
          hBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-zinc-800 text-zinc-400 border border-zinc-700';
          hBadge.textContent = 'IDLE';
        }

        // 2. Scheduler Lock
        const lock = data.scheduler_lock || {};
        const lockDot = document.getElementById('lockDot');
        const lockText = document.getElementById('lockStatusText');
        const lockSub = document.getElementById('lockSubText');
        if (lock.active) {
          lockDot.className = 'w-2 h-2 rounded-full bg-indigo-500 animate-pulse';
          lockText.textContent = 'RUNNING';
          lockText.className = 'text-2xl font-bold font-mono tracking-tight text-indigo-400';
          lockSub.textContent = `PID ${lock.pid} (Task Scheduler)`;
        } else {
          lockDot.className = 'w-2 h-2 rounded-full bg-zinc-600';
          lockText.textContent = 'IDLE';
          lockText.className = 'text-2xl font-bold font-mono tracking-tight text-white';
          lockSub.textContent = 'Lock File: Clean';
        }

        // 3. Circuit Breaker
        const cb = data.circuit_breaker || {};
        const cbBadge = document.getElementById('cbBadge');
        const cbFailures = document.getElementById('cbFailures');
        if (cb.is_tripped) {
          cbBadge.className = 'px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20';
          cbBadge.textContent = 'TRIPPED';
          cbFailures.className = 'text-2xl font-bold font-mono tracking-tight text-rose-400';
        } else {
          cbBadge.className = 'px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
          cbBadge.textContent = 'CLOSED';
          cbFailures.className = 'text-2xl font-bold font-mono tracking-tight text-emerald-400';
        }
        cbFailures.textContent = `${cb.consecutive_failures || 0} / ${cb.max_failures || 3}`;

        // 4. Quotas
        const q = data.quota || {};
        const usedUnits = q.youtube_data_units_used || 0;
        const dailyLimit = q.youtube_data_daily_limit || 10000;
        document.getElementById('quotaUsed').textContent = usedUnits.toLocaleString();
        const pct = Math.min(100, Math.round((usedUnits / dailyLimit) * 100));
        document.getElementById('quotaBar').style.width = pct + '%';
        document.getElementById('analyticsUsed').textContent = (q.youtube_analytics_queries_used || 0).toLocaleString();

        // 5. Latest Run Telemetry
        const r = data.latest_run;
        if (r) {
          document.getElementById('latestTopic').textContent = r.topic || 'Auto-Ideated Concept';
          document.getElementById('latestNiche').textContent = r.niche || 'tech';
          document.getElementById('latestVideoPath').textContent = r.video_path || '--';
          document.getElementById('latestVideoSize').textContent = (r.video_size_mb || 0) + ' MB';
          document.getElementById('latestDuration').textContent = (r.duration_seconds || 0).toFixed(1) + 's';
          document.getElementById('latestQuotaSpent').textContent = (r.quota_spent || 0) + ' units';
          document.getElementById('latestRunTimestamp').textContent = r.timestamp || '--';

          // YouTube button
          const ytLink = document.getElementById('latestRunYoutubeLink');
          if (ytLink) {
            if (r.youtube_url) {
              ytLink.href = r.youtube_url;
              ytLink.classList.remove('hidden');
              ytLink.classList.add('flex');
            } else {
              ytLink.classList.add('hidden');
              ytLink.classList.remove('flex');
            }
          }

          // Download Agent Info button
          const dlBtn = document.getElementById('latestRunDownloadBtn');
          if (dlBtn) {
            const fname = r.video_path ? r.video_path.split(/[\\/]/).pop() : '';
            if (fname) {
              dlBtn.href = `/api/videos/${encodeURIComponent(fname)}/simulation/download`;
              dlBtn.classList.remove('hidden');
              dlBtn.classList.add('flex');
            } else if (r.id) {
              dlBtn.href = `/api/jobs/${r.id}/simulation/download`;
              dlBtn.classList.remove('hidden');
              dlBtn.classList.add('flex');
            } else {
              dlBtn.classList.add('hidden');
              dlBtn.classList.remove('flex');
            }
          }

          const runBadge = document.getElementById('latestRunBadge');
          if (r.status === 'SUCCESS') {
            runBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
            runBadge.textContent = 'SUCCESS';
          } else if (r.status === 'FAILED') {
            runBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-rose-500/10 text-rose-400 border border-rose-500/20';
            runBadge.textContent = 'FAILED';
          } else {
            runBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-amber-500/10 text-amber-400 border border-amber-500/20';
            runBadge.textContent = r.status || 'UNKNOWN';
          }

          const stages = r.stages || {};
          document.getElementById('stageIdeation').textContent = (stages.ideation !== undefined ? stages.ideation.toFixed(2) + 's' : '--');
          document.getElementById('stageScript').textContent = (stages.scriptwriting !== undefined ? stages.scriptwriting.toFixed(2) + 's' : '--');
          document.getElementById('stageProd').textContent = (stages.production !== undefined ? stages.production.toFixed(2) + 's' : '--');
          document.getElementById('stageYt').textContent = (stages.youtube_publish !== undefined ? stages.youtube_publish.toFixed(2) + 's' : '0.00s');
          document.getElementById('stageIg').textContent = (stages.instagram_publish !== undefined ? stages.instagram_publish.toFixed(2) + 's' : '0.00s');

          const errBox = document.getElementById('errorBox');
          if (r.error_message) {
            errBox.classList.remove('hidden');
            document.getElementById('errorText').textContent = r.error_message;
          } else {
            errBox.classList.add('hidden');
          }
        }
      } catch (err) {
        console.warn('Status poller error:', err);
      }
    }

    setInterval(fetchStatus, 4000);
    fetchStatus();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Renders the executive read-only dashboard UI or returns JSON if requested."""
    if request.headers.get("Accept") == "application/json":
        return status()
    return render_template_string(HTML_TEMPLATE), 200


@app.route("/videos/<path:filename>")
def serve_video(filename: str):
    """Serves rendered MP4 videos with HTTP byte-range support."""
    return send_from_directory(str(OUTPUT_DIR), filename, mimetype="video/mp4")


@app.route("/thumbnails/<path:filename>")
def serve_thumbnail(filename: str):
    """Serves cached poster thumbnails for video assets."""
    stem = Path(filename).stem
    target_jpg = THUMBNAILS_DIR / f"{stem}.jpg"
    if target_jpg.exists() and target_jpg.stat().st_size > 0:
        return send_from_directory(str(THUMBNAILS_DIR), f"{stem}.jpg", mimetype="image/jpeg")
    return ("Thumbnail Not Found", 404)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Faceless Executive Read-Only Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", default=5001, type=int, help="Port number (default: 5001)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print("=" * 75)
    print(f"CEO READ-ONLY CONTROL NODE ACTIVE -> http://{args.host}:{args.port}")
    print("Rule 7 Enforcement: Windows Task Scheduler remains the sole trigger.")
    print("This dashboard is strictly read-only and will never execute main.py.")
    print("=" * 75)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
