"""Master Orchestrator for Autonomous Faceless Video Automation.

Entry point designed for automated execution via Windows Task Scheduler.
Complies with:
- Rule 1 (GPU MEMORY): Subprocess isolation via gpu_lock.py.
- Rule 2 (NO SILENT CRASHES): Bounded retries, fallbacks, and explicit error handling.
- Rule 3 (QUOTA AWARENESS): Central quota tracker gating for YouTube uploads.
- Rule 4 (VALIDATE LLM OUTPUT): Pydantic validation across all LLM stages.
- Rule 5 (edge-tts IS UNOFFICIAL): Sentence-chunked TTS with local Piper fallback.
- Rule 6 (INSTAGRAM NEEDS A PUBLIC URL): Ephemeral B2 public hosting with guaranteed cleanup.
- Rule 7 (SCHEDULER SAFETY): Single-instance lock file (pipeline.lock) with stale-lock detection.
- Rule 8 (NO UNBOUNDED RETRIES): Durable circuit breaker halts runs after 3 consecutive failures.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.circuit_breaker import circuit_breaker, CircuitBreakerOpenError
from pipeline.core.quota_tracker import quota_tracker, YOUTUBE_VIDEO_UPLOAD_COST
from pipeline.core.schema import IdeaConcept, Script
from pipeline.agents.ideator import ideator
from pipeline.agents.scriptwriter import get_valid_script
from pipeline.agents.director import orchestrate_video
from pipeline.agents.publisher import publisher, PublishResult, PublishStatus

# Setup master logger
logger = logging.getLogger("orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [orchestrator] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

LOCK_FILE = ROOT_DIR / "pipeline.lock"
OUTPUT_DIR = ROOT_DIR / "pipeline" / "output"


# ==============================================================================
# SCHEDULER SAFETY LOCK (RULE 7 COMPLIANCE)
# ==============================================================================

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


class SchedulerLock:
    """Guarantees single-instance execution per Rule 7 (SCHEDULER SAFETY)."""

    def __init__(self, lock_path: Path = LOCK_FILE):
        self.lock_path = lock_path

    def acquire(self) -> bool:
        """Acquires lock or cleans up stale lock.
        
        Returns:
            True if lock acquired, False if an active run is already in progress.
        """
        if self.lock_path.exists():
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recorded_pid = int(data.get("pid", -1))
                started_at = data.get("started_at", "unknown")
            except Exception:
                recorded_pid = -1
                started_at = "corrupt"

            if is_pid_alive(recorded_pid):
                logger.warning(
                    f"[SCHEDULER_LOCK] Active pipeline execution already in progress "
                    f"(PID {recorded_pid}, started at {started_at}). "
                    f"Exiting immediately to prevent concurrent execution (Rule 7)."
                )
                return False
            else:
                logger.info(
                    f"[SCHEDULER_LOCK] Stale lock detected from dead PID {recorded_pid} "
                    f"(crashed or terminated). Purging stale lock."
                )
                try:
                    self.lock_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not unlink stale lock: {e}")

        # Write fresh lock
        lock_data = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        logger.info(f"[SCHEDULER_LOCK] Acquired run lock (PID {os.getpid()}).")
        return True

    def release(self) -> None:
        """Releases the lock file if owned by current PID."""
        if self.lock_path.exists():
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if int(data.get("pid", -1)) == os.getpid():
                    self.lock_path.unlink()
                    logger.info(f"[SCHEDULER_LOCK] Released run lock (PID {os.getpid()}).")
            except Exception as e:
                logger.warning(f"Failed to cleanly release lock file: {e}")


@contextmanager
def scheduler_lock(lock_path: Path = LOCK_FILE) -> Generator[bool, None, None]:
    """Context manager for scheduler lock lifecycle."""
    lock = SchedulerLock(lock_path)
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


# ==============================================================================
# PIPELINE STAGE COORDINATION & TELEMETRY
# ==============================================================================

def get_gpu_memory_allocated_mb() -> float:
    """Returns current PyTorch GPU allocated memory in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def run_pipeline(
    niche: str = "tech",
    topic_override: Optional[str] = None,
    dry_run: bool = False,
    skip_publish: bool = False,
) -> dict:
    """Executes the complete end-to-end faceless video production pipeline.
    
    Stages:
    1. Ideator Agent -> IdeaConcept (validated)
    2. Scriptwriter Agent -> Script (validated)
    3. Director Agent -> Video Render (Whisper + ComfyUI + Render workers via GPU lock)
    4. Publisher Agent -> YouTube Shorts (quota-gated) + Instagram Reels (ephemeral hosting)
    """
    telemetry = {
        "start_time": time.time(),
        "stages": {},
        "vram": {},
        "quota_spent": 0,
        "results": {}
    }

    # -------------------------------------------------------------------------
    # STAGE 0: Circuit Breaker Pre-Check (Rule 8)
    # -------------------------------------------------------------------------
    allowed, breaker_msg = circuit_breaker.can_execute()
    if not allowed:
        logger.error(f"[CIRCUIT_BREAKER_BLOCKED]\n{breaker_msg}")
        raise CircuitBreakerOpenError(circuit_breaker.get_last_failure_reasons())

    try:
        logger.info("=" * 80)
        logger.info(f"STARTING AUTONOMOUS FACELESS VIDEO PIPELINE (Niche: {niche})")
        logger.info(f"System Baseline VRAM: {get_gpu_memory_allocated_mb():.2f} MB")
        logger.info("=" * 80)

        # ---------------------------------------------------------------------
        # STAGE 1: Ideation (Ideator Agent)
        # ---------------------------------------------------------------------
        t0 = time.time()
        logger.info("\n>>> [STAGE 1] Concept Ideation (Ideator Agent)...")
        concept: IdeaConcept = ideator.generate_idea(
            niche=niche,
            custom_topic=topic_override,
            dry_run=dry_run
        )
        if not isinstance(concept, IdeaConcept) or not concept.topic:
            raise ValueError(f"Ideator returned invalid concept contract: {concept}")
        
        telemetry["stages"]["ideation"] = time.time() - t0
        telemetry["concept"] = concept.model_dump()
        logger.info(f"Concept approved: '{concept.topic}' ({concept.angle}) [{telemetry['stages']['ideation']:.2f}s]")

        # ---------------------------------------------------------------------
        # STAGE 2: Scriptwriting (Scriptwriter Agent)
        # ---------------------------------------------------------------------
        t0 = time.time()
        logger.info("\n>>> [STAGE 2] Script Breakdown & Drafting (Scriptwriter Agent)...")
        script: Script = get_valid_script(
            topic=concept.topic,
            niche=concept.niche
        )
        if not isinstance(script, Script) or not script.segments:
            raise ValueError(f"Scriptwriter returned empty or invalid script: {script}")

        telemetry["stages"]["scriptwriting"] = time.time() - t0
        telemetry["script"] = {
            "hook": script.hook,
            "segments_count": len(script.segments),
            "estimated_duration": script.total_estimated_duration()
        }
        logger.info(
            f"Script structured: {len(script.segments)} scenes, "
            f"~{script.total_estimated_duration():.1f}s total duration [{telemetry['stages']['scriptwriting']:.2f}s]"
        )

        # ---------------------------------------------------------------------
        # STAGE 3: Direction & Production (Director Agent + GPU Workers)
        # ---------------------------------------------------------------------
        t0 = time.time()
        logger.info("\n>>> [STAGE 3] Asset Direction & Video Render (Director Agent)...")
        safe_slug = "".join(c if c.isalnum() else "_" for c in concept.topic)[:30].strip("_")
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rendered_output_path = OUTPUT_DIR / f"{timestamp_str}_{safe_slug}.mp4"

        # Record baseline before GPU stages
        telemetry["vram"]["pre_render_mb"] = get_gpu_memory_allocated_mb()

        # Execute orchestration through GPU lock
        video_path = orchestrate_video(
            script=script,
            output_path=rendered_output_path,
            voice="en-US-ChristopherNeural"
        )
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Director failed to produce valid video file at {video_path}")

        telemetry["stages"]["production"] = time.time() - t0
        telemetry["video_path"] = str(video_path)
        telemetry["video_size_mb"] = round(video_path.stat().st_size / (1024 * 1024), 2)
        telemetry["vram"]["post_render_mb"] = get_gpu_memory_allocated_mb()
        logger.info(
            f"Video rendered successfully: {video_path.name} "
            f"({telemetry['video_size_mb']} MB) [{telemetry['stages']['production']:.2f}s]"
        )

        # ---------------------------------------------------------------------
        # STAGE 4: Multi-Platform Publishing (Publisher Agent)
        # ---------------------------------------------------------------------
        if not skip_publish:
            logger.info("\n>>> [STAGE 4] Multi-Platform Publishing (Publisher Agent)...")
            
            # YouTube Shorts (Quota-gated)
            t0 = time.time()
            yt_res: PublishResult = publisher.publish_to_youtube(
                video_path=video_path,
                title=concept.topic,
                description=f"{script.full_narration()}\n\n#shorts #{concept.niche} #facts",
                tags=[f"#{concept.niche}", "#shorts", "#education"],
                privacy_status="public",
                dry_run=dry_run
            )
            telemetry["stages"]["youtube_publish"] = time.time() - t0
            telemetry["results"]["youtube"] = yt_res.model_dump()
            telemetry["quota_spent"] += yt_res.units_spent

            # Instagram Reels (Ephemeral Hosting)
            t0 = time.time()
            ig_caption = f"{concept.topic}\n\n{script.hook}\n\nFollow for more daily insights! #reels #{concept.niche}"
            ig_res: PublishResult = publisher.publish_to_instagram(
                video_path=video_path,
                caption=ig_caption,
                dry_run=dry_run
            )
            telemetry["stages"]["instagram_publish"] = time.time() - t0
            telemetry["results"]["instagram"] = ig_res.model_dump()
        else:
            logger.info("\n>>> [STAGE 4] Publishing skipped (--skip-publish requested).")

        # Record total runtime
        telemetry["total_wall_clock_time"] = time.time() - telemetry["start_time"]

        # ---------------------------------------------------------------------
        # SUCCESS: Reset Circuit Breaker (Rule 8)
        # ---------------------------------------------------------------------
        circuit_breaker.record_success()

        # Print Final Telemetry Summary
        _print_run_summary(telemetry)
        return telemetry

    except Exception as exc:
        logger.error(f"Pipeline run encountered a fatal exception: {exc}", exc_info=True)
        # Record failure into circuit breaker (Rule 8)
        circuit_breaker.record_failure(str(exc))
        raise


def _print_run_summary(telemetry: dict) -> None:
    """Formats and prints executive telemetry table."""
    concept = telemetry.get("concept", {})
    stages = telemetry.get("stages", {})
    results = telemetry.get("results", {})

    print("\n" + "=" * 80)
    print("           AUTONOMOUS PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Topic:                 {concept.get('topic')}")
    print(f"Niche:                 {concept.get('niche')}")
    print(f"Output Video:          {telemetry.get('video_path')} ({telemetry.get('video_size_mb')} MB)")
    print("-" * 80)
    print("STAGE WALL-CLOCK DURATIONS:")
    print(f"  Stage 1 (Ideation):       {stages.get('ideation', 0.0):6.2f}s")
    print(f"  Stage 2 (Scriptwriting):  {stages.get('scriptwriting', 0.0):6.2f}s")
    print(f"  Stage 3 (Production):     {stages.get('production', 0.0):6.2f}s")
    if "youtube_publish" in stages:
        print(f"  Stage 4 (YouTube Pub):    {stages.get('youtube_publish', 0.0):6.2f}s")
    if "instagram_publish" in stages:
        print(f"  Stage 4 (Instagram Pub):  {stages.get('instagram_publish', 0.0):6.2f}s")
    print(f"TOTAL PIPELINE RUNTIME:     {telemetry.get('total_wall_clock_time', 0.0):6.2f}s")
    print("-" * 80)
    print("RESOURCE & QUOTA TELEMETRY:")
    print(f"  YouTube Quota Units Spent Today:  {quota_tracker.get_used_units()} / {quota_tracker.daily_limit} units")
    print(f"  YouTube Available Quota:          {quota_tracker.get_available_quota()} units")
    if "youtube" in results:
        yt = results["youtube"]
        print(f"  YouTube Post Status:              {yt.get('status')} (Post ID: {yt.get('post_id')})")
    if "instagram" in results:
        ig = results["instagram"]
        print(f"  Instagram Post Status:            {ig.get('status')} (Post ID: {ig.get('post_id')})")
    print("=" * 80 + "\n")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous Faceless Video Pipeline Entry Point")
    parser.add_argument("--niche", type=str, default="tech", help="Content niche (tech, finance, science, history)")
    parser.add_argument("--topic", type=str, default=None, help="Explicit topic override (bypasses ideator generation)")
    parser.add_argument("--dry-run", action="store_true", help="Simulates external publishing & uses template fallback")
    parser.add_argument("--skip-publish", action="store_true", help="Renders video locally without publishing")
    parser.add_argument("--reset-circuit-breaker", action="store_true", help="Manually re-arms the circuit breaker")
    parser.add_argument("--reset-quota", action="store_true", help="Manually resets today's YouTube quota tracker")
    parser.add_argument("--status", action="store_true", help="Displays current circuit breaker and quota status")
    args = parser.parse_args()

    # Administrative flags
    if args.reset_circuit_breaker:
        circuit_breaker.reset()
        print("Circuit breaker successfully reset to CLOSED state.")
        return

    if args.reset_quota:
        quota_tracker.reset()
        print("YouTube quota tracker successfully reset to 0 units.")
        return

    if args.status:
        print("\n--- Pipeline Operational Status ---")
        print(f"Circuit Breaker: {'OPEN (TRIPPED)' if circuit_breaker.is_tripped() else 'CLOSED (HEALTHY)'}")
        print(f"Consecutive Failures: {circuit_breaker.get_consecutive_failures()}/{circuit_breaker.max_failures}")
        print(f"YouTube Quota Used: {quota_tracker.get_used_units()} / {quota_tracker.daily_limit} units")
        print(f"YouTube Quota Available: {quota_tracker.get_available_quota()} units")
        print(f"Lock File Present: {LOCK_FILE.exists()}")
        return

    # Scheduler Safety Lock Lifecycle (Rule 7)
    with scheduler_lock() as acquired:
        if not acquired:
            sys.exit(0)

        try:
            run_pipeline(
                niche=args.niche,
                topic_override=args.topic,
                dry_run=args.dry_run,
                skip_publish=args.skip_publish
            )
        except CircuitBreakerOpenError:
            sys.exit(2)
        except Exception:
            sys.exit(1)


if __name__ == "__main__":
    main()
