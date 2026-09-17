"""Mock pipeline entry point for testing the Command Center Job Queue (Mission H).

Simulates the multi-stage pipeline with telemetry stage markers and human-in-the-loop review.
Strict log formatting: [YYYY-MM-DD HH:MM:SS] [AGENT/WORKER] Message
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "pipeline" / "output"


def log(agent: str, message: str) -> None:
    """Prints strictly formatted log line."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] [{agent}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Mock Autonomous Faceless Video Pipeline")
    parser.add_argument("--topic", type=str, default="Mock Test Concept", help="Topic name")
    parser.add_argument("--niche", type=str, default="tech", help="Niche")
    parser.add_argument("--require-approval", action="store_true", help="Pause pipeline for human script review")
    parser.add_argument("--approved-script", type=str, default=None, help="Path to approved script JSON file")
    parser.add_argument("--publish-target", type=str, default="youtube", choices=["youtube", "disk_only"], help="Publish destination")
    parser.add_argument("--skip-publish", action="store_true", help="Skip publishing")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    args, _ = parser.parse_known_args()

    topic = args.topic or "Autonomous Ideation Breakthrough"
    niche = args.niche or "tech"
    step_sleep = float(os.environ.get("MOCK_STEP_SLEEP", "1.5"))

    log("ORCHESTRATOR", f"Starting pipeline execution: '{topic}' [niche: {niche}]")
    log("ORCHESTRATOR", "Preflight check: VRAM and Scheduler safety verified.")

    # -------------------------------------------------------------------------
    # STAGE 1 & 2: IDEATION & SCRIPTWRITING (or load approved script)
    # -------------------------------------------------------------------------
    if args.approved_script and Path(args.approved_script).exists():
        try:
            with open(args.approved_script, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            log("SCRIPTWRITER", f"Loaded approved script from human review: '{topic}'")
            topic = script_data.get("topic", topic)
        except Exception as e:
            log("SCRIPTWRITER", f"Failed to load approved script ({e}), using default.")
    else:
        # STAGE 1: IDEATION
        print("[STAGE:IDEATION]", flush=True)
        log("IDEATOR", f"Synthesizing high-retention concepts for niche '{niche}'...")
        time.sleep(step_sleep)
        log("IDEATOR", f"Concept approved: '{topic}' (Predicted Retention: 94.8%)")

        # STAGE 2: SCRIPTWRITING
        print("[STAGE:SCRIPTWRITING]", flush=True)
        log("SCRIPTWRITER", "Generating 3-segment short with zero-breath pacing and strong hook...")
        time.sleep(step_sleep)

        mock_script = {
            "topic": topic,
            "niche": niche,
            "hook": f"Did you know the secret behind {topic}?",
            "target_duration": 42,
            "segments": [
                {
                    "segment_index": 1,
                    "duration_seconds": 12.0,
                    "narration": f"Here is the shocking reality behind {topic} that most people ignore.",
                    "visual_query": "cinematic macro shot of advanced glowing technology"
                },
                {
                    "segment_index": 2,
                    "duration_seconds": 18.0,
                    "narration": "Engineers discovered that subtle shifts in electrical frequency alter the entire system dynamics.",
                    "visual_query": "high tech holographic data visualization interface"
                },
                {
                    "segment_index": 3,
                    "duration_seconds": 12.0,
                    "narration": "Stay ahead of the curve: follow for daily breakthroughs and innovations.",
                    "visual_query": "futuristic city skyline sunset hyperlapse"
                }
            ]
        }

        # Human-in-the-loop review check
        if args.require_approval:
            print(f"[SCRIPT_JSON] {json.dumps(mock_script)}", flush=True)
            print("[STAGE:PENDING_REVIEW]", flush=True)
            log("WORKER", "Human script review required: Pipeline paused in PENDING_REVIEW status.")
            sys.exit(10)

    # -------------------------------------------------------------------------
    # STAGE 3: AUDIO_TTS
    # -------------------------------------------------------------------------
    print("[STAGE:AUDIO_TTS]", flush=True)
    log("AUDIO_TTS", "Synthesizing voiceover with edge-tts (+15% speed, zero-breath trim)...")
    time.sleep(step_sleep)
    log("AUDIO_TTS", "Speech synthesis complete: 3 audio stems aligned (duration: 41.6s).")

    # -------------------------------------------------------------------------
    # STAGE 4: RENDER
    # -------------------------------------------------------------------------
    print("[STAGE:RENDER]", flush=True)
    log("RENDER", "Normalizing visual clips to 1080x1920 30fps CFR...")
    time.sleep(step_sleep)
    log("RENDER", "Burning word-level styled captions in bottom safe zone...")
    time.sleep(step_sleep)

    # Create valid playable output video for gallery testing if not present
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_slug = "".join(c if c.isalnum() else "_" for c in topic)[:25].strip("_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dummy_video = OUTPUT_DIR / f"{ts}_{safe_slug}.mp4"
    if not dummy_video.exists():
        try:
            sample_videos = [p for p in OUTPUT_DIR.glob("*.mp4") if p != dummy_video and p.stat().st_size > 100000]
            if sample_videos:
                import shutil
                shutil.copy2(sample_videos[0], dummy_video)
            else:
                import subprocess
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=0x18181b:s=1080x1920:d=2",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(dummy_video)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    log("RENDER", f"Video render completed successfully: {dummy_video.name} (18.4 MB)")

    # -------------------------------------------------------------------------
    # STAGE 5: PUBLISH
    # -------------------------------------------------------------------------
    print("[STAGE:PUBLISH]", flush=True)
    if args.publish_target == "disk_only" or args.skip_publish:
        log("PUBLISHER", f"Video saved to disk ({dummy_video.name}). External publishing skipped per settings.")
    else:
        log("PUBLISHER", f"Uploading short to YouTube channel (Title: '{topic}')...")
        time.sleep(step_sleep)
        log("PUBLISHER", f"YouTube Shorts upload verified (Video ID: mock_{int(time.time())}).")

    log("ORCHESTRATOR", f"SUCCESS: Autonomous pipeline run complete for '{topic}'.")
    try:
        from pipeline.dashboard.database import record_run_telemetry
        mock_telem = {
            "concept": {"topic": topic, "niche": niche},
            "stages": {"ideation": step_sleep, "scriptwriting": step_sleep, "production": step_sleep * 2, "publish": step_sleep},
            "vram": {"pre_render_mb": 128.0, "post_render_mb": 150.0},
            "quota_spent": 0 if args.skip_publish or args.publish_target == "disk_only" else 50,
            "video_path": str(dummy_video),
            "video_size_mb": 18.4,
            "total_wall_clock_time": step_sleep * 5
        }
        record_run_telemetry(mock_telem, status="SUCCESS")
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
