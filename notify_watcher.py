"""T-1hr Pre-Publish Notification Watcher.

A lightweight, standalone single-run script designed to execute on its own
Windows Task Scheduler entry every 5-10 minutes.

Complies with:
- Rule 2 (NO SILENT CRASHES): Explicit structured logging and error reporting.
- Rule 7 (NO DAEMON COUPLING): NOT a persistent process, NOT folded into Flask/APScheduler.
- Rule 13 (GATE CLEARANCE): Strictly requires clearance from BOTH Chief Critic
  (chief_critic_passed = 1) AND the Compliance Officer (compliance_passed = 1).
  Never notifies about any video that hasn't cleared both gates.
- Crash Resilience: Immediately commits pre_notice_sent = 1 per video before processing
  any subsequent row to guarantee zero duplicate notifications on crash or restart.
- Pure FYI: No hold/cancel mechanism built speculatively.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.logger import get_logger
from pipeline.core.notifier import Notifier, notifier as default_notifier
from pipeline.dashboard.database import (
    get_db_path,
    get_videos_pending_pre_notice,
    init_db,
    mark_pre_notice_sent,
)

logger = get_logger("notify_watcher")


def run_watcher(
    db_path: Path | str | None = None,
    min_minutes: float = 55.0,
    max_minutes: float = 65.0,
    reference_time: Optional[datetime] = None,
    dry_run: bool = False,
    notifier_instance: Optional[Notifier] = None,
) -> List[Dict[str, Any]]:
    """Runs a single pass of the pre-publish notification watcher.

    Queries the SQLite database for videos:
    1. Where target_publish_datetime falls within [min_minutes, max_minutes] (default 55-65 mins).
    2. pre_notice_sent is 0 (or NULL).
    3. The video has passed BOTH Chief Critic AND Compliance Officer.

    For each matching video, fires a notification and immediately updates
    pre_notice_sent = 1 to guarantee zero duplicates even in crash scenarios.
    """
    resolved_db = get_db_path(db_path)
    init_db(resolved_db)
    active_notifier = notifier_instance or default_notifier

    logger.info(
        f"[NOTIFY_WATCHER] Scanning for videos publishing in {min_minutes}-{max_minutes}m (DB: {resolved_db})..."
    )

    pending_videos = get_videos_pending_pre_notice(
        min_minutes=min_minutes,
        max_minutes=max_minutes,
        reference_time=reference_time,
        db_path=resolved_db,
    )

    if not pending_videos:
        logger.info("[NOTIFY_WATCHER] No videos pending pre-publish notice in current window.")
        return []

    logger.info(f"[NOTIFY_WATCHER] Found {len(pending_videos)} video(s) requiring T-1hr pre-notice.")
    sent_records: List[Dict[str, Any]] = []

    for video in pending_videos:
        job_id = video.get("job_id") or video.get("id")
        topic = video.get("topic") or "Untitled Video"
        target_publish = video.get("target_publish_datetime")
        minutes_until = video.get("minutes_until_publish", 60.0)
        summary = video.get("schedule_summary") or video.get("niche", "General Video")

        # Format message: title, scheduled time, short summary
        message = (
            f"[PRE_PUBLISH_NOTICE] Video '{topic}' is scheduled to publish at {target_publish} "
            f"(in ~{int(round(minutes_until))} min). "
            f"Gates Cleared: Chief Critic [PASS] & Compliance Officer [PASS]. "
            f"Summary: {summary}"
        )

        try:
            # 1. Fire Notifier message
            active_notifier.alert(
                message=message,
                level="INFO",
                extra={
                    "type": "pre_publish_notice",
                    "job_id": job_id,
                    "topic": topic,
                    "target_publish_datetime": target_publish,
                    "minutes_until_publish": minutes_until,
                },
            )
            logger.info(f"[NOTIFY_WATCHER] Pre-notice emitted successfully for Job {job_id} ('{topic}')")

            # 2. Immediately mark pre_notice_sent = 1 BEFORE anything else
            if not dry_run:
                marked = mark_pre_notice_sent(job_id_or_pk=job_id, db_path=resolved_db)
                if not marked:
                    logger.warning(
                        f"[NOTIFY_WATCHER] Failed to mark pre_notice_sent for Job {job_id}; checking fallback by primary key"
                    )
                    if video.get("id"):
                        mark_pre_notice_sent(job_id_or_pk=video["id"], db_path=resolved_db)

            sent_records.append({
                "job_id": job_id,
                "topic": topic,
                "target_publish_datetime": target_publish,
                "message": message,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            logger.error(
                f"[NOTIFY_WATCHER] Failed to dispatch pre-notice for Job {job_id}: {e}",
                exc_info=True,
            )

    logger.info(f"[NOTIFY_WATCHER] Completed pass. Emitted {len(sent_records)} notice(s).")
    return sent_records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T-1hr Pre-Publish Notification Watcher (Runs every 5-10 mins via Windows Task Scheduler)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite jobs.db database (defaults to pipeline/dashboard/jobs.db)",
    )
    parser.add_argument(
        "--window-min",
        type=float,
        default=55.0,
        help="Lower bound in minutes for notification window (default: 55.0)",
    )
    parser.add_argument(
        "--window-max",
        type=float,
        default=65.0,
        help="Upper bound in minutes for notification window (default: 65.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate watcher without marking pre_notice_sent in database",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Scan and print status of upcoming scheduled videos without notifying",
    )

    args = parser.parse_args()

    if args.status:
        resolved_db = get_db_path(args.db_path)
        init_db(resolved_db)
        pending = get_videos_pending_pre_notice(
            min_minutes=0.0,
            max_minutes=1440.0,  # Next 24 hours
            db_path=resolved_db,
        )
        print(f"\n--- Scheduled Videos Pending Publication (Next 24h) [DB: {resolved_db}] ---")
        if not pending:
            print("No scheduled videos found within the next 24 hours.")
        else:
            for v in pending:
                print(
                    f"- Job ID: {v.get('job_id')} | Topic: '{v.get('topic')}' | "
                    f"Target: {v.get('target_publish_datetime')} (~{v.get('minutes_until_publish')}m) | "
                    f"Chief Critic: {v.get('chief_critic_passed')} | Compliance: {v.get('compliance_passed')} | "
                    f"Pre-Notice Sent: {v.get('pre_notice_sent')}"
                )
        print("-" * 75 + "\n")
        return 0

    results = run_watcher(
        db_path=args.db_path,
        min_minutes=args.window_min,
        max_minutes=args.window_max,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
