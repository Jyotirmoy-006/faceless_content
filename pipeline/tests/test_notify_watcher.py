"""Test Suite for T-1hr Pre-Publish Notification Watcher.

Complies with Acceptance Criteria:
- [x] A test row 58 minutes out fires exactly one notification
- [x] Running again immediately after fires no duplicate
- [x] A video that hasn't cleared Compliance Officer does NOT trigger a notice
- [x] A video that hasn't cleared Chief Critic does NOT trigger a notice
- [x] Videos outside the 55-65 minute window do NOT trigger a notice
- [x] Crash resilience: atomic pre_notice_sent update prevents duplicates across iterations
- [x] Strategist OptimalSchedule persistence directly updates SQLite
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from notify_watcher import run_watcher
from pipeline.agents.strategist import StrategistAgent, OptimalSchedule
from pipeline.core.notifier import Notifier
from pipeline.dashboard.database import (
    enqueue_job,
    get_job,
    get_videos_pending_pre_notice,
    init_db,
    mark_pre_notice_sent,
    persist_video_schedule,
    update_video_gate_status,
)


class TestNotifyWatcher(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_jobs.db"
        init_db(self.db_path)
        self.test_notifier = Notifier()
        self.fixed_now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

    def tearDown(self):
        import gc
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_row_58_minutes_out_fires_exactly_one_notification(self):
        """Acceptance Criteria 1: A test row 58 minutes out fires exactly one notification."""
        # 1. Insert test job
        job = enqueue_job(topic="58 Min Test Video", niche="tech", db_path=self.db_path)
        job_id = job["job_id"]

        # 2. Set publish time to 58 minutes in future
        target_time = self.fixed_now + timedelta(minutes=58)
        persist_video_schedule(
            job_id_or_pk=job_id,
            target_publish_datetime=target_time.isoformat(),
            summary="Quantum Computing Breakthrough",
            db_path=self.db_path
        )

        # 3. Mark both Chief Critic and Compliance Officer gates passed
        update_video_gate_status(
            job_id_or_pk=job_id,
            chief_critic_passed=True,
            compliance_passed=True,
            db_path=self.db_path
        )

        # 4. Run watcher
        results = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )

        # 5. Assert exactly one notification fired
        self.assertEqual(len(results), 1, "Expected exactly 1 notification sent")
        sent = results[0]
        self.assertEqual(sent["job_id"], job_id)
        self.assertIn("58 Min Test Video", sent["message"])
        self.assertIn(target_time.isoformat(), sent["message"])
        self.assertIn("Gates Cleared: Chief Critic [PASS] & Compliance Officer [PASS]", sent["message"])

        # 6. Verify Notifier received alert
        alerts = self.test_notifier.get_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["extra"]["type"], "pre_publish_notice")

        # 7. Verify pre_notice_sent is now marked 1 in the database
        updated_job = get_job(job_id, db_path=self.db_path)
        self.assertEqual(updated_job["pre_notice_sent"], 1)

    def test_immediate_rerun_fires_no_duplicate(self):
        """Acceptance Criteria 2: Running again immediately after fires no duplicate."""
        job = enqueue_job(topic="Deduplication Test Video", niche="tech", db_path=self.db_path)
        job_id = job["job_id"]
        target_time = self.fixed_now + timedelta(minutes=60)
        persist_video_schedule(
            job_id_or_pk=job_id,
            target_publish_datetime=target_time.isoformat(),
            summary="Fast Dedup Test",
            db_path=self.db_path
        )
        update_video_gate_status(
            job_id_or_pk=job_id,
            chief_critic_passed=True,
            compliance_passed=True,
            db_path=self.db_path
        )

        # First run fires notification
        first_run = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )
        self.assertEqual(len(first_run), 1)
        self.assertEqual(len(self.test_notifier.get_alerts()), 1)

        # Second run immediately after
        second_run = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )
        self.assertEqual(len(second_run), 0, "Second run must not fire duplicate notices")
        self.assertEqual(len(self.test_notifier.get_alerts()), 1, "No additional alerts should exist")

    def test_unapproved_compliance_officer_does_not_trigger_notice(self):
        """Acceptance Criteria 3: A video that hasn't cleared Compliance Officer does NOT trigger a notice."""
        job = enqueue_job(topic="Risky Compliance Test Video", niche="tech", db_path=self.db_path)
        job_id = job["job_id"]
        target_time = self.fixed_now + timedelta(minutes=58)
        persist_video_schedule(
            job_id_or_pk=job_id,
            target_publish_datetime=target_time.isoformat(),
            summary="Compliance Block Test",
            db_path=self.db_path
        )
        # Chief critic passed, but Compliance Officer has NOT passed (0)
        update_video_gate_status(
            job_id_or_pk=job_id,
            chief_critic_passed=True,
            compliance_passed=False,
            db_path=self.db_path
        )

        results = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )
        self.assertEqual(len(results), 0, "Video without compliance clearance must be blocked from notification")
        self.assertEqual(len(self.test_notifier.get_alerts()), 0)

        # Verify DB still has pre_notice_sent = 0
        job_record = get_job(job_id, db_path=self.db_path)
        self.assertEqual(job_record["pre_notice_sent"], 0)

    def test_unapproved_chief_critic_does_not_trigger_notice(self):
        """A video that hasn't cleared Chief Critic does NOT trigger a notice."""
        job = enqueue_job(topic="Poor Quality Video", niche="tech", db_path=self.db_path)
        job_id = job["job_id"]
        target_time = self.fixed_now + timedelta(minutes=58)
        persist_video_schedule(
            job_id_or_pk=job_id,
            target_publish_datetime=target_time.isoformat(),
            summary="Critic Block Test",
            db_path=self.db_path
        )
        # Compliance passed, but Chief Critic has NOT passed (0)
        update_video_gate_status(
            job_id_or_pk=job_id,
            chief_critic_passed=False,
            compliance_passed=True,
            db_path=self.db_path
        )

        results = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )
        self.assertEqual(len(results), 0, "Video without critic clearance must be blocked from notification")

    def test_outside_window_does_not_fire(self):
        """Videos outside the 55-65 minute window must not fire."""
        # 40 minutes out (< 55 min)
        job_too_soon = enqueue_job(topic="40 Min Video", niche="tech", db_path=self.db_path)
        persist_video_schedule(
            job_id_or_pk=job_too_soon["job_id"],
            target_publish_datetime=(self.fixed_now + timedelta(minutes=40)).isoformat(),
            db_path=self.db_path
        )
        update_video_gate_status(job_too_soon["job_id"], chief_critic_passed=True, compliance_passed=True, db_path=self.db_path)

        # 80 minutes out (> 65 min)
        job_too_late = enqueue_job(topic="80 Min Video", niche="tech", db_path=self.db_path)
        persist_video_schedule(
            job_id_or_pk=job_too_late["job_id"],
            target_publish_datetime=(self.fixed_now + timedelta(minutes=80)).isoformat(),
            db_path=self.db_path
        )
        update_video_gate_status(job_too_late["job_id"], chief_critic_passed=True, compliance_passed=True, db_path=self.db_path)

        results = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )
        self.assertEqual(len(results), 0, "No notices should fire for videos outside 55-65m")

    def test_strategist_optimal_schedule_persistence(self):
        """Strategist OptimalSchedule persists target_publish_datetime and pre_notice_sent=0."""
        agent = StrategistAgent()
        job = enqueue_job(topic="AI Future Predictions", niche="tech", db_path=self.db_path)
        job_id = job["job_id"]

        target_time = self.fixed_now + timedelta(hours=2)
        schedule = agent.schedule_video(
            job_id_or_pk=job_id,
            topic="AI Future Predictions",
            niche="tech",
            target_datetime=target_time,
            db_path=self.db_path
        )

        self.assertIsInstance(schedule, OptimalSchedule)
        self.assertEqual(schedule.target_publish_datetime, target_time.isoformat())
        self.assertTrue(len(schedule.recommended_tags) > 0)
        self.assertTrue(len(schedule.primary_hashtags) > 0)

        # Check DB row
        db_job = get_job(job_id, db_path=self.db_path)
        self.assertEqual(db_job["target_publish_datetime"], target_time.isoformat())
        self.assertEqual(db_job["pre_notice_sent"], 0)
        self.assertIn("Scheduled for", db_job["schedule_summary"])

    def test_crash_resilience_multi_row(self):
        """Crash resilience: row N is committed immediately before row N+1 is processed."""
        # Row 1: 56 mins out
        job1 = enqueue_job(topic="Video 1", niche="tech", db_path=self.db_path)
        persist_video_schedule(job1["job_id"], (self.fixed_now + timedelta(minutes=56)).isoformat(), db_path=self.db_path)
        update_video_gate_status(job1["job_id"], chief_critic_passed=True, compliance_passed=True, db_path=self.db_path)

        # Row 2: 62 mins out
        job2 = enqueue_job(topic="Video 2", niche="tech", db_path=self.db_path)
        persist_video_schedule(job2["job_id"], (self.fixed_now + timedelta(minutes=62)).isoformat(), db_path=self.db_path)
        update_video_gate_status(job2["job_id"], chief_critic_passed=True, compliance_passed=True, db_path=self.db_path)

        # Simulate crash on row 2 notification
        original_alert = self.test_notifier.alert
        call_count = 0

        def flaky_alert(message, level="ERROR", extra=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated transient network drop during row 2 alert")
            return original_alert(message, level=level, extra=extra)

        with patch.object(self.test_notifier, "alert", side_effect=flaky_alert):
            results = run_watcher(
                db_path=self.db_path,
                reference_time=self.fixed_now,
                notifier_instance=self.test_notifier
            )

        # Row 1 succeeded, Row 2 failed with exception caught & logged
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["job_id"], job1["job_id"])

        # Row 1 MUST be marked pre_notice_sent = 1
        db_job1 = get_job(job1["job_id"], db_path=self.db_path)
        self.assertEqual(db_job1["pre_notice_sent"], 1)

        # Row 2 MUST NOT be marked pre_notice_sent = 1
        db_job2 = get_job(job2["job_id"], db_path=self.db_path)
        self.assertEqual(db_job2["pre_notice_sent"], 0)

        # Next run when network is restored:
        second_run = run_watcher(
            db_path=self.db_path,
            reference_time=self.fixed_now,
            notifier_instance=self.test_notifier
        )
        # Exactly row 2 is sent now, NO duplicate for row 1
        self.assertEqual(len(second_run), 1)
        self.assertEqual(second_run[0]["job_id"], job2["job_id"])

        # Both rows now marked sent
        self.assertEqual(get_job(job1["job_id"], db_path=self.db_path)["pre_notice_sent"], 1)
        self.assertEqual(get_job(job2["job_id"], db_path=self.db_path)["pre_notice_sent"], 1)


if __name__ == "__main__":
    unittest.main()
