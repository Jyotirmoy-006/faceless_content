"""Unit test proving YouTube Quota Tracker resets at midnight Pacific Time with DST handling.

Verifies:
1. Midnight PT Boundary Rollover: Quota usage spent at 23:59:59 PT resets cleanly to 0 at 00:00:01 PT.
2. UTC vs Pacific Discrepancy (Bug Fix Proof): Quota does NOT reset when UTC midnight passes (17:00 PT);
   it resets ONLY when Pacific midnight passes (07:00 UTC in PDT, 08:00 UTC in PST).
3. Daylight Saving Time (DST) Transitions:
   - Spring-forward (PST -> PDT): Clocks advance from 02:00 to 03:00 without premature reset.
   - Fall-back (PDT -> PST): Repeated 01:00 hour remains within the same Pacific date without premature reset.
"""

import json
import shutil
import tempfile
import unittest
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.quota_tracker import QuotaTracker, PACIFIC_TZ, YOUTUBE_VIDEO_UPLOAD_COST


class TestQuotaDST(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_quota_dst_"))
        self.state_file = self.temp_dir / "quota_state.json"
        self.tracker = QuotaTracker(state_file=self.state_file, daily_limit=10_000)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_midnight_pt_boundary_rollover(self):
        """Proves quota resets at 00:00 Pacific Time, not at UTC midnight."""
        # 1. Simulate 23:59:59 PT on June 15, 2026 (PDT, UTC-7)
        t_before_midnight = datetime(2026, 6, 15, 23, 59, 59, tzinfo=PACIFIC_TZ)
        self.tracker.consume(8000, reason="videos.insert", current_time=t_before_midnight)

        self.assertEqual(self.tracker.get_used_units(current_time=t_before_midnight), 8000)
        self.assertEqual(self.tracker.get_available_quota(current_time=t_before_midnight), 2000)
        self.assertTrue(self.tracker.has_budget(1600, current_time=t_before_midnight))

        # 2. Advance 2 seconds into June 16, 2026 at 00:00:01 PT
        t_after_midnight = datetime(2026, 6, 16, 0, 0, 1, tzinfo=PACIFIC_TZ)

        # Quota must automatically detect new Pacific day and reset to 0 used
        self.assertEqual(self.tracker.get_used_units(current_time=t_after_midnight), 0)
        self.assertEqual(self.tracker.get_available_quota(current_time=t_after_midnight), 10000)
        self.assertTrue(self.tracker.has_budget(YOUTUBE_VIDEO_UPLOAD_COST, current_time=t_after_midnight))

        # Verify state file on disk
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state["date"], "2026-06-16")
        self.assertEqual(state["used_units"], 0)
        self.assertEqual(state["timezone"], "America/Los_Angeles")

    def test_utc_rollover_does_not_prematurely_reset_quota(self):
        """Proves that when UTC midnight rolls over, the tracker DOES NOT reset if it is still afternoon PT."""
        # In summer (PDT, UTC-7), 00:30 UTC on June 16 is 17:30 PDT (5:30 PM) on June 15!
        utc_midnight_time = datetime(2026, 6, 16, 0, 30, 0, tzinfo=timezone.utc)
        pt_equivalent = utc_midnight_time.astimezone(PACIFIC_TZ)
        self.assertEqual(pt_equivalent.strftime("%Y-%m-%d %H:%M"), "2026-06-15 17:30")

        # Spend quota at 16:00 PT
        t_afternoon = datetime(2026, 6, 15, 16, 0, 0, tzinfo=PACIFIC_TZ)
        self.tracker.consume(9000, reason="videos.insert", current_time=t_afternoon)
        self.assertEqual(self.tracker.get_used_units(current_time=t_afternoon), 9000)

        # Check at 00:30 UTC (June 16 in UTC, but STILL June 15 in PT!)
        # The OLD UTC tracker would have prematurely reset to 0!
        # The NEW Pacific tracker MUST maintain 9000 used units!
        used_at_utc_midnight = self.tracker.get_used_units(current_time=utc_midnight_time)
        self.assertEqual(
            used_at_utc_midnight, 9000,
            "Bug check: Quota must NOT reset at 00:30 UTC because Pacific date is still June 15!"
        )

        # Now advance to 07:01 UTC on June 16 (= 00:01 PDT June 16)
        pt_midnight_utc = datetime(2026, 6, 16, 7, 1, 0, tzinfo=timezone.utc)
        self.assertEqual(self.tracker.get_used_units(current_time=pt_midnight_utc), 0)
        self.assertEqual(self.tracker.get_available_quota(current_time=pt_midnight_utc), 10000)

    def test_spring_forward_dst_transition(self):
        """Proves correct behavior during Spring Forward (PST -> PDT, 1-hour jump)."""
        # On March 8, 2026, clocks jump from 02:00 PST to 03:00 PDT
        t_march7_night = datetime(2026, 3, 7, 23, 50, 0, tzinfo=PACIFIC_TZ)
        self.tracker.consume(3200, reason="videos.insert", current_time=t_march7_night)
        self.assertEqual(self.tracker.get_used_units(current_time=t_march7_night), 3200)

        # 00:05 March 8: resets for March 8
        t_march8_early = datetime(2026, 3, 8, 0, 5, 0, tzinfo=PACIFIC_TZ)
        self.assertEqual(self.tracker.get_used_units(current_time=t_march8_early), 0)
        self.tracker.consume(1600, reason="videos.insert", current_time=t_march8_early)

        # 03:30 PDT March 8 (after 1-hour spring forward skip): must remain March 8 usage
        t_march8_after_jump = datetime(2026, 3, 8, 3, 30, 0, tzinfo=PACIFIC_TZ)
        self.assertEqual(self.tracker.get_used_units(current_time=t_march8_after_jump), 1600)

        # 00:01 PDT March 9: rolls over to March 9
        t_march9 = datetime(2026, 3, 9, 0, 1, 0, tzinfo=PACIFIC_TZ)
        self.assertEqual(self.tracker.get_used_units(current_time=t_march9), 0)

    def test_fall_back_dst_transition(self):
        """Proves correct behavior during Fall Back (PDT -> PST, 1-hour repeat)."""
        # On November 1, 2026, clocks fall back from 02:00 PDT to 01:00 PST
        t_oct31 = datetime(2026, 10, 31, 23, 55, 0, tzinfo=PACIFIC_TZ)
        self.tracker.consume(4800, reason="videos.insert", current_time=t_oct31)
        self.assertEqual(self.tracker.get_used_units(current_time=t_oct31), 4800)

        # 00:01 Nov 1 (PDT, UTC-7): rolls over to Nov 1
        t_nov1_early = datetime(2026, 11, 1, 0, 1, 0, tzinfo=PACIFIC_TZ)
        self.assertEqual(self.tracker.get_used_units(current_time=t_nov1_early), 0)
        self.tracker.consume(1600, reason="videos.insert", current_time=t_nov1_early)

        # Nov 1 afternoon (PST, UTC-8): remains Nov 1 usage
        t_nov1_afternoon = datetime(2026, 11, 1, 14, 0, 0, tzinfo=PACIFIC_TZ)
        self.assertEqual(self.tracker.get_used_units(current_time=t_nov1_afternoon), 1600)

        # 00:01 Nov 2 (PST, UTC-8): rolls over to Nov 2
        t_nov2 = datetime(2026, 11, 2, 0, 1, 0, tzinfo=PACIFIC_TZ)
        self.assertEqual(self.tracker.get_used_units(current_time=t_nov2), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
