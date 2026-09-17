"""Comprehensive Unit and Integration Tests for Phase 3C Publisher Agent.

Acceptance Criteria Verified:
1. Simulated quota exhaustion results in clean "DEFERRED" state (no exceptions escape, no upload attempted).
2. YouTube upload increments quota tracker by exactly 1600 units, detects unverified project privacy restriction.
3. Written justification for Instagram hosting + real external HTTPS reachability demonstration on Backblaze B2.
4. Two-phase Graph API container polling respects timeout and avoids tight looping (paced backoff).
5. Ephemeral B2 hosted video is guaranteed purged after successful publish AND after failure/exception.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
from dotenv import load_dotenv

from pipeline.agents.publisher import PublisherAgent, PublishResult, PublishStatus
from pipeline.core.b2_hosting import B2HostingBridge, b2_bridge
from pipeline.core.quota_tracker import QuotaTracker, YOUTUBE_VIDEO_UPLOAD_COST

load_dotenv()


class TestPublisherAgent(unittest.TestCase):
    """Test suite covering YouTube quota safety and Instagram hosting/publishing flows."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_pub_"))
        self.state_file = self.temp_dir / "quota_state.json"
        self.quota_tracker = QuotaTracker(state_file=self.state_file, daily_limit=10_000)
        self.publisher = PublisherAgent(quota_mgr=self.quota_tracker, hosting_bridge=b2_bridge)

        # Create a dummy test video file
        self.dummy_video = self.temp_dir / "test_video.mp4"
        with open(self.dummy_video, "wb") as f:
            # Minimal MP4 header bytes
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00isommp42\x00\x00\x00\x08free")

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # CRITERIA 1: YouTube Quota Budget Exhaustion -> Clean Deferred State
    # -------------------------------------------------------------------------
    def test_youtube_quota_exhaustion_clean_deferred_state(self):
        """Simulate a day where quota budget is exhausted (< 1600 units).
        
        Verifies:
        - Result is cleanly PublishStatus.DEFERRED
        - No exception escapes to caller
        - Zero units spent
        - No network upload call is attempted
        """
        # Simulate having only 500 units left out of 10,000
        self.quota_tracker.simulate_quota(used_units=9500)
        self.assertEqual(self.quota_tracker.get_available_quota(), 500)
        self.assertFalse(self.quota_tracker.has_budget(YOUTUBE_VIDEO_UPLOAD_COST))

        # Attempt to publish
        with patch.object(self.publisher, "_execute_youtube_upload") as mock_upload:
            result = self.publisher.publish_to_youtube(
                video_path=self.dummy_video,
                title="AI Revolution in 60 Seconds",
                description="Testing quota gating #shorts",
            )

            # Assert upload was NOT called
            mock_upload.assert_not_called()

        # Assert clean deferred state
        self.assertIsInstance(result, PublishResult)
        self.assertEqual(result.status, PublishStatus.DEFERRED)
        self.assertEqual(result.units_spent, 0)
        self.assertIn("exhausted", result.message.lower())
        self.assertIn("deferring", result.message.lower())
        print("\n[CRITERIA 1 PASS] Quota exhaustion clean deferral verified: no exception, status=DEFERRED.")

    # -------------------------------------------------------------------------
    # CRITERIA 2: YouTube Upload Consumes 1600 Units & Detects Unverified Project
    # -------------------------------------------------------------------------
    def test_youtube_upload_consumes_1600_units(self):
        """Verify that a successful upload increments the quota tracker by 1600 units,
        and surfaces visibility restrictions for unverified projects.
        """
        initial_used = self.quota_tracker.get_used_units()
        self.assertEqual(initial_used, 0)

        mock_upload_resp = {
            "id": "YT_TEST_12345",
            "snippet": {"title": "Test Title"},
            "status": {"privacyStatus": "private"}  # Simulating unverified project restriction
        }

        with patch.object(self.publisher, "_execute_youtube_upload", return_value=mock_upload_resp):
            result = self.publisher.publish_to_youtube(
                video_path=self.dummy_video,
                title="Autonomous AI Test",
                privacy_status="public"
            )

        self.assertEqual(result.status, PublishStatus.PUBLISHED)
        self.assertEqual(result.post_id, "YT_TEST_12345")
        self.assertEqual(result.units_spent, 1600)
        # Verify unverified project privacy restriction was surfaced
        self.assertTrue(result.is_restricted)
        self.assertEqual(result.visibility, "private")

        # Verify quota tracker was updated immediately
        self.assertEqual(self.quota_tracker.get_used_units(), 1600)
        self.assertEqual(self.quota_tracker.get_available_quota(), 8400)

        # Verify disk persistence in quota_state.json
        with open(self.state_file, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        self.assertEqual(persisted["used_units"], 1600)
        self.assertEqual(persisted["history"][-1]["units"], 1600)
        print("\n[CRITERIA 2 PASS] Quota consumption (1600 units) and unverified project warning verified.")

    def test_youtube_upload_failure_refunds_quota(self):
        """Verify that a failed upload automatically refunds the 1600 reserved quota units."""
        self.assertEqual(self.quota_tracker.get_used_units(), 0)

        with patch.object(self.publisher, "_execute_youtube_upload", side_effect=RuntimeError("Simulated upload network crash")):
            result = self.publisher.publish_to_youtube(
                video_path=self.dummy_video,
                title="Failed Upload Concept"
            )

        self.assertEqual(result.status, PublishStatus.FAILED)
        self.assertEqual(result.units_spent, 0)
        # Quota must be safely refunded
        self.assertEqual(self.quota_tracker.get_used_units(), 0)
        self.assertEqual(self.quota_tracker.get_available_quota(), 10_000)


    # -------------------------------------------------------------------------
    # CRITERIA 3: Instagram Hosting Justification & External HTTPS Reachability
    # -------------------------------------------------------------------------
    def test_instagram_hosting_justification_and_external_reachability(self):
        """Verifies written justification and demonstrates real external HTTPS
        reachability of Backblaze B2 temporary pre-signed URL (fetchable outside localhost).
        """
        # 1. Verify Architectural Justification in Code Documentation
        doc = (B2HostingBridge.__doc__ or "").lower()
        self.assertIn("cloudflare tunnel", doc)
        self.assertIn("backblaze b2", doc)
        self.assertIn("reliability", doc)
        self.assertIn("crawler", doc)

        # 2. Live B2 Demonstration
        if not b2_bridge.is_configured():
            self.skipTest("Backblaze B2 credentials not populated in .env; skipping live network test.")

        presigned_url, object_key = b2_bridge.upload_temporary_media(
            file_path=self.dummy_video,
            expires_in_seconds=180
        )

        try:
            # Prove URL is not localhost / loopback
            self.assertTrue(presigned_url.startswith("https://"))
            self.assertIn("backblazeb2.com", presigned_url)
            self.assertNotIn("localhost", presigned_url)
            self.assertNotIn("127.0.0.1", presigned_url)

            # Fetch genuinely from public internet without any auth headers
            res = requests.get(presigned_url, timeout=15)
            self.assertEqual(res.status_code, 200)
            self.assertGreater(len(res.content), 0)
            print(f"\n[CRITERIA 3 PASS] B2 public URL verified externally reachable: HTTP {res.status_code} ({len(res.content)} bytes).")
        finally:
            # Purge
            b2_bridge.delete_media(object_key)
            self.assertFalse(b2_bridge.check_exists(object_key))

    # -------------------------------------------------------------------------
    # CRITERIA 4: Container Polling Respects Max Timeout & Does Not Tight-Loop
    # -------------------------------------------------------------------------
    def test_instagram_container_polling_timeout_and_non_tight_loop(self):
        """Verifies container status polling has a max timeout and paces requests with backoff."""
        poll_timestamps = []

        def mock_poll_get(url, params=None, timeout=None):
            poll_timestamps.append(time.monotonic())
            mock_res = MagicMock()
            mock_res.status_code = 200
            # Always return IN_PROGRESS to test timeout trigger
            mock_res.json.return_value = {"status_code": "IN_PROGRESS"}
            return mock_res

        with patch("requests.get", side_effect=mock_poll_get):
            with self.assertRaises(TimeoutError) as ctx:
                # Test with short timeout of 0.6s and 0.15s initial interval
                self.publisher._poll_container_status(
                    container_id="CONTAINER_MOCK_1",
                    access_token="TEST_TOKEN",
                    timeout_seconds=0.5,
                    initial_poll_interval=0.15,
                    max_poll_interval=0.3
                )

        self.assertIn("Timed out", str(ctx.exception))
        # Ensure it polled at least once but did NOT tight loop (e.g. not hundreds of calls)
        self.assertGreaterEqual(len(poll_timestamps), 2)
        self.assertLessEqual(len(poll_timestamps), 6)

        # Verify time gap between successive polls is at least the initial interval
        for i in range(1, len(poll_timestamps)):
            gap = poll_timestamps[i] - poll_timestamps[i - 1]
            self.assertGreaterEqual(gap, 0.12)  # Allow small clock tolerance

        print(f"\n[CRITERIA 4 PASS] Polling pace verified: {len(poll_timestamps)} polls in ~0.5s (no tight loop, timeout enforced).")

    # -------------------------------------------------------------------------
    # CRITERIA 5: Temporary Hosted File Cleanup After Publish & After Failure
    # -------------------------------------------------------------------------
    def test_hosted_file_cleanup_on_publish_and_failure(self):
        """Demonstrates that temporary B2 asset is guaranteed purged both when publish
        succeeds AND when an unexpected error occurs during processing.
        """
        if not b2_bridge.is_configured():
            self.skipTest("Backblaze B2 credentials not populated in .env; skipping live cleanup test.")

        # Test Case A: Cleanup on Success Path
        captured_key = None
        with b2_bridge.temporary_public_url(self.dummy_video, expires_in_seconds=120) as (url, key):
            captured_key = key
            self.assertTrue(b2_bridge.check_exists(captured_key))

        # Outside context manager: must be deleted
        self.assertFalse(b2_bridge.check_exists(captured_key))

        # Test Case B: Cleanup on Failure / Exception Path
        captured_fail_key = None
        try:
            with b2_bridge.temporary_public_url(self.dummy_video, expires_in_seconds=120) as (url, key):
                captured_fail_key = key
                self.assertTrue(b2_bridge.check_exists(captured_fail_key))
                # Simulate an unexpected failure during processing
                raise ValueError("Simulated network drop during container ingestion")
        except ValueError:
            pass

        # Outside exception: must STILL be purged!
        self.assertFalse(b2_bridge.check_exists(captured_fail_key))
        print("\n[CRITERIA 5 PASS] Temporary B2 object cleanup verified on both success and failure paths.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
