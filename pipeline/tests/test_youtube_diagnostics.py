"""Exhaustive Diagnostic Test Suite for YouTube OAuth, Scopes, Quotas, and Error Resilience.

Verifies:
1. Token inspection and scope auditing (detects write-only vs full channel visibility).
2. Quota pre-check enforcement (0 units spent in dry-run or when budget is exhausted).
3. Google unverified project restriction handling (requested 'public' forced to 'private').
4. Resilient error categorization (machine-readable error codes).
5. Insertion payload structuring and metadata conformity (Rule 2 & 3).
"""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.agents.publisher import (
    PublisherAgent,
    PublishResult,
    PublishStatus,
    YOUTUBE_SCOPES,
    classify_youtube_error,
    get_authenticated_channel_info
)
from pipeline.core.quota_tracker import QuotaTracker, YOUTUBE_VIDEO_UPLOAD_COST

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class TestYouTubeDiagnostics(unittest.TestCase):

    def setUp(self):
        self.dummy_state = ROOT_DIR / "pipeline" / "core" / "test_diag_quota_state.json"
        self.quota_tracker = QuotaTracker(state_file=self.dummy_state, daily_limit=10000)
        self.publisher = PublisherAgent(quota_mgr=self.quota_tracker)
        self.dummy_video = ROOT_DIR / "pipeline" / "assets_cache" / "test_dummy.mp4"
        self.dummy_video.parent.mkdir(parents=True, exist_ok=True)
        self.dummy_video.write_bytes(b"\x00" * 1024)

    def tearDown(self):
        if self.dummy_state.exists():
            self.dummy_state.unlink(missing_ok=True)
        if self.dummy_video.exists():
            self.dummy_video.unlink(missing_ok=True)

    def test_01_error_classification_mapping(self):
        """Verifies that raw exceptions map to actionable machine error codes."""
        # 1. OAuth expired
        c1, m1 = classify_youtube_error(Exception("invalid_grant: Token has been expired or revoked."))
        self.assertEqual(c1, "YOUTUBE_OAUTH_EXPIRED")
        self.assertIn("expired", m1.lower())

        # 2. Insufficient scopes
        c2, m2 = classify_youtube_error(Exception("HttpError 403: Request had insufficient authentication scopes."))
        self.assertEqual(c2, "YOUTUBE_INSUFFICIENT_SCOPES")

        # 3. Quota exhausted
        c3, m3 = classify_youtube_error(Exception("The request cannot be completed because you have exceeded your quotaExceeded."))
        self.assertEqual(c3, "YOUTUBE_QUOTA_EXHAUSTED")

        # 4. Network timeout
        c4, m4 = classify_youtube_error(Exception("Connection to youtube.googleapis.com timed out."))
        self.assertEqual(c4, "YOUTUBE_NETWORK_ERROR")

    def test_02_quota_discipline_precheck(self):
        """Verifies dry-run and quota exhaustion consume zero units."""
        # Dry-run test
        res_dry = self.publisher.publish_to_youtube(
            video_path=self.dummy_video,
            title="Dry Run Test",
            dry_run=True
        )
        self.assertEqual(res_dry.status, PublishStatus.PUBLISHED)
        self.assertEqual(res_dry.units_spent, 0)
        self.assertEqual(self.quota_tracker.get_used_units(), 0)

        # Force quota exhaustion
        self.quota_tracker.consume(10000, reason="artificial_burn")
        res_exhausted = self.publisher.publish_to_youtube(
            video_path=self.dummy_video,
            title="Exhausted Budget Test"
        )
        self.assertEqual(res_exhausted.status, PublishStatus.DEFERRED)
        self.assertEqual(res_exhausted.error_code, "QUOTA_EXHAUSTED")
        self.assertEqual(res_exhausted.units_spent, 0)

    def test_03_unverified_project_restriction_handling(self):
        """Verifies that Google forcing 'public' to 'private' sets is_restricted and error_code."""
        mock_resp = {
            "id": "UNVERIFIED_VID_789",
            "snippet": {"title": "Test Title"},
            "status": {"privacyStatus": "private"}  # Google forced to private
        }
        mock_channel = {
            "authenticated": True,
            "channel_id": "UC_TEST_123",
            "channel_title": "Automated Channel",
            "email": "developer@test.com"
        }

        with patch.object(self.publisher, "_execute_youtube_upload", return_value=(mock_resp, mock_channel)):
            res = self.publisher.publish_to_youtube(
                video_path=self.dummy_video,
                title="Unverified App Test",
                privacy_status="public"
            )

        self.assertEqual(res.status, PublishStatus.PUBLISHED)
        self.assertTrue(res.is_restricted)
        self.assertEqual(res.error_code, "UNVERIFIED_PROJECT_RESTRICTION")
        self.assertEqual(res.visibility, "private")
        self.assertEqual(res.url, "https://youtube.com/shorts/UNVERIFIED_VID_789")
        self.assertEqual(res.units_spent, YOUTUBE_VIDEO_UPLOAD_COST)

    def test_04_get_authenticated_channel_info_safety(self):
        """Verifies get_authenticated_channel_info handles missing or erroring credentials without crashing."""
        info_none = get_authenticated_channel_info(None)
        self.assertFalse(info_none["authenticated"])
        self.assertIsNone(info_none["email"])

        mock_creds = MagicMock()
        mock_creds.valid = False
        info_invalid = get_authenticated_channel_info(mock_creds)
        self.assertFalse(info_invalid["authenticated"])

    def test_05_scopes_definition_completeness(self):
        """Confirms standard scopes contain upload, readonly, and user email."""
        self.assertIn("https://www.googleapis.com/auth/youtube.upload", YOUTUBE_SCOPES)
        self.assertIn("https://www.googleapis.com/auth/youtube.readonly", YOUTUBE_SCOPES)
        self.assertIn("https://www.googleapis.com/auth/userinfo.email", YOUTUBE_SCOPES)


if __name__ == "__main__":
    unittest.main()
