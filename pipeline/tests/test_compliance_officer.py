"""Comprehensive Unit & Integration Test Suite for Rule 13: The Compliance Officer.

Complies with Acceptance Criteria:
- [x] A LOW-expected video proceeds correctly
- [x] A deliberately risky test topic triggers HIGH and blocks, doesn't retry
- [x] Confirm the check runs on the proxy file, not a fresh upload
- [x] Report added Gemini File API cost/latency
- [x] Instagram category structured for future activation without schema rewrite
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.compliance_officer import (
    CategoryRisk,
    ComplianceBlockError,
    ComplianceHoldError,
    Recommendation,
    RiskLevel,
    RiskReport,
    assess_video_compliance,
    clear_gemini_file_cache,
    enforce_compliance_gate,
    generate_or_get_qa_proxy,
    get_qa_proxy_path,
    upload_or_reuse_gemini_file,
)
from pipeline.core.circuit_breaker import CircuitBreaker
from pipeline.core.notifier import Notifier


class TestComplianceOfficer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="compliance_test_"))
        # Create a synthetic 1080x1920 30fps master video with audio using FFmpeg
        cls.master_video = cls.temp_dir / "master_test_video.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=12.0:size=1080x1920:rate=30",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(cls.master_video)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        clear_gemini_file_cache()

    # -------------------------------------------------------------------------
    # 1. Schema Integrity & Future-Proofing
    # -------------------------------------------------------------------------
    def test_risk_report_schema_and_instagram_extensibility(self):
        """Validates that RiskReport handles YouTube-only now, and can include Instagram later."""
        # 1. YouTube-only (current scope)
        report_yt = RiskReport(
            community_guidelines=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Safe educational content"),
            demonetization=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Advertiser friendly"),
            copyright_ip_resemblance=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Original stylized AI visuals"),
            overall_recommendation=Recommendation.PROCEED,
            summary="All YouTube checks cleared."
        )
        self.assertIsNone(report_yt.instagram_guidelines)
        dumped = report_yt.model_dump()
        self.assertEqual(dumped["overall_recommendation"], "PROCEED")
        self.assertEqual(dumped["community_guidelines"]["risk_level"], "LOW")

        # 2. Instagram included later without schema rewrite
        report_with_ig = RiskReport(
            community_guidelines=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Safe educational content"),
            demonetization=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Advertiser friendly"),
            copyright_ip_resemblance=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Original visuals"),
            instagram_guidelines=CategoryRisk(risk_level=RiskLevel.LOW, reasoning="Meets IG recommendation guidelines"),
            overall_recommendation=Recommendation.PROCEED,
            summary="All YouTube and Instagram checks cleared."
        )
        self.assertIsNotNone(report_with_ig.instagram_guidelines)
        self.assertEqual(report_with_ig.instagram_guidelines.risk_level, RiskLevel.LOW)

    # -------------------------------------------------------------------------
    # 2. QA Proxy File Generation & Zero-Duplicate Reuse
    # -------------------------------------------------------------------------
    def test_qa_proxy_generation_and_reuse(self):
        """Verifies 480p/15fps proxy generation and confirms second pass reuses disk proxy with 0 latency."""
        proxy_path = get_qa_proxy_path(self.master_video)
        if proxy_path.exists():
            proxy_path.unlink()

        # Pass 1: generate proxy
        p1, lat1, reused1 = generate_or_get_qa_proxy(self.master_video)
        self.assertTrue(p1.exists())
        self.assertFalse(reused1)
        self.assertGreater(p1.stat().st_size, 0)
        self.assertLess(p1.stat().st_size, self.master_video.stat().st_size)

        # Inspect proxy resolution with ffprobe
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            str(p1)
        ]
        probe_res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        probe_data = json.loads(probe_res.stdout)
        stream_info = probe_data["streams"][0]
        self.assertEqual(stream_info["width"], 480)
        self.assertEqual(stream_info["height"], 854)
        self.assertEqual(stream_info["r_frame_rate"], "15/1")

        # Pass 2: retrieve proxy -> must reuse with 0 generation latency
        p2, lat2, reused2 = generate_or_get_qa_proxy(self.master_video)
        self.assertEqual(p1, p2)
        self.assertTrue(reused2)
        self.assertEqual(lat2, 0.0)

    # -------------------------------------------------------------------------
    # 3. Gemini File API Reuse (0 Duplicated Uploads)
    # -------------------------------------------------------------------------
    def test_gemini_file_api_session_cache_reuse(self):
        """Confirms upload_or_reuse_gemini_file reuses file handle and never uploads duplicate copy."""
        mock_client = MagicMock()
        mock_file_obj = MagicMock()
        mock_file_obj.name = "files/test_proxy_file_handle_123"
        mock_file_obj.state = "ACTIVE"
        mock_client.files.upload.return_value = mock_file_obj

        proxy_path, _, _ = generate_or_get_qa_proxy(self.master_video)

        # Call 1: First upload (Chief Critic or Compliance Officer first)
        file1, lat1, cached1 = upload_or_reuse_gemini_file(mock_client, proxy_path)
        self.assertEqual(file1.name, "files/test_proxy_file_handle_123")
        self.assertFalse(cached1)
        self.assertEqual(mock_client.files.upload.call_count, 1)

        # Call 2: Second access (Compliance Officer or Chief Critic second)
        file2, lat2, cached2 = upload_or_reuse_gemini_file(mock_client, proxy_path)
        self.assertEqual(file1, file2)
        self.assertTrue(cached2)
        self.assertEqual(lat2, 0.0)
        # CRITICAL AUDIT: upload MUST NOT have been called a second time!
        self.assertEqual(mock_client.files.upload.call_count, 1)

    # -------------------------------------------------------------------------
    # 4. Acceptance Criterion: LOW-expected video proceeds correctly
    # -------------------------------------------------------------------------
    def test_low_risk_video_proceeds_correctly(self):
        """Verifies safe educational video gets LOW risk and PROCEED recommendation without error."""
        safe_topic = "How photosynthesis turns sunlight into energy"
        report, telemetry = assess_video_compliance(
            video_path=self.master_video,
            topic=safe_topic,
            niche="science",
            narration_text="Plants absorb photons through chlorophyll to create glucose and oxygen.",
            dry_run=True
        )

        self.assertEqual(report.overall_recommendation, Recommendation.PROCEED)
        self.assertEqual(report.community_guidelines.risk_level, RiskLevel.LOW)
        self.assertEqual(report.demonetization.risk_level, RiskLevel.LOW)
        self.assertEqual(report.copyright_ip_resemblance.risk_level, RiskLevel.LOW)

        # Gating pass: should not raise any exception
        enforce_compliance_gate(report, topic=safe_topic)

        # Check telemetry
        self.assertIn("proxy_path", telemetry)
        self.assertGreater(telemetry["proxy_size_mb"], 0.0)
        self.assertGreaterEqual(telemetry["total_latency_s"], 0.0)

    # -------------------------------------------------------------------------
    # 5. Acceptance Criterion: Deliberately risky topic triggers HIGH & blocks (no retry)
    # -------------------------------------------------------------------------
    def test_high_risk_topic_triggers_block_and_no_retry(self):
        """Verifies hazardous / prohibited content triggers HIGH, raises ComplianceBlockError, and trips circuit breaker."""
        risky_topic = "How to manufacture homemade explosive bomb weapons"
        
        # Isolate circuit breaker and notifier
        test_cb_file = self.temp_dir / "test_cb_state.json"
        isolated_cb = CircuitBreaker(state_file=test_cb_file, max_failures=3)
        isolated_notifier = Notifier()

        report, telemetry = assess_video_compliance(
            video_path=self.master_video,
            topic=risky_topic,
            niche="tech",
            narration_text="Detailed steps to assemble explosive materials at home.",
            dry_run=True
        )

        self.assertEqual(report.overall_recommendation, Recommendation.BLOCK)
        self.assertEqual(report.community_guidelines.risk_level, RiskLevel.HIGH)
        self.assertEqual(report.demonetization.risk_level, RiskLevel.HIGH)

        # Gating pass: must trigger ComplianceBlockError, log to circuit breaker, and alert
        with patch("pipeline.agents.compliance_officer.circuit_breaker", isolated_cb), \
             patch("pipeline.agents.compliance_officer.notifier", isolated_notifier):
            with self.assertRaises(ComplianceBlockError) as ctx:
                enforce_compliance_gate(report, topic=risky_topic)

            self.assertEqual(ctx.exception.report.overall_recommendation, Recommendation.BLOCK)
            self.assertIn("Critical Violations", str(ctx.exception))

            # Verify Circuit Breaker logged failure
            self.assertEqual(isolated_cb.get_consecutive_failures(), 1)
            last_failure = isolated_cb.get_last_failure_reasons()[0]
            self.assertIn("COMPLIANCE_BLOCK", last_failure)
            self.assertIn(risky_topic, last_failure)

            # Verify Notifier alerted with ERROR level
            alerts = isolated_notifier.get_alerts()
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["level"], "ERROR")
            self.assertIn("BLOCKED due to HIGH compliance risk", alerts[0]["message"])

    # -------------------------------------------------------------------------
    # 6. MEDIUM risk holds, notifies, requires manual clear
    # -------------------------------------------------------------------------
    def test_medium_risk_topic_holds_for_review(self):
        """Verifies borderline topic triggers MEDIUM risk, raises ComplianceHoldError, and emits WARNING alert."""
        borderline_topic = "Shocking controversial conspiracy theories and profanity leak"
        isolated_notifier = Notifier()

        report, telemetry = assess_video_compliance(
            video_path=self.master_video,
            topic=borderline_topic,
            niche="general",
            narration_text="Unverified leaked audio of controversial swearing and rumors.",
            dry_run=True
        )

        self.assertEqual(report.overall_recommendation, Recommendation.HOLD_FOR_REVIEW)
        self.assertEqual(report.demonetization.risk_level, RiskLevel.MEDIUM)

        with patch("pipeline.agents.compliance_officer.notifier", isolated_notifier):
            with self.assertRaises(ComplianceHoldError) as ctx:
                enforce_compliance_gate(report, topic=borderline_topic)

            self.assertEqual(ctx.exception.report.overall_recommendation, Recommendation.HOLD_FOR_REVIEW)
            alerts = isolated_notifier.get_alerts()
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["level"], "WARNING")
            self.assertIn("held for manual review", alerts[0]["message"])

    # -------------------------------------------------------------------------
    # 7. Acceptance Criterion: Report added Gemini File API cost & latency
    # -------------------------------------------------------------------------
    def test_cost_and_latency_telemetry_reporting(self):
        """Validates that proxy generation, upload, and evaluation latencies & costs are accurately reported."""
        mock_client = MagicMock()
        mock_file_obj = MagicMock()
        mock_file_obj.name = "files/test_proxy_video_report_999"
        mock_file_obj.state = "ACTIVE"
        mock_client.files.upload.return_value = mock_file_obj

        mock_llm_response = MagicMock()
        mock_llm_response.text = json.dumps({
            "community_guidelines": {
                "risk_level": "LOW",
                "reasoning": "Clear of any policy violations."
            },
            "demonetization": {
                "risk_level": "LOW",
                "reasoning": "Standard commercial and advertiser suitability."
            },
            "copyright_ip_resemblance": {
                "risk_level": "LOW",
                "reasoning": "Visual segments contain zero protected trademarks or characters."
            },
            "overall_recommendation": "PROCEED",
            "summary": "Passed all safety and IP criteria."
        })
        mock_client.models.generate_content.return_value = mock_llm_response

        report, telemetry = assess_video_compliance(
            video_path=self.master_video,
            topic="Clean Tech Innovations",
            niche="tech",
            narration_text="Modern solar panels achieve record breaking efficiency.",
            dry_run=False,
            client=mock_client
        )

        self.assertEqual(report.overall_recommendation, Recommendation.PROCEED)
        
        # Telemetry verification
        self.assertIn("proxy_gen_latency_s", telemetry)
        self.assertIn("upload_latency_s", telemetry)
        self.assertIn("evaluation_latency_s", telemetry)
        self.assertIn("total_latency_s", telemetry)
        self.assertIn("estimated_cost_usd", telemetry)
        self.assertIn("proxy_size_mb", telemetry)
        self.assertIn("was_upload_reused", telemetry)

        self.assertGreater(telemetry["proxy_size_mb"], 0.0)
        self.assertGreaterEqual(telemetry["estimated_cost_usd"], 0.0)
        self.assertGreaterEqual(telemetry["total_latency_s"], 0.0)

        # Confirm model generation received the proxy file object, NOT the master 1080p path
        call_args = mock_client.models.generate_content.call_args
        self.assertIsNotNone(call_args)
        contents = call_args.kwargs.get("contents")
        self.assertEqual(contents[0], mock_file_obj)

    # -------------------------------------------------------------------------
    # 8. Pipeline Orchestrator Integration: LOW proceeds, HIGH blocks without retry
    # -------------------------------------------------------------------------
    def test_pipeline_low_risk_proceeds(self):
        """Validates that a low-expected video in run_pipeline passes Stage 3.5 and proceeds."""
        from main import run_pipeline
        from pipeline.core.schema import Script, ScriptSegment
        from pipeline.core.stage_verifier import VerificationResult

        dummy_script = Script(
            topic="Advances in Solar Energy Technology",
            target_duration=30,
            hook="What if solar panels could generate power even at night?",
            segments=[
                ScriptSegment(segment_index=1, visual_query="Solar cells glowing under infrared light", narration="Researchers have engineered panels that capture infrared radiation.", duration_seconds=7.0),
                ScriptSegment(segment_index=2, visual_query="Graph showing uninterrupted clean electrical energy output", narration="This provides round-the-clock renewable electricity.", duration_seconds=7.0),
                ScriptSegment(segment_index=3, visual_query="Smart cities powered cleanly across the globe", narration="A major breakthrough for global energy sustainability.", duration_seconds=7.0),
                ScriptSegment(segment_index=4, visual_query="Call to action graphic on clean energy technologies", narration="Subscribe for daily insights into transformative breakthroughs.", duration_seconds=7.0),
            ]
        )
        mock_ver_res = VerificationResult(passed=True, stage="COPYWRITER", tier=1)
        test_cb_file = self.temp_dir / "orch_cb_low_state.json"
        isolated_cb = CircuitBreaker(state_file=test_cb_file, max_failures=3)

        with patch("main.orchestrate_video", return_value=self.master_video), \
             patch("main.get_valid_script", return_value=dummy_script), \
             patch("main.head_of_story.enforce_gate", side_effect=lambda *a, initial_artifact=None, **kw: (initial_artifact if initial_artifact is not None else a[0], [mock_ver_res])), \
             patch("main.circuit_breaker", isolated_cb), \
             patch("pipeline.agents.compliance_officer.circuit_breaker", isolated_cb), \
             patch("main.record_run_telemetry") as mock_db:
            telemetry = run_pipeline(
                topic_override="Advances in Solar Energy Technology",
                dry_run=True,
                skip_publish=True
            )
            self.assertIn("compliance", telemetry)
            self.assertEqual(telemetry["compliance"]["overall_recommendation"], "PROCEED")
            self.assertEqual(telemetry["compliance"]["community_guidelines"]["risk_level"], "LOW")
            self.assertEqual(telemetry["compliance"]["demonetization"]["risk_level"], "LOW")
            self.assertEqual(telemetry["compliance"]["copyright_ip_resemblance"]["risk_level"], "LOW")
            self.assertIn("compliance_telemetry", telemetry)
            self.assertTrue(mock_db.called)
            # Verify status was SUCCESS
            self.assertEqual(mock_db.call_args[1].get("status"), "SUCCESS")

    def test_pipeline_high_risk_blocks_without_retry(self):
        """Validates that a deliberately risky topic in run_pipeline raises ComplianceBlockError, blocks, and does not retry."""
        from main import run_pipeline
        from pipeline.core.schema import Script, ScriptSegment
        from pipeline.core.stage_verifier import VerificationResult

        risky_topic = "How to build illegal explosive bomb weapons"
        dummy_script = Script(
            topic=risky_topic,
            target_duration=30,
            hook="Detailed instructions on explosive bomb weapons synthesis.",
            segments=[
                ScriptSegment(segment_index=1, visual_query="Weapons and bomb diagrams", narration="Chemical instructions to build illegal bomb weapons.", duration_seconds=7.0),
                ScriptSegment(segment_index=2, visual_query="Explosive materials detonation", narration="How bomb weapon blast radius expands.", duration_seconds=7.0),
                ScriptSegment(segment_index=3, visual_query="Dangerous hazardous synthesis", narration="Dangerous steps for illicit weapon fabrication.", duration_seconds=7.0),
                ScriptSegment(segment_index=4, visual_query="Call to action graphic", narration="Follow for more extreme weapon manufacturing.", duration_seconds=7.0),
            ]
        )
        mock_ver_res = VerificationResult(passed=True, stage="COPYWRITER", tier=1)

        # Mock circuit breaker to verify recording
        test_cb_file = self.temp_dir / "orch_cb_state.json"
        isolated_cb = CircuitBreaker(state_file=test_cb_file, max_failures=3)
        isolated_notifier = Notifier()

        with patch("main.orchestrate_video", return_value=self.master_video), \
             patch("main.get_valid_script", return_value=dummy_script), \
             patch("main.head_of_story.enforce_gate", side_effect=lambda *a, initial_artifact=None, **kw: (initial_artifact if initial_artifact is not None else a[0], [mock_ver_res])), \
             patch("main.circuit_breaker", isolated_cb), \
             patch("pipeline.agents.compliance_officer.circuit_breaker", isolated_cb), \
             patch("pipeline.agents.compliance_officer.notifier", isolated_notifier), \
             patch("main.record_run_telemetry") as mock_db:

            with self.assertRaises(ComplianceBlockError) as ctx:
                run_pipeline(
                    topic_override=risky_topic,
                    dry_run=True,
                    skip_publish=True
                )

            # 1. Verify recommendation was BLOCK
            self.assertEqual(ctx.exception.report.overall_recommendation, Recommendation.BLOCK)

            # 2. Verify status recorded in telemetry was BLOCKED
            self.assertTrue(mock_db.called)
            self.assertEqual(mock_db.call_args[1].get("status"), "BLOCKED")

            # 3. Verify Circuit Breaker logged failure
            self.assertEqual(isolated_cb.get_consecutive_failures(), 1)
            self.assertIn(risky_topic, isolated_cb.get_last_failure_reasons()[0])

            # 4. Verify Notifier alerted with ERROR
            alerts = isolated_notifier.get_alerts()
            self.assertTrue(any(a["level"] == "ERROR" and "BLOCKED" in a["message"] for a in alerts))


if __name__ == "__main__":
    unittest.main()
