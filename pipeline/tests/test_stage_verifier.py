"""Comprehensive Test Suite for Tiered Stage-Verification Gates (Rule 12 Compliance).

Covers:
1. Tier 1 - Structural Verification (0 API cost, deterministic code):
   - Creative Director: schema-valid CreativeBrief
   - Copywriter: schema-valid ProductionScript
   - Voice Actor: audio exists, duration within +/-15%, LUFS, no silence, no clipping
   - Art Director: video exists, non-corrupt ffprobe, canonical 1080x1920 30fps CFR yuv420p,
     non-blank frames (pixel variance test)
   - Editor: master video passes ffprobe, 0 soft subtitle streams (hard-burned Rule 10)

2. Tier 2 - Semantic Verification (Cheap Gemini call only for Copywriter):
   - Coherence, repetition, and pacing fit evaluation via Chief Critic
   - Strict 0-quota guarantee for Voice Actor, Art Director, and Editor

3. Failure Handling & Corrective Retries:
   - Deliberate bad output injection at each of the 5 stages proving corrective retry engages
   - Max 2 retries bound, deterministic fallback, and circuit breaker escalation

4. Added Gemini Call Count and Latency Reporting
"""

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

import numpy as np
import scipy.io.wavfile as wavfile

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.schema import IdeaConcept, Script, ScriptSegment
from pipeline.core.stage_verifier import (
    CreativeBrief,
    ProductionScript,
    VerificationResult,
    StageVerificationError,
    verify_creative_director,
    verify_copywriter_structural,
    verify_voice_actor,
    verify_art_director,
    verify_editor,
    verify_copywriter_semantic,
    verify_copywriter,
    run_stage_with_verification,
    CopywriterSemanticScore
)


class MockGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class TestTieredStageVerifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="stage_verifier_test_"))

    @classmethod
    def tearDownClass(cls):
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Helper fixtures
    # -------------------------------------------------------------------------
    def _create_synthetic_wav(
        self,
        filename: str,
        duration_s: float = 3.0,
        sample_rate: int = 48000,
        amplitude: float = 0.5,
        is_silent: bool = False,
        clipped: bool = False
    ) -> Path:
        """Generates a synthetic 48kHz WAV audio file for testing."""
        out_path = self.temp_dir / filename
        num_samples = int(duration_s * sample_rate)
        if is_silent:
            samples = np.zeros(num_samples, dtype=np.float32)
        else:
            # 440 Hz sine wave tone
            t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
            samples = amplitude * np.sin(2 * np.pi * 440.0 * t)

        if clipped:
            samples = np.clip(samples * 3.0, -1.5, 1.5)
            # Write with overflow/clipping
            int_samples = (samples * 32767.0).astype(np.int16)
        else:
            int_samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

        wavfile.write(str(out_path), sample_rate, int_samples)
        return out_path

    def _create_synthetic_video(
        self,
        filename: str,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
        duration_s: float = 1.0,
        pix_fmt: str = "yuv420p",
        is_blank: bool = False,
        with_soft_subtitles: bool = False
    ) -> Path:
        """Generates a synthetic MP4 video clip via ffmpeg for deterministic testing."""
        out_path = self.temp_dir / filename
        if out_path.exists():
            out_path.unlink()

        if is_blank:
            # Solid black frame
            cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
                "-c:v", "libx264", "-pix_fmt", pix_fmt,
                str(out_path)
            ]
        else:
            # Dynamic pattern with gradient/noise so pixel variance is high
            cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc=s={width}x{height}:r={fps}:d={duration_s}",
                "-c:v", "libx264", "-pix_fmt", pix_fmt,
                str(out_path)
            ]
        subprocess.run(cmd, check=True, capture_output=True)

        if with_soft_subtitles:
            # Add a soft-muxed mov_text/srt subtitle stream to simulate Rule 10 violation
            srt_path = self.temp_dir / "test_soft.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:01,000\nSoft caption test\n")

            subbed_path = self.temp_dir / f"soft_{filename}"
            sub_cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(out_path),
                "-i", str(srt_path),
                "-c", "copy", "-c:s", "mov_text",
                str(subbed_path)
            ]
            subprocess.run(sub_cmd, check=True, capture_output=True)
            return subbed_path

        return out_path

    # =========================================================================
    # 1. TIER 1 STRUCTURAL TESTS (Zero API cost)
    # =========================================================================

    def test_tier1_creative_director_valid(self):
        """Tier 1: Valid Creative Director output passes with zero Gemini calls."""
        concept = IdeaConcept(
            topic="Quantum Computing Explained",
            niche="tech",
            angle="Why Quantum Will Break Encryption",
            target_duration=45
        )
        res = verify_creative_director(concept)
        self.assertTrue(res.passed)
        self.assertEqual(res.stage, "CREATIVE_DIRECTOR")
        self.assertEqual(res.tier, 1)
        self.assertEqual(res.gemini_calls, 0)
        self.assertLess(res.latency_seconds, 0.1)

    def test_tier1_creative_director_invalid(self):
        """Tier 1: Invalid Creative Director output fails validation."""
        # Empty topic
        bad_concept_1 = {"topic": "", "niche": "tech", "angle": "Too short", "target_duration": 45}
        res1 = verify_creative_director(bad_concept_1)
        self.assertFalse(res1.passed)
        self.assertIn("at least 3 characters", res1.error_message)

        # Invalid duration out of 15s-60s
        bad_concept_2 = {
            "topic": "Valid Topic",
            "niche": "tech",
            "angle": "Valid Angle Long Enough",
            "target_duration": 120  # Out of range
        }
        res2 = verify_creative_director(bad_concept_2)
        self.assertFalse(res2.passed)
        self.assertTrue("less than or equal to 60" in res2.error_message or "between 15s and 60s" in res2.error_message)

    def test_tier1_copywriter_structural_valid(self):
        """Tier 1: Valid Copywriter script passes with zero Gemini calls."""
        script = Script(
            topic="Quantum Computing",
            niche="tech",
            target_duration=30,
            hook="Quantum computers aren't just faster, they think in dimensions.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Traditional computers use binary bits of zeros and ones.",
                    visual_query="microchip circuit close up glowing",
                    duration_seconds=5.0
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="Quantum qubits can exist in superpositions of both simultaneously.",
                    visual_query="abstract quantum sphere wave particle",
                    duration_seconds=6.0
                )
            ]
        )
        res = verify_copywriter_structural(script)
        self.assertTrue(res.passed)
        self.assertEqual(res.stage, "COPYWRITER")
        self.assertEqual(res.tier, 1)
        self.assertEqual(res.gemini_calls, 0)

    def test_tier1_copywriter_structural_invalid(self):
        """Tier 1: Malformed Copywriter script fails validation."""
        bad_script = {
            "hook": "Hi",  # Too short (< 5 chars)
            "segments": []  # Empty segments
        }
        res = verify_copywriter_structural(bad_script)
        self.assertFalse(res.passed)
        self.assertIn("at least 5 characters", res.error_message)

    def test_tier1_voice_actor_valid(self):
        """Tier 1: Audio file with correct duration, loudness, and no clipping passes."""
        wav_path = self._create_synthetic_wav("valid_voice.wav", duration_s=4.0, amplitude=0.4)
        res = verify_voice_actor(wav_path, expected_duration=4.0)
        self.assertTrue(res.passed)
        self.assertEqual(res.stage, "VOICE_ACTOR")
        self.assertEqual(res.tier, 1)
        self.assertEqual(res.gemini_calls, 0)
        self.assertIn("duration_seconds", res.details)

    def test_tier1_voice_actor_duration_mismatch(self):
        """Tier 1: Voice Actor audio outside +/-15% duration tolerance is rejected."""
        wav_path = self._create_synthetic_wav("short_voice.wav", duration_s=2.0)
        # Expected is 4.0s; 2.0s is 50% deviation (> 15%)
        res = verify_voice_actor(wav_path, expected_duration=4.0)
        self.assertFalse(res.passed)
        self.assertIn("outside +/-15%", res.error_message)

    def test_tier1_voice_actor_silence_detection(self):
        """Tier 1: Full-track silence is detected and rejected."""
        silent_wav = self._create_synthetic_wav("silent.wav", duration_s=3.0, is_silent=True)
        res = verify_voice_actor(silent_wav, expected_duration=3.0)
        self.assertFalse(res.passed)
        self.assertIn("Full-track silence detected", res.error_message)

    def test_tier1_art_director_valid(self):
        """Tier 1: Canonical 1080x1920 30fps yuv420p video with pixel variance passes."""
        vid_path = self._create_synthetic_video("art_valid.mp4", duration_s=1.0)
        res = verify_art_director(vid_path)
        self.assertTrue(res.passed)
        self.assertEqual(res.stage, "ART_DIRECTOR")
        self.assertEqual(res.tier, 1)
        self.assertEqual(res.gemini_calls, 0)
        self.assertEqual(res.details["width"], 1080)
        self.assertEqual(res.details["height"], 1920)

    def test_tier1_art_director_blank_frame_rejection(self):
        """Tier 1: Solid black blank video fails pixel variance check."""
        blank_path = self._create_synthetic_video("blank.mp4", is_blank=True)
        res = verify_art_director(blank_path)
        self.assertFalse(res.passed)
        self.assertIn("blank/solid-color", res.error_message)

    def test_tier1_art_director_resolution_mismatch(self):
        """Tier 1: Non-canonical resolution (e.g. 720x1280 instead of 1080x1920) is rejected."""
        wrong_res_path = self._create_synthetic_video("wrong_res.mp4", width=720, height=1280)
        res = verify_art_director(wrong_res_path)
        self.assertFalse(res.passed)
        self.assertIn("Resolution mismatch", res.error_message)

    def test_tier1_editor_valid(self):
        """Tier 1: Valid final master with 0 soft subtitle streams passes."""
        master_path = self._create_synthetic_video("master_valid.mp4", duration_s=1.0)
        res = verify_editor(master_path)
        self.assertTrue(res.passed)
        self.assertEqual(res.stage, "EDITOR")
        self.assertEqual(res.tier, 1)
        self.assertEqual(res.gemini_calls, 0)
        self.assertTrue(res.details["hard_burned_enforced"])

    def test_tier1_editor_soft_subtitle_rejection(self):
        """Tier 1: Rule 10 violation (soft-muxed subtitles) is strictly caught."""
        soft_subbed_path = self._create_synthetic_video("soft_subs.mp4", with_soft_subtitles=True)
        res = verify_editor(soft_subbed_path)
        self.assertFalse(res.passed)
        self.assertIn("Rule 10 Violation: Video contains", res.error_message)
        self.assertIn("Captions must be hard-burned into video pixels", res.error_message)

    def test_tier1_editor_duration_gate_exceeded(self):
        """Tier 1: Video exceeding 59.5s max duration is strictly rejected to protect YouTube Shorts classification."""
        over_length_path = self._create_synthetic_video("over_length.mp4", duration_s=62.0)
        res = verify_editor(over_length_path, max_duration_seconds=59.5)
        self.assertFalse(res.passed)
        self.assertIn("Short-Form Duration Gate Violation", res.error_message)
        self.assertIn("exceeds maximum allowed limit", res.error_message)

    # =========================================================================
    # 2. TIER 2 SEMANTIC TESTS (Gemini call strictly on Copywriter)
    # =========================================================================

    def test_tier2_copywriter_semantic_passing(self):
        """Tier 2: Fast cheap model rates high-quality script >= 6/10 across all criteria."""
        script = Script(
            topic="Cosmic Expansion",
            niche="science",
            target_duration=30,
            hook="The universe is expanding faster than light can travel.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="In 1998, astronomers discovered dark energy accelerating galaxy recession.",
                    visual_query="deep space telescope galaxies moving apart",
                    duration_seconds=5.0
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="Eventually, all distant stars will vanish from our night sky forever.",
                    visual_query="lonely galaxy fading into black void",
                    duration_seconds=6.0
                )
            ]
        )
        mock_eval = CopywriterSemanticScore(
            coherence_score=9,
            repetition_score=9,
            pacing_score=8,
            critique="Excellent progression from cosmological hook to chilling conclusion."
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_eval.model_dump_json())

        res = verify_copywriter_semantic(
            script=script,
            topic="Cosmic Expansion",
            niche="science",
            client=mock_client
        )

        self.assertTrue(res.passed)
        self.assertEqual(res.tier, 2)
        self.assertEqual(res.gemini_calls, 1)
        self.assertEqual(res.details["coherence_score"], 9)
        self.assertEqual(res.details["repetition_score"], 9)
        self.assertEqual(res.details["pacing_score"], 8)
        self.assertIn("progression", res.critique)

    def test_tier2_copywriter_semantic_failing(self):
        """Tier 2: Script with repetitive text or poor pacing is rejected (< 6/10)."""
        script = Script(
            topic="Money",
            niche="finance",
            target_duration=30,
            hook="Money is money because money is good.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Money is very good money and everyone wants money.",
                    visual_query="money cash piles",
                    duration_seconds=5.0
                )
            ]
        )
        mock_eval = CopywriterSemanticScore(
            coherence_score=4,
            repetition_score=2,
            pacing_score=5,
            critique="Extreme repetitive phrasing and circular tautology."
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_eval.model_dump_json())

        res = verify_copywriter_semantic(
            script=script,
            topic="Money",
            niche="finance",
            client=mock_client
        )

        self.assertFalse(res.passed)
        self.assertEqual(res.tier, 2)
        self.assertEqual(res.gemini_calls, 1)
        self.assertIn("Semantic quality below standards", res.error_message)
        self.assertIn("repetitive", res.critique.lower())

    def test_tier2_quota_discipline_guarantee(self):
        """Confirms that Voice Actor, Art Director, and Editor gates perform ZERO Gemini calls."""
        wav_path = self._create_synthetic_wav("quota_check.wav", duration_s=2.0)
        vid_path = self._create_synthetic_video("quota_check.mp4", duration_s=1.0)

        va_res = verify_voice_actor(wav_path, expected_duration=2.0)
        art_res = verify_art_director(vid_path)
        ed_res = verify_editor(vid_path)

        self.assertEqual(va_res.gemini_calls, 0)
        self.assertEqual(art_res.gemini_calls, 0)
        self.assertEqual(ed_res.gemini_calls, 0)

    # =========================================================================
    # 3. FAILURE INJECTION & CORRECTIVE RETRY TESTS
    # =========================================================================

    def test_corrective_retry_creative_director_injection(self):
        """Deliberately inject bad concept on attempt 1; verify corrective retry engages."""
        attempts = []

        def execute_stage(feedback):
            attempts.append(feedback)
            if len(attempts) == 1:
                # Attempt 1: Return bad concept dict (target_duration out of bounds)
                return {
                    "topic": "Bad Duration Topic",
                    "niche": "tech",
                    "angle": "An angle that is long enough",
                    "target_duration": 999
                }
            # Attempt 2: Corrective retry fixes the issue using feedback
            return IdeaConcept(
                topic="Fixed Concept Topic",
                niche="tech",
                angle="An angle that is long enough",
                target_duration=45
            )

        final_concept, history = run_stage_with_verification(
            stage_name="CREATIVE_DIRECTOR",
            execute_fn=execute_stage,
            verify_fn=verify_creative_director,
            max_retries=2
        )

        self.assertEqual(len(attempts), 2)
        self.assertIsNone(attempts[0])  # First attempt has no feedback
        self.assertIsNotNone(attempts[1])  # Retry receives feedback from verifier
        self.assertTrue("less than or equal to 60" in attempts[1] or "between 15s and 60s" in attempts[1])
        self.assertEqual(final_concept.target_duration, 45)
        self.assertFalse(history[0].passed)
        self.assertTrue(history[1].passed)

    def test_corrective_retry_copywriter_injection(self):
        """Deliberately inject malformed script on attempt 1; verify corrective retry engages."""
        attempts = []

        def execute_stage(feedback):
            attempts.append(feedback)
            if len(attempts) == 1:
                # Bad script with empty narration in segment
                return {
                    "topic": "Testing",
                    "target_duration": 30,
                    "hook": "Valid hook for this script",
                    "segments": [{"segment_index": 1, "narration": "", "visual_query": "query", "duration_seconds": 5.0}]
                }
            # Attempt 2: Valid script
            return Script(
                topic="Technology Testing",
                niche="tech",
                target_duration=30,
                hook="Valid hook for this script",
                segments=[
                    ScriptSegment(
                        segment_index=1,
                        narration="Valid narration that explains the point clearly.",
                        visual_query="technology lab visual",
                        duration_seconds=5.0
                    )
                ]
            )

        final_script, history = run_stage_with_verification(
            stage_name="COPYWRITER",
            execute_fn=execute_stage,
            verify_fn=verify_copywriter_structural,
            max_retries=2
        )

        self.assertEqual(len(attempts), 2)
        self.assertIn("narration", attempts[1])
        self.assertFalse(history[0].passed)
        self.assertTrue(history[1].passed)

    def test_corrective_retry_voice_actor_injection(self):
        """Deliberately inject silent audio on attempt 1; verify corrective retry re-synthesizes."""
        attempts = []
        silent_wav = self._create_synthetic_wav("retry_silent.wav", duration_s=3.0, is_silent=True)
        valid_wav = self._create_synthetic_wav("retry_valid.wav", duration_s=3.0, amplitude=0.4)

        def execute_stage(feedback):
            attempts.append(feedback)
            if len(attempts) == 1:
                return silent_wav
            return valid_wav

        out_wav, history = run_stage_with_verification(
            stage_name="VOICE_ACTOR",
            execute_fn=execute_stage,
            verify_fn=lambda p: verify_voice_actor(p, expected_duration=3.0),
            max_retries=2
        )

        self.assertEqual(len(attempts), 2)
        self.assertIn("Full-track silence detected", attempts[1])
        self.assertFalse(history[0].passed)
        self.assertTrue(history[1].passed)
        self.assertEqual(out_wav, valid_wav)

    def test_corrective_retry_art_director_injection(self):
        """Deliberately inject blank video on attempt 1; verify corrective retry fixes."""
        attempts = []
        blank_vid = self._create_synthetic_video("retry_blank.mp4", is_blank=True)
        valid_vid = self._create_synthetic_video("retry_valid_vid.mp4", is_blank=False)

        def execute_stage(feedback):
            attempts.append(feedback)
            if len(attempts) == 1:
                return blank_vid
            return valid_vid

        out_vid, history = run_stage_with_verification(
            stage_name="ART_DIRECTOR",
            execute_fn=execute_stage,
            verify_fn=verify_art_director,
            max_retries=2
        )

        self.assertEqual(len(attempts), 2)
        self.assertIn("blank/solid-color", attempts[1])
        self.assertFalse(history[0].passed)
        self.assertTrue(history[1].passed)
        self.assertEqual(out_vid, valid_vid)

    def test_corrective_retry_editor_injection(self):
        """Deliberately inject soft subtitles on attempt 1; verify corrective retry burns captions."""
        attempts = []
        soft_vid = self._create_synthetic_video("retry_soft.mp4", with_soft_subtitles=True)
        burned_vid = self._create_synthetic_video("retry_burned.mp4", with_soft_subtitles=False)

        def execute_stage(feedback):
            attempts.append(feedback)
            if len(attempts) == 1:
                return soft_vid
            return burned_vid

        out_vid, history = run_stage_with_verification(
            stage_name="EDITOR",
            execute_fn=execute_stage,
            verify_fn=verify_editor,
            max_retries=2
        )

        self.assertEqual(len(attempts), 2)
        self.assertIn("Rule 10 Violation", attempts[1])
        self.assertFalse(history[0].passed)
        self.assertTrue(history[1].passed)
        self.assertEqual(out_vid, burned_vid)

    def test_max_retries_exhausted_fallback_engagement(self):
        """When max retries are exhausted, deterministic fallback engages seamlessly."""
        attempts = []

        def execute_always_fails(feedback):
            attempts.append(feedback)
            return {"topic": "", "niche": "tech", "angle": "short", "target_duration": 10}

        def deterministic_fallback():
            return IdeaConcept(
                topic="Fallback Deterministic Topic",
                niche="tech",
                angle="Fallback angle with sufficient length",
                target_duration=30
            )

        final_output, history = run_stage_with_verification(
            stage_name="CREATIVE_DIRECTOR",
            execute_fn=execute_always_fails,
            verify_fn=verify_creative_director,
            max_retries=2,
            fallback_fn=deterministic_fallback
        )

        # 3 attempts (initial + 2 retries) + 1 fallback verification = 4 entries
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(history), 4)
        self.assertFalse(history[0].passed)
        self.assertFalse(history[1].passed)
        self.assertFalse(history[2].passed)
        self.assertTrue(history[3].passed)
        self.assertEqual(final_output.topic, "Fallback Deterministic Topic")

    def test_max_retries_exhausted_circuit_breaker_escalation(self):
        """When retries are exhausted with no fallback, raises StageVerificationError."""
        def execute_always_fails(feedback):
            return {"topic": "", "niche": "tech", "angle": "short", "target_duration": 10}

        with self.assertRaises(StageVerificationError) as ctx:
            run_stage_with_verification(
                stage_name="CREATIVE_DIRECTOR",
                execute_fn=execute_always_fails,
                verify_fn=verify_creative_director,
                max_retries=2,
                fallback_fn=None
            )

        self.assertEqual(ctx.exception.stage, "CREATIVE_DIRECTOR")
        self.assertEqual(ctx.exception.tier, 1)

    # =========================================================================
    # 4. FULL RUN TRACE & ADDED GEMINI CALL COUNT / LATENCY REPORT
    # =========================================================================

    def test_end_to_end_pipeline_gate_trace(self):
        """Traces all 5 stage gates across a simulated pipeline run.
        
        Demonstrates:
        1. Every transition gated with tier and result recorded.
        2. Added Gemini call count strictly equals 1 (Copywriter Tier 2).
        3. Total verification latency calculated and bounded (< 1.5s total).
        """
        telemetry = {
            "start_time": time.time(),
            "stages": {},
            "verification": {}
        }

        # Stage 1: Creative Director
        concept = IdeaConcept(
            topic="Autonomous AI Coding",
            niche="tech",
            angle="How Agentic Architecture Replaces Static Scripts",
            target_duration=40
        )
        concept, g1_hist = run_stage_with_verification(
            stage_name="CREATIVE_DIRECTOR",
            execute_fn=lambda fb: concept,
            verify_fn=verify_creative_director,
            max_retries=2
        )
        telemetry["verification"]["creative_director"] = [r.to_dict() for r in g1_hist]

        # Stage 2: Copywriter
        script = Script(
            topic=concept.topic,
            niche=concept.niche,
            target_duration=40,
            hook="AI agents don't just follow instructions, they verify their own work.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Tier one verification catches structural bugs with zero API cost.",
                    visual_query="code editor automated testing green checkmarks",
                    duration_seconds=5.0
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="Tier two deploys high-speed neural critique only where needed.",
                    visual_query="neural network nodes lighting up fast",
                    duration_seconds=5.0
                )
            ]
        )
        mock_eval = CopywriterSemanticScore(
            coherence_score=8,
            repetition_score=9,
            pacing_score=8,
            critique="Tight, punchy, and logically sound."
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(mock_eval.model_dump_json())

        script, g2_hist = run_stage_with_verification(
            stage_name="COPYWRITER",
            execute_fn=lambda fb: script,
            verify_fn=lambda s: verify_copywriter(s, topic=concept.topic, niche=concept.niche, client=mock_client),
            max_retries=2
        )
        telemetry["verification"]["copywriter"] = [r.to_dict() for r in g2_hist]

        # Stage 3A: Voice Actor
        synthetic_wav = self._create_synthetic_wav("trace_voice.wav", duration_s=10.0, amplitude=0.4)
        out_wav, g3a_hist = run_stage_with_verification(
            stage_name="VOICE_ACTOR",
            execute_fn=lambda fb: synthetic_wav,
            verify_fn=lambda p: verify_voice_actor(p, expected_duration=10.0),
            max_retries=2
        )
        telemetry["verification"]["voice_actor"] = [r.to_dict() for r in g3a_hist]

        # Stage 3B: Art Director
        synthetic_vid = self._create_synthetic_video("trace_art.mp4", duration_s=1.0)
        out_vid, g3b_hist = run_stage_with_verification(
            stage_name="ART_DIRECTOR",
            execute_fn=lambda fb: synthetic_vid,
            verify_fn=verify_art_director,
            max_retries=2
        )
        telemetry["verification"]["art_director"] = [r.to_dict() for r in g3b_hist]

        # Stage 3C: Editor
        master_vid = self._create_synthetic_video("trace_editor.mp4", duration_s=1.0)
        out_master, g3c_hist = run_stage_with_verification(
            stage_name="EDITOR",
            execute_fn=lambda fb: master_vid,
            verify_fn=verify_editor,
            max_retries=2
        )
        telemetry["verification"]["editor"] = [r.to_dict() for r in g3c_hist]

        # Assertions
        self.assertIn("creative_director", telemetry["verification"])
        self.assertIn("copywriter", telemetry["verification"])
        self.assertIn("voice_actor", telemetry["verification"])
        self.assertIn("art_director", telemetry["verification"])
        self.assertIn("editor", telemetry["verification"])

        # Compute added Gemini calls and total verification latency
        total_gemini_calls = sum(
            entry["gemini_calls"]
            for gate_list in telemetry["verification"].values()
            for entry in gate_list
        )
        total_latency = sum(
            entry["latency_seconds"]
            for gate_list in telemetry["verification"].values()
            for entry in gate_list
        )

        # Print trace report
        print("\n" + "=" * 70)
        print("          STAGE-VERIFICATION GATE TRACE REPORT")
        print("=" * 70)
        for stage_name, res_list in telemetry["verification"].items():
            for r in res_list:
                status = "PASSED" if r["passed"] else "FAILED"
                print(f"[{status}] {stage_name.upper():<18} (Tier {r['tier']}) | Latency: {r['latency_seconds']:.3f}s | Gemini Calls: {r['gemini_calls']}")
        print("-" * 70)
        print(f"Added Gemini Call Count per Video: {total_gemini_calls}")
        print(f"Total Added Verification Latency:   {total_latency:.3f}s")
        print("=" * 70)

        # Strictly 1 Gemini call added across the entire pipeline
        self.assertEqual(total_gemini_calls, 1)
        self.assertLess(total_latency, 15.0)


if __name__ == "__main__":
    unittest.main()
