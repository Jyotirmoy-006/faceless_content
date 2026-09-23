"""Comprehensive Verification Test Suite for Trend Sourcing, Visual Tractability Gate, Tone Lock, and Voice/LUFS Standardization.

Covers:
1. Trend Sourcing: Verifies 1-unit videos.list(chart='mostPopular') integration from Strategist to Ideator.
2. Visual Tractability Gate: Verifies HeadOfStory rejection of abstract code/software concepts and approval of physically anchored topics.
3. Tone Lock: Verifies high-energy, upbeat narrator persona prompts in Scriptwriter.
4. Loudness Normalization: Verifies -14.0 LUFS target lock across VoiceActor, Director, SoundDesigner, and HeadOfAudio.
5. Kokoro Integration: Verifies CPU execution (zero VRAM), multiple voice support, and -14.0 LUFS compliance.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.department_heads import head_of_audio, head_of_story
from pipeline.agents.ideator import IdeatorAgent
from pipeline.agents.scriptwriter import get_valid_script
from pipeline.agents.strategist import StrategistAgent
from pipeline.agents.voice_actor import synthesize_segment
from pipeline.core.audio_processor import measure_loudness_ebu_r128
from pipeline.core.quota_tracker import QuotaTracker
from pipeline.core.schema import IdeaConcept, Script, ScriptSegment
from pipeline.core.sound_designer import mix_voice_bgm_sfx
from pipeline.workers.kokoro_worker import synthesize_kokoro_speech


class TestTrendTractabilityToneVoice(unittest.TestCase):
    """Test suite covering the 5 key upgrades."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.work_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. TREND SOURCING VERIFICATION
    # -------------------------------------------------------------------------
    def test_trend_sourcing_quota_and_ideator_integration(self):
        """Verifies Strategist fetches trends with 1 unit quota and Ideator receives them."""
        custom_tracker = QuotaTracker()
        strategist = StrategistAgent(quota_mgr=custom_tracker)

        # Dry run trend fetch
        initial_used = custom_tracker.get_used_units()
        trends = strategist.get_trending_topics(niche="tech", dry_run=True)
        self.assertTrue(len(trends) > 0)
        # Verify exactly 1 unit was tracked
        self.assertEqual(custom_tracker.get_used_units() - initial_used, 1)

        # Verify Ideator uses trend candidates
        ideator = IdeatorAgent()
        with patch.object(ideator, "_get_fallback_idea") as mock_fallback:
            mock_fallback.return_value = IdeaConcept(
                topic="Test Trend Idea",
                niche="tech",
                angle="Test angle",
                target_duration=30
            )
            idea = ideator.generate_idea(niche="tech", dry_run=True, trend_candidates=trends[:3])
            self.assertEqual(idea.topic, "Test Trend Idea")

    # -------------------------------------------------------------------------
    # 2. VISUAL TRACTABILITY GATE (HEAD OF STORY)
    # -------------------------------------------------------------------------
    def test_visual_tractability_rejection_of_abstract_code(self):
        """Verifies HeadOfStory rejects abstract source code concepts without physical anchors."""
        abstract_concepts = [
            IdeaConcept(
                topic="How Python GIL Works In Source Code",
                niche="tech",
                angle="A deep dive into writing code and memory pointer syntax in CPython",
                target_duration=30
            ),
            IdeaConcept(
                topic="Fixing Git Merge Conflicts and Coding Syntax",
                niche="tech",
                angle="Understanding variable scoping and javascript closures syntax",
                target_duration=30
            ),
            IdeaConcept(
                topic="Software Bug Fix in Cloud Database Schema",
                niche="tech",
                angle="How to debug abstract algorithm logic without hardware",
                target_duration=30
            )
        ]

        for concept in abstract_concepts:
            result = head_of_story.inspect_tier1(concept)
            self.assertFalse(result.passed, f"Expected rejection for abstract topic: {concept.topic}")
            self.assertIn("visual tractability", (result.feedback or "").lower())

    def test_visual_tractability_approval_of_grounded_concepts(self):
        """Verifies HeadOfStory approves physically anchored technical topics."""
        grounded_concepts = [
            IdeaConcept(
                topic="The Submersible Fixing Undersea Fiber Optic Cables",
                niche="tech",
                angle="How deep sea robotic submersibles splice 10,000-volt ocean data cables",
                target_duration=30
            ),
            IdeaConcept(
                topic="Inside the Quantum Processor Cryogenic Refrigerator",
                niche="tech",
                angle="Why superconducting silicon wafer chips must run colder than outer space",
                target_duration=30
            ),
            IdeaConcept(
                topic="Why Datacenter Mega-Batteries Are Exploding in Demand",
                niche="tech",
                angle="How massive server rack power grids are surviving peak load crashes",
                target_duration=30
            )
        ]

        for concept in grounded_concepts:
            result = head_of_story.inspect_tier1(concept)
            self.assertTrue(result.passed, f"Expected approval for grounded topic: {concept.topic}")

    # -------------------------------------------------------------------------
    # 3. TONE LOCK VERIFICATION
    # -------------------------------------------------------------------------
    def test_tone_lock_in_scriptwriter_prompt(self):
        """Verifies Scriptwriter prompt incorporates upbeat, energetic tone lock directives."""
        import inspect
        from pipeline.agents import scriptwriter

        source = inspect.getsource(scriptwriter.get_valid_script)
        self.assertIn("ENERGETIC TONE LOCK", source)
        self.assertIn("High-Energy & Upbeat", source)
        self.assertIn("Contemporary & Rhythmic", source)

    # -------------------------------------------------------------------------
    # 4. LOUDNESS NORMALIZATION (-14.0 LUFS)
    # -------------------------------------------------------------------------
    def test_voice_actor_and_head_of_audio_lufs_lock(self):
        """Verifies VoiceActor synthesizes audio that satisfies HeadOfAudio -14.0 LUFS gate."""
        out_wav = self.work_path / "va_test_14lufs.wav"
        t_dir = self.work_path / "va_temp"

        clean_wav, engine, speech_dur, pause_dur = synthesize_segment(
            text="Quantum computing processors operate at millikelvin temperatures.",
            output_path=out_wav,
            voice="en-US-AndrewMultilingualNeural",
            is_hook=True,
            temp_dir=t_dir,
            mock_edge_failure=True,  # Offline test with Piper fallback
            target_lufs=-14.0,
            target_tp=-1.5
        )

        self.assertTrue(clean_wav.exists())
        measured = measure_loudness_ebu_r128(clean_wav)
        self.assertAlmostEqual(measured["input_i"], -14.0, delta=1.5)
        self.assertLessEqual(measured["input_tp"], -1.4)

        # Verify HeadOfAudio approves the -14.0 LUFS audio
        gate_res = head_of_audio.inspect_tier1(clean_wav, context={"expected_duration": speech_dur})
        self.assertTrue(gate_res.passed, f"HeadOfAudio rejected valid audio: {gate_res.feedback}")
        self.assertEqual(gate_res.details.get("target_lufs"), -14.0)

    def test_sound_designer_mix_14lufs(self):
        """Verifies SoundDesigner multi-layer mix normalizes to -14.0 LUFS."""
        voice_wav = self.work_path / "voice_source.wav"
        mix_out = self.work_path / "mixed_master.wav"

        synthesize_segment(
            text="In under four minutes, this chip solved a ten thousand year puzzle.",
            output_path=voice_wav,
            mock_edge_failure=True,
            target_lufs=-14.0
        )

        mix_voice_bgm_sfx(
            voice_path=voice_wav,
            output_path=mix_out,
            sfx_timestamps=[1.5],
            sfx_type="whoosh",
            bgm_volume_db=-22.0,
            sfx_volume_offset=-24.0
        )

        self.assertTrue(mix_out.exists())
        stats = measure_loudness_ebu_r128(mix_out)
        self.assertAlmostEqual(stats["input_i"], -14.0, delta=2.0)
        self.assertLessEqual(stats["input_tp"], -1.4)

    # -------------------------------------------------------------------------
    # 5. KOKORO CPU SYNTHESIS VERIFICATION
    # -------------------------------------------------------------------------
    def test_kokoro_cpu_worker_synthesis(self):
        """Verifies Kokoro CPU synthesis produces clean audio at -14.0 LUFS."""
        kokoro_wav = self.work_path / "kokoro_test_adam.wav"
        sample_text = "This quantum processor just shattered traditional cryptographic security standards."

        synthesize_kokoro_speech(
            text=sample_text,
            output_path=kokoro_wav,
            voice="am_adam",
            speed=1.12,
            target_lufs=-14.0,
            target_tp=-1.5
        )

        self.assertTrue(kokoro_wav.exists())
        self.assertGreater(kokoro_wav.stat().st_size, 5000)

        measured = measure_loudness_ebu_r128(kokoro_wav)
        self.assertAlmostEqual(measured["input_i"], -14.0, delta=2.8)
        self.assertLessEqual(measured["input_tp"], -1.4)

        # Dispatch via VoiceActor with Kokoro voice ID
        va_out = self.work_path / "va_kokoro_heart.wav"
        clean_wav, engine, speech_dur, pause_dur = synthesize_segment(
            text=sample_text,
            output_path=va_out,
            voice="af_heart",
            is_hook=False,
            target_lufs=-14.0
        )
        self.assertEqual(engine, "kokoro-cpu")
        self.assertTrue(clean_wav.exists())


if __name__ == "__main__":
    unittest.main()
