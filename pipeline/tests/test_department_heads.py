"""Unit & Integration Tests for Studio Department Head Gatekeepers.

Tests:
1. HeadOfStory: Validates hook brevity, word density, banned opener detection, schema contracts.
2. HeadOfAudio: Validates silence detection, peak clipping, duration alignment, WAV specs.
3. HeadOfArt: Validates 1080x1920 30fps CFR yuv420p, blank frame rejection via pixel variance.
4. HeadOfPost: Validates master container, 0 soft subtitle streams, duration limits, audio stream mapping.
5. HeadOfCompliance: Validates YouTube 1,600-unit quota gating, title bounds, '#shorts' metadata tags.
"""

from __future__ import annotations

import math
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from unittest import TestCase

from pipeline.agents.department_heads import (
    head_of_art,
    head_of_audio,
    head_of_compliance,
    head_of_post,
    head_of_story,
)
from pipeline.core.quota_tracker import quota_tracker
from pipeline.core.schema import IdeaConcept, Script, ScriptSegment


class TestDepartmentHeads(TestCase):
    """Test suite for autonomous Department Head gatekeepers."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. HEAD OF STORY TESTS
    # -------------------------------------------------------------------------
    def test_head_of_story_concept_approval_and_rejection(self) -> None:
        """Verifies HeadOfStory concept approval and structural rejection."""
        # Valid Concept
        valid_concept = IdeaConcept(
            topic="Quantum Computing Secrets",
            niche="tech",
            angle="Why Silicon Valley is secretly terrified of quantum supremacy",
            target_duration=45
        )
        res_valid = head_of_story.inspect(valid_concept)
        self.assertTrue(res_valid.passed)
        self.assertEqual(res_valid.department, "story")

        # Invalid Concept (topic too short passed as dict)
        invalid_concept_dict = {
            "topic": "QC",
            "niche": "tech",
            "angle": "Quantum computing overview",
            "target_duration": 45
        }
        res_invalid = head_of_story.inspect(invalid_concept_dict)
        self.assertFalse(res_invalid.passed)
        self.assertTrue("characters" in (res_invalid.feedback or "") or "short" in (res_invalid.feedback or ""))

    def test_head_of_story_script_hook_and_opener_checks(self) -> None:
        """Verifies HeadOfStory rejects banned openers and overly verbose hooks."""
        # Banned opener: "Did you know..."
        banned_script = Script(
            topic="Cosmic Physics",
            target_duration=30,
            hook="Did you know that black holes can swallow light?",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Nothing escapes an event horizon.",
                    duration_seconds=10.0,
                    visual_query="black hole accretion disk",
                    visual_shots=["black hole accretion disk", "space warping"]
                )
            ]
        )
        res_banned = head_of_story.inspect(banned_script)
        self.assertFalse(res_banned.passed)
        self.assertIn("low-retention filler", res_banned.feedback or "")

        # Verbose hook (> 15 words)
        verbose_script = Script(
            topic="Productivity Secrets",
            target_duration=30,
            hook="This is an extremely long and unnecessarily verbose opening hook that goes on and on without ever getting to the point quickly.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Speed is essential.",
                    duration_seconds=10.0,
                    visual_query="running stopwatch",
                    visual_shots=["running stopwatch", "ticking clock"]
                )
            ]
        )
        res_verbose = head_of_story.inspect(verbose_script)
        self.assertFalse(res_verbose.passed)
        self.assertIn("Hook too verbose", res_verbose.feedback or "")

        # Valid high-retention script with escalating narrative progression
        good_segments = [
            ScriptSegment(
                segment_index=1,
                narration="Right now, Shor's algorithm on a 100,000 qubit system can crack RSA-2048 encryption in hours.",
                duration_seconds=6.0,
                visual_query="superconducting quantum processor glowing",
                visual_shots=["superconducting quantum processor glowing", "silicon chip macro"]
            ),
            ScriptSegment(
                segment_index=2,
                narration="Every financial system, medical record, and government database on earth is suddenly defenseless.",
                duration_seconds=6.0,
                visual_query="digital banking vault red alert breach",
                visual_shots=["digital banking vault red alert breach", "cyber security warning"]
            ),
            ScriptSegment(
                segment_index=3,
                narration="Attackers are already harvesting encrypted internet traffic today to decrypt it the second quantum arrives.",
                duration_seconds=6.0,
                visual_query="dark web server racks glowing",
                visual_shots=["dark web server racks glowing", "data theft cables"]
            ),
            ScriptSegment(
                segment_index=4,
                narration="Engineers are racing into post-quantum lattice cryptography to shield global infrastructure before zero day.",
                duration_seconds=6.0,
                visual_query="lattice cryptography code matrix green",
                visual_shots=["lattice cryptography code matrix green", "secure network grid"]
            ),
            ScriptSegment(
                segment_index=5,
                narration="Upgrade your enterprise keys to lattice standards now before your private data belongs to everyone.",
                duration_seconds=6.0,
                visual_query="cyber defense shield active",
                visual_shots=["cyber defense shield active", "secure quantum shield"]
            ),
        ]
        valid_script = Script(
            topic="Quantum Supremacy Breakthrough",
            target_duration=30,
            hook="Quantum computers just broke all modern encryption.",
            segments=good_segments
        )
        res_valid = head_of_story.inspect(valid_script)
        self.assertTrue(res_valid.passed)

    # -------------------------------------------------------------------------
    # 2. HEAD OF AUDIO TESTS
    # -------------------------------------------------------------------------
    def test_head_of_audio_silence_rejection(self) -> None:
        """Verifies HeadOfAudio rejects pure silent audio tracks."""
        silent_wav = self.work_dir / "silent.wav"
        with wave.open(str(silent_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(b"\x00\x00" * 48000)  # 1.0 second of silence

        res = head_of_audio.inspect(silent_wav, context={"expected_duration": 1.0})
        self.assertFalse(res.passed)
        self.assertIn("silent", res.feedback or "")

    def test_head_of_audio_valid_sine_approval(self) -> None:
        """Verifies HeadOfAudio approves valid normalized WAV audio."""
        tone_wav = self.work_dir / "tone.wav"
        with wave.open(str(tone_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            # Generate 2.0 seconds of 440Hz sine wave with non-zero amplitude
            frames = bytearray()
            for i in range(96000):
                val = int(12000 * math.sin(2 * math.pi * 440 * (i / 48000.0)))
                frames.extend(struct.pack("<h", val))
            wf.writeframes(frames)

        res = head_of_audio.inspect(tone_wav, context={"expected_duration": 2.0})
        self.assertTrue(res.passed)
        self.assertGreater(res.details.get("rms_amplitude", 0.0), 0.001)

    # -------------------------------------------------------------------------
    # 3. HEAD OF ART TESTS
    # -------------------------------------------------------------------------
    def test_head_of_art_blank_frame_rejection(self) -> None:
        """Verifies HeadOfArt rejects a solid flat black video (variance < 12.0)."""
        black_mp4 = self.work_dir / "solid_black.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30",
            "-t", "0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(black_mp4)
        ]
        subprocess.run(cmd, check=True)

        res = head_of_art.inspect(black_mp4)
        self.assertFalse(res.passed)
        self.assertIn("Blank or corrupt render", res.feedback or "")

    def test_head_of_art_valid_video_approval(self) -> None:
        """Verifies HeadOfArt approves a 1080x1920 30fps CFR video with texture."""
        texture_mp4 = self.work_dir / "texture.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=1080x1920:r=30",
            "-t", "0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(texture_mp4)
        ]
        subprocess.run(cmd, check=True)

        res = head_of_art.inspect(texture_mp4)
        self.assertTrue(res.passed)
        self.assertEqual(res.details.get("resolution"), "1080x1920")

    # -------------------------------------------------------------------------
    # 4. HEAD OF POST TESTS
    # -------------------------------------------------------------------------
    def test_head_of_post_soft_subtitles_rejection(self) -> None:
        """Verifies HeadOfPost rejects master videos containing soft subtitle streams."""
        soft_sub_mp4 = self.work_dir / "soft_sub.mp4"
        # Create MP4 with dummy text subtitle stream
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=1080x1920:r=30",
            "-f", "lavfi", "-i", "sine=f=440:r=48000",
            "-t", "1.0",
            "-c:v", "libx264", "-c:a", "aac",
            str(soft_sub_mp4)
        ]
        subprocess.run(cmd, check=True)

        # Mux soft subtitle stream via mov_text
        sub_srt = self.work_dir / "test.srt"
        sub_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nSoft caption\n", encoding="utf-8")

        muxed_mp4 = self.work_dir / "soft_sub_muxed.mp4"
        cmd_mux = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(soft_sub_mp4),
            "-i", str(sub_srt),
            "-c", "copy", "-c:s", "mov_text",
            str(muxed_mp4)
        ]
        subprocess.run(cmd_mux, check=True)

        res = head_of_post.inspect(muxed_mp4)
        self.assertFalse(res.passed)
        self.assertIn("soft subtitle stream", res.feedback or "")

    def test_head_of_post_valid_master_approval(self) -> None:
        """Verifies HeadOfPost approves a clean master MP4 with audio and hard-burned visual."""
        master_mp4 = self.work_dir / "clean_master.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=s=1080x1920:r=30",
            "-f", "lavfi", "-i", "sine=f=440:r=48000",
            "-t", "15.0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(master_mp4)
        ]
        subprocess.run(cmd, check=True)

        res = head_of_post.inspect(master_mp4)
        self.assertTrue(res.passed)
        self.assertTrue(res.details.get("hard_burned_confirmed"))

    # -------------------------------------------------------------------------
    # 5. HEAD OF COMPLIANCE TESTS
    # -------------------------------------------------------------------------
    def test_head_of_compliance_metadata_and_quota(self) -> None:
        """Verifies HeadOfCompliance enforces title bounds and '#shorts' tags."""
        fake_video = self.work_dir / "placeholder.mp4"
        fake_video.touch()

        # Missing #shorts
        res_no_shorts = head_of_compliance.inspect(
            fake_video,
            context={"title": "Tech Insights", "description": "No hashtags here", "skip_publish": True}
        )
        self.assertFalse(res_no_shorts.passed)
        self.assertIn("missing mandatory '#shorts'", res_no_shorts.feedback or "")

        # Title > 100 chars
        res_long_title = head_of_compliance.inspect(
            fake_video,
            context={
                "title": "A" * 105,
                "description": "Valid description with #shorts tag",
                "skip_publish": True
            }
        )
        self.assertFalse(res_long_title.passed)
        self.assertIn("exceeds YouTube 100-character limit", res_long_title.feedback or "")

        # Valid metadata
        res_valid = head_of_compliance.inspect(
            fake_video,
            context={
                "title": "Quantum Computing Secrets Revealed",
                "description": "Full narration breakdown #shorts #tech",
                "skip_publish": True
            }
        )
        self.assertTrue(res_valid.passed)

        # Valid metadata with YouTube quota check (skip_publish=False)
        res_quota = head_of_compliance.inspect_tier1(
            fake_video,
            context={
                "title": "Quantum Computing Secrets Revealed",
                "description": "Full narration breakdown #shorts #tech",
                "skip_publish": False
            }
        )
        self.assertTrue(res_quota.passed)
