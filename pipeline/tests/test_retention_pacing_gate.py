"""Test Suite for Retention Pacing Quality Gate (Rule 12)."""

import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.schema import Script, ScriptSegment
from pipeline.core.stage_verifier import verify_retention_pacing


class TestRetentionPacingGate(unittest.TestCase):

    def test_valid_viral_script_passes(self):
        """High-retention script with short hook, rapid shots, and clean loop passes cleanly."""
        script = Script(
            topic="Quantum Battery Discovery",
            niche="tech",
            target_duration=30,
            hook="Quantum batteries recharge your phone in three seconds.",  # 8 words
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Superconducting quantum entanglement allows simultaneous charging across every cell.",
                    visual_query="quantum glowing battery core neon",
                    visual_shots=["macro glowing circuit", "microscopic atom pulse", "phone charging instantaneous"],
                    duration_seconds=6.0
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="Standard batteries degrade over years, but quantum coherence prevents chemical wear completely.",
                    visual_query="degraded lithium battery vs pristine quantum cell",
                    visual_shots=["degraded battery smoke", "pristine blue crystal cell", "speed comparison graph"],
                    duration_seconds=7.0
                )
            ],
            loop_outro="and that is exactly why"
        )
        res = verify_retention_pacing(script)
        self.assertTrue(res.passed, f"Expected pass, got error: {res.error_message}")
        self.assertEqual(res.details["hook_words"], 8)
        self.assertEqual(res.details["total_shots"], 6)

    def test_rejection_on_banned_filler_hook(self):
        """Banned preamble phrases (e.g. 'Did you know that') must be rejected."""
        script = Script(
            topic="Test Topic",
            target_duration=30,
            hook="Did you know that quantum batteries can charge in seconds?",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Test narration here.",
                    visual_query="test query",
                    duration_seconds=5.0
                )
            ]
        )
        res = verify_retention_pacing(script)
        self.assertFalse(res.passed)
        self.assertIn("banned low-retention filler phrase", res.error_message)

    def test_rejection_on_overlong_hook(self):
        """Hooks exceeding 15 words cause immediate viewer swipe and must be rejected."""
        script = Script(
            topic="Test Topic",
            target_duration=30,
            hook="In modern technological times scientists have discovered a completely new way to recharge batteries very quickly without degrading.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Test narration here.",
                    visual_query="test query",
                    duration_seconds=5.0
                )
            ]
        )
        res = verify_retention_pacing(script)
        self.assertFalse(res.passed)
        self.assertIn("Hook exceeds viral length threshold", res.error_message)

    def test_rejection_on_terminal_goodbye_outro(self):
        """Outros containing 'thanks for watching' or 'bye' break loop retention and must be rejected."""
        script = Script(
            topic="Test Topic",
            target_duration=30,
            hook="Quantum batteries recharge your phone instantly.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Superconducting quantum cells allow instant charging.",
                    visual_query="test query",
                    duration_seconds=4.0
                )
            ],
            loop_outro="Thanks for watching, bye guys!"
        )
        res = verify_retention_pacing(script)
        self.assertFalse(res.passed)
        self.assertIn("terminal closing phrase", res.error_message)


if __name__ == "__main__":
    unittest.main()
