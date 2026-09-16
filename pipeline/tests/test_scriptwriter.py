"""Unit tests for Scriptwriter Agent and schema-enforced Gemini structured output.

Per Rule 4 & Rule 2:
- Deliberately feeds malformed JSON (missing field, wrong type, truncated syntax).
- Proves the corrective retry and fallback path triggers with zero unhandled exceptions.
- Validates WARNING logging on fallback events.
- Validates live end-to-end call with real topic.
"""

import json
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.schema import Script, ScriptSegment
from pipeline.agents.scriptwriter import get_valid_script, load_fallback_template, logger


class MockGeminiResponse:
    """Mock Gemini API response object."""
    def __init__(self, text: str, parsed=None):
        self.text = text
        self.parsed = parsed


class TestScriptwriter(unittest.TestCase):

    def setUp(self):
        self.topic = "The Mystery of Black Holes"
        self.niche = "tech"

    def test_malformed_missing_field(self):
        """Case 1: JSON missing mandatory 'hook' and 'segments' fields."""
        bad_json = json.dumps({
            "topic": self.topic,
            "niche": self.niche,
            "target_duration": 30
            # Missing 'hook' and 'segments'
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(text=bad_json)

        # Must not raise any unhandled exception
        script = get_valid_script(
            topic=self.topic,
            niche=self.niche,
            max_retries=2,
            client=mock_client
        )

        self.assertIsInstance(script, Script)
        self.assertTrue(len(script.hook) > 0)
        self.assertTrue(len(script.segments) >= 1)
        self.assertIsInstance(script.segments[0], ScriptSegment)

    def test_malformed_wrong_type(self):
        """Case 2: JSON with invalid field types (segments is a string, target_duration is boolean)."""
        bad_json = json.dumps({
            "topic": self.topic,
            "niche": self.niche,
            "target_duration": "not_an_int",
            "hook": "What happens when light cannot escape?",
            "segments": "this_should_be_a_list_not_a_string"
        })

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(text=bad_json)

        script = get_valid_script(
            topic=self.topic,
            niche=self.niche,
            max_retries=2,
            client=mock_client
        )

        self.assertIsInstance(script, Script)
        self.assertIsInstance(script.target_duration, int)
        self.assertIsInstance(script.segments, list)

    def test_malformed_truncated_json(self):
        """Case 3: Truncated JSON string (syntax decode error)."""
        truncated_json = '{"topic": "Black Holes", "hook": "Did you know that gravity'

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(text=truncated_json)

        script = get_valid_script(
            topic=self.topic,
            niche=self.niche,
            max_retries=2,
            client=mock_client
        )

        self.assertIsInstance(script, Script)
        self.assertTrue(len(script.segments) > 0)

    def test_fallback_warning_logged(self):
        """Verify that fallback events are logged at WARNING level with topic & reason."""
        truncated_json = '{"topic": "Invalid JSON'
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MockGeminiResponse(text=truncated_json)

        with self.assertLogs(logger, level="WARNING") as log_context:
            script = get_valid_script(
                topic="Quantum Entanglement Test",
                niche="tech",
                max_retries=1,
                client=mock_client
            )
            self.assertIsInstance(script, Script)

        # Check log output
        warning_records = [msg for msg in log_context.output if "WARNING" in msg]
        self.assertTrue(len(warning_records) > 0, "Expected at least one WARNING log record")
        self.assertTrue(any("Quantum Entanglement Test" in msg for msg in warning_records), "Log must contain topic")
        self.assertTrue(any("fallback" in msg.lower() for msg in warning_records), "Log must mention fallback")

    def test_corrective_retry_recovers(self):
        """Test that a malformed attempt 1 followed by a valid attempt 2 succeeds without fallback."""
        bad_json = '{"topic": "AI", "hook": "Short"}'  # missing segments
        valid_script = Script(
            topic=self.topic,
            niche=self.niche,
            target_duration=30,
            hook="Artificial intelligence just changed forever.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="New models demonstrate emergent reasoning.",
                    visual_query="neural network glowing futuristic visualization",
                    duration_seconds=10.0
                )
            ]
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            MockGeminiResponse(text=bad_json),
            MockGeminiResponse(text=valid_script.model_dump_json(), parsed=valid_script)
        ]

        with self.assertLogs(logger, level="INFO") as log_context:
            script = get_valid_script(
                topic=self.topic,
                niche=self.niche,
                max_retries=2,
                client=mock_client
            )

        self.assertIsInstance(script, Script)
        self.assertEqual(script.hook, "Artificial intelligence just changed forever.")
        # Ensure fallback WARNING was NOT logged because attempt 2 recovered
        warning_logs = [msg for msg in log_context.output if "WARNING" in msg]
        self.assertEqual(len(warning_logs), 0, "No warning should be logged when retry succeeds")

    def test_live_gemini_api_call(self):
        """Real end-to-end integration test querying Gemini API with a real topic."""
        import dotenv
        dotenv.load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY not found in environment")

        real_topic = "The Deepest Trench in the Ocean"
        script = get_valid_script(topic=real_topic, niche="history")

        self.assertIsInstance(script, Script)
        self.assertTrue(len(script.hook) >= 5, "Hook should be substantial")
        self.assertTrue(len(script.segments) >= 2, "Should have multiple narrative segments")
        for seg in script.segments:
            self.assertTrue(len(seg.narration) >= 3, "Narration should be non-empty")
            self.assertTrue(len(seg.visual_query) >= 3, "Visual query should be non-empty")
            self.assertGreater(seg.duration_seconds, 0.0, "Duration must be positive")
        self.assertGreater(script.target_duration, 0)
        print(f"\n[LIVE TEST SUCCESS] Generated script for '{script.topic}' with {len(script.segments)} segments.")


if __name__ == "__main__":
    unittest.main()
