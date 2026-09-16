"""Unit and integration tests for Director Agent.

Acceptance Criteria:
1. Deliberately-induced edge-tts failure triggers Piper fallback for only the affected chunk.
2. Full narration audio track for a real script is generated end-to-end and is continuous.
3. GPU worker invocations strictly go through gpu_lock.py.
4. Asset fetch demonstrates cache-hit avoidance of duplicate Pexels API calls.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.director import (
    chunk_text_by_sentence,
    synthesize_narration,
    fetch_visual_asset,
    run_gpu_worker,
    DEFAULT_ASSETS_CACHE
)
from pipeline.core.gpu_lock import GPULockError
from pipeline.core.schema import Script, ScriptSegment


class TestDirector(unittest.TestCase):

    def setUp(self):
        self.test_dir = ROOT_DIR / "pipeline" / "assets_cache" / "test_director_tmp"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_chunking_under_300_chars(self):
        """Verify sentence-level chunking respects the ~300 character bound."""
        long_text = (
            "The universe is constantly expanding into the unknown. "
            "Galaxies are drifting apart faster than the speed of light at extreme cosmic distances! "
            "When physicists first observed redshift anomalies, the scientific consensus was completely shattered. "
            "Yet today, dark energy remains the biggest enigma in astrophysics, pushing space apart at an accelerating rate. "
            "Will the cosmos end in a cold, silent freeze, or will gravity pull everything back together in a violent crunch?"
        )
        chunks = chunk_text_by_sentence(long_text, max_chars=300)
        self.assertTrue(len(chunks) >= 2)
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk), 300, f"Chunk {i} exceeded 300 chars: {len(chunk)}")
            self.assertTrue(len(chunk) > 10, f"Chunk {i} was too short")

    def test_mock_edgetts_failure_triggers_piper_fallback_for_single_chunk(self):
        """CRITERIA 1: Deliberately-induced edge-tts failure triggers Piper fallback ONLY for that chunk."""
        # 3 distinct sentences creating 3 chunks
        script_text = (
            "Chunk one introduces the mysterious premise of the topic. "
            "Chunk two represents the middle evidence that experiences a network drop. "
            "Chunk three delivers the final thought and closing takeaway."
        )

        out_audio = self.test_dir / "fallback_narration.mp3"

        # Deliberately fail chunk index 1 (the middle chunk)
        master_audio, sources = synthesize_narration(
            text=script_text,
            voice="en-US-ChristopherNeural",
            output_path=out_audio,
            min_delay=0.1,  # fast test pacing
            jitter_range=(0.01, 0.05),
            temp_dir=self.test_dir / "tts_chunks",
            mock_edge_failure_chunk=1
        )

        self.assertTrue(out_audio.exists())
        self.assertGreater(out_audio.stat().st_size, 5000)

        # Confirm exact source per chunk: Chunk 0 = edge-tts, Chunk 1 = piper-fallback, Chunk 2 = edge-tts
        self.assertEqual(len(sources), 3, "Expected 3 chunks synthesized")
        self.assertEqual(sources[0], "edge-tts", "Chunk 1 should use edge-tts")
        self.assertEqual(sources[1], "piper-fallback", "Chunk 2 must trigger Piper fallback")
        self.assertEqual(sources[2], "edge-tts", "Chunk 3 should use edge-tts")
        print(f"\n[CRITERIA 1 PASS] Chunk sources verified: {sources}")

    def test_full_narration_audio_continuous(self):
        """CRITERIA 2: Full narration audio track for real script is continuous without dropped chunks."""
        narration_text = (
            "Artificial intelligence is progressing at an astonishing pace. "
            "In just a few years, computational power has scaled exponentially. "
            "The future belongs to those who adapt and build with these tools today."
        )
        out_audio = self.test_dir / "full_narration.mp3"

        master_audio, sources = synthesize_narration(
            text=narration_text,
            voice="en-US-ChristopherNeural",
            output_path=out_audio,
            min_delay=0.2,
            jitter_range=(0.05, 0.1),
            temp_dir=self.test_dir / "tts_real"
        )

        self.assertTrue(out_audio.exists())
        # Inspect audio length via ffprobe or moviepy
        from moviepy import AudioFileClip
        clip = AudioFileClip(str(out_audio))
        duration = clip.duration
        clip.close()

        self.assertGreater(duration, 8.0, "Expected narration duration to be at least 8 seconds")
        self.assertEqual(len(sources), 3, "All 3 chunks must be present")
        print(f"\n[CRITERIA 2 PASS] Master narration generated: {duration:.2f}s ({len(sources)} chunks, all present)")

    def test_gpu_lock_structural_enforcement(self):
        """CRITERIA 3: Whisper & ComfyUI worker execution MUST go through gpu_lock.py."""
        # Test 1: Verify run_gpu_worker uses gpu_lock
        with patch("pipeline.agents.director.gpu_lock") as mock_gpu_lock, \
             patch("subprocess.run") as mock_subproc:

            mock_subproc.return_value = subprocess.CompletedProcess(
                args=["dummy"], returncode=0, stdout="Mock success", stderr=""
            )

            res = run_gpu_worker(
                worker_name="whisper_worker.py",
                worker_args=["--audio", "dummy.mp3"]
            )

            # Assert gpu_lock was called with worker_name
            mock_gpu_lock.assert_called_once()
            _, kwargs = mock_gpu_lock.call_args
            self.assertEqual(kwargs.get("worker_name"), "whisper_worker.py")
            print("\n[CRITERIA 3 PASS] Director dispatch verified: run_gpu_worker() acquires gpu_lock for whisper_worker.py.")

    def test_pexels_cache_hit_avoids_duplicate_api_call(self):
        """CRITERIA 4: Asset fetch demonstrates cache-hit avoidance of duplicate Pexels API calls."""
        query = "futuristic autonomous electric car on highway"
        cache_dir = self.test_dir / "pexels_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        query_slug = re.sub(r'[^a-zA-Z0-9]', '_', query.lower())[:24]
        query_hash = hashlib.md5(query.lower().strip().encode()).hexdigest()[:8]
        seeded_file = cache_dir / f"{query_slug}_{query_hash}.mp4"

        # Pre-seed the cache with a dummy video file
        with open(seeded_file, "wb") as f:
            f.write(b"MOCK_VIDEO_DATA" * 500)  # > 1024 bytes

        # Call fetch_visual_asset with patch on requests.get to prove 0 network calls occur
        with patch("requests.get") as mock_requests_get:
            asset_path, was_hit = fetch_visual_asset(
                visual_query=query,
                cache_dir=cache_dir
            )

            self.assertTrue(was_hit, "Expected cache hit")
            self.assertEqual(asset_path, seeded_file)
            mock_requests_get.assert_not_called()
            print(f"\n[CRITERIA 4 PASS] Cache HIT verified for '{query}': 0 network calls made.")


if __name__ == "__main__":
    unittest.main()
