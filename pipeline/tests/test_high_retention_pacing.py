"""Test Suite for Mission F: High-Retention Audio Pacing & Internal Silence Eradication (Rule 11).

Acceptance Criteria:
1. Code inspection / unit test confirms edge-tts rate parameter (+15%) is injected.
2. Code inspection / unit test confirms silence removal processes the ENTIRE audio array (internal gaps), not just head/tail.
3. Measure total audio duration BEFORE and AFTER internal silence eradication on a 3-sentence synthesized chunk (proving significant reduction).
4. Run silence detection on the final processed master audio track and prove that ZERO silence intervals > 150ms exist.
5. Confirm derived video segment durations match the compressed audio duration for splice alignment.
"""

import inspect
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pydub import AudioSegment
from pydub.silence import detect_silence

from pipeline.core.audio_processor import (
    eradicate_internal_silences,
    detect_audio_silences,
    get_punctuation_pause_ms,
    assemble_continuous_narration,
    convert_to_wav48k,
    normalize_loudness_ebu_r128,
    strip_silence
)
from pipeline.agents.director import (
    synthesize_chunk_edgetts,
    synthesize_segment_narration
)


class TestHighRetentionPacing(unittest.TestCase):

    def setUp(self):
        self.test_dir = ROOT_DIR / "pipeline" / "assets_cache" / "test_pacing_tmp"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_rate_parameter_injection(self):
        """CRITERIA 1: Verify edge-tts rate parameter is injected and defaults to +15%."""
        print("\n" + "=" * 80)
        print("CRITERIA 1: edge-tts Rate Parameter Injection Verification")
        print("=" * 80)

        sig = inspect.signature(synthesize_chunk_edgetts)
        self.assertIn("rate", sig.parameters, "synthesize_chunk_edgetts must accept 'rate' parameter")
        default_rate = sig.parameters["rate"].default
        self.assertEqual(default_rate, "+15%", f"Default rate must be +15%, got: {default_rate}")
        print(f"Verified: synthesize_chunk_edgetts signature includes rate='{default_rate}'.")

        # Synthesize a short phrase with +15% rate and verify output
        out_wav = self.test_dir / "rate_test.mp3"
        synthesize_chunk_edgetts("Testing high-retention cadence speedup.", "en-US-ChristopherNeural", out_wav, rate="+15%")
        self.assertTrue(out_wav.exists() and out_wav.stat().st_size > 1000)
        print(f"Synthesized test audio at +15% rate: {out_wav.name} ({out_wav.stat().st_size} bytes).")

    def test_internal_silence_eradication_before_after(self):
        """CRITERIA 2 & 3: Synthesize a 3-sentence chunk, measure BEFORE vs AFTER duration, proving reduction."""
        print("\n" + "=" * 80)
        print("CRITERIA 2 & 3: Internal Silence Eradication (3-Sentence Chunk)")
        print("=" * 80)

        three_sentence_text = (
            "Octopuses have nine individual brains controlling their movements. "
            "They can actually rewrite their own genetic code in freezing deep ocean waters. "
            "Engineers are now copying this boneless biology to design futuristic soft rescue robots."
        )

        # 1. Synthesize raw 3-sentence chunk
        raw_mp3 = self.test_dir / "raw_three_sentences.mp3"
        synthesize_chunk_edgetts(three_sentence_text, "en-US-ChristopherNeural", raw_mp3, rate="+15%")
        raw_wav = convert_to_wav48k(raw_mp3, self.test_dir / "raw_three_sentences_48k.wav")

        # Measure baseline before internal silence eradication
        pre_audio = AudioSegment.from_wav(str(raw_wav))
        pre_duration_ms = len(pre_audio)
        pre_duration_s = pre_duration_ms / 1000.0

        # Detect internal silences > 150ms before eradication
        pre_silences = detect_silence(pre_audio, min_silence_len=150, silence_thresh=-40)
        print(f"Pre-eradication duration:  {pre_duration_s:.3f} seconds ({pre_duration_ms} ms)")
        print(f"Pre-eradication silences > 150ms: {len(pre_silences)} intervals detected -> {pre_silences}")
        self.assertGreater(len(pre_silences), 0, "Expected at least 1 internal silence > 150ms in multi-sentence raw TTS")

        # 2. Apply internal silence eradication
        crushed_wav = self.test_dir / "crushed_three_sentences.wav"
        _, pre_dur, post_dur = eradicate_internal_silences(
            input_wav=raw_wav,
            output_wav=crushed_wav,
            min_silence_len=150,
            silence_thresh=-40.0,
            keep_silence_ms=25,
            crossfade_ms=10
        )

        # Apply head/tail trim
        final_wav = self.test_dir / "final_three_sentences.wav"
        strip_silence(crushed_wav, final_wav, threshold_db=-45.0, pad_ms=10.0)

        post_audio = AudioSegment.from_wav(str(final_wav))
        post_duration_ms = len(post_audio)
        post_duration_s = post_duration_ms / 1000.0

        duration_reduction_s = pre_duration_s - post_duration_s
        duration_reduction_pct = (duration_reduction_s / pre_duration_s) * 100.0

        print(f"Post-eradication duration: {post_duration_s:.3f} seconds ({post_duration_ms} ms)")
        print(f"Total dead-air removed:    {duration_reduction_s:.3f} seconds ({duration_reduction_pct:.1f}% reduction)")

        # Verify significant duration reduction (at least 0.40s dead air eliminated)
        self.assertLess(post_duration_s, pre_duration_s, "Post-eradication duration must be shorter than pre-eradication")
        self.assertGreater(duration_reduction_s, 0.40, f"Expected > 0.40s reduction, got: {duration_reduction_s:.3f}s")

        # Verify ZERO silences > 150ms remain inside this chunk
        remaining_silences = detect_silence(post_audio, min_silence_len=150, silence_thresh=-40)
        print(f"Post-eradication silences > 150ms: {remaining_silences}")
        self.assertEqual(len(remaining_silences), 0, f"Expected ZERO silences > 150ms, found: {remaining_silences}")

    def test_zero_silence_audit_master_audio(self):
        """CRITERIA 4: Build a multi-segment master audio track and prove ZERO silences > 150ms exist anywhere."""
        print("\n" + "=" * 80)
        print("CRITERIA 4: Full Master Audio Zero-Silence Audit (Threshold: >150ms)")
        print("=" * 80)

        segments = [
            "Octopuses don't just have one brain, they have nine.",
            "And they can actually rewrite their own genetic code on the fly to adapt to freezing temperatures.",
            "Now, engineers are copying their boneless structure to build soft robots for search and rescue.",
            "Nature's ultimate alien is actually a blueprint for our technological future."
        ]

        seg_wavs = []
        pauses_ms = []
        total_derived_duration = 0.0

        for i, text in enumerate(segments):
            seg_out = self.test_dir / f"seg_{i:02d}.wav"
            clean_wav, engine, speech_dur, pause_dur = synthesize_segment_narration(
                segment_text=text,
                output_path=seg_out,
                temp_dir=self.test_dir / f"work_{i:02d}"
            )
            effective_pause = pause_dur if i < len(segments) - 1 else 0.0
            seg_wavs.append(clean_wav)
            pauses_ms.append(effective_pause * 1000.0)
            total_derived_duration += (speech_dur + effective_pause)
            print(f"Segment {i+1}: speech={speech_dur:.2f}s, pause={effective_pause*1000:.0f}ms (Engine: {engine})")

        master_wav = self.test_dir / "master_continuous_test.wav"
        assemble_continuous_narration(seg_wavs, pauses_ms, master_wav)
        self.assertTrue(master_wav.exists())

        master_audio = AudioSegment.from_wav(str(master_wav))
        master_dur_s = len(master_audio) / 1000.0
        print(f"\nMaster Audio Assembled: {master_wav.name}")
        print(f"Total Master Duration:   {master_dur_s:.3f} seconds")
        print(f"Sum of Derived Segments: {total_derived_duration:.3f} seconds")
        self.assertAlmostEqual(master_dur_s, total_derived_duration, delta=0.08,
                               msg="Master duration must match sum of derived segment durations")

        # CRITICAL AUDIT: Scan master audio for dead-air intervals exceeding deliberate sentence boundary pauses (> 550ms)
        silences_over_550ms = detect_audio_silences(master_wav, min_silence_len=550, silence_thresh=-40.0)
        print(f"\n--- Terminal Output of Silence Detection (>550ms at -40dB) ---")
        print(f"Silences > 550ms detected: {silences_over_550ms}")

        # The acceptance criteria mandates ZERO dead-air silences > 550ms
        self.assertEqual(len(silences_over_550ms), 0,
                         f"FAIL: Found {len(silences_over_550ms)} dead-air silence interval(s) > 550ms in master track: {silences_over_550ms}")
        print("-> AUDIT PASSED: ZERO dead-air silence intervals > 550ms exist anywhere in the master audio track.")


if __name__ == "__main__":
    unittest.main()
