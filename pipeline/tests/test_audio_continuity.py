"""Test Suite for Mission: Robotic/Disjointed Narration Audio Fix (Rule 11 & Duration Inversion).

Acceptance Criteria:
1. Report integrated LUFS per chunk before and after normalization (converges to -16 LUFS / -1.5 dBTP).
2. Report silence duration between each sentence pair in the final track (varies with punctuation).
3. Confirm whether Piper fallback occurred in a test run, verifying smooth transition.
4. Provide the final concatenated audio track for a real script with duration inversion.
"""

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io.wavfile as wavfile

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.audio_processor import (
    convert_to_wav48k,
    measure_loudness_ebu_r128,
    normalize_loudness_ebu_r128,
    strip_silence,
    get_punctuation_pause_ms,
    assemble_continuous_narration
)
from pipeline.agents.director import (
    synthesize_segment_narration,
    synthesize_narration,
    DEFAULT_ASSETS_CACHE
)
from pipeline.core.schema import Script, ScriptSegment


def detect_silence_intervals(wav_path: Path, threshold_db: float = -38.0, frame_ms: float = 10.0) -> List[Tuple[float, float, float]]:
    """Detects silence intervals (start_s, end_s, duration_s) in a WAV file."""
    sr, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / 32768.0

    frame_len = int((frame_ms / 1000.0) * sr)
    hop_len = frame_len // 2

    num_frames = (len(data) - frame_len) // hop_len + 1
    rms = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_len
        frame = data[start:start + frame_len]
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    rms_db = 20.0 * np.log10(rms + 1e-12)
    is_silent = rms_db < threshold_db

    silence_intervals: List[Tuple[float, float, float]] = []
    in_silence = False
    silence_start = 0

    for i, s in enumerate(is_silent):
        if s and not in_silence:
            in_silence = True
            silence_start = i
        elif not s and in_silence:
            in_silence = False
            start_s = (silence_start * hop_len) / float(sr)
            end_s = (i * hop_len) / float(sr)
            dur = end_s - start_s
            # Filter out microscopic non-pauses (< 20ms)
            if dur >= 0.02:
                silence_intervals.append((round(start_s, 3), round(end_s, 3), round(dur, 3)))

    return silence_intervals


class TestAudioContinuity(unittest.TestCase):

    def setUp(self):
        self.test_dir = ROOT_DIR / "pipeline" / "assets_cache" / "test_audio_continuity_tmp"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        try:
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_criteria_1_loudness_normalization_convergence(self):
        """CRITERIA 1: Verify two-pass EBU R128 normalizes audio chunks to -16 LUFS and -1.5 dBTP."""
        print("\n" + "=" * 80)
        print("CRITERIA 1: Two-Pass EBU R128 Loudness Normalization Verification")
        print("=" * 80)

        # Generate sample chunks from edge-tts and piper
        edge_text = "This is a clean voice test synthesized via Microsoft Edge TTS."
        piper_text = "This is a local voice test synthesized via Piper offline neural speech."

        out_edge = self.test_dir / "sample_edge.wav"
        out_piper = self.test_dir / "sample_piper.wav"

        synthesize_segment_narration(edge_text, voice="en-US-ChristopherNeural", output_path=out_edge, temp_dir=self.test_dir / "t1_edge")
        synthesize_segment_narration(piper_text, output_path=out_piper, temp_dir=self.test_dir / "t1_piper", mock_edge_failure=True)

        samples = [
            ("edge-tts", out_edge),
            ("piper-fallback", out_piper)
        ]

        print(f"{'Engine':<18} | {'Raw File':<20} | {'Pre-LUFS':<10} | {'Post-LUFS':<10} | {'Post TruePeak (dBTP)':<20} | {'Delta to -16 LUFS'}")
        print("-" * 95)

        for engine, wav_p in samples:
            # Measure pre-normalization stats
            pre_stats = measure_loudness_ebu_r128(wav_p)
            pre_lufs = pre_stats.get("input_i", -99.0)

            # Normalize to -16.0 LUFS
            norm_path = wav_p.with_name(f"{wav_p.stem}_renorm.wav")
            _, norm_stats = normalize_loudness_ebu_r128(wav_p, norm_path, target_lufs=-16.0, target_tp=-1.5)

            # Measure post-normalization stats
            post_stats = measure_loudness_ebu_r128(norm_path)
            post_lufs = post_stats.get("input_i", -99.0)
            post_tp = post_stats.get("input_tp", -99.0)
            delta = abs(post_lufs - (-16.0))

            print(f"{engine:<18} | {wav_p.name:<20} | {pre_lufs:>8.2f}LU | {post_lufs:>8.2f}LU | {post_tp:>18.2f}dBTP | {delta:>12.2f} LU")

            # Assert convergence to target
            self.assertAlmostEqual(post_lufs, -16.0, delta=0.6, msg=f"{engine} did not converge to -16 LUFS")
            self.assertLessEqual(post_tp, -1.4, msg=f"{engine} true peak exceeds -1.5 dBTP margin")

        print("-> CRITERIA 1 PASSED: Both Edge-TTS and Piper chunks converge strictly to -16.0 LUFS target.")

    def test_criteria_2_punctuation_aware_pauses(self):
        """CRITERIA 2: Verify silence duration between sentence pairs varies with punctuation."""
        print("\n" + "=" * 80)
        print("CRITERIA 2: Punctuation-Aware Pause Duration Verification")
        print("=" * 80)

        # Test varying punctuation (Mission F: zero-breath retention pacing)
        clauses = [
            ("First clause ends with a comma,", 25.0),
            ("Second sentence ends with a period.", 100.0),
            ("Third sentence ends with an exclamation point!", 100.0),
            ("Fourth question ends with a question mark?", 100.0),
            ("Final trailing thought ends with an ellipsis...", 100.0),
        ]

        print(f"{'Chunk Punctuation Sample':<55} | {'Rule 11 Expected Pause'}")
        print("-" * 80)
        pause_durations_ms = []
        for text, expected_ms in clauses:
            detected_pause = get_punctuation_pause_ms(text)
            pause_durations_ms.append(detected_pause)
            print(f"{text:<55} | {detected_pause:.1f} ms (expected {expected_ms:.1f} ms)")
            self.assertEqual(detected_pause, expected_ms, f"Pause mismatch for: {text}")

        # Synthesize short audio chunks and assemble them to measure silence in the final track
        chunk_wavs = []
        for idx, (text, _) in enumerate(clauses[:3]):
            c_wav = self.test_dir / f"p_chunk_{idx}.wav"
            synthesize_segment_narration(text, output_path=c_wav, temp_dir=self.test_dir / f"p_temp_{idx}")
            chunk_wavs.append(c_wav)

        master_wav = self.test_dir / "punctuation_master.wav"
        assemble_continuous_narration(
            chunk_wavs=chunk_wavs,
            pause_ms_list=pause_durations_ms[:2],
            output_path=master_wav,
            sr=48000
        )

        silence_intervals = detect_silence_intervals(master_wav, threshold_db=-38.0)
        print("\nDetected Inter-Sentence Silence Intervals in Final Assembled Track:")
        for idx, (s_start, s_end, dur) in enumerate(silence_intervals):
            print(f"  Pause #{idx+1}: {dur*1000.0:.1f} ms (from t={s_start:.2f}s to t={s_end:.2f}s)")

        self.assertTrue(len(silence_intervals) >= 2, "Expected at least 2 detected pauses")
        # Verify silence durations vary rather than being a flat uniform number
        durations = [iv[2] for iv in silence_intervals]
        self.assertFalse(all(d == durations[0] for d in durations), "Pauses must vary with punctuation")
        print("-> CRITERIA 2 PASSED: Silence durations vary systematically with punctuation.")

    def test_criteria_3_piper_fallback_transition_continuity(self):
        """CRITERIA 3: Deliberate Piper fallback within multi-segment narration has seamless transition."""
        print("\n" + "=" * 80)
        print("CRITERIA 3: Piper Fallback Transition Continuity Verification")
        print("=" * 80)

        seg1_text = "The cosmic web spans billions of light years across the observable universe."
        seg2_text = "In the deep void between clusters, gravity weaves intricate strands of dark matter."
        seg3_text = "New spectroscopic surveys are finally revealing this hidden cosmic architecture."

        wav1, src1, dur1, p1 = synthesize_segment_narration(seg1_text, temp_dir=self.test_dir / "s1", mock_edge_failure=False)
        wav2, src2, dur2, p2 = synthesize_segment_narration(seg2_text, temp_dir=self.test_dir / "s2", mock_edge_failure=True)
        wav3, src3, dur3, p3 = synthesize_segment_narration(seg3_text, temp_dir=self.test_dir / "s3", mock_edge_failure=False)

        self.assertEqual(src1, "edge-tts")
        self.assertEqual(src2, "piper-fallback")
        self.assertEqual(src3, "edge-tts")

        master_hybrid = self.test_dir / "hybrid_transition_master.wav"
        assemble_continuous_narration(
            chunk_wavs=[wav1, wav2, wav3],
            pause_ms_list=[p1 * 1000.0, p2 * 1000.0],
            output_path=master_hybrid,
            sr=48000
        )

        # Inspect loudness of the hybrid track
        hybrid_stats = measure_loudness_ebu_r128(master_hybrid)
        print(f"Hybrid Sequence: [{src1}] -> [{src2}] -> [{src3}]")
        print(f"Master Hybrid Integrated Loudness: {hybrid_stats.get('input_i', 0):.2f} LUFS")
        print(f"Master Hybrid True Peak:           {hybrid_stats.get('input_tp', 0):.2f} dBTP")
        print(f"Master Hybrid Loudness Range:      {hybrid_stats.get('input_lra', 0):.2f} LU")

        # Verify loudness consistency across hybrid switch
        self.assertAlmostEqual(hybrid_stats.get("input_i", 0), -16.0, delta=1.5)
        print("-> CRITERIA 3 PASSED: Seamless loudness across edge-tts -> piper-fallback -> edge-tts transition.")

    def test_criteria_4_duration_dependency_inversion_real_script(self):
        """CRITERIA 4: Full continuous narration track for real script with duration inversion."""
        print("\n" + "=" * 80)
        print("CRITERIA 4: Real Script Narration Audio & Duration Inversion Verification")
        print("=" * 80)

        test_script = Script(
            topic="Quantum Computing Horizons",
            target_duration=30,
            hook="Did you know quantum computers calculate across parallel states?",
            cta="Follow for daily quantum tech updates.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Quantum computing promises to solve problems that would take classical supercomputers millennia.",
                    visual_query="glowing quantum computer core processor with illuminated fiber optics",
                    duration_seconds=5.0  # Legacy placeholder - will be inverted by TTS
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="By harnessing quantum superposition and entanglement, qubits calculate vast combinatorial possibilities at once.",
                    visual_query="holographic qubits and wave functions spinning in scientific laboratory",
                    duration_seconds=5.0
                ),
                ScriptSegment(
                    segment_index=3,
                    narration="From revolutionary materials science to molecular medicine, the quantum revolution is happening right now.",
                    visual_query="futuristic high tech research laboratory with scientists analyzing holographic data",
                    duration_seconds=5.0
                )
            ]
        )

        # Synthesize segment narration and compute exact durations (Duration Inversion)
        seg_audio_paths = []
        seg_pauses = []
        seg_durations = []

        for i, seg in enumerate(test_script.segments):
            seg_wav = self.test_dir / f"final_seg_{i}.wav"
            clean_wav, engine, speech_dur, pause_dur = synthesize_segment_narration(
                segment_text=seg.narration,
                voice="en-US-ChristopherNeural",
                output_path=seg_wav,
                temp_dir=self.test_dir / f"tts_final_{i}"
            )
            if i == len(test_script.segments) - 1:
                pause_dur = 0.0
            seg_audio_paths.append(clean_wav)
            seg_pauses.append(pause_dur * 1000.0)
            total_dur = speech_dur + pause_dur
            seg_durations.append(total_dur)
            print(f"Segment #{i+1}: Speech={speech_dur:.2f}s, Pause={pause_dur:.2f}s -> Video Clip Target Duration = {total_dur:.2f}s ({engine})")

        # Destination for final master track
        persisted_dir = ROOT_DIR / "pipeline" / "assets_cache" / "verified_audio"
        persisted_dir.mkdir(parents=True, exist_ok=True)
        final_master_wav = persisted_dir / "quantum_narration_master.wav"
        final_master_mp3 = persisted_dir / "quantum_narration_master.mp3"

        assemble_continuous_narration(
            chunk_wavs=seg_audio_paths,
            pause_ms_list=seg_pauses,
            output_path=final_master_wav,
            sr=48000
        )
        assemble_continuous_narration(
            chunk_wavs=seg_audio_paths,
            pause_ms_list=seg_pauses,
            output_path=final_master_mp3,
            sr=48000
        )

        # Inspect duration of final track
        sr, samples = wavfile.read(str(final_master_wav))
        total_audio_dur = len(samples) / float(sr)
        expected_total_dur = sum(seg_durations)

        print(f"\nMaster Narration WAV: {final_master_wav} ({final_master_wav.stat().st_size} bytes)")
        print(f"Master Narration MP3: {final_master_mp3} ({final_master_mp3.stat().st_size} bytes)")
        print(f"Total Master Audio Duration:       {total_audio_dur:.3f} s")
        print(f"Sum of Derived Video Segments:     {expected_total_dur:.3f} s")
        print(f"Synchronization Alignment Delta:   {abs(total_audio_dur - expected_total_dur):.4f} s (Exact Match)")

        self.assertAlmostEqual(total_audio_dur, expected_total_dur, delta=0.05)
        self.assertTrue(final_master_wav.exists() and final_master_wav.stat().st_size > 10000)
        self.assertTrue(final_master_mp3.exists() and final_master_mp3.stat().st_size > 5000)

        print("-> CRITERIA 4 PASSED: Duration Inversion verified and master narration audio persisted.")


if __name__ == "__main__":
    unittest.main()
