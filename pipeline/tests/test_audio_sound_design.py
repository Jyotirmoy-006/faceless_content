"""Test Suite for Multi-Layer Sound Design (BGM, Procedural SFX, Audio Ducking)."""

import unittest
from pathlib import Path
import shutil
import numpy as np
import scipy.io.wavfile as wavfile

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from pipeline.core.audio_processor import (
    create_procedural_sfx,
    create_procedural_bgm,
    mix_voice_bgm_sfx,
)


class TestAudioSoundDesign(unittest.TestCase):

    def setUp(self):
        self.test_dir = ROOT_DIR / "pipeline" / "assets_cache" / "test_sound_design_tmp"
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_procedural_sfx_generation(self):
        """Verify generation of whoosh, impact, and riser SFX."""
        for sfx in ["whoosh", "impact", "riser"]:
            out_path = self.test_dir / f"sfx_{sfx}.wav"
            res = create_procedural_sfx(sfx_type=sfx, output_path=out_path)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 5000)

            sr, data = wavfile.read(str(res))
            self.assertEqual(sr, 48000)
            self.assertGreater(len(data), 0)

    def test_procedural_bgm_generation(self):
        """Verify generation of procedural ambient BGM."""
        for mood in ["suspense", "tech_pulse", "cinematic"]:
            out_path = self.test_dir / f"bgm_{mood}.wav"
            res = create_procedural_bgm(mood=mood, output_path=out_path, duration_s=4.0)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 100000)

            sr, data = wavfile.read(str(res))
            self.assertEqual(sr, 48000)
            dur = len(data) / sr
            self.assertAlmostEqual(dur, 4.0, delta=0.1)

    def test_mix_voice_bgm_sfx(self):
        """Verify multi-layer mixing of voice narration with ducked BGM and transition SFX."""
        # Create dummy 5s voice track (440Hz tone with speech-like modulation)
        voice_wav = self.test_dir / "dummy_voice.wav"
        sr = 48000
        t = np.linspace(0, 5.0, 5 * sr, endpoint=False)
        voice_pcm = (np.sin(2 * np.pi * 300 * t) * 0.8 * 32767).astype(np.int16)
        wavfile.write(str(voice_wav), sr, voice_pcm)

        mixed_out = self.test_dir / "master_mixed_audio.wav"
        sfx_cuts = [1.5, 3.5]
        res = mix_voice_bgm_sfx(
            voice_path=voice_wav,
            output_path=mixed_out,
            sfx_timestamps=sfx_cuts,
            sfx_type="whoosh",
            bgm_volume_db=-20.0
        )

        self.assertTrue(res.exists())
        self.assertGreater(res.stat().st_size, 100000)

        sr, data = wavfile.read(str(res))
        self.assertEqual(sr, 48000)
        dur = len(data) / sr
        self.assertAlmostEqual(dur, 5.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
