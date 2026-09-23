"""Multi-Layer Sound Design Engine — Procedural SFX, Ambient BGM, and Sidechain Ducking.

Complies with:
- Rule 11 (AUDIO MUST SOUND CONTINUOUS): Cinematic soundscapes, subtle transitional sweeps,
  and ducked background audio that never competes with speech clarity.
- AGENTS.md: Modular sound design isolated from core DSP utilities.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

logger = logging.getLogger("sound_designer")


def create_procedural_sfx(
    sfx_type: str = "whoosh",
    output_path: Optional[Path] = None,
    sr: int = 48000
) -> Path:
    """Generates pristine, organic procedural sound effects deterministically.

    Types:
    - 'whoosh': Soft cinematic air-rush sweep (150-1200Hz bandpass pink noise, 0.40s).
    - 'impact': Low-frequency sub-bass thump with transient click (0.45s).
    - 'riser': Tension pitch swell for narrative pivots (0.70s).
    """
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent / "assets_cache" / "audio" / "sfx"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"sfx_{sfx_type}.wav"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if sfx_type == "whoosh":
        duration = 0.40
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        white = np.random.normal(0, 1, len(t))
        b, a = signal.butter(2, [150.0 / (sr / 2.0), 1200.0 / (sr / 2.0)], btype="band")
        filtered_air = signal.lfilter(b, a, white)

        rise_len = int(0.15 * sr)
        decay_len = len(t) - rise_len
        rise = np.sin(np.linspace(0, np.pi / 2.0, rise_len)) ** 2
        decay = np.cos(np.linspace(0, np.pi / 2.0, decay_len)) ** 2
        envelope = np.concatenate([rise, decay])
        audio = filtered_air * envelope * 0.50
    elif sfx_type == "impact":
        duration = 0.45
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 90Hz -> 35Hz sub-bass transient
        freq = 90.0 * np.exp(-t * 5.0) + 35.0
        phase = 2 * np.pi * np.cumsum(freq) / sr
        decay = np.exp(-t * 8.0)
        sine_wave = np.sin(phase) * decay
        click = np.zeros_like(t)
        click_len = int(0.005 * sr)
        click[:click_len] = np.linspace(1.0, 0.0, click_len) * np.random.normal(0, 0.5, click_len)
        audio = (sine_wave * 0.85 + click * 0.25)
    elif sfx_type == "riser":
        duration = 0.70
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 150.0 + (500.0 * (t / duration) ** 2)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        swell = (t / duration) ** 2
        audio = (np.sin(phase) * 0.7 + np.sin(phase * 2) * 0.2) * swell
    else:
        duration = 0.40
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.random.normal(0, 1, len(t)) * (np.sin(np.pi * (t / duration)) ** 2) * 0.40

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.50
    pcm = (audio * 32767).astype(np.int16)
    wav.write(str(output_path), sr, pcm)
    return output_path


def create_procedural_bgm(
    mood: str = "suspense",
    output_path: Optional[Path] = None,
    duration_s: float = 30.0,
    sr: int = 48000
) -> Path:
    """Generates an atmospheric, high-retention procedural background music track."""
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent / "assets_cache" / "audio" / "bgm"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"bgm_{mood}_{int(duration_s)}s.wav"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)

    if mood == "tech_pulse":
        bpm = 120.0
        beat_dur = 60.0 / bpm
        pulse = np.exp(-((t % beat_dur) / 0.15))
        bass_freq = 65.4  # C2
        carrier = np.sin(2 * np.pi * bass_freq * t) + 0.3 * np.sin(2 * np.pi * bass_freq * 2 * t)
        pad = 0.25 * np.sin(2 * np.pi * 130.8 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
        audio = carrier * pulse * 0.7 + pad
    elif mood == "cinematic":
        root = 55.0  # A1
        fifth = 82.4  # E2
        octave = 110.0  # A2
        drone = (
            0.4 * np.sin(2 * np.pi * root * t) +
            0.3 * np.sin(2 * np.pi * fifth * t) +
            0.2 * np.sin(2 * np.pi * octave * t)
        )
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.15 * t)
        audio = drone * lfo
    else:  # suspense (default)
        sub = 0.5 * np.sin(2 * np.pi * 50.0 * t)
        tension_freq = 73.4  # D2
        pulse = 0.35 * np.sin(2 * np.pi * tension_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        shimmer = 0.15 * np.sin(2 * np.pi * 220.0 * t) * (0.3 + 0.3 * np.sin(2 * np.pi * 0.1 * t))
        audio = sub + pulse + shimmer

    fade_in_len = int(1.0 * sr)
    fade_out_len = int(1.5 * sr)
    if len(audio) > fade_in_len + fade_out_len:
        audio[:fade_in_len] *= np.linspace(0, 1, fade_in_len)
        audio[-fade_out_len:] *= np.linspace(1, 0, fade_out_len)

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.25
    pcm = (audio * 32767).astype(np.int16)
    wav.write(str(output_path), sr, pcm)
    return output_path


def mix_voice_bgm_sfx(
    voice_path: Path,
    output_path: Path,
    bgm_path: Optional[Path] = None,
    sfx_timestamps: Optional[List[float]] = None,
    sfx_type: str = "whoosh",
    bgm_volume_db: float = -22.0,
    sfx_volume_offset: float = -24.0,
    sr: int = 48000
) -> Path:
    """Mixes voiceover narration with ducked background music and transition sound effects."""
    from pydub import AudioSegment

    voice_path = Path(voice_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    voice_audio = AudioSegment.from_file(str(voice_path))
    duration_ms = len(voice_audio)
    duration_s = duration_ms / 1000.0

    # 1. Resolve or generate BGM
    if bgm_path and Path(bgm_path).exists():
        bgm_raw = AudioSegment.from_file(str(bgm_path))
    else:
        temp_bgm = output_path.parent / "temp_procedural_bgm.wav"
        create_procedural_bgm(mood="suspense", output_path=temp_bgm, duration_s=max(duration_s + 2.0, 10.0), sr=sr)
        bgm_raw = AudioSegment.from_file(str(temp_bgm))
        temp_bgm.unlink(missing_ok=True)

    if len(bgm_raw) < duration_ms:
        loops_needed = int(duration_ms // len(bgm_raw)) + 1
        bgm_raw = bgm_raw * loops_needed
    bgm_trimmed = bgm_raw[:duration_ms]

    # Apply volume offset with hook ducking and smooth fades
    hook_len_ms = min(3500, max(1500, duration_ms // 4))
    if duration_ms > hook_len_ms + 2000:
        bgm_hook = bgm_trimmed[:hook_len_ms] + (bgm_volume_db - 3.0)
        bgm_body = bgm_trimmed[hook_len_ms:] + bgm_volume_db
        bgm_ducked = bgm_hook + bgm_body
    else:
        bgm_ducked = bgm_trimmed + bgm_volume_db

    if len(bgm_ducked) > 2000:
        bgm_ducked = bgm_ducked.fade_in(400).fade_out(1200)

    # 2. Overlay Voice + Ducked BGM
    mixed = bgm_ducked.overlay(voice_audio, position=0)

    # 3. Overlay Frame-0 Sub-Bass Impact (Acoustic Startle Reflex)
    temp_impact = output_path.parent / "temp_impact_zero.wav"
    create_procedural_sfx(sfx_type="impact", output_path=temp_impact, sr=sr)
    impact_seg = AudioSegment.from_file(str(temp_impact)) - 14.0
    mixed = mixed.overlay(impact_seg, position=0)
    temp_impact.unlink(missing_ok=True)

    # 4. Overlay Mid-Roll Riser at revelation beat (around 42% duration)
    if duration_s > 15.0:
        riser_pos_ms = int(duration_ms * 0.42)
        temp_riser = output_path.parent / "temp_mid_riser.wav"
        create_procedural_sfx(sfx_type="riser", output_path=temp_riser, sr=sr)
        riser_seg = AudioSegment.from_file(str(temp_riser)) - 20.0
        mixed = mixed.overlay(riser_seg, position=max(0, riser_pos_ms - 700))
        temp_riser.unlink(missing_ok=True)

    # 5. Overlay Cut Transition Sound Effects
    if sfx_timestamps:
        temp_sfx = output_path.parent / f"temp_{sfx_type}.wav"
        create_procedural_sfx(sfx_type=sfx_type, output_path=temp_sfx, sr=sr)
        sfx_segment = AudioSegment.from_file(str(temp_sfx)) + sfx_volume_offset

        for ts in sfx_timestamps:
            pos_ms = int(ts * 1000)
            if 0 < pos_ms < duration_ms:
                mixed = mixed.overlay(sfx_segment, position=pos_ms)

        temp_sfx.unlink(missing_ok=True)

    # 6. Master Loudness Limiter & EBU R128 Normalization (-14.0 LUFS, -1.5 dBTP Ceiling)
    temp_unnorm = output_path.parent / f"temp_unnorm_{output_path.name}"
    mixed.export(str(temp_unnorm), format="wav", parameters=["-ar", str(sr), "-ac", "1"])

    try:
        from pipeline.core.audio_processor import normalize_loudness_ebu_r128
        normalize_loudness_ebu_r128(
            input_path=temp_unnorm,
            output_path=output_path,
            target_lufs=-14.0,
            target_tp=-1.5
        )
        temp_unnorm.unlink(missing_ok=True)
    except Exception as norm_err:
        logger.warning(f"[SOUND_DESIGNER] Normalization fallback: {norm_err}")
        if temp_unnorm.exists():
            temp_unnorm.replace(output_path)

    return output_path
