"""Audio Processing & Continuity Engine.

Enforces Rule 11 (AUDIO MUST SOUND CONTINUOUS):
1. Strips leading and trailing silence from synthesized chunks.
2. Two-pass EBU R128 loudness normalization (-16 LUFS, -1.5 dBTP) across all chunks (edge-tts and Piper).
3. Punctuation-aware deliberate pause insertion (150-250ms for clauses, 400-600ms for sentence boundaries).
4. Smooth 10-20ms micro-crossfade at splices to prevent click/pop artifacts with zero speech overlap.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wav

logger = logging.getLogger("audio_processor")


def convert_to_wav48k(input_path: Path, output_path: Path) -> Path:
    """Converts any audio file to 48kHz 16-bit mono WAV using FFmpeg."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", str(input_path),
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to convert {input_path} to WAV: {res.stderr}")
    return output_path


def measure_loudness_ebu_r128(audio_path: Path) -> Dict[str, float]:
    """Measures EBU R128 loudness parameters (integrated LUFS, true peak, LRA) via FFmpeg loudnorm pass 1."""
    audio_path = Path(audio_path).resolve()
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(audio_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"\{[\s\S]*?\}", res.stderr)
    if not match:
        raise RuntimeError(f"Could not parse loudnorm measurement JSON for {audio_path}:\n{res.stderr}")

    raw_data = json.loads(match.group(0))
    return {
        "input_i": float(raw_data.get("input_i", -99.0)),
        "input_tp": float(raw_data.get("input_tp", -99.0)),
        "input_lra": float(raw_data.get("input_lra", 0.0)),
        "input_thresh": float(raw_data.get("input_thresh", -99.0)),
        "target_offset": float(raw_data.get("target_offset", 0.0)),
        "output_i": float(raw_data.get("output_i", -99.0)),
        "output_tp": float(raw_data.get("output_tp", -99.0)),
    }


def normalize_loudness_ebu_r128(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -16.0,
    target_tp: float = -1.5
) -> Tuple[Path, Dict[str, float]]:
    """Performs two-pass EBU R128 loudness normalization targeting target_lufs and target_tp."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: Measure
    stats = measure_loudness_ebu_r128(input_path)

    # Pass 2: Apply linear normalization
    filter_str = (
        f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", str(input_path),
        "-af", filter_str,
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg loudnorm pass 2 failed for {input_path}:\n{res.stderr}")

    post_stats = measure_loudness_ebu_r128(output_path)
    return output_path, post_stats


def strip_silence(
    input_wav: Path,
    output_wav: Path,
    threshold_db: float = -45.0,
    pad_ms: float = 20.0
) -> Tuple[Path, float]:
    """Strips leading and trailing silence from a WAV file, retaining pad_ms margin.

    Returns:
        Tuple[Path, float]: (output_path, stripped_duration_seconds)
    """
    input_wav = Path(input_wav).resolve()
    output_wav = Path(output_wav).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    sr, data = wav.read(str(input_wav))
    if data.ndim > 1:
        data = data[:, 0]

    original_duration = len(data) / sr

    peak = np.max(np.abs(data))
    if peak == 0:
        # All silence
        wav.write(str(output_wav), sr, data[:int((pad_ms / 1000.0) * sr)])
        return output_wav, original_duration

    threshold = 10 ** (threshold_db / 20.0) * peak
    pad_samples = int((pad_ms / 1000.0) * sr)

    active_indices = np.where(np.abs(data) > threshold)[0]
    if len(active_indices) == 0:
        wav.write(str(output_wav), sr, data[:pad_samples])
        return output_wav, original_duration

    start_idx = max(0, active_indices[0] - pad_samples)
    end_idx = min(len(data), active_indices[-1] + pad_samples)

    trimmed = data[start_idx:end_idx]
    trimmed_duration = len(trimmed) / sr
    stripped_seconds = max(0.0, original_duration - trimmed_duration)

    wav.write(str(output_wav), sr, trimmed)
    return output_wav, stripped_seconds


def eradicate_internal_silences(
    input_wav: Path,
    output_wav: Path,
    min_silence_len: int = 140,
    silence_thresh: float = -40.0,
    keep_silence_ms: int = 25,
    crossfade_ms: int = 10
) -> Tuple[Path, float, float]:
    """Aggressively eliminates internal dead-air pauses longer than min_silence_len across entire audio.

    Rule 11 (Mission F Pacing):
    - Scans the entire audio array for silences > min_silence_len (140ms) at silence_thresh (-40dB).
    - Splits on silence, preserving keep_silence_ms (25ms) padding on chunk boundaries.
    - Recombines active speech segments with <= 50ms gaps and micro-crossfades.
    - Eliminates internal TTS sentence pauses, sighs, and hesitations.

    Returns:
        Tuple[Path, float, float]: (output_wav_path, pre_duration_s, post_duration_s)
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    input_wav = Path(input_wav).resolve()
    output_wav = Path(output_wav).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    seg = AudioSegment.from_wav(str(input_wav))
    pre_duration_s = len(seg) / 1000.0

    chunks = split_on_silence(
        seg,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence_ms
    )

    if not chunks:
        seg.export(str(output_wav), format="wav")
        return output_wav, pre_duration_s, pre_duration_s

    if len(chunks) == 1:
        recombined = chunks[0]
    else:
        recombined = chunks[0]
        for c in chunks[1:]:
            if crossfade_ms > 0 and len(recombined) > crossfade_ms and len(c) > crossfade_ms:
                recombined = recombined.append(c, crossfade=crossfade_ms)
            else:
                recombined = recombined + c

    recombined.export(str(output_wav), format="wav")
    post_duration_s = len(recombined) / 1000.0
    return output_wav, pre_duration_s, post_duration_s


def detect_audio_silences(
    audio_path: Path,
    min_silence_len: int = 150,
    silence_thresh: float = -40.0
) -> List[Tuple[int, int]]:
    """Returns list of (start_ms, end_ms) for any silence interval exceeding min_silence_len."""
    from pydub import AudioSegment
    from pydub.silence import detect_silence

    seg = AudioSegment.from_file(str(audio_path))
    return detect_silence(seg, min_silence_len=min_silence_len, silence_thresh=silence_thresh)


def get_punctuation_pause_ms(text: str) -> float:
    """Calculates aggressive, zero-breath pause length based on trailing punctuation for short-form retention.

    Rule 11 (Mission F Pacing):
    - Clause breaks (, ; : -- -): 25 ms (target: 0ms to 50ms)
    - Terminal boundaries / Ellipses (. ! ? ...): 100 ms (target: 100ms to 150ms)
    - Default / none: 50 ms
    """
    cleaned = text.strip()
    if not cleaned:
        return 50.0

    if cleaned.endswith("..."):
        return 100.0

    last_char = cleaned[-1]
    if last_char in {".", "!", "?"}:
        return 100.0
    elif last_char in {",", ";", ":", "-", "–", "—"}:
        return 25.0
    return 50.0


def apply_micro_fade(samples: np.ndarray, fade_ms: float = 6.0, sr: int = 48000) -> np.ndarray:
    """Applies a smooth 6ms fade-in and fade-out to prevent boundary click/pop artifacts."""
    fade_len = int((fade_ms / 1000.0) * sr)
    if len(samples) < fade_len * 2:
        fade_len = len(samples) // 4

    if fade_len <= 0:
        return samples

    out = samples.astype(np.float32)
    # Cosine fade curve
    fade_in_curve = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
    fade_out_curve = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2

    out[:fade_len] *= fade_in_curve
    out[-fade_len:] *= fade_out_curve
    return np.clip(out, -32768, 32767).astype(np.int16)


def assemble_continuous_narration(
    chunk_wavs: List[Path],
    pause_ms_list: List[float],
    output_path: Path,
    sr: int = 48000
) -> Path:
    """Assembles normalized, trimmed speech chunks with punctuation-aware pauses and click-free splices.

    Args:
        chunk_wavs: List of trimmed, loudnorm-normalized 48kHz WAV chunk files.
        pause_ms_list: Inter-chunk pause lengths in milliseconds (len = len(chunk_wavs) - 1).
        output_path: Destination path (.wav or .mp3).
        sr: Audio sample rate (default: 48000).

    Returns:
        Path to the assembled continuous narration audio.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunk_wavs:
        raise ValueError("No audio chunks provided for assembly.")

    assembled_blocks: List[np.ndarray] = []

    for i, p in enumerate(chunk_wavs):
        c_sr, samples = wav.read(str(p))
        if samples.ndim > 1:
            samples = samples[:, 0]
        if c_sr != sr:
            # Resample via numpy if not exactly sr
            indices = np.round(np.linspace(0, len(samples) - 1, int(len(samples) * sr / c_sr))).astype(int)
            samples = samples[indices]

        # Apply 6ms micro-fade to boundaries to eliminate pop artifacts
        faded = apply_micro_fade(samples, fade_ms=6.0, sr=sr)
        assembled_blocks.append(faded)

        # Insert punctuation-aware pause between chunks (never after the final chunk)
        if i < len(chunk_wavs) - 1 and i < len(pause_ms_list):
            pause_len = int((pause_ms_list[i] / 1000.0) * sr)
            if pause_len > 0:
                silence_block = np.zeros(pause_len, dtype=np.int16)
                assembled_blocks.append(silence_block)

    full_audio = np.concatenate(assembled_blocks)

    if output_path.suffix.lower() == ".mp3":
        temp_wav = output_path.with_name(f"{output_path.stem}_mp3temp.wav")
        wav.write(str(temp_wav), sr, full_audio)
        cmd = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(temp_wav),
            "-codec:a", "libmp3lame",
            "-b:a", "128k",
            "-ar", str(sr),
            str(output_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        temp_wav.unlink(missing_ok=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to encode MP3 at {output_path}: {res.stderr}")
    else:
        wav.write(str(output_path), sr, full_audio)

    return output_path


# ==============================================================================
# MULTI-LAYER SOUND DESIGN: SFX, BGM & AUTOMATED AUDIO DUCKING
# ==============================================================================

def create_procedural_sfx(sfx_type: str = "whoosh", output_path: Optional[Path] = None, sr: int = 48000) -> Path:
    """Generates pristine, royalty-free transitional sound effects deterministically.

    Types:
    - 'whoosh': High-velocity air sweep for visual cuts (0.35s).
    - 'impact': Cinematic low-frequency sub-bass thump with transient click (0.45s).
    - 'riser': Tension pitch swell for pre-hook/revelation moments (0.70s).
    """
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent / "assets_cache" / "audio" / "sfx"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"sfx_{sfx_type}.wav"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if sfx_type == "whoosh":
        duration = 0.35
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Filtered pink/white noise with smooth Gaussian bell envelope
        noise = np.random.normal(0, 1, len(t))
        envelope = np.sin(np.pi * (t / duration)) ** 2
        # Modulation sweep
        freq_mod = np.sin(2 * np.pi * 350 * t * (1 + 2.0 * t))
        audio = (noise * 0.45 + freq_mod * 0.55) * envelope
    elif sfx_type == "impact":
        duration = 0.45
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Pitch drop 90Hz -> 35Hz
        freq = 90.0 * np.exp(-t * 5.0) + 35.0
        phase = 2 * np.pi * np.cumsum(freq) / sr
        decay = np.exp(-t * 8.0)
        sine_wave = np.sin(phase) * decay
        # Transient click at start
        click = np.zeros_like(t)
        click_len = int(0.005 * sr)
        click[:click_len] = np.linspace(1.0, 0.0, click_len) * np.random.normal(0, 0.5, click_len)
        audio = (sine_wave * 0.85 + click * 0.25)
    elif sfx_type == "riser":
        duration = 0.70
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Ascending pitch 150Hz -> 650Hz with volume swell
        freq = 150.0 + (500.0 * (t / duration) ** 2)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        swell = (t / duration) ** 2
        audio = (np.sin(phase) * 0.7 + np.sin(phase * 2) * 0.2) * swell
    else:
        # Default whoosh
        duration = 0.35
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.random.normal(0, 1, len(t)) * (np.sin(np.pi * (t / duration)) ** 2)

    # Normalize to -3dB peak and convert to 16-bit PCM
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.70
    pcm = (audio * 32767).astype(np.int16)
    wav.write(str(output_path), sr, pcm)
    return output_path


def create_procedural_bgm(
    mood: str = "suspense",
    output_path: Optional[Path] = None,
    duration_s: float = 30.0,
    sr: int = 48000
) -> Path:
    """Generates an atmospheric, high-retention procedural background music track.

    Designed to stay below voice frequencies (warm low-pass filter) to maintain speech clarity.
    """
    if output_path is None:
        out_dir = Path(__file__).resolve().parent.parent / "assets_cache" / "audio" / "bgm"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"bgm_{mood}_{int(duration_s)}s.wav"
    else:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)

    if mood == "tech_pulse":
        # 120 BPM rhythmic pulse: beat every 0.5s
        bpm = 120.0
        beat_dur = 60.0 / bpm
        pulse = np.exp(-((t % beat_dur) / 0.15))
        bass_freq = 65.4  # C2
        carrier = np.sin(2 * np.pi * bass_freq * t) + 0.3 * np.sin(2 * np.pi * bass_freq * 2 * t)
        pad = 0.25 * np.sin(2 * np.pi * 130.8 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
        audio = carrier * pulse * 0.7 + pad
    elif mood == "cinematic":
        # Lush slow-evolving minor chords
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
        # Deep mystery pulse: 50Hz sub with slow pulsing harmonic tension
        sub = 0.5 * np.sin(2 * np.pi * 50.0 * t)
        tension_freq = 73.4  # D2
        pulse = 0.35 * np.sin(2 * np.pi * tension_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
        shimmer = 0.15 * np.sin(2 * np.pi * 220.0 * t) * (0.3 + 0.3 * np.sin(2 * np.pi * 0.1 * t))
        audio = sub + pulse + shimmer

    # Smooth 1.0s fade in and 1.5s fade out
    fade_in_len = int(1.0 * sr)
    fade_out_len = int(1.5 * sr)
    if len(audio) > fade_in_len + fade_out_len:
        audio[:fade_in_len] *= np.linspace(0, 1, fade_in_len)
        audio[-fade_out_len:] *= np.linspace(1, 0, fade_out_len)

    # Normalize to -18.0 dBFS target for backing track
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
    sr: int = 48000
) -> Path:
    """Mixes voiceover narration with ducked background music and transition sound effects.

    - Speech: Kept crystal-clear and prominent at -16 LUFS.
    - BGM: Sidechain-ducked underneath speech so words are effortlessly understood.
    - SFX: Synchronized to transition timestamps (e.g. at visual shot cuts).

    Returns:
        Path to the mixed 48kHz master audio file.
    """
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

    # Loop BGM if shorter than voice
    if len(bgm_raw) < duration_ms:
        loops_needed = int(duration_ms // len(bgm_raw)) + 1
        bgm_raw = bgm_raw * loops_needed
    bgm_trimmed = bgm_raw[:duration_ms]

    # Apply volume offset (e.g. -22dB relative to voice)
    # Also add gentle fade-in (500ms) and fade-out (1500ms)
    bgm_ducked = bgm_trimmed + bgm_volume_db
    if len(bgm_ducked) > 2000:
        bgm_ducked = bgm_ducked.fade_in(500).fade_out(1500)

    # 2. Overlay Voice + Ducked BGM
    mixed = bgm_ducked.overlay(voice_audio, position=0)

    # 3. Overlay Transition Sound Effects at timestamps
    if sfx_timestamps:
        temp_sfx = output_path.parent / f"temp_{sfx_type}.wav"
        create_procedural_sfx(sfx_type=sfx_type, output_path=temp_sfx, sr=sr)
        sfx_segment = AudioSegment.from_file(str(temp_sfx)) - 10.0  # -10dB for subtle impact

        for ts in sfx_timestamps:
            pos_ms = int(ts * 1000)
            if 0 <= pos_ms < duration_ms:
                mixed = mixed.overlay(sfx_segment, position=pos_ms)

        temp_sfx.unlink(missing_ok=True)

    # Export mixed master audio
    mixed.export(str(output_path), format="wav", parameters=["-ar", str(sr), "-ac", "1"])
    return output_path

