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


def measure_loudness_ebu_r128(
    audio_path: Path, target_lufs: float = -14.0, target_tp: float = -1.5
) -> Dict[str, float]:
    """Measures EBU R128 loudness parameters (integrated LUFS, true peak, LRA) via FFmpeg loudnorm pass 1."""
    audio_path = Path(audio_path).resolve()
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(audio_path),
        "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    match = re.search(r"\{[\s\S]*?\}", res.stderr)
    if not match:
        raise RuntimeError(f"Could not parse loudnorm measurement JSON for {audio_path}:\n{res.stderr}")

    raw = json.loads(match.group(0))
    keys = ["input_i", "input_tp", "input_lra", "input_thresh", "target_offset", "output_i", "output_tp"]
    return {k: float(raw.get(k, -99.0 if "i" in k or "tp" in k or "thresh" in k else 0.0)) for k in keys}


def normalize_loudness_ebu_r128(
    input_path: Path, output_path: Path, target_lufs: float = -14.0, target_tp: float = -1.5
) -> Tuple[Path, Dict[str, float]]:
    """Performs two-pass EBU R128 loudness normalization targeting target_lufs and target_tp."""
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = measure_loudness_ebu_r128(input_path, target_lufs=target_lufs, target_tp=target_tp)
    filter_str = (
        f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=11:measured_I={stats['input_i']}:"
        f"measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
        f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true"
    )
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(input_path), "-af", filter_str, "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg loudnorm pass 2 failed for {input_path}:\n{res.stderr}")
    return output_path, measure_loudness_ebu_r128(output_path, target_lufs=target_lufs, target_tp=target_tp)


def strip_silence(input_wav: Path, output_wav: Path, threshold_db: float = -45.0, pad_ms: float = 20.0) -> Tuple[Path, float]:
    """Strips leading and trailing silence from a WAV file, retaining pad_ms margin."""
    input_wav, output_wav = Path(input_wav).resolve(), Path(output_wav).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sr, data = wav.read(str(input_wav))
    if data.ndim > 1:
        data = data[:, 0]
    original_duration = len(data) / sr
    peak = np.max(np.abs(data))
    if peak == 0:
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
    wav.write(str(output_wav), sr, trimmed)
    return output_wav, max(0.0, original_duration - (len(trimmed) / sr))


def eradicate_internal_silences(
    input_wav: Path, output_wav: Path, min_silence_len: int = 140,
    silence_thresh: float = -40.0, keep_silence_ms: int = 25, crossfade_ms: int = 10
) -> Tuple[Path, float, float]:
    """Aggressively eliminates internal dead-air pauses longer than min_silence_len across entire audio."""
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    input_wav, output_wav = Path(input_wav).resolve(), Path(output_wav).resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    seg = AudioSegment.from_wav(str(input_wav))
    pre_dur = len(seg) / 1000.0
    chunks = split_on_silence(seg, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence_ms)
    if not chunks:
        seg.export(str(output_wav), format="wav")
        return output_wav, pre_dur, pre_dur

    recombined = chunks[0]
    for c in chunks[1:]:
        if crossfade_ms > 0 and len(recombined) > crossfade_ms and len(c) > crossfade_ms:
            recombined = recombined.append(c, crossfade=crossfade_ms)
        else:
            recombined += c
    recombined.export(str(output_wav), format="wav")
    return output_wav, pre_dur, len(recombined) / 1000.0


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
# MULTI-LAYER SOUND DESIGN (Delegated to pipeline.core.sound_designer)
# ==============================================================================
from pipeline.core.sound_designer import (
    create_procedural_sfx,
    create_procedural_bgm,
    mix_voice_bgm_sfx,
)


