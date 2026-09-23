"""Standalone CPU-only Kokoro TTS Synthesis Worker.

Per Rule 1 (GPU ISOLATION) & Rule 2 (NO SILENT CRASHES):
- Runs strictly on CPU (`device="cpu"`), zero GPU lock requirement, zero VRAM usage.
- High-fidelity 24kHz/48kHz neural voice synthesis across multiple English voices.
- Normalizes output to -14.0 LUFS / -1.5 dBTP standard.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.audio_processor import convert_to_wav48k, normalize_loudness_ebu_r128


DEFAULT_VOICE = "af_bella*0.6+bf_isabella*0.4"
KOKORO_VOICES = {
    # Blends / Fusions
    "af_bella*0.6+bf_isabella*0.4": "Bella 60% + Isabella 40% (Emotive Gamer / Speedrunner - Default Fusion 4)",
    "af_bella*0.7+bf_emma*0.3": "Bella 70% + Maisie/Emma 30% (Crisp American with British Lilt)",
    "af_bella*0.5+bf_emma*0.5": "Bella 50% + Maisie/Emma 50% (Mid-Atlantic Gamer Streamer)",
    # American English
    "af_heart": "American Female - Warm & Expressive",
    "af_bella": "American Female - Bright & Dynamic",
    "af_sarah": "American Female - Professional & Narrative",
    "af_nicole": "American Female - Conversational",
    "am_adam": "American Male - Energetic & Punchy",
    "am_michael": "American Male - Deep & Authoritative",
    "am_eric": "American Male - Fast-Paced & Tech",
    # British English
    "bf_emma": "British Female - Clear & Sophisticated",
    "bf_isabella": "British Female - Dramatic & Narrative",
    "bm_george": "British Male - Crisp & Cinematic",
    "bm_lewis": "British Male - Storyteller & Intense",
}


def parse_voice_blend(voice_str: str, pipeline) -> Union[str, torch.Tensor]:
    """Parses single voice name or blended voice specification into a Kokoro voice tensor."""
    import re
    # Formats:
    # 1. "af_bella*0.6+bf_isabella*0.4"
    # 2. "fusion:af_bella:0.6:bf_isabella:0.4"
    # 3. "blend:af_bella:0.6:bf_isabella:0.4"
    if "fusion:" in voice_str or "blend:" in voice_str:
        parts = voice_str.split(":")
        v1_name = parts[1]
        w1 = float(parts[2])
        v2_name = parts[3]
        w2 = float(parts[4])
        v1 = pipeline.load_voice(v1_name)
        v2 = pipeline.load_voice(v2_name)
        min_len = min(v1.shape[0], v2.shape[0])
        return (w1 * v1[:min_len]) + (w2 * v2[:min_len])

    if "*" in voice_str and "+" in voice_str:
        match = re.match(r'([a-zA-Z0-9_]+)\*([0-9.]+)\+([a-zA-Z0-9_]+)\*([0-9.]+)', voice_str.strip())
        if match:
            v1_name, w1_str, v2_name, w2_str = match.groups()
            w1 = float(w1_str)
            w2 = float(w2_str)
            v1 = pipeline.load_voice(v1_name)
            v2 = pipeline.load_voice(v2_name)
            min_len = min(v1.shape[0], v2.shape[0])
            return (w1 * v1[:min_len]) + (w2 * v2[:min_len])

    return voice_str


def synthesize_kokoro_speech(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.18,
    target_lufs: float = -14.0,
    target_tp: float = -1.5,
    lang_code: str = "a"
) -> Path:
    """Synthesizes speech on CPU using Kokoro neural pipeline with -14 LUFS normalization.
    
    Supports single voices ('af_bella') or blended styles ('af_bella*0.6+bf_isabella*0.4').
    
    Args:
        text: Input narration text.
        output_path: Destination WAV file path.
        voice: Kokoro voice identifier or blend formula.
        speed: Speech rate multiplier (1.0 = normal, 1.18 = high retention).
        target_lufs: Target EBU R128 integrated loudness (standardized to -14.0 LUFS).
        target_tp: True Peak ceiling (standardized to -1.5 dBTP).
        lang_code: Language code ('a' = American English, 'b' = British English).
    """
    from kokoro import KPipeline

    start_time = time.time()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve language code based on primary voice prefix if not overridden
    if voice.startswith("b") and not ("af_" in voice or "*" in voice):
        lang_code = "b"
    else:
        lang_code = "a"

    print(f"[KOKORO_WORKER] Initializing Kokoro on CPU (voice={voice}, lang={lang_code}, speed={speed:.2f})...", flush=True)
    pipeline = KPipeline(lang_code=lang_code, device="cpu")

    voice_arg = parse_voice_blend(voice, pipeline)

    generator = pipeline(text, voice=voice_arg, speed=speed, split_pattern=r'\n+')
    audio_chunks = []
    for gs, ps, audio in generator:
        if audio is not None and len(audio) > 0:
            audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro synthesis yielded no audio frames.")

    full_audio = np.concatenate(audio_chunks)
    raw_wav = output_path.parent / f"raw_kokoro_{output_path.stem}.wav"
    sf.write(str(raw_wav), full_audio, 24000)

    # Standardize to 48kHz mono
    wav_48k = output_path.parent / f"wav48k_kokoro_{output_path.stem}.wav"
    convert_to_wav48k(raw_wav, wav_48k)

    # EBU R128 Normalization to -14.0 LUFS
    final_wav, stats = normalize_loudness_ebu_r128(
        input_path=wav_48k,
        output_path=output_path,
        target_lufs=target_lufs,
        target_tp=target_tp
    )

    # Cleanup temp intermediates
    raw_wav.unlink(missing_ok=True)
    wav_48k.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    duration_s = len(full_audio) / 24000.0
    print(
        f"[KOKORO_WORKER] Synthesis complete in {elapsed:.2f}s -> {output_path.name} "
        f"(dur={duration_s:.2f}s, LUFS={stats.get('output_i', target_lufs):.1f}, "
        f"TP={stats.get('output_tp', target_tp):.1f} dBTP)",
        flush=True
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Standalone CPU-only Kokoro TTS Synthesis Worker")
    parser.add_argument("--text", type=str, required=True, help="Input text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file path")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help=f"Kokoro voice ID (default: {DEFAULT_VOICE})")
    parser.add_argument("--speed", type=float, default=1.12, help="Speech rate multiplier (default: 1.12)")
    parser.add_argument("--target-lufs", type=float, default=-14.0, help="Target LUFS (default: -14.0)")
    args = parser.parse_args()

    synthesize_kokoro_speech(
        text=args.text,
        output_path=Path(args.output),
        voice=args.voice,
        speed=args.speed,
        target_lufs=args.target_lufs
    )


if __name__ == "__main__":
    main()
