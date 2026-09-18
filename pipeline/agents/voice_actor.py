"""VoiceActor Agent — High-Retention Cadence, Dynamic SSML, and EBU R128 Normalization.

Complies with:
- Rule 2 (NO SILENT CRASHES): Bounded retries and fallback to local Piper TTS.
- Rule 5 (edge-tts IS UNOFFICIAL): Resilient sentence/clause chunking, exponential backoff, Piper fallback.
- Rule 11 (AUDIO MUST SOUND CONTINUOUS): Expressive pacing, deliberate punctuation pauses,
  and EBU R128 (-14.0 LUFS / -1.5 dBTP) normalization.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import edge_tts
from pydub import AudioSegment, effects

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.audio_processor import (
    convert_to_wav48k,
    normalize_loudness_ebu_r128,
    strip_silence,
)

DEFAULT_PIPER_MODEL = ROOT_DIR / "pipeline" / "models" / "piper" / "en_US-lessac-low.onnx"
DEFAULT_PIPER_CONFIG = ROOT_DIR / "pipeline" / "models" / "piper" / "en_US-lessac-low.onnx.json"

logger = logging.getLogger("voice_actor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# High-retention narrative-driven voices
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
FALLBACK_VOICE = "en-US-BrianMultilingualNeural"


def generate_ssml(
    text: str,
    is_hook: bool = False,
    voice: str = DEFAULT_VOICE
) -> str:
    """Wraps text in dynamic SSML with prosody control and punctuation break tags.

    - Hooks inject urgency: rate="+8%", pitch="-2Hz".
    - Standard narration: rate="+4%", pitch="+0Hz".
    - Commas, semicolons, em-dashes: <break time="180ms"/>
    - Full stops, exclamation marks, question marks: <break time="320ms"/>
    """
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    rate = "+8%" if is_hook else "+4%"
    pitch = "-2Hz" if is_hook else "+0Hz"

    # Insert deliberate break tags after punctuation boundaries
    # Commas, semicolons, em-dashes
    formatted = re.sub(r'([,;—\-])\s*', r'\1 <break time="180ms"/> ', cleaned_text)
    # Sentence boundaries (. ! ?)
    formatted = re.sub(r'([.!?])\s*', r'\1 <break time="320ms"/> ', formatted)
    formatted = re.sub(r'\s+', ' ', formatted).strip()

    ssml = (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">'
        f'<prosody rate="{rate}" pitch="{pitch}">'
        f'{formatted}'
        f'</prosody>'
        f'</voice>'
        f'</speak>'
    )
    return ssml


def parse_ssml_tokens(ssml_or_text: str) -> Tuple[List[Tuple[str, int]], str, str]:
    """Parses SSML string into spoken sub-phrases and millisecond pause intervals.

    Returns:
        Tuple[List[Tuple[str, int]], rate_str, pitch_str]:
        A list of (phrase_text, pause_after_ms), and the prosody rate/pitch parameters.
    """
    # Extract prosody attributes if present
    rate = "+4%"
    pitch = "+0Hz"
    rate_match = re.search(r'rate="([^"]+)"', ssml_or_text)
    if rate_match:
        rate = rate_match.group(1)
    pitch_match = re.search(r'pitch="([^"]+)"', ssml_or_text)
    if pitch_match:
        pitch = pitch_match.group(1)

    # Strip speak, voice, prosody tags to focus on the text and break tags
    inner_text = re.sub(r'</?(?:speak|voice|prosody)[^>]*>', '', ssml_or_text).strip()

    # Split on <break time="(\\d+)ms"\\s*/>
    tokens = re.split(r'<break\s+time="(\d+)ms"\s*/>', inner_text)
    phrases_with_breaks: List[Tuple[str, int]] = []

    i = 0
    while i < len(tokens):
        phrase = tokens[i].strip()
        # Clean any remaining stray XML
        phrase = re.sub(r'<[^>]+>', '', phrase).strip()
        pause_ms = 0
        if i + 1 < len(tokens) and tokens[i + 1].isdigit():
            pause_ms = int(tokens[i + 1])
        if phrase:
            phrases_with_breaks.append((phrase, pause_ms))
        elif pause_ms > 0 and phrases_with_breaks:
            # Attach pause to previous phrase if current is empty
            last_p, last_pause = phrases_with_breaks[-1]
            phrases_with_breaks[-1] = (last_p, last_pause + pause_ms)
        i += 2

    return phrases_with_breaks, rate, pitch


def synthesize_phrase_edgetts(
    phrase: str,
    voice: str,
    rate: str,
    pitch: str,
    output_path: Path,
    max_attempts: int = 3,
    base_backoff: float = 1.5
) -> Path:
    """Synthesizes an individual clean phrase chunk via Edge-TTS."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None

    for attempt in range(1, max_attempts + 1):
        try:
            communicate = edge_tts.Communicate(
                phrase,
                voice=voice,
                rate=rate,
                pitch=pitch
            )
            asyncio.run(communicate.save(str(output_path)))
            if output_path.exists() and output_path.stat().st_size > 100:
                return output_path
            raise ValueError("Edge-TTS generated empty or corrupt audio file")
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                sleep_time = base_backoff ** attempt
                logger.warning(f"Edge-TTS attempt {attempt} failed ({e}). Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    raise last_err or RuntimeError(f"Edge-TTS synthesis failed after {max_attempts} attempts")


def synthesize_piper_fallback(
    text: str,
    output_path: Path,
    model_path: Optional[Path] = None,
    length_scale: float = 0.85
) -> Path:
    """Synthesizes narration locally using Piper TTS when Edge-TTS is unavailable (Rule 5)."""
    import wave
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    m_path = Path(model_path) if model_path else DEFAULT_PIPER_MODEL
    c_path = m_path.with_suffix(".onnx.json") if m_path.suffix == ".onnx" else Path(str(m_path) + ".json")

    if not m_path.exists():
        raise FileNotFoundError(f"Piper model not found at {m_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(m_path, config_path=c_path, use_cuda=False)
    syn_config = SynthesisConfig(length_scale=length_scale)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        # Clean text from break tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        for audio_chunk in voice.synthesize(clean_text, syn_config=syn_config):
            wav_file.writeframes(audio_chunk.audio_int16_bytes)

    return output_path


def synthesize_segment(
    text: str,
    output_path: Path,
    voice: str = DEFAULT_VOICE,
    is_hook: bool = False,
    temp_dir: Optional[Path] = None,
    mock_edge_failure: bool = False,
    target_lufs: float = -14.0,
    target_tp: float = -1.5
) -> Tuple[Path, str, float, float]:
    """Main VoiceActor synthesis pipeline for a script segment.

    1. Formats text with dynamic SSML (<prosody rate/pitch> and <break time="..."/>).
    2. Synthesizes phrases through Edge-TTS (with fallback to Piper).
    3. Injects precise audio silence buffers (180ms for commas/em-dashes, 320ms for sentences).
    4. Applies true-peak limiting at -1.5 dBFS via pydub.effects.normalize.
    5. Normalizes to EBU R128 (-14.0 LUFS) at 48kHz mono WAV.

    Returns:
        Tuple[Path, str, float, float]:
        (processed_wav_path, engine_used, speech_duration_s, trailing_pause_s)
    """
    output_path = Path(output_path).resolve()
    t_dir = Path(temp_dir) if temp_dir else output_path.parent / f"va_temp_{abs(hash(text)) % 100000}"
    t_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate standard SSML
    ssml_text = generate_ssml(text, is_hook=is_hook, voice=voice)
    logger.debug(f"[VOICE_ACTOR] Generated SSML: {ssml_text}")

    # 2. Parse phrases and break intervals
    phrases_with_breaks, rate, pitch = parse_ssml_tokens(ssml_text)
    combined_audio = AudioSegment.empty()
    engine_used = "edge-tts"

    if not mock_edge_failure:
        try:
            for idx, (phrase, pause_ms) in enumerate(phrases_with_breaks):
                chunk_file = t_dir / f"chunk_{idx}.mp3"
                synthesize_phrase_edgetts(
                    phrase=phrase,
                    voice=voice,
                    rate=rate,
                    pitch=pitch,
                    output_path=chunk_file
                )
                seg_audio = AudioSegment.from_file(str(chunk_file))
                combined_audio += seg_audio

                # Inject deliberate SSML punctuation break
                if pause_ms > 0:
                    combined_audio += AudioSegment.silent(duration=pause_ms)

        except Exception as err:
            logger.warning(f"[VOICE_ACTOR] Edge-TTS synthesis failed ({err}). Falling back to local Piper...")
            combined_audio = None
            engine_used = "piper-fallback"
    else:
        combined_audio = None
        engine_used = "piper-fallback"

    # Fallback to local Piper TTS
    if combined_audio is None or len(combined_audio) < 100:
        piper_wav = t_dir / "piper_raw.wav"
        synthesize_piper_fallback(text, piper_wav, length_scale=0.85 if is_hook else 0.90)
        combined_audio = AudioSegment.from_file(str(piper_wav))
        engine_used = "piper-fallback"

    # 3. Peak normalization with -1.5 dBFS true-peak headroom via pydub
    pre_norm_wav = t_dir / "pre_norm.wav"
    peak_normalized = effects.normalize(combined_audio, headroom=abs(target_tp))
    peak_normalized.export(str(pre_norm_wav), format="wav")

    # 4. Standardize to 48kHz mono
    wav_48k = t_dir / "wav_48k.wav"
    convert_to_wav48k(pre_norm_wav, wav_48k)

    # 5. Two-pass EBU R128 loudness normalization (-14.0 LUFS)
    norm_wav = t_dir / "ebu_norm.wav"
    final_wav, stats = normalize_loudness_ebu_r128(
        input_path=wav_48k,
        output_path=output_path,
        target_lufs=target_lufs,
        target_tp=target_tp
    )

    # 6. Measure clean duration and trailing pause
    speech_duration = len(peak_normalized) / 1000.0
    trailing_pause_ms = phrases_with_breaks[-1][1] if phrases_with_breaks else 320
    trailing_pause_s = trailing_pause_ms / 1000.0

    logger.info(
        f"[VOICE_ACTOR] Audio ready ({engine_used}, voice={voice}): dur={speech_duration:.2f}s, "
        f"measured_LUFS={stats.get('output_i', target_lufs):.1f}, TP={stats.get('output_tp', target_tp):.1f}dBTP"
    )
    return final_wav, engine_used, speech_duration, trailing_pause_s
