"""Director Agent: Resilient TTS narration, asset caching, and GPU-lock orchestration.

Per Rule 1 (GPU MEMORY):
- All GPU-bound workers (Whisper, ComfyUI, Render) MUST be called through `run_gpu_worker()`.
- Direct execution without acquiring `gpu_lock` is structurally prohibited.

Per Rule 5 (edge-tts IS UNOFFICIAL):
- Text is chunked into sentence-level pieces under ~300 characters.
- Jittered rate-limiting (min delay 1.5s + jitter) between calls.
- Exponential backoff per chunk (base 2s, up to 3 attempts).
- Per-chunk fallback to local Piper TTS if edge-tts fails (minimal quality loss).
- Concatenates audio chunks in order into a continuous master narration track.

Per Rule 2 & Rule 8:
- Pexels client checks local cache first to avoid duplicate API requests.
- Rate-limited and bounded retries across all external services.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import scipy.io.wavfile as wavfile

import dotenv
import requests

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.gpu_lock import gpu_lock, GPULockError, preflight_vram_check
from pipeline.core.schema import Script, ScriptSegment
from pipeline.core.audio_processor import (
    convert_to_wav48k,
    measure_loudness_ebu_r128,
    normalize_loudness_ebu_r128,
    strip_silence,
    eradicate_internal_silences,
    get_punctuation_pause_ms,
    assemble_continuous_narration
)
from pipeline.core.stage_verifier import (
    verify_voice_actor,
    verify_art_director,
    verify_editor,
    estimate_spoken_length,
    run_stage_with_verification,
    StageVerificationError
)

# Configure module logger
logger = logging.getLogger("director")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEFAULT_ASSETS_CACHE = ROOT_DIR / "pipeline" / "assets_cache"
DEFAULT_PIPER_MODEL = ROOT_DIR / "pipeline" / "models" / "piper" / "en_US-lessac-low.onnx"
DEFAULT_PIPER_CONFIG = ROOT_DIR / "pipeline" / "models" / "piper" / "en_US-lessac-low.onnx.json"
WORKERS_DIR = ROOT_DIR / "pipeline" / "workers"


# ==============================================================================
# 1. TEXT CHUNKING & RESILIENT TTS (Rule 5 Compliance)
# ==============================================================================

def chunk_text_by_sentence(text: str, max_chars: int = 300) -> List[str]:
    """Splits text into sentence-level pieces, each under max_chars."""
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks = []
    for sentence in sentences:
        if len(sentence) <= max_chars:
            chunks.append(sentence)
        else:
            # Sentence exceeds max_chars; break down into sub-chunks on comma or words
            words = sentence.split()
            sub_chunk = []
            sub_len = 0
            for w in words:
                if sub_len + len(w) + 1 > max_chars and sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = [w]
                    sub_len = len(w)
                else:
                    sub_chunk.append(w)
                    sub_len += len(w) + 1
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

    return chunks


def synthesize_chunk_piper(
    text: str,
    output_path: Path,
    model_path: Optional[Path] = None,
    length_scale: float = 0.85
) -> Path:
    """Synthesizes an audio chunk locally using Piper TTS with accelerated length_scale."""
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    m_path = Path(model_path) if model_path else DEFAULT_PIPER_MODEL
    c_path = m_path.with_suffix(".onnx.json") if m_path.suffix == ".onnx" else Path(str(m_path) + ".json")

    if not m_path.exists():
        raise FileNotFoundError(f"Piper model not found at {m_path}. Please download voice model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(m_path, config_path=c_path, use_cuda=False)
    syn_config = SynthesisConfig(length_scale=length_scale)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        for audio_chunk in voice.synthesize(text, syn_config=syn_config):
            wav_file.writeframes(audio_chunk.audio_int16_bytes)

    return output_path


def synthesize_chunk_edgetts(
    text: str,
    voice: str,
    output_path: Path,
    max_attempts: int = 3,
    base_backoff: float = 2.0,
    rate: str = "+15%"
) -> Path:
    """Synthesizes an audio chunk via edge-tts with exponential backoff and rate speedup."""
    import edge_tts

    last_error = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            asyncio.run(communicate.save(str(output_path)))

            if not output_path.exists() or output_path.stat().st_size < 100:
                raise ValueError("edge-tts generated empty or truncated audio file")

            return output_path
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                sleep_time = base_backoff ** attempt
                logger.info(f"edge-tts attempt {attempt} failed ({e}). Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    raise last_error or RuntimeError("edge-tts synthesis failed all attempts")


def synthesize_segment_narration(
    segment_text: str,
    voice: str = "en-US-ChristopherNeural",
    output_path: Optional[Path] = None,
    temp_dir: Optional[Path] = None,
    mock_edge_failure: bool = False
) -> Tuple[Path, str, float, float]:
    """Synthesizes an entire script segment's voiceover.

    Rule 11 Compliance (Mission F Pacing):
    - Attempts synthesis of the FULL segment in a single edge-tts call first (natural prosody).
    - Injects rate="+15%" (or Piper length_scale=0.85) for fast-paced short-form cadence.
    - Two-pass EBU R128 loudness normalization to -16.0 LUFS / -1.5 dBTP.
    - Aggressive internal silence eradication: crushes pauses > 150ms to <= 50ms.
    - Zero-breath punctuation pauses (30ms clauses, 120ms sentences).
    - Measures exact clean speech duration and punctuation-aware pause.

    Returns:
        Tuple[Path, str, float, float]:
        (processed_chunk_wav, source_engine, speech_duration_s, pause_duration_s)
    """
    t_dir = Path(temp_dir) if temp_dir else DEFAULT_ASSETS_CACHE / "tts_temp"
    t_dir.mkdir(parents=True, exist_ok=True)
    out_wav = Path(output_path) if output_path else t_dir / "segment_processed.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    raw_edge_target = t_dir / f"seg_raw_{abs(hash(segment_text)) % 100000}_edge.mp3"
    raw_piper_target = t_dir / f"seg_raw_{abs(hash(segment_text)) % 100000}_piper.wav"

    source = "edge-tts"
    raw_path = None

    if not mock_edge_failure:
        # 1. Attempt synthesis of the FULL narration in a single edge-tts call (+15% rate)
        try:
            synthesize_chunk_edgetts(
                text=segment_text,
                voice=voice,
                output_path=raw_edge_target,
                max_attempts=3,
                base_backoff=2.0,
                rate="+15%"
            )
            raw_path = raw_edge_target
            source = "edge-tts"
        except Exception as e:
            logger.warning(f"Single-call edge-tts failed for segment: {e}. Trying adaptive sub-chunks...")
            # Adaptive fallback: split into 2-3 sentence chunks
            sub_chunks = chunk_text_by_sentence(segment_text, max_chars=400)
            if len(sub_chunks) > 1:
                try:
                    sub_wavs = []
                    for s_idx, sc in enumerate(sub_chunks):
                        sc_path = t_dir / f"sub_{s_idx}_{abs(hash(sc)) % 10000}.mp3"
                        synthesize_chunk_edgetts(sc, voice, sc_path, max_attempts=2, rate="+15%")
                        sc_wav = convert_to_wav48k(sc_path, t_dir / f"sub_{s_idx}.wav")
                        sub_wavs.append(sc_wav)
                    # Merge sub-chunks
                    sub_merged = t_dir / f"sub_merged_{abs(hash(segment_text)) % 10000}.wav"
                    assemble_continuous_narration(sub_wavs, [30.0] * (len(sub_wavs) - 1), sub_merged)
                    raw_path = sub_merged
                    source = "edge-tts-adaptive"
                except Exception as sub_err:
                    logger.warning(f"Adaptive edge-tts failed: {sub_err}. Falling back to local Piper.")

    # Fallback to local Piper if edge-tts failed or mocked
    if raw_path is None:
        logger.warning("[FALLBACK] Synthesizing segment locally using Piper TTS.")
        synthesize_chunk_piper(segment_text, raw_piper_target, length_scale=0.85)
        raw_path = raw_piper_target
        source = "piper-fallback"

    # 2. Standardize to 48kHz WAV
    wav_48k = convert_to_wav48k(raw_path, t_dir / f"seg_48k_{abs(hash(segment_text)) % 100000}.wav")

    # 3. Two-pass EBU R128 loudness normalization (-16.0 LUFS, -1.5 dBTP)
    norm_wav, stats = normalize_loudness_ebu_r128(wav_48k, t_dir / f"seg_norm_{abs(hash(segment_text)) % 100000}.wav", target_lufs=-16.0, target_tp=-1.5)

    # 4. Aggressive Internal Silence Eradication (Rule 11 - Mission F)
    crushed_wav = t_dir / f"seg_crushed_{abs(hash(segment_text)) % 100000}.wav"
    _, pre_dur, post_dur = eradicate_internal_silences(
        input_wav=norm_wav,
        output_wav=crushed_wav,
        min_silence_len=150,
        silence_thresh=-40.0,
        keep_silence_ms=25,
        crossfade_ms=10
    )

    # 5. Silence trimming on remaining head/tail edges (down to 5ms pad)
    trimmed_wav, stripped_s = strip_silence(crushed_wav, out_wav, threshold_db=-45.0, pad_ms=5.0)

    # 6. Measure exact speech duration strictly on the compressed audio array
    sr, samples = wavfile.read(str(trimmed_wav))
    speech_duration = len(samples) / float(sr)

    # 7. Compute zero-breath, punctuation-aware trailing pause length (25-75ms)
    raw_pause_ms = get_punctuation_pause_ms(segment_text)
    pause_ms = min(raw_pause_ms, 75.0)
    pause_duration = pause_ms / 1000.0

    logger.info(
        f"Segment audio ready ({source}): speech={speech_duration:.2f}s (crushed from {pre_dur:.2f}s), "
        f"pause={pause_ms:.0f}ms, LUFS={stats.get('input_i', -16.0):.1f} -> -16.0 (stripped {stripped_s:.2f}s dead-air)"
    )
    return trimmed_wav, source, speech_duration, pause_duration


def synthesize_narration(
    text: str,
    voice: str = "en-US-ChristopherNeural",
    output_path: Optional[Path] = None,
    min_delay: float = 1.5,
    jitter_range: Tuple[float, float] = (0.2, 0.8),
    temp_dir: Optional[Path] = None,
    mock_edge_failure_chunk: Optional[int] = None
) -> Tuple[Path, List[str]]:
    """Synthesizes narration with sentence-level chunking, EBU R128 loudnorm, silence trimming, and Piper fallback.

    Maintains backward compatibility with unit tests while enforcing Rule 11.
    """
    chunks = chunk_text_by_sentence(text, max_chars=300)
    if not chunks:
        raise ValueError("Cannot synthesize empty text narration.")

    out_path = Path(output_path) if output_path else DEFAULT_ASSETS_CACHE / "master_narration.mp3"
    t_dir = Path(temp_dir) if temp_dir else DEFAULT_ASSETS_CACHE / "tts_chunks"
    t_dir.mkdir(parents=True, exist_ok=True)

    processed_wavs: List[Path] = []
    chunk_sources: List[str] = []
    pauses_ms: List[float] = []

    logger.info(f"Synthesizing narration ({len(chunks)} sentence chunks, target voice: {voice})...")

    for i, chunk in enumerate(chunks):
        if i > 0:
            delay = min_delay + random.uniform(jitter_range[0], jitter_range[1])
            time.sleep(delay)

        chunk_edge_target = t_dir / f"chunk_{i:03d}_edge.mp3"
        chunk_piper_target = t_dir / f"chunk_{i:03d}_piper.wav"

        use_fallback = (mock_edge_failure_chunk is not None and i == mock_edge_failure_chunk)
        raw_target = None
        source = "edge-tts"

        if not use_fallback:
            try:
                synthesize_chunk_edgetts(
                    text=chunk,
                    voice=voice,
                    output_path=chunk_edge_target,
                    max_attempts=3,
                    base_backoff=2.0
                )
                raw_target = chunk_edge_target
                source = "edge-tts"
                logger.info(f"Chunk {i+1}/{len(chunks)} synthesized successfully via edge-tts.")
            except Exception as edge_err:
                logger.warning(
                    f"edge-tts failed for chunk {i+1}: {edge_err}. Falling back to local Piper TTS."
                )

        if raw_target is None:
            logger.warning(f"[FALLBACK] Synthesizing chunk {i+1}/{len(chunks)} locally using Piper TTS.")
            synthesize_chunk_piper(chunk, chunk_piper_target)
            raw_target = chunk_piper_target
            source = "piper-fallback"

        chunk_sources.append(source)

        # Standardize to 48kHz WAV
        wav_48 = convert_to_wav48k(raw_target, t_dir / f"chunk_{i:03d}_48k.wav")

        # Two-pass EBU R128 loudness normalization to -16 LUFS
        norm_wav, stats = normalize_loudness_ebu_r128(wav_48, t_dir / f"chunk_{i:03d}_norm.wav", target_lufs=-16.0)

        # Strip leading/trailing dead air
        trim_wav, _ = strip_silence(norm_wav, t_dir / f"chunk_{i:03d}_trim.wav", threshold_db=-45.0, pad_ms=20.0)
        processed_wavs.append(trim_wav)

        if i < len(chunks) - 1:
            pauses_ms.append(get_punctuation_pause_ms(chunk))

    # Assemble with punctuation-aware pauses and 15ms click-prevention crossfades
    logger.info(f"Assembling {len(processed_wavs)} audio chunks into master track via assemble_continuous_narration...")
    assemble_continuous_narration(
        chunk_wavs=processed_wavs,
        pause_ms_list=pauses_ms,
        output_path=out_path,
        sr=48000
    )

    logger.info(f"Master narration audio created at {out_path} ({len(chunks)} chunks, sources: {chunk_sources}).")
    return out_path, chunk_sources


# ==============================================================================
# 2. PEXELS ASSET CACHING & FETCHING (Rate Limiter + Cache Hit Avoidance)
# ==============================================================================

def fetch_visual_asset(
    visual_query: str,
    duration: float = 4.0,
    cache_dir: Optional[Path] = None,
    pexels_api_key: Optional[str] = None,
    min_delay: float = 0.5
) -> Tuple[Path, bool]:
    """Fetches visual footage for a query, checking local cache first.

    Args:
        visual_query: Search query for footage.
        duration: Target duration in seconds for fallback generation.
        cache_dir: Directory to cache downloaded clips.
        pexels_api_key: Optional API key override.
        min_delay: Rate-limiting delay.

    Returns:
        Tuple[Path, bool]: (asset_path, was_cache_hit)
    """
    dotenv.load_dotenv(override=True)
    c_dir = Path(cache_dir) if cache_dir else DEFAULT_ASSETS_CACHE / "pexels"
    c_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic query hash for cache file
    query_slug = re.sub(r'[^a-zA-Z0-9]', '_', visual_query.lower())[:24]
    query_hash = hashlib.md5(visual_query.lower().strip().encode()).hexdigest()[:8]
    cache_filename = f"{query_slug}_{query_hash}.mp4"
    cache_path = c_dir / cache_filename

    # 1. Local Cache Check
    if cache_path.exists() and cache_path.stat().st_size > 1024:
        logger.info(f"Cache HIT for query '{visual_query}' -> {cache_path.name}. Avoiding network call.")
        return cache_path, True

    logger.info(f"Cache MISS for query '{visual_query}'. Fetching from Pexels API...")
    api_key = pexels_api_key or os.getenv("PEXELS_API_KEY", "").strip("'\"")

    if not api_key:
        logger.warning("PEXELS_API_KEY not found. Generating fallback visual asset via ComfyUI worker...")
        return _generate_fallback_clip(visual_query, cache_path, duration=duration), False

    # 2. Query Pexels API with rate-limiting pacing
    time.sleep(min_delay)
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {
        "query": visual_query,
        "orientation": "portrait",
        "per_page": 3
    }

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        videos = data.get("videos", [])

        if not videos:
            logger.warning(f"No Pexels videos found for '{visual_query}'. Using fallback generator.")
            return _generate_fallback_clip(visual_query, cache_path, duration=duration), False

        # Select highest quality download link
        video_files = videos[0].get("video_files", [])
        # Prefer HD 1080x1920 or 720x1280
        best_link = None
        for vf in video_files:
            if vf.get("file_type") == "video/mp4":
                best_link = vf.get("link")
                if vf.get("width", 0) >= 720:
                    break

        if not best_link:
            best_link = video_files[0].get("link")

        # Download and cache video
        logger.info(f"Downloading Pexels clip for '{visual_query}'...")
        v_resp = requests.get(best_link, timeout=30.0, stream=True)
        v_resp.raise_for_status()

        with open(cache_path, "wb") as f:
            for chunk in v_resp.iter_content(chunk_size=65536):
                f.write(chunk)

        logger.info(f"Saved asset to cache: {cache_path.name} ({cache_path.stat().st_size // 1024} KB).")
        return cache_path, False

    except Exception as e:
        logger.warning(f"Pexels fetch failed for '{visual_query}': {e}. Using fallback generator.")
        return _generate_fallback_clip(visual_query, cache_path, duration=duration), False


def _generate_fallback_clip(prompt: str, output_path: Path, duration: float = 4.0) -> Path:
    """Generates a fallback visual clip via comfyui_worker (governed by GPU lock)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allow_mock = os.getenv("ALLOW_MOCK_ASSETS", "false").lower() in ("true", "1")
    worker_args = [
        "--prompt", prompt,
        "--output", str(output_path),
        "--duration", f"{duration:.2f}",
    ]
    if allow_mock:
        worker_args.append("--mock-on-error")
    else:
        worker_args.append("--disable-mock")

    run_gpu_worker(
        worker_name="comfyui_worker.py",
        worker_args=worker_args
    )
    return output_path


# ==============================================================================
# 3. STRUCTURAL GPU-LOCK ORCHESTRATION (Rule 1 Compliance)
# ==============================================================================

def run_gpu_worker(
    worker_name: str,
    worker_args: List[str],
    timeout: float = 180.0
) -> subprocess.CompletedProcess:
    """The sole authorized execution path for GPU workers in the Director pipeline.

    Structurally mandates acquiring the GPU lock before launching any GPU-bound
    subprocess (Whisper, ComfyUI, Render). Direct bypass is impossible.
    """
    worker_file = WORKERS_DIR / worker_name
    if not worker_file.exists():
        raise FileNotFoundError(f"GPU worker script not found at {worker_file}")

    cmd = [sys.executable, str(worker_file)] + worker_args

    if "comfyui" in worker_name.lower():
        logger.info(f"[GPU_DISPATCHER] Performing preflight VRAM check (threshold: 1024 MB) for {worker_name}...")
        preflight_vram_check(min_free_mb=1024, worker_name=worker_name)

    logger.info(f"[GPU_DISPATCHER] Acquiring lock for worker: {worker_name}...")
    with gpu_lock(timeout=timeout, worker_name=worker_name):
        logger.info(f"[GPU_DISPATCHER] Executing GPU subprocess: {worker_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"[GPU_DISPATCHER] Worker {worker_name} failed (code {result.returncode}):\n{result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)

        return result


# ==============================================================================
# 4. MASTER VIDEO PRODUCTION ORCHESTRATION
# ==============================================================================

def orchestrate_video(
    script: Script,
    output_path: Path,
    voice: str = "en-US-ChristopherNeural",
    verification_history: Optional[Dict[str, Any]] = None
) -> Path:
    """Coordinates full video assembly from Script through narration, assets, and render."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = DEFAULT_ASSETS_CACHE / f"run_{int(time.time())}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Video Orchestration for: '{script.topic}' ===")

    # -------------------------------------------------------------------------
    # STAGE 3A: Voice Actor (Narration Audio Synthesis & Pacing)
    # -------------------------------------------------------------------------
    seg_audio_paths: List[Path] = []
    seg_pauses_ms: List[float] = []
    seg_durations: List[float] = []

    def _execute_voice_actor(feedback: Optional[str] = None) -> Path:
        if feedback:
            logger.info(f"[VOICE_ACTOR_RETRY] Engaging corrective voice synthesis: {feedback}")
        seg_audio_paths.clear()
        seg_pauses_ms.clear()
        seg_durations.clear()
        logger.info(f"Synthesizing narration for {len(script.segments)} script segments (Duration Inversion)...")
        for i, seg in enumerate(script.segments):
            seg_audio_out = staging_dir / "tts" / f"tts_seg_{i:02d}.wav"
            clean_wav, engine, speech_dur, pause_dur = synthesize_segment_narration(
                segment_text=seg.narration,
                voice=voice,
                output_path=seg_audio_out,
                temp_dir=staging_dir / "tts" / f"work_{i:02d}"
            )
            if i == len(script.segments) - 1:
                pause_dur = 0.0  # Zero trailing dead-air at the conclusion of the video
            seg_audio_paths.append(clean_wav)
            seg_pauses_ms.append(pause_dur * 1000.0)

            seg_total_duration = speech_dur + pause_dur
            seg_durations.append(seg_total_duration)
            logger.info(
                f"Segment {i+1}/{len(script.segments)}: speech={speech_dur:.2f}s, pause={pause_dur:.2f}s "
                f"-> Target Video Duration={seg_total_duration:.2f}s (Engine: {engine})"
            )

        assembled_audio = staging_dir / "narration.wav"
        assemble_continuous_narration(
            chunk_wavs=seg_audio_paths,
            pause_ms_list=seg_pauses_ms,
            output_path=assembled_audio,
            sr=48000
        )
        logger.info(f"Master continuous narration audio assembled at {assembled_audio}")
        return assembled_audio

    expected_spoken_len = estimate_spoken_length(script)
    audio_path, va_history = run_stage_with_verification(
        stage_name="VOICE_ACTOR",
        execute_fn=_execute_voice_actor,
        verify_fn=lambda p: verify_voice_actor(p, expected_duration=expected_spoken_len),
        max_retries=2
    )
    if verification_history is not None:
        verification_history["voice_actor"] = [r.to_dict() for r in va_history]

    # Step 2: Transcribe narration audio with Whisper (THROUGH GPU LOCK)
    srt_path = staging_dir / "subtitles.srt"
    logger.info("Invoking Whisper worker through GPU lock...")
    run_gpu_worker(
        worker_name="whisper_worker.py",
        worker_args=[
            "--audio", str(audio_path),
            "--output", str(srt_path),
            "--device", "cuda"
        ]
    )

    # Step 3: Fetch visual assets for segments matching derived TTS durations
    logger.info(f"Fetching assets for {len(script.segments)} script segments (matched to TTS durations)...")
    segment_clips: List[Path] = []
    for i, seg in enumerate(script.segments):
        target_dur = seg_durations[i]
        clip_path, was_cached = fetch_visual_asset(
            visual_query=seg.visual_query,
            duration=target_dur,
            cache_dir=DEFAULT_ASSETS_CACHE / "pexels"
        )
        segment_clips.append(clip_path)

    # -------------------------------------------------------------------------
    # STAGE 3B: Art Director (Normalize & Compose Master Visual Track)
    # -------------------------------------------------------------------------
    from pipeline.workers.normalize_worker import normalize_clip
    from moviepy import VideoFileClip, concatenate_videoclips

    def _execute_art_director(feedback: Optional[str] = None) -> Path:
        if feedback:
            logger.info(f"[ART_DIRECTOR_RETRY] Engaging corrective visual assembly: {feedback}")
        normalized_clips: List[Path] = []
        for i, raw_clip in enumerate(segment_clips):
            target_dur = seg_durations[i]
            norm_path = staging_dir / f"norm_seg_{i:02d}.mp4"
            logger.info(f"  -> Normalizing clip {i+1}/{len(segment_clips)}: {raw_clip.name} (duration={target_dur:.2f}s)")
            normalize_clip(
                input_path=raw_clip,
                output_path=norm_path,
                target_w=1080,
                target_h=1920,
                fps=30,
                duration=target_dur
            )
            normalized_clips.append(norm_path)

        master_visual_path = staging_dir / "master_visual.mp4"
        logger.info(f"Assembling {len(normalized_clips)} normalized visual clips with method='compose'...")
        v_clips = [VideoFileClip(str(p)) for p in normalized_clips]
        concatenated_visuals = concatenate_videoclips(v_clips, method="compose")
        concatenated_visuals.write_videofile(
            str(master_visual_path),
            fps=30,
            codec="libx264",
            preset="ultrafast",
            logger=None,
            ffmpeg_params=[
                "-g", "48",
                "-keyint_min", "48",
                "-sc_threshold", "0",
                "-pix_fmt", "yuv420p",
                "-color_range", "tv",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-movflags", "+faststart"
            ]
        )
        for vc in v_clips:
            vc.close()
        concatenated_visuals.close()
        return master_visual_path

    master_visual, art_history = run_stage_with_verification(
        stage_name="ART_DIRECTOR",
        execute_fn=_execute_art_director,
        verify_fn=verify_art_director,
        max_retries=2
    )
    if verification_history is not None:
        verification_history["art_director"] = [r.to_dict() for r in art_history]

    # -------------------------------------------------------------------------
    # STAGE 3C: Editor (Final Master Render with Hard-Burned Subtitles)
    # -------------------------------------------------------------------------
    def _execute_editor(feedback: Optional[str] = None) -> Path:
        if feedback:
            logger.info(f"[EDITOR_RETRY] Engaging corrective render: {feedback}")
        logger.info("Invoking Render worker through GPU lock...")
        run_gpu_worker(
            worker_name="render_worker.py",
            worker_args=[
                "--video", str(master_visual),
                "--audio", str(audio_path),
                "--output", str(output_path),
                "--srt", str(srt_path)
            ]
        )
        return output_path

    final_rendered_path, editor_history = run_stage_with_verification(
        stage_name="EDITOR",
        execute_fn=_execute_editor,
        verify_fn=verify_editor,
        max_retries=2
    )
    if verification_history is not None:
        verification_history["editor"] = [r.to_dict() for r in editor_history]

    logger.info(f"=== Video Orchestration Completed Successfully: {output_path} ===")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Director Agent CLI")
    parser.add_argument("--script", type=str, required=True, help="Path to input script JSON")
    parser.add_argument("--output", type=str, required=True, help="Path for rendered video output")
    parser.add_argument("--voice", type=str, default="en-US-ChristopherNeural", help="TTS Voice")
    args = parser.parse_args()

    with open(args.script, "r", encoding="utf-8") as f:
        data = json.load(f)

    script = Script.model_validate(data)
    out = orchestrate_video(script=script, output_path=Path(args.output), voice=args.voice)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
