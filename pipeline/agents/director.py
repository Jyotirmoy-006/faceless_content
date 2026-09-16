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
from typing import List, Optional, Tuple

import dotenv
import requests

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.gpu_lock import gpu_lock, GPULockError
from pipeline.core.schema import Script, ScriptSegment

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
    # Split on sentence boundaries (period, exclamation, question mark, or newline)
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


def synthesize_chunk_piper(text: str, output_path: Path, model_path: Optional[Path] = None) -> Path:
    """Synthesizes an audio chunk locally using Piper TTS."""
    from piper import PiperVoice

    m_path = Path(model_path) if model_path else DEFAULT_PIPER_MODEL
    c_path = m_path.with_suffix(".onnx.json") if m_path.suffix == ".onnx" else Path(str(m_path) + ".json")

    if not m_path.exists():
        raise FileNotFoundError(f"Piper model not found at {m_path}. Please download voice model.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    voice = PiperVoice.load(m_path, config_path=c_path, use_cuda=False)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        for audio_chunk in voice.synthesize(text):
            wav_file.writeframes(audio_chunk.audio_int16_bytes)

    return output_path


def synthesize_chunk_edgetts(
    text: str,
    voice: str,
    output_path: Path,
    max_attempts: int = 3,
    base_backoff: float = 2.0
) -> Path:
    """Synthesizes an audio chunk via edge-tts with exponential backoff."""
    import edge_tts

    last_error = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
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


def synthesize_narration(
    text: str,
    voice: str = "en-US-ChristopherNeural",
    output_path: Optional[Path] = None,
    min_delay: float = 1.5,
    jitter_range: Tuple[float, float] = (0.2, 0.8),
    temp_dir: Optional[Path] = None,
    mock_edge_failure_chunk: Optional[int] = None
) -> Tuple[Path, List[str]]:
    """Synthesizes narration with sentence-level chunking, jitter, and per-chunk Piper fallback.

    Args:
        text: Complete narration text.
        voice: edge-tts voice name.
        output_path: Destination for concatenated audio file.
        min_delay: Minimum delay between edge-tts API calls (Rule 5).
        jitter_range: (min_jitter, max_jitter) added to delay.
        temp_dir: Staging folder for chunk audio files.
        mock_edge_failure_chunk: 0-based chunk index to deliberately fail for testing.

    Returns:
        Tuple[Path, List[str]]: (output_audio_path, list_of_chunk_sources)
        where each source is either 'edge-tts' or 'piper-fallback'.
    """
    chunks = chunk_text_by_sentence(text, max_chars=300)
    if not chunks:
        raise ValueError("Cannot synthesize empty text narration.")

    out_path = Path(output_path) if output_path else DEFAULT_ASSETS_CACHE / "master_narration.mp3"
    t_dir = Path(temp_dir) if temp_dir else DEFAULT_ASSETS_CACHE / "tts_chunks"
    t_dir.mkdir(parents=True, exist_ok=True)

    chunk_files: List[Path] = []
    chunk_sources: List[str] = []

    logger.info(f"Synthesizing narration ({len(chunks)} sentence chunks, target voice: {voice})...")

    for i, chunk in enumerate(chunks):
        # Apply jittered rate limiting prior to calling edge-tts (skip first chunk)
        if i > 0:
            delay = min_delay + random.uniform(jitter_range[0], jitter_range[1])
            time.sleep(delay)

        chunk_edge_target = t_dir / f"chunk_{i:03d}_edge.mp3"
        chunk_piper_target = t_dir / f"chunk_{i:03d}_piper.wav"

        use_fallback = (mock_edge_failure_chunk is not None and i == mock_edge_failure_chunk)

        if not use_fallback:
            try:
                synthesize_chunk_edgetts(
                    text=chunk,
                    voice=voice,
                    output_path=chunk_edge_target,
                    max_attempts=3,
                    base_backoff=2.0
                )
                chunk_files.append(chunk_edge_target)
                chunk_sources.append("edge-tts")
                logger.info(f"Chunk {i+1}/{len(chunks)} synthesized successfully via edge-tts.")
                continue
            except Exception as edge_err:
                logger.warning(
                    f"edge-tts failed for chunk {i+1} after retries: {edge_err}. "
                    f"Falling back to local Piper TTS for this chunk."
                )

        # Fallback to local Piper TTS for this specific chunk
        logger.warning(f"[FALLBACK] Synthesizing chunk {i+1}/{len(chunks)} locally using Piper TTS.")
        synthesize_chunk_piper(chunk, chunk_piper_target)
        chunk_files.append(chunk_piper_target)
        chunk_sources.append("piper-fallback")

    # Concatenate chunk audio files in strict sequential order
    from moviepy import AudioFileClip, concatenate_audioclips

    logger.info(f"Concatenating {len(chunk_files)} chunk audio files into master track...")
    clips = []
    try:
        for p in chunk_files:
            clips.append(AudioFileClip(str(p)))

        final_clip = concatenate_audioclips(clips)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        audio_codec = "libmp3lame" if out_path.suffix.lower() == ".mp3" else "aac"
        final_clip.write_audiofile(
            str(out_path),
            fps=48000,
            codec=audio_codec,
            bitrate="128k",
            logger=None
        )
        final_clip.close()
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass

    logger.info(f"Master narration audio created at {out_path} ({len(chunks)} chunks, sources: {chunk_sources}).")
    return out_path, chunk_sources


# ==============================================================================
# 2. PEXELS ASSET CACHING & FETCHING (Rate Limiter + Cache Hit Avoidance)
# ==============================================================================

def fetch_visual_asset(
    visual_query: str,
    cache_dir: Optional[Path] = None,
    pexels_api_key: Optional[str] = None,
    min_delay: float = 0.5
) -> Tuple[Path, bool]:
    """Fetches visual footage for a query, checking local cache first.

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
        return _generate_fallback_clip(visual_query, cache_path), False

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
            return _generate_fallback_clip(visual_query, cache_path), False

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
        return _generate_fallback_clip(visual_query, cache_path), False


def _generate_fallback_clip(prompt: str, output_path: Path) -> Path:
    """Generates a fallback visual clip via comfyui_worker (governed by GPU lock)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_gpu_worker(
        worker_name="comfyui_worker.py",
        worker_args=[
            "--prompt", prompt,
            "--output", str(output_path),
            "--duration", "4.0",
            "--mock-on-error"
        ]
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
    voice: str = "en-US-ChristopherNeural"
) -> Path:
    """Coordinates full video assembly from Script through narration, assets, and render."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = DEFAULT_ASSETS_CACHE / f"run_{int(time.time())}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Starting Video Orchestration for: '{script.topic}' ===")

    # Step 1: Synthesize narration audio
    audio_path = staging_dir / "narration.mp3"
    synthesize_narration(
        text=script.full_narration(),
        voice=voice,
        output_path=audio_path,
        temp_dir=staging_dir / "tts"
    )

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

    # Step 3: Fetch visual assets for segments (with local cache check)
    logger.info(f"Fetching assets for {len(script.segments)} script segments...")
    segment_clips: List[Path] = []
    for seg in script.segments:
        clip_path, was_cached = fetch_visual_asset(
            visual_query=seg.visual_query,
            cache_dir=DEFAULT_ASSETS_CACHE / "pexels"
        )
        segment_clips.append(clip_path)

    # Step 4: Concatenate segment clips into master visual track
    from moviepy import VideoFileClip, concatenate_videoclips

    master_visual = staging_dir / "master_visual.mp4"
    logger.info(f"Assembling {len(segment_clips)} visual clips...")
    v_clips = [VideoFileClip(str(p)) for p in segment_clips]
    concatenated_visuals = concatenate_videoclips(v_clips)
    concatenated_visuals.write_videofile(
        str(master_visual),
        fps=30,
        codec="libx264",
        preset="ultrafast",
        logger=None,
        ffmpeg_params=[
            "-g", "48",
            "-keyint_min", "48",
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"
        ]
    )
    for vc in v_clips:
        vc.close()
    concatenated_visuals.close()

    # Step 5: Final Instagram-compliant NVENC Render (THROUGH GPU LOCK)
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
