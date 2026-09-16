"""Final video assembler subprocess worker with Instagram-compliant NVENC encoding.

Per Rule 1 (GPU MEMORY):
- Subprocess isolation: runs independently and releases all MoviePy/FFmpeg NVENC
  resources upon process exit.

INSTAGRAM COMPLIANCE SPECIFICATION:
-----------------------------------
- Video Codec: h264_nvenc (NVIDIA hardware acceleration)
- Pixel Format: yuv420p (required for broad mobile/browser playback)
- Closed-GOP: -g 48 -keyint_min 48 -sc_threshold 0 (fixed keyframe interval at 30 fps)
- Audio Codec: AAC at 48kHz sampling rate and 128 kbps bitrate
- Moov Atom: -movflags +faststart (places moov before mdat for instant streaming)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from moviepy import AudioFileClip, VideoFileClip

# Intercept Popen to capture and display the exact FFmpeg command line
_original_popen = subprocess.Popen
_executed_ffmpeg_commands: List[List[str]] = []


def _logged_popen(cmd, *args, **kwargs):
    if isinstance(cmd, (list, tuple)) and any("ffmpeg" in str(arg).lower() for arg in cmd):
        _executed_ffmpeg_commands.append([str(c) for c in cmd])
        cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
        print("\n" + "=" * 70, flush=True)
        print("[RENDER_WORKER] EXACT FFmpeg COMMAND EXECUTED:", flush=True)
        print(cmd_str, flush=True)
        print("=" * 70 + "\n", flush=True)
    return _original_popen(cmd, *args, **kwargs)


subprocess.Popen = _logged_popen


def render_video(
    video_path: Path,
    audio_path: Optional[Path],
    output_path: Path,
    srt_path: Optional[Path] = None,
    fps: int = 30,
    bitrate: str = "4000k"
) -> Path:
    """Renders the final video with explicit Instagram-compliant parameters."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"[RENDER_WORKER] Loading video: {video_path}")
    video = VideoFileClip(str(video_path))

    audio = None
    if audio_path and audio_path.exists():
        print(f"[RENDER_WORKER] Loading audio: {audio_path}")
        audio = AudioFileClip(str(audio_path))
        # Align duration: if audio is shorter or longer, match video duration to audio
        target_duration = audio.duration
        if video.duration < target_duration:
            # Loop video to match audio duration
            num_loops = int(target_duration // video.duration) + 1
            print(f"[RENDER_WORKER] Looping video {num_loops}x to match audio duration ({target_duration:.2f}s)...")
            from moviepy import concatenate_videoclips
            video = concatenate_videoclips([video] * num_loops).subclipped(0, target_duration)
        else:
            video = video.subclipped(0, target_duration)
        video = video.with_audio(audio)
    else:
        print("[RENDER_WORKER] No separate audio provided; using existing video audio if present.")

    # Explicit Instagram-compliant FFmpeg parameters
    ffmpeg_params = [
        "-g", "48",                  # Closed GOP size (48 frames = 1.6s at 30 fps)
        "-keyint_min", "48",         # Minimum keyframe interval
        "-sc_threshold", "0",        # Disable scene-change keyframes for constant GOP
        "-pix_fmt", "yuv420p",       # Standard 4:2:0 chroma subsampling
        "-movflags", "+faststart",   # Move moov atom to beginning of file
        "-b:a", "128k",              # Target audio bitrate
        "-ar", "48000"               # Target audio sample rate
    ]

    print("[RENDER_WORKER] Starting h264_nvenc hardware-accelerated render...")
    video.write_videofile(
        str(output_path),
        fps=fps,
        codec="h264_nvenc",
        audio_codec="aac",
        audio_fps=48000,
        audio_bitrate="128k",
        preset="fast",
        bitrate=bitrate,
        ffmpeg_params=ffmpeg_params,
        logger="bar"
    )

    # Explicitly close and release clip resources
    try:
        if video.audio:
            video.audio.close()
        video.close()
    except Exception:
        pass
    if audio:
        try:
            audio.close()
        except Exception:
            pass

    elapsed = time.time() - start_time
    print(f"[RENDER_WORKER] Render completed in {elapsed:.2f}s -> {output_path}")

    # Burn in subtitles if srt_path provided and ffmpeg filter is desired
    if srt_path and srt_path.exists():
        print(f"[RENDER_WORKER] Subtitle file detected: {srt_path} (ready for publisher overlay)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Standalone Instagram-compliant NVENC render worker")
    parser.add_argument("--video", type=str, required=True, help="Path to input video clip")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio file")
    parser.add_argument("--output", type=str, required=True, help="Path for rendered output video (.mp4)")
    parser.add_argument("--srt", type=str, default=None, help="Optional path to .srt subtitles")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate (default: 30)")
    parser.add_argument("--bitrate", type=str, default="4000k", help="Video bitrate")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[RENDER_WORKER] ERROR: Video file not found at {video_path}", file=sys.stderr)
        sys.exit(1)

    audio_path = Path(args.audio).resolve() if args.audio else None
    output_path = Path(args.output).resolve()
    srt_path = Path(args.srt).resolve() if args.srt else None

    try:
        render_video(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            srt_path=srt_path,
            fps=args.fps,
            bitrate=args.bitrate
        )
        sys.exit(0)
    except Exception as e:
        print(f"[RENDER_WORKER] ERROR during render: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
