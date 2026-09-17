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

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from moviepy import AudioFileClip, VideoFileClip
from pipeline.workers.normalize_worker import normalize_clip

# Intercept Popen to capture and display the exact FFmpeg command line
_original_popen = subprocess.Popen
_executed_ffmpeg_commands: List[List[str]] = []


class _LoggedPopen(_original_popen):
    def __init__(self, cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and any("ffmpeg" in str(arg).lower() for arg in cmd):
            _executed_ffmpeg_commands.append([str(c) for c in cmd])
            cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
            print("\n" + "=" * 70, flush=True)
            print("[RENDER_WORKER] EXACT FFmpeg COMMAND EXECUTED:", flush=True)
            print(cmd_str, flush=True)
            print("=" * 70 + "\n", flush=True)
        super().__init__(cmd, *args, **kwargs)


subprocess.Popen = _LoggedPopen


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
            video = concatenate_videoclips([video] * num_loops, method="compose").subclipped(0, target_duration)
        else:
            video = video.subclipped(0, target_duration)
        video = video.with_audio(audio)
    else:
        print("[RENDER_WORKER] No separate audio provided; using existing video audio if present.")

    # Explicit Instagram-compliant FFmpeg parameters (Rule 9: bt709/tv range)
    ffmpeg_params = [
        "-g", "48",                  # Closed GOP size (48 frames = 1.6s at 30 fps)
        "-keyint_min", "48",         # Minimum keyframe interval
        "-sc_threshold", "0",        # Disable scene-change keyframes for constant GOP
        "-pix_fmt", "yuv420p",       # Standard 4:2:0 chroma subsampling
        "-color_range", "tv",        # Standard broadcast dynamic range
        "-colorspace", "bt709",      # Standard HD/Vertical color space
        "-color_primaries", "bt709", # Standard color primaries
        "-color_trc", "bt709",       # Standard transfer characteristics
        "-movflags", "+faststart",   # Move moov atom to beginning of file
        "-b:a", "128k",              # Target audio bitrate
        "-ar", "48000"               # Target audio sample rate
    ]

    # Rule 10: Hard-burn captions directly into video frame pixels
    if srt_path and Path(srt_path).exists():
        srt_p = Path(srt_path).resolve()
        ass_p = srt_p.with_suffix(".ass")
        if not ass_p.exists():
            from pipeline.core.caption_styler import srt_to_styled_ass
            print(f"[RENDER_WORKER] Converting {srt_p.name} to styled 1080x1920 ASS format...")
            ass_p = srt_to_styled_ass(srt_p, ass_p)

        from pipeline.core.caption_styler import escape_ffmpeg_path
        escaped_ass = escape_ffmpeg_path(ass_p)
        print(f"[RENDER_WORKER] Hard-burning styled captions directly into video frame: {ass_p.name}")
        ffmpeg_params.extend(["-vf", f"ass='{escaped_ass}'"])
    else:
        print("[RENDER_WORKER] No subtitles file provided; rendering clean video without burned captions.")

    print("[RENDER_WORKER] Starting h264_nvenc hardware-accelerated render...")
    try:
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
    except Exception as nvenc_err:
        print(f"[RENDER_WORKER] h264_nvenc failed ({nvenc_err}). Falling back to libx264...")
        video.write_videofile(
            str(output_path),
            fps=fps,
            codec="libx264",
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
