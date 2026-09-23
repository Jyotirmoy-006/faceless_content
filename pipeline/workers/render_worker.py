"""Final video assembler subprocess worker with Instagram-compliant NVENC encoding.

Per Rule 1 (GPU MEMORY):
- Subprocess isolation: runs independently and releases all FFmpeg NVENC
  resources upon process exit.

INSTAGRAM & BROADCAST COMPLIANCE SPECIFICATION:
----------------------------------------------
- Video Codec: h264_nvenc (NVIDIA hardware acceleration) with libx264 fallback
- Pixel Format: yuv420p (required for broad mobile/browser playback)
- Color Space: bt709 / bt709 / bt709 (Rule 9: HD / Vertical video standard)
- Dynamic Range: tv (standard broadcast 16-235)
- Closed-GOP: -g 48 -keyint_min 48 -sc_threshold 0 (fixed keyframe interval at 30 fps)
- Audio Codec: AAC at 48kHz sampling rate and 128 kbps bitrate
- Moov Atom: -movflags +faststart (places moov before mdat for instant streaming)
"""

from __future__ import annotations

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


def check_nvenc_available() -> bool:
    """Checks whether h264_nvenc encoder is available in local FFmpeg installation."""
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5.0)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False


def build_ffmpeg_cmd(
    video_path: Path,
    audio_path: Optional[Path],
    output_path: Path,
    ass_path: Optional[Path] = None,
    audio_duration: Optional[float] = None,
    fps: int = 30,
    bitrate: str = "4000k"
) -> List[str]:
    """Constructs the exact FFmpeg CLI command list for rendering with NVENC or libx264."""
    use_nvenc = check_nvenc_available()
    cmd = ["ffmpeg", "-y", "-hide_banner", "-i", str(video_path)]
    if audio_path and Path(audio_path).exists():
        cmd.extend(["-i", str(audio_path)])
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        cmd.extend(["-map", "0:v:0", "-map", "0:a?"])

    filter_complex: List[str] = []
    if ass_path and Path(ass_path).exists():
        from pipeline.core.caption_styler import escape_ffmpeg_path
        escaped_ass = escape_ffmpeg_path(Path(ass_path))
        filter_complex.append(f"ass='{escaped_ass}'")

    if filter_complex:
        cmd.extend(["-vf", ",".join(filter_complex)])

    cmd.extend([
        "-g", "48",
        "-keyint_min", "48",
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "48000",
        "-movflags", "+faststart"
    ])

    if use_nvenc:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "23",
            "-b:v", bitrate
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-b:v", bitrate
        ])

    if audio_duration:
        cmd.extend(["-t", f"{audio_duration:.3f}"])

    cmd.append(str(output_path))
    return cmd


def render_video(
    video_path: Path,
    audio_path: Optional[Path],
    output_path: Path,
    srt_path: Optional[Path] = None,
    fps: int = 30,
    bitrate: str = "4000k"
) -> Path:
    """Renders the final video with explicit Instagram-compliant parameters via direct FFmpeg."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    ass_path = None
    if srt_path and Path(srt_path).exists():
        srt_p = Path(srt_path).resolve()
        ass_p = srt_p.with_suffix(".ass")
        if not ass_p.exists():
            from pipeline.core.caption_styler import srt_to_styled_ass
            print(f"[RENDER_WORKER] Converting {srt_p.name} to styled 1080x1920 ASS format...")
            ass_p = srt_to_styled_ass(srt_p, ass_p)
        ass_path = ass_p

    # Extract audio duration to ensure exact stream alignment
    audio_dur = None
    if audio_path and Path(audio_path).exists():
        try:
            import wave
            with wave.open(str(audio_path), "r") as wf:
                audio_dur = wf.getnframes() / float(wf.getframerate())
        except Exception:
            pass

    cmd = build_ffmpeg_cmd(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        ass_path=ass_path,
        audio_duration=audio_dur,
        fps=fps,
        bitrate=bitrate
    )

    print("\n" + "=" * 70, flush=True)
    print("[RENDER_WORKER] EXACT FFmpeg COMMAND EXECUTED:", flush=True)
    print(" ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    print("=" * 70 + "\n", flush=True)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        if "h264_nvenc" in cmd:
            print(f"[RENDER_WORKER] h264_nvenc failed ({res.stderr[:200]}). Retrying with libx264...")
            cmd_fallback = [c if c != "h264_nvenc" else "libx264" for c in cmd]
            res = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg render failed: {res.stderr}")
        else:
            raise RuntimeError(f"FFmpeg render failed: {res.stderr}")

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
