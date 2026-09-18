"""Video Segment Normalization Worker.

Enforces Rule 9 (VISUAL CONSISTENCY):
- Normalizes any video clip (Pexels stock footage, AI-generated Ken Burns, etc.)
  to one canonical resolution (1080x1920), constant frame rate (30 fps CFR),
  pixel format (yuv420p), and color range (tv / bt709) before concatenation.
- Uses scale+crop (never stretch) to fill the 9:16 vertical canvas cleanly.
- Preserves high visual fidelity using Lanczos interpolation.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def normalize_clip(
    input_path: Path,
    output_path: Path,
    target_w: int = 1080,
    target_h: int = 1920,
    fps: int = 30,
    duration: Optional[float] = None,
    crf: int = 18,
    preset: str = "veryfast",
    motion: str = "none"
) -> Path:
    """Normalizes an individual video clip to canonical specs using FFmpeg.

    Args:
        input_path: Path to raw input video file.
        output_path: Destination path for normalized .mp4.
        target_w: Canonical width (default 1080).
        target_h: Canonical height (default 1920).
        fps: Constant frame rate (default 30).
        duration: Optional exact segment duration to trim/loop to.
        crf: Quality level (default 18 for visually lossless).
        preset: x264 speed preset.
        motion: Camera motion type ('none', 'zoom_in', 'zoom_out').

    Returns:
        Path to the normalized video file.
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Base scaling & cropping
    base_filter = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={target_w}:{target_h}:(iw-{target_w})/2:(ih-{target_h})/2,"
        f"setsar=1"
    )

    if motion in ("zoom_in", "zoom_out") and duration and duration > 0:
        total_frames = max(1, int(fps * duration))
        step = 0.12 / total_frames
        if motion == "zoom_in":
            vf_filter = (
                f"{base_filter},"
                f"zoompan=z='min(zoom+{step:.6f},1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={target_w}x{target_h}:fps={fps}"
            )
        else:
            vf_filter = (
                f"{base_filter},"
                f"zoompan=z='max(1.15-{step:.6f}*on,1.0)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={target_w}x{target_h}:fps={fps}"
            )
    else:
        vf_filter = base_filter


    # Detect if input has an audio stream
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(input_path)
    ]
    has_audio = False
    try:
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=False)
        has_audio = "audio" in res.stdout.lower()
    except Exception:
        has_audio = False

    cmd = ["ffmpeg", "-y"]
    if duration is not None and duration > 0:
        cmd.extend(["-stream_loop", "-1"])
    cmd.extend(["-i", str(input_path)])
    if duration is not None and duration > 0:
        cmd.extend(["-t", f"{duration:.3f}"])

    cmd.extend([
        "-vf", vf_filter,
        "-r", str(fps),
        "-fps_mode", "cfr",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-g", str(fps),
        "-keyint_min", str(fps),
        "-sc_threshold", "0"
    ])

    if has_audio:
        cmd.extend(["-c:a", "aac", "-ar", "48000", "-b:a", "128k"])
    else:
        cmd.extend(["-an"])

    cmd.append(str(output_path))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fallback for older ffmpeg versions that don't support -fps_mode
        if "-fps_mode" in proc.stderr:
            cmd_alt = [c if c != "-fps_mode" else "-vsync" for c in cmd]
            proc = subprocess.run(cmd_alt, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"FFmpeg normalization failed (code {proc.returncode}) for {input_path}:\n{proc.stderr}"
            )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Normalized output missing or empty at {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Standalone video clip normalization worker (Rule 9)")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output normalized video path")
    parser.add_argument("--width", type=int, default=1080, help="Target width (default: 1080)")
    parser.add_argument("--height", type=int, default=1920, help="Target height (default: 1920)")
    parser.add_argument("--fps", type=int, default=30, help="Target constant frame rate (default: 30)")
    parser.add_argument("--duration", type=float, default=None, help="Target duration in seconds to trim/loop")
    parser.add_argument(
        "--motion",
        choices=["none", "zoom_in", "zoom_out"],
        default="none",
        help="Camera motion effect (default: none)"
    )
    args = parser.parse_args()

    out = normalize_clip(
        input_path=Path(args.input),
        output_path=Path(args.output),
        target_w=args.width,
        target_h=args.height,
        fps=args.fps,
        duration=args.duration,
        motion=args.motion
    )
    print(f"[NORMALIZE_WORKER] Successfully normalized {args.input} -> {out}")


if __name__ == "__main__":
    main()
