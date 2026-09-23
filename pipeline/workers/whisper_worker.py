"""Whisper isolated subprocess worker.

Per Rule 1 (GPU MEMORY):
- Runs as an isolated standalone script.
- Releases 100% of VRAM by exiting the process when transcription completes.
- Logs peak VRAM usage (torch.cuda.max_memory_allocated()) to stdout on exit.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def format_srt_timestamp(seconds: float) -> str:
    """Converts seconds into SRT timestamp format: HH:MM:SS,mmm"""
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: list, output_path: Path):
    """Writes transcription segments to an SRT file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_str = format_srt_timestamp(segment["start"])
            end_str = format_srt_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Standalone Whisper transcription subprocess worker")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, default=None, help="Path to output .srt file")
    parser.add_argument("--model", type=str, default="base", help="Whisper model name (default: base)")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda or cpu)")
    parser.add_argument("--language", type=str, default=None, help="Language code (e.g. 'en')")
    parser.add_argument("--script-text", type=str, default=None, help="Ground truth script text for phonetic reconciliation")
    args = parser.parse_args()

    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        print(f"[WHISPER_WORKER] Error: Audio file not found at {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else audio_path.with_suffix(".srt")

    import torch
    import whisper

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        initial_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[WHISPER_WORKER] Running on GPU: {torch.cuda.get_device_name(0)} (initial allocated: {initial_vram_mb:.2f} MB)")
    else:
        print(f"[WHISPER_WORKER] Warning: CUDA unavailable or CPU selected; running on {device}", file=sys.stderr)

    start_time = time.time()
    print(f"[WHISPER_WORKER] Loading Whisper model '{args.model}' on {device}...")
    model = whisper.load_model(args.model, device=device)

    transcribe_options = {"verbose": False, "word_timestamps": True}
    if args.language:
        transcribe_options["language"] = args.language

    result = model.transcribe(str(audio_path), **transcribe_options)
    duration = time.time() - start_time

    write_srt(result["segments"], output_path)
    print(f"[WHISPER_WORKER] Successfully wrote subtitles to {output_path} ({len(result['segments'])} segments, {duration:.2f}s elapsed)")

    try:
        from pipeline.core.caption_styler import generate_karaoke_ass
        ass_path = output_path.with_suffix(".ass")
        generate_karaoke_ass(result["segments"], ass_path, script_text=args.script_text)
        print(f"[WHISPER_WORKER] Successfully generated styled ASS captions to {ass_path}")
    except Exception as ass_err:
        print(f"[WHISPER_WORKER] Warning: Failed to generate ASS captions ({ass_err})", file=sys.stderr)

    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        final_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[WHISPER_WORKER] Peak VRAM allocated: {peak_vram_mb:.2f} MB (active at exit: {final_vram_mb:.2f} MB)")
    else:
        print("[WHISPER_WORKER] Peak VRAM: N/A (CPU execution)")

    # Clean exit guarantees full OS VRAM cleanup
    sys.exit(0)


if __name__ == "__main__":
    main()
