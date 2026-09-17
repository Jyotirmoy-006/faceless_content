"""Caption Styling and ASS Formatting Engine for Short-Form Video (Rule 10).

Features:
- Enforces 1080x1920 canvas resolution (PlayResX=1080, PlayResY=1920).
- Styles captions for short-form mobile viewing (bold, high-contrast, black outline, drop shadow).
- Places subtitles in the vertical safe zone (avoiding top 10% and bottom 15-20% platform UI overlays).
- Escapes Windows absolute paths for FFmpeg filtergraph parsing (C\\:/path/to/subs.ass).
- Generates word-by-word karaoke highlight ASS captions from Whisper word-level timestamps.
- Converts existing SRT files into styled, safe-zone ASS format.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pysubs2


def escape_ffmpeg_path(path: Union[str, Path]) -> str:
    """Escapes a Windows or POSIX path for FFmpeg's filtergraph parser.

    FFmpeg's filter parser uses colons as option separators. On Windows,
    a path like 'C:\\path\\file.ass' contains a drive colon which breaks parsing.
    This function converts backslashes to forward slashes and escapes the colon:
    'C\\:/path/file.ass'.
    """
    p = Path(path).resolve()
    posix_path = p.as_posix()
    # Escape drive colon (e.g., C: -> C\:)
    escaped = posix_path.replace(":", r"\:")
    return escaped


def create_shortform_style(
    font_name: str = "Arial",
    font_size: float = 62.0,
    outline: float = 4.5,
    shadow: float = 2.0,
    margin_v: int = 400
) -> pysubs2.SSAStyle:
    """Creates an ASS style tuned for 1080x1920 vertical mobile viewports."""
    style = pysubs2.SSAStyle()
    style.fontname = font_name
    style.fontsize = font_size
    style.bold = True
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)      # Pure White (&H00FFFFFF)
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)          # Solid Black (&H00000000)
    style.backcolor = pysubs2.Color(0, 0, 0, 140)           # Semi-transparent drop shadow
    style.outline = outline
    style.shadow = shadow
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER       # Alignment 2 (Bottom-center)
    style.marginl = 60
    style.marginr = 60
    style.marginv = margin_v                                 # Vertical safe zone (above bottom 15-20%)
    return style


def wrap_text_lines(text: str, max_chars: int = 28) -> str:
    """Wraps text into concise lines separated by ASS linebreaks (\\N)."""
    words = text.strip().split()
    if not words:
        return ""

    lines = []
    current_line = []
    current_len = 0

    for w in words:
        if current_len + len(w) + 1 > max_chars and current_line:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + 1

    if current_line:
        lines.append(" ".join(current_line))

    # Join with ASS hard line break
    return r"\N".join(lines)


def srt_to_styled_ass(
    srt_path: Path,
    output_ass_path: Optional[Path] = None,
    font_size: float = 62.0,
    margin_v: int = 400
) -> Path:
    """Converts a standard .srt file into a styled, safe-zone .ass file."""
    srt_path = Path(srt_path).resolve()
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    out_path = Path(output_ass_path).resolve() if output_ass_path else srt_path.with_suffix(".ass")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subs = pysubs2.load(str(srt_path), encoding="utf-8")
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920

    style = create_shortform_style(font_size=font_size, margin_v=margin_v)
    subs.styles["Default"] = style

    for ev in subs.events:
        # Clean text and format with uppercase and balanced wrapping
        cleaned = re.sub(r'\s+', ' ', ev.text.replace(r'\N', ' ').replace('\n', ' ')).strip()
        ev.text = wrap_text_lines(cleaned.upper(), max_chars=30)
        ev.style = "Default"

    subs.save(str(out_path))
    return out_path


def generate_karaoke_ass(
    segments: List[dict],
    output_ass_path: Path,
    words_per_chunk: int = 3,
    font_size: float = 62.0,
    margin_v: int = 400,
    highlight_bgr: str = "&H002BF7FF&"  # Vibrant Gold/Yellow in BGR format
) -> Path:
    """Builds a word-by-word karaoke-highlight caption track from Whisper segments.

    Each spoken word glows in highlight_bgr while other words in the active 2-4 word
    chunk remain bold white with a thick black outline.
    """
    output_ass_path = Path(output_ass_path).resolve()
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920

    style = create_shortform_style(font_size=font_size, margin_v=margin_v)
    subs.styles["Default"] = style

    highlight_tag = f"{{\\c{highlight_bgr}}}"
    normal_tag = r"{\c&H00FFFFFF&}"

    # Extract all words across all segments
    all_words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                cleaned_word = w.get("word", "").strip().upper()
                if cleaned_word:
                    all_words.append({
                        "word": cleaned_word,
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0))
                    })
        else:
            # Fallback if words not present in segment: split segment text
            text = seg.get("text", "").strip()
            if text:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                words_in_text = text.split()
                dur_per_word = (seg_end - seg_start) / max(1, len(words_in_text))
                for idx, w in enumerate(words_in_text):
                    all_words.append({
                        "word": w.strip().upper(),
                        "start": seg_start + idx * dur_per_word,
                        "end": seg_start + (idx + 1) * dur_per_word
                    })

    if not all_words:
        # Write empty valid ASS file
        subs.save(str(output_ass_path))
        return output_ass_path

    # Chunk words into groups of `words_per_chunk`
    i = 0
    while i < len(all_words):
        chunk = all_words[i:i + words_per_chunk]
        i += words_per_chunk

        # For each word in this chunk, create an event where that word is highlighted
        for active_idx, active_word in enumerate(chunk):
            w_start_ms = int(active_word["start"] * 1000)
            w_end_ms = int(active_word["end"] * 1000)

            if w_end_ms <= w_start_ms:
                w_end_ms = w_start_ms + 250

            # Construct formatted line
            line_parts = []
            for j, w in enumerate(chunk):
                if j == active_idx:
                    line_parts.append(f"{highlight_tag}{w['word']}{normal_tag}")
                else:
                    line_parts.append(f"{normal_tag}{w['word']}")

            event_text = " ".join(line_parts)
            subs.events.append(pysubs2.SSAEvent(
                start=w_start_ms,
                end=w_end_ms,
                text=event_text,
                style="Default"
            ))

    subs.save(str(output_ass_path))
    return output_ass_path
