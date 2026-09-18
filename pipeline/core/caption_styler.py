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
    font_size: float = 64.0,
    margin_v: int = 520,
    highlight_bgr: str = "&H0000FFFF&"  # Vivid Yellow (&H00BBGGRR&)
) -> Path:
    """Builds an animated, high-contrast kinetic ASS caption track centered at Y ≈ 1400.

    Delegates to pipeline.subtitles.generator.generate_kinetic_ass for kinetic word scaling
    and positioning while preserving backwards compatibility.
    """
    from pipeline.subtitles.generator import generate_kinetic_ass, VIVID_YELLOW_BGR, TARGET_Y
    color = highlight_bgr if highlight_bgr != "&H002BF7FF&" else VIVID_YELLOW_BGR
    return generate_kinetic_ass(
        segments=segments,
        output_ass_path=output_ass_path,
        words_per_chunk=words_per_chunk,
        font_name="Montserrat Black",
        font_size=font_size,
        highlight_color_bgr=color,
        center_y=TARGET_Y
    )
