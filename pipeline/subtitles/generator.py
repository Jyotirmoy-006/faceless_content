"""Kinetic Subtitle Generator — High-Contrast ASS Captions with Word Pop-In (Rule 10).

Features:
- Enforces 1080x1920 canvas resolution (PlayResX=1080, PlayResY=1920).
- Centered at Y ≈ 1400 (mobile safe viewing zone, avoiding platform UI overlays).
- Font: 'Montserrat Black' (with fallback 'Arial Black' or 'TheBoldFont'), uppercase.
- High-contrast: Pure White (&H00FFFFFF&), 4px Solid Black outline (&H00000000&), 2px shadow.
- Active Word Highlight: Spoken word instantly scales to 110% (\\fscx110\\fscy110) and turns
  Vivid Yellow (&H0000FFFF&) or Electric Green (&H0000FF00&).
- Strict maximum of 3 words on screen simultaneously for rapid, non-cluttered viewing pace.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pysubs2

logger = logging.getLogger("subtitles")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Canonical visual parameters
TARGET_Y = 1400
TARGET_X = 540  # 1080 / 2 (Center)
MAX_WORDS_ON_SCREEN = 3
PRIMARY_COLOR_BGR = "&H00FFFFFF&"    # Pure White
OUTLINE_COLOR_BGR = "&H00000000&"    # Solid Black
VIVID_YELLOW_BGR = "&H0000FFFF&"     # Vivid Yellow in BGR (&H00BBGGRR&)
ELECTRIC_GREEN_BGR = "&H0000FF00&"   # Electric Green in BGR


def create_kinetic_style(
    font_name: str = "Montserrat Black",
    font_size: float = 64.0,
    outline: float = 4.0,
    shadow: float = 2.0
) -> pysubs2.SSAStyle:
    """Creates ASS style definition matching kinetic high-contrast short-form specs."""
    style = pysubs2.SSAStyle()
    style.fontname = font_name
    style.fontsize = font_size
    style.bold = True
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)      # Pure White
    style.secondarycolor = pysubs2.Color(255, 255, 0, 0)     # Vivid Yellow
    style.outlinecolor = pysubs2.Color(0, 0, 0, 0)          # Solid Black
    style.backcolor = pysubs2.Color(0, 0, 0, 150)           # Shadow
    style.outline = outline
    style.shadow = shadow
    style.alignment = pysubs2.Alignment.MIDDLE_CENTER       # Alignment 5 (Middle Center)
    style.marginl = 40
    style.marginr = 40
    style.marginv = 520                                     # 1920 - 1400 = 520
    return style


def generate_kinetic_ass(
    segments: List[Dict[str, Any]],
    output_ass_path: Path,
    words_per_chunk: int = MAX_WORDS_ON_SCREEN,
    font_name: str = "Montserrat Black",
    font_size: float = 64.0,
    highlight_color_bgr: str = VIVID_YELLOW_BGR,
    center_y: int = TARGET_Y
) -> Path:
    """Generates kinetic, animated ASS subtitles with word-level highlight and scaling.

    Args:
        segments: Whisper transcription segments containing word-level timestamps.
        output_ass_path: Destination path for the .ass subtitle script.
        words_per_chunk: Maximum words visible on screen at once (strictly <= 3).
        font_name: Font name ('Montserrat Black', 'Arial Black', etc.).
        font_size: Point size of the caption text.
        highlight_color_bgr: Color string for active spoken word (&H0000FFFF&).
        center_y: Vertical pixel coordinate for caption center (default 1400).

    Returns:
        Path to generated ASS subtitle file.
    """
    output_ass_path = Path(output_ass_path).resolve()
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)

    words_per_chunk = min(MAX_WORDS_ON_SCREEN, max(1, words_per_chunk))

    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["WrapStyle"] = 2  # No auto line break

    # Primary Style
    style = create_kinetic_style(font_name=font_name, font_size=font_size)
    subs.styles["Default"] = style

    # Extract clean words across all segments
    all_words: List[Dict[str, Any]] = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                raw_word = str(w.get("word", "")).strip()
                # Clean punctuation for display while maintaining uppercase impact
                cleaned_word = re.sub(r'[\'\"`,;:]', '', raw_word).strip().upper()
                if cleaned_word:
                    all_words.append({
                        "word": cleaned_word,
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0))
                    })
        else:
            # Fallback if Whisper word_timestamps was not populated
            text = seg.get("text", "").strip()
            if text:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                raw_token_list = text.split()
                if raw_token_list:
                    dur_per_word = (seg_end - seg_start) / float(len(raw_token_list))
                    for idx, token in enumerate(raw_token_list):
                        w_clean = re.sub(r'[\'\"`,;:]', '', token).strip().upper()
                        if w_clean:
                            all_words.append({
                                "word": w_clean,
                                "start": seg_start + idx * dur_per_word,
                                "end": seg_start + (idx + 1) * dur_per_word
                            })

    if not all_words:
        # Write empty ASS file
        subs.save(str(output_ass_path))
        return output_ass_path

    # Chunk words into groups of at most `words_per_chunk` (default: 3)
    chunk_idx = 0
    while chunk_idx < len(all_words):
        chunk = all_words[chunk_idx:chunk_idx + words_per_chunk]
        chunk_idx += words_per_chunk

        # For each word in the chunk, generate an active highlight event
        for active_idx, active_word in enumerate(chunk):
            w_start_ms = int(active_word["start"] * 1000)
            w_end_ms = int(active_word["end"] * 1000)

            if w_end_ms <= w_start_ms:
                w_end_ms = w_start_ms + 250

            # Construct kinetic formatted ASS line
            # Active word: scaled to 110% with highlight color
            # Inactive words: 100% scale with pure white color
            formatted_words: List[str] = []
            for j, item in enumerate(chunk):
                word_text = item["word"]
                if j == active_idx:
                    # Pop-in scale and vivid highlight
                    tag = f"{{\\c{highlight_color_bgr}}}{{\\fscx110\\fscy110}}{word_text}{{\\fscx100\\fscy100}}{{\\c{PRIMARY_COLOR_BGR}}}"
                    formatted_words.append(tag)
                else:
                    tag = f"{{\\fscx100\\fscy100}}{{\\c{PRIMARY_COLOR_BGR}}}{word_text}"
                    formatted_words.append(tag)

            event_body = " ".join(formatted_words)
            # Centered at (540, center_y) via middle-center positioning tag \an5\pos
            pos_tag = f"{{\\an5\\pos({TARGET_X},{center_y})}}"
            full_text = f"{pos_tag}{event_body}"

            subs.events.append(pysubs2.SSAEvent(
                start=w_start_ms,
                end=w_end_ms,
                text=full_text,
                style="Default"
            ))

    subs.save(str(output_ass_path))
    logger.info(
        f"[SUBTITLES] Generated kinetic ASS captions: {output_ass_path.name} "
        f"({len(subs.events)} word events, max {words_per_chunk} words/chunk, Y={center_y})"
    )
    return output_ass_path
