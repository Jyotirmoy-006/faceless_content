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

import difflib
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
TARGET_Y = 1180  # Safe zone: centered in golden ratio lower-middle zone, clear of UI overlays
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
    return style


def reconcile_whisper_with_script(
    whisper_words: List[Dict[str, Any]],
    script_text: str
) -> List[Dict[str, Any]]:
    """Reconciles Whisper phonetic mishearings with ground-truth script text.

    Preserves millisecond-accurate Whisper timestamps while substituting
    ground-truth vocabulary using SequenceMatcher alignment.
    """
    if not whisper_words or not script_text or not script_text.strip():
        return whisper_words

    raw_script_tokens = script_text.strip().split()
    script_items: List[Dict[str, Any]] = []
    for t in raw_script_tokens:
        clean = re.sub(r'[\'\"`,;:.]', '', t).strip().upper()
        if clean:
            is_end = bool(re.search(r'[.!?]$', t))
            script_items.append({"word": clean, "is_sentence_end": is_end})

    if not script_items:
        return whisper_words

    # Pre-sanitize known phonetic slur patterns in Whisper tokens
    for w_dict in whisper_words:
        w_up = w_dict.get("word", "").upper()
        if w_up == "OF" and "BILLION" in [nxt.get("word", "").upper() for nxt in whisper_words]:
            w_dict["word"] = "A"

    w_tokens = [w["word"].upper() for w in whisper_words]
    s_tokens = [s["word"] for s in script_items]

    matcher = difflib.SequenceMatcher(None, w_tokens, s_tokens)
    reconciled: List[Dict[str, Any]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for idx, w_idx in enumerate(range(i1, i2)):
                s_idx = j1 + idx
                item = dict(whisper_words[w_idx])
                item["word"] = s_tokens[s_idx]
                item["is_sentence_end"] = script_items[s_idx]["is_sentence_end"]
                reconciled.append(item)
        elif tag == "replace":
            w_span = whisper_words[i1:i2]
            s_span = script_items[j1:j2]
            if len(w_span) == len(s_span):
                for w_item, s_item in zip(w_span, s_span):
                    item = dict(w_item)
                    item["word"] = s_item["word"]
                    item["is_sentence_end"] = s_item["is_sentence_end"]
                    reconciled.append(item)
            else:
                start_time = w_span[0]["start"]
                end_time = w_span[-1]["end"]
                dur = max(0.2, end_time - start_time)
                w_dur = dur / float(len(s_span))
                for s_idx, s_item in enumerate(s_span):
                    reconciled.append({
                        "word": s_item["word"],
                        "start": start_time + s_idx * w_dur,
                        "end": start_time + (s_idx + 1) * w_dur,
                        "is_sentence_end": s_item["is_sentence_end"]
                    })
        elif tag == "delete":
            for w_idx in range(i1, i2):
                reconciled.append(dict(whisper_words[w_idx]))
        elif tag == "insert":
            s_span = script_items[j1:j2]
            prev_end = reconciled[-1]["end"] if reconciled else 0.0
            next_start = whisper_words[i2]["start"] if i2 < len(whisper_words) else prev_end + 0.3
            dur = max(0.15 * len(s_span), next_start - prev_end)
            w_dur = dur / float(len(s_span))
            for s_idx, s_item in enumerate(s_span):
                reconciled.append({
                    "word": s_item["word"],
                    "start": prev_end + s_idx * w_dur,
                    "end": prev_end + (s_idx + 1) * w_dur,
                    "is_sentence_end": s_item["is_sentence_end"]
                })

    return reconciled if reconciled else whisper_words


def generate_kinetic_ass(
    segments: List[Dict[str, Any]],
    output_ass_path: Union[Path, str],
    words_per_chunk: int = 3,
    font_name: str = "Montserrat Black",
    font_size: float = 64.0,
    highlight_color_bgr: str = VIVID_YELLOW_BGR,
    center_y: int = TARGET_Y,
    script_text: Optional[str] = None,
    active_scale: int = 115
) -> Path:
    """Generates an animated kinetic typography ASS subtitle file with active word scaling."""
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
                is_sentence_end = bool(re.search(r'[.!?]$', raw_word))
                cleaned_word = re.sub(r'[\'\"`,;:.]', '', raw_word).strip().upper()
                if cleaned_word:
                    all_words.append({
                        "word": cleaned_word,
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                        "is_sentence_end": is_sentence_end
                    })
        else:
            text = seg.get("text", "").strip()
            if text:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", 0.0))
                raw_token_list = text.split()
                if raw_token_list:
                    dur_per_word = (seg_end - seg_start) / float(len(raw_token_list))
                    for idx, token in enumerate(raw_token_list):
                        is_sentence_end = bool(re.search(r'[.!?]$', token))
                        w_clean = re.sub(r'[\'\"`,;:.]', '', token).strip().upper()
                        if w_clean:
                            all_words.append({
                                "word": w_clean,
                                "start": seg_start + idx * dur_per_word,
                                "end": seg_start + (idx + 1) * dur_per_word,
                                "is_sentence_end": is_sentence_end
                            })

    if not all_words:
        subs.save(str(output_ass_path))
        return output_ass_path

    if script_text and script_text.strip():
        all_words = reconcile_whisper_with_script(all_words, script_text)

    # Chunk words into groups and eliminate single-word orphan delays
    raw_chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    for item in all_words:
        current_chunk.append(item)
        if len(current_chunk) >= words_per_chunk or item.get("is_sentence_end"):
            raw_chunks.append(current_chunk)
            current_chunk = []
    if current_chunk:
        raw_chunks.append(current_chunk)

    chunks: List[List[Dict[str, Any]]] = []
    for ch in raw_chunks:
        # Merge orphan single-word chunk into previous chunk only if within same sentence and total <= 3 words
        if len(ch) == 1 and chunks and not chunks[-1][-1].get("is_sentence_end") and len(chunks[-1]) < 3:
            chunks[-1].extend(ch)
        else:
            chunks.append(ch)

    magnitude_keywords = {
        "TRILLIONS", "BILLIONS", "MILLIONS", "INSTANTLY", "COLLAPSE",
        "QUANTUM", "SHREDS", "ATOMIC", "BREAK", "ZERO", "EXPOSING"
    }

    for c_idx, chunk in enumerate(chunks):
        next_chunk_start_ms = int(chunks[c_idx + 1][0]["start"] * 1000) if c_idx < len(chunks) - 1 else None
        for active_idx, active_word in enumerate(chunk):
            w_start_ms = int(active_word["start"] * 1000)
            w_end_ms = int(active_word["end"] * 1000)

            if w_end_ms <= w_start_ms:
                w_end_ms = w_start_ms + 250

            # Bridge micro-gaps between words within the same chunk
            if active_idx < len(chunk) - 1:
                next_w_start_ms = int(chunk[active_idx + 1]["start"] * 1000)
                if next_w_start_ms > w_start_ms and next_w_start_ms - w_end_ms < 450:
                    w_end_ms = next_w_start_ms
            elif next_chunk_start_ms is not None:
                # Bridge short pause to next chunk (up to 550ms) to prevent subtitle box blinking
                if next_chunk_start_ms > w_start_ms and (next_chunk_start_ms - w_end_ms) < 550:
                    w_end_ms = next_chunk_start_ms

            formatted_words: List[str] = []
            for j, item in enumerate(chunk):
                word_text = item["word"]
                if j == active_idx:
                    scale = active_scale + 10 if word_text in magnitude_keywords else active_scale
                    tag = f"{{\\c{highlight_color_bgr}}}{{\\fscx{scale}\\fscy{scale}}}{word_text}{{\\fscx100\\fscy100}}{{\\c{PRIMARY_COLOR_BGR}}}"
                    formatted_words.append(tag)
                else:
                    tag = f"{{\\fscx100\\fscy100}}{{\\c{PRIMARY_COLOR_BGR}}}{word_text}"
                    formatted_words.append(tag)

            event_body = " ".join(formatted_words)
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
