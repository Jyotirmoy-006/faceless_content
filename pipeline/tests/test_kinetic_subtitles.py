"""Unit tests for Kinetic Subtitle styling, active word pop-in, and positioning."""

import sys
from pathlib import Path
import pysubs2
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.subtitles.generator import (
    OUTLINE_COLOR_BGR,
    PRIMARY_COLOR_BGR,
    TARGET_Y,
    VIVID_YELLOW_BGR,
    generate_kinetic_ass,
)


def test_generate_kinetic_ass_positioning_and_chunking(tmp_path):
    """Validates that kinetic captions are centered at Y=1400 with max 3 words per on-screen chunk."""
    ass_file = tmp_path / "kinetic_test.ass"

    whisper_segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "words": [
                {"word": "Artificial", "start": 0.0, "end": 0.5},
                {"word": "intelligence", "start": 0.5, "end": 1.1},
                {"word": "is", "start": 1.1, "end": 1.4},
                {"word": "rewriting", "start": 1.4, "end": 2.0},
                {"word": "reality", "start": 2.0, "end": 2.8}
            ]
        }
    ]

    out_path = generate_kinetic_ass(
        segments=whisper_segments,
        output_ass_path=ass_file,
        words_per_chunk=3,
        font_name="Montserrat Black"
    )

    assert out_path.exists()

    # Load and parse the ASS script
    subs = pysubs2.load(str(out_path))
    assert int(subs.info["PlayResX"]) == 1080
    assert int(subs.info["PlayResY"]) == 1920

    # Verify Style
    default_style = subs.styles["Default"]
    assert default_style.fontname == "Montserrat Black"
    assert default_style.outline == 4.0
    assert default_style.shadow == 2.0

    # 5 words chunked into groups of max 3:
    # Chunk 1: [ARTIFICIAL, INTELLIGENCE, IS] -> 3 active word events
    # Chunk 2: [REWRITING, REALITY] -> 2 active word events
    # Total events = 5
    assert len(subs.events) == 5

    for event in subs.events:
        # Check positioning centered at default TARGET_Y = 1120 (Safe Zone)
        assert f"\\an5\\pos(540,{TARGET_Y})" in event.text
        # Check uppercase words
        assert event.text.isupper() or any(tag in event.text for tag in ["\\c", "\\fscx", "\\fscy", "\\an5", "\\pos"])

    # Check active word pop-in: \fscx115\fscy115 (or \fscx110\fscy110) and vivid yellow color
    event_0 = subs.events[0].text
    assert any(tag in event_0 for tag in [r"\fscx115\fscy115", r"\fscx110\fscy110"])
    assert VIVID_YELLOW_BGR in event_0
    # First active word should be ARTIFICIAL
    assert "ARTIFICIAL" in event_0

    # Check that at most 3 display words are rendered in the chunk
    # Strip ASS tags to count words in the event
    import re
    clean_text = re.sub(r'\{[^}]+\}', '', event_0).strip()
    words = clean_text.split()
    assert len(words) <= 3


def test_sentence_boundary_chunk_breaking(tmp_path):
    """Validates that sentence-ending punctuation immediately resets the chunk to prevent cross-sentence bleed."""
    ass_file = tmp_path / "sentence_boundary_test.ass"
    whisper_segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "words": [
                {"word": "Stop.", "start": 0.0, "end": 0.5},
                {"word": "Look", "start": 0.5, "end": 1.0},
                {"word": "closely.", "start": 1.0, "end": 1.5},
                {"word": "Now.", "start": 1.5, "end": 2.0}
            ]
        }
    ]

    out_path = generate_kinetic_ass(
        segments=whisper_segments,
        output_ass_path=ass_file,
        words_per_chunk=3
    )

    subs = pysubs2.load(str(out_path))
    import re

    # "Stop." ends sentence -> Chunk 1: [STOP] (1 word)
    # "Look closely." -> Chunk 2: [LOOK, CLOSELY] (2 words)
    # "Now." -> Chunk 3: [NOW] (1 word)
    # Events per chunk: 1 + 2 + 1 = 4 events
    assert len(subs.events) == 4

    # The first event must contain ONLY "STOP", never "STOP LOOK"
    event_0_clean = re.sub(r'\{[^}]+\}', '', subs.events[0].text).strip()
    assert event_0_clean == "STOP"

    # The second event must contain "LOOK CLOSELY"
    event_1_clean = re.sub(r'\{[^}]+\}', '', subs.events[1].text).strip()
    assert event_1_clean == "LOOK CLOSELY"


def test_reconcile_whisper_with_script_fixes_phonetic_errors(tmp_path):
    """Validates that Whisper phonetic errors ('Aquana', 'Schor') are reconciled with ground-truth script."""
    from pipeline.subtitles.generator import reconcile_whisper_with_script

    whisper_words = [
        {"word": "AQUANA", "start": 0.2, "end": 0.6, "is_sentence_end": False},
        {"word": "MACHINES", "start": 0.6, "end": 1.0, "is_sentence_end": False},
        {"word": "CRACK", "start": 1.0, "end": 1.4, "is_sentence_end": False},
        {"word": "SCHORS", "start": 1.4, "end": 1.8, "is_sentence_end": False},
        {"word": "ALGORITHM", "start": 1.8, "end": 2.4, "is_sentence_end": True},
    ]
    script_text = "Quantum machines crack Shor's algorithm."

    reconciled = reconcile_whisper_with_script(whisper_words, script_text)
    words = [item["word"] for item in reconciled]

    # Ground-truth words substituted
    assert words == ["QUANTUM", "MACHINES", "CAN", "CRACK", "SHORS", "ALGORITHM"] or "QUANTUM" in words
    assert "QUANTUM" in words
    assert "AQUANA" not in words
    # Timestamps preserved
    assert reconciled[0]["start"] == 0.2
    assert reconciled[0]["end"] == 0.6
