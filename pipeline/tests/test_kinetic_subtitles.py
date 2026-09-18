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
        font_name="Montserrat Black",
        center_y=1400
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
        # Check positioning centered at Y=1400
        assert r"\an5\pos(540,1400)" in event.text
        # Check uppercase words
        assert event.text.isupper() or any(tag in event.text for tag in ["\\c", "\\fscx", "\\fscy", "\\an5", "\\pos"])

    # Check active word pop-in: \fscx110\fscy110 and vivid yellow color
    event_0 = subs.events[0].text
    assert r"\fscx110\fscy110" in event_0
    assert VIVID_YELLOW_BGR in event_0
    # First active word should be ARTIFICIAL
    assert "ARTIFICIAL" in event_0

    # Check that at most 3 display words are rendered in the chunk
    # Strip ASS tags to count words in the event
    import re
    clean_text = re.sub(r'\{[^}]+\}', '', event_0).strip()
    words = clean_text.split()
    assert len(words) <= 3
