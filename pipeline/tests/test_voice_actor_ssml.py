"""Unit tests for VoiceActor dynamic SSML pacing, cadence, and EBU R128 audio normalization."""

import os
import sys
from pathlib import Path
import pytest
from pydub import AudioSegment

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.voice_actor import (
    DEFAULT_VOICE,
    FALLBACK_VOICE,
    generate_ssml,
    parse_ssml_tokens,
    synthesize_segment,
)
from pipeline.core.audio_processor import measure_loudness_ebu_r128


def test_generate_ssml_hook():
    """Validates that hooks inject urgency with rate="+16%" and pitch="-2Hz"."""
    hook_text = "Stop scrolling! This hidden secret will shock you."
    ssml = generate_ssml(hook_text, is_hook=True, voice=DEFAULT_VOICE)

    assert 'rate="+16%"' in ssml
    assert 'pitch="-2Hz"' in ssml
    assert f'<voice name="{DEFAULT_VOICE}">' in ssml
    assert '<break time="200ms"/>' in ssml


def test_generate_ssml_punctuation_breaks():
    """Validates that commas/dashes inject 120ms breaks and sentences inject 200ms breaks."""
    text = "Wait, listen closely — this is huge. Are you ready?"
    ssml = generate_ssml(text, is_hook=False)

    assert 'rate="+12%"' in ssml
    assert 'pitch="+0Hz"' in ssml
    assert '<break time="120ms"/>' in ssml
    assert '<break time="200ms"/>' in ssml


def test_parse_ssml_tokens():
    """Validates that SSML is parsed into phrases and exact pause millisecond intervals."""
    ssml = (
        '<speak><voice name="en-US-AndrewMultilingualNeural">'
        '<prosody rate="+8%" pitch="-2Hz">'
        'Attention, <break time="180ms"/> look at this. <break time="320ms"/>'
        '</prosody></voice></speak>'
    )
    phrases, rate, pitch = parse_ssml_tokens(ssml)

    assert rate == "+8%"
    assert pitch == "-2Hz"
    assert len(phrases) == 2
    assert phrases[0][0] == "Attention,"
    assert phrases[0][1] == 180
    assert "look at this." in phrases[1][0]
    assert phrases[1][1] == 320


def test_synthesize_segment_audio_normalization(tmp_path):
    """Validates audio synthesis, silence insertion, and strict EBU R128 -14 LUFS normalization."""
    out_wav = tmp_path / "voice_actor_test.wav"
    t_dir = tmp_path / "va_temp"

    # Use mock edge failure to force local piper synthesis or fallback
    final_path, engine, duration, pause = synthesize_segment(
        text="The quantum revolution is already here, changing everything forever.",
        output_path=out_wav,
        voice=DEFAULT_VOICE,
        is_hook=True,
        temp_dir=t_dir,
        mock_edge_failure=True,  # Test offline resilience
        target_lufs=-14.0,
        target_tp=-1.5
    )

    assert final_path.exists()
    assert final_path.stat().st_size > 1000
    assert engine == "piper-fallback"
    assert duration > 1.0
    assert pause >= 0.18

    # Verify audio properties: 48kHz, mono
    audio = AudioSegment.from_file(str(final_path))
    assert audio.frame_rate == 48000
    assert audio.channels == 1

    # Verify EBU R128 loudness
    measured = measure_loudness_ebu_r128(final_path)
    assert abs(measured["input_i"] - (-14.0)) <= 1.5
    assert measured["input_tp"] <= -1.4
