"""Unit tests for Editor micro-cuts pacing, alternating zoom motion, and NVENC render parameters."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.editor import (
    HARD_MAX_CUT_DURATION_S,
    MAX_CUT_DURATION_S,
    MIN_CUT_DURATION_S,
    calculate_beat_cuts,
)
from pipeline.core.schema import Script, ScriptSegment
from pipeline.workers.render_worker import build_ffmpeg_cmd


def test_calculate_beat_cuts_pacing_bounds():
    """Validates that all micro-cuts are strictly bounded within 1.2s to 2.2s (hard max 2.5s)."""
    segments = [
        ScriptSegment(
            segment_index=1,
            narration="Hook segment narration.",
            visual_query="Topic 1",
            visual_shots=["Shot 1A", "Shot 1B", "Shot 1C"],
            duration_seconds=5.5
        ),
        ScriptSegment(
            segment_index=2,
            narration="Segment 2 narration.",
            visual_query="Topic 2",
            visual_shots=["Shot 2A", "Shot 2B", "Shot 2C", "Shot 2D"],
            duration_seconds=6.8
        ),
        ScriptSegment(
            segment_index=3,
            narration="Segment 3 narration.",
            visual_query="Topic 3",
            visual_shots=["Shot 3A", "Shot 3B"],
            duration_seconds=4.2
        )
    ]
    script = Script(
        topic="Editor pacing test",
        hook="Hook for editor microcuts",
        segments=segments,
        target_duration=16,
        cta="Follow for more"
    )

    shot_assets = [
        (Path(f"asset_{i}.mp4"), f"Shot {i}") for i in range(15)
    ]

    micro_cuts, cut_timestamps = calculate_beat_cuts(
        script=script,
        segment_durations=[5.5, 6.8, 4.2],
        shot_assets=shot_assets
    )

    assert len(micro_cuts) >= 8
    for cut in micro_cuts:
        assert MIN_CUT_DURATION_S <= cut.duration <= HARD_MAX_CUT_DURATION_S, (
            f"Cut duration {cut.duration}s exceeds required bounds [{MIN_CUT_DURATION_S}, {HARD_MAX_CUT_DURATION_S}]"
        )

    # Verify alternating zoom motions
    for i in range(len(micro_cuts) - 1):
        assert micro_cuts[i].motion_type != micro_cuts[i + 1].motion_type


def test_render_worker_nvenc_flags(tmp_path):
    """Validates that build_ffmpeg_cmd configures -preset p4 -rc vbr -cq 23 -pix_fmt yuv420p when NVENC is available."""
    video_path = tmp_path / "master.mp4"
    video_path.touch()
    audio_path = tmp_path / "audio.wav"
    audio_path.touch()
    ass_path = tmp_path / "captions.ass"
    ass_path.touch()
    out_path = tmp_path / "final.mp4"

    with patch("pipeline.workers.render_worker.check_nvenc_available", return_value=True):
        cmd = build_ffmpeg_cmd(
            video_path=video_path,
            audio_path=audio_path,
            ass_path=ass_path,
            output_path=out_path,
            audio_duration=15.0
        )

    cmd_str = " ".join(cmd)
    assert "-c:v h264_nvenc" in cmd_str
    assert "-preset p4" in cmd_str
    assert "-rc vbr" in cmd_str
    assert "-cq 23" in cmd_str
    assert "-pix_fmt yuv420p" in cmd_str
    # Explicit stream mappings for video input 0 and audio input 1
    assert "-map 0:v:0" in cmd_str
    assert "-map 1:a:0" in cmd_str

    # Without separate audio input, map video from 0 and optional audio from 0
    cmd_no_audio = build_ffmpeg_cmd(
        video_path=video_path,
        audio_path=None,
        output_path=out_path
    )
    cmd_no_audio_str = " ".join(cmd_no_audio)
    assert "-map 0:v:0" in cmd_no_audio_str
    assert "-map 0:a?" in cmd_no_audio_str
