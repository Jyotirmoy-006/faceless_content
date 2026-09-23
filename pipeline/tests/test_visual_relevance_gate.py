"""Unit and Integration Tests for HeadOfVisualRelevance Quality Gate.

Verifies:
1. Rejection of deliberately mismatched clips (e.g. Roman pottery for quantum microchip).
2. Rejection of forbidden drift clips (e.g. beach party for server infrastructure).
3. Approval of semantically aligned clips.
4. Corrective rerouting to ComfyUI upon visual gate rejection.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.agents.art_director import ShotPlan, source_shot_asset
from pipeline.agents.department_heads.head_of_visual_relevance import (
    HeadOfVisualRelevance,
    head_of_visual_relevance,
)


def test_visual_relevance_catches_deliberate_mismatch():
    """Demonstrates catching a deliberately mismatched test case.

    Injects an ancient Roman pottery clip against a 'quantum microchip' query,
    confirming uncompromised rejection by HeadOfVisualRelevance.
    """
    gate = HeadOfVisualRelevance()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        dummy_clip_path = Path(tf.name)
        dummy_clip_path.write_bytes(b"\x00" * 2048)

    try:
        # Context with quantum computing query but Roman pottery clip
        mismatched_context = {
            "visual_query": "quantum microchip circuit silicon semiconductor",
            "narration": "Engineers engineered a breakthrough quantum computing chip.",
            "clip_metadata": {
                "title": "Ancient Roman Pottery Exhibition",
                "tags": ["roman pottery", "ancient clay vase", "museum ceramics", "archaeology"],
                "url": "https://www.pexels.com/video/ancient-greek-clay-vase-museum-12345/"
            }
        }

        result = gate.inspect(dummy_clip_path, context=mismatched_context)

        assert not result.passed, "Head of Visual Relevance should have REJECTED the mismatched pottery clip."
        assert result.tier == 1
        assert "mismatch" in result.feedback.lower() or "zero semantic overlap" in result.feedback.lower()
        assert result.details["overlap_ratio"] == 0.0

    finally:
        dummy_clip_path.unlink(missing_ok=True)


def test_visual_relevance_catches_forbidden_drift():
    """Validates that forbidden topics (party, beach, dance) are instantly rejected."""
    gate = HeadOfVisualRelevance()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        dummy_clip_path = Path(tf.name)
        dummy_clip_path.write_bytes(b"\x00" * 2048)

    try:
        drift_context = {
            "visual_query": "artificial intelligence datacenter server racks",
            "narration": "Massive compute clusters are processing neural algorithms.",
            "clip_metadata": {
                "title": "Summer Beach Party Dance",
                "tags": ["beach", "party", "dance", "vacation", "summer"],
                "url": "https://www.pexels.com/video/people-dancing-at-beach-club-99999/"
            }
        }

        result = gate.inspect(dummy_clip_path, context=drift_context)

        assert not result.passed, "Head of Visual Relevance should have rejected forbidden drift."
        assert "forbidden" in result.feedback.lower() or "drift" in result.feedback.lower()

    finally:
        dummy_clip_path.unlink(missing_ok=True)


def test_visual_relevance_approves_aligned_clip():
    """Validates that semantically relevant stock assets pass Tier 1."""
    gate = HeadOfVisualRelevance()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        dummy_clip_path = Path(tf.name)
        dummy_clip_path.write_bytes(b"\x00" * 2048)

    try:
        aligned_context = {
            "visual_query": "microchip circuit board silicon",
            "narration": "High density circuit boards are routing trillions of operations.",
            "clip_metadata": {
                "title": "Macro Shot of Microchip Circuit",
                "tags": ["microchip", "circuit board", "silicon", "hardware", "technology"],
                "url": "https://www.pexels.com/video/macro-view-of-computer-chip-and-circuit-54321/"
            }
        }

        result = gate.inspect(dummy_clip_path, context=aligned_context)

        assert result.passed, f"Aligned clip should have PASSED. Feedback: {result.feedback}"
        assert result.score is not None and result.score >= 7.0
        assert "circuit" in result.details["overlap_tokens"] or "microchip" in result.details["overlap_tokens"]

    finally:
        dummy_clip_path.unlink(missing_ok=True)


def test_source_shot_asset_reroutes_to_comfyui_on_relevance_rejection():
    """Tests end-to-end corrective reroute: mismatched Pexels candidate is rejected

    and execution falls back to generating a ComfyUI shot.
    """
    shot = ShotPlan(
        segment_index=1,
        shot_index=1,
        query="quantum microchip circuit silicon semiconductor",
        target_duration=2.0,
        motion_type="zoom_in",
        source_type="pexels",
        narration="A quantum processor running at near zero Kelvin."
    )

    # Mock Pexels returning a mismatched candidate (pottery clip)
    mismatched_candidate = {
        "url": "https://www.pexels.com/video/ancient-greek-clay-vase-museum-12345/",
        "tags": ["roman pottery", "ancient clay vase", "museum ceramics"],
        "video_files": [{"link": "https://fake.url/video.mp4"}]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        with patch("pipeline.agents.art_director.is_comfyui_online", return_value=True), \
             patch("pipeline.agents.art_director.find_best_pexels_candidate", return_value=(mismatched_candidate, 0.45)), \
             patch("pipeline.agents.art_director.generate_comfyui_shot") as mock_comfy:

            # Create dummy output file when mock_comfy is invoked
            def fake_comfy(prompt, output_path, duration, **kwargs):
                p = Path(output_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x00" * 4096)
                return p

            mock_comfy.side_effect = fake_comfy

            with patch.dict("os.environ", {"PEXELS_API_KEY": "dummy_pexels_key"}):
                res_path, src_type = source_shot_asset(
                    shot=shot,
                    cache_dir=cache_dir,
                    allow_pexels_fallback=True
                )

                # Confirm that the mismatched candidate was rejected and ComfyUI was called
                assert src_type == "comfyui", f"Expected 'comfyui' reroute but got '{src_type}'"
                assert mock_comfy.called, "ComfyUI generation worker should have been called upon gate rejection."
                assert shot.source_type == "comfyui"
