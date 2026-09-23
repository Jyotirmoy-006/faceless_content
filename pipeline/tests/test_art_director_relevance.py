"""Unit tests for ArtDirector visual shot unpacking, semantic relevance gating, and ComfyUI generation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.art_director import (
    CURATED_NEGATIVE_PROMPT,
    RELEVANCE_THRESHOLD,
    ShotPlan,
    compute_relevance_score,
    is_concrete_entity,
    source_shot_asset,
    unpack_script_shots,
)
from pipeline.core.schema import Script, ScriptSegment


def test_unpack_script_shots_completeness():
    """Validates that unpack_script_shots unpacks 100% of visual_shots across all segments (12-15 shots)."""
    segments = [
        ScriptSegment(
            segment_index=i,
            narration=f"Segment {i} narration text.",
            visual_query=f"Technology topic {i}",
            visual_shots=[
                f"Angle A for segment {i}",
                f"Angle B for segment {i}",
                f"Angle C for segment {i}",
            ],
            duration_seconds=6.0
        )
        for i in range(1, 6)
    ]
    script = Script(
        topic="Future of AI technology",
        hook="Hook title for testing",
        segments=segments,
        target_duration=30,
        cta="Subscribe now"
    )

    plans = unpack_script_shots(script, target_total_duration=30.0)

    # Exactly 5 segments * 3 shots = 15 distinct shots
    assert len(plans) == 15
    # Verify durations are bounded between 1.2s and 2.2s (max 2.5s)
    for p in plans:
        assert 1.2 <= p.target_duration <= 2.5
    # Verify alternating motions
    motions = [p.motion_type for p in plans]
    for i in range(len(motions) - 1):
        assert motions[i] != motions[i + 1]


def test_is_concrete_entity():
    """Validates discrimination between concrete physical entities and abstract concepts."""
    assert is_concrete_entity("silicon microchip glowing circuit") is True
    assert is_concrete_entity("modern skyscraper city skyline night") is True
    assert is_concrete_entity("server rack blinking green leds") is True
    assert is_concrete_entity("philosophy of human wisdom and logic") is False
    assert is_concrete_entity("abstract thought process and intelligence") is False


def test_compute_relevance_score_and_rejection():
    """Validates that relevant stock footage scores >= 0.78 while drift (snow/clubbers) is rejected (< 0.78)."""
    query = "futuristic microchip glowing silicon circuit"

    # Matching candidate
    good_cand = {
        "url": "https://www.pexels.com/video/futuristic-microchip-circuit-board-12345/",
        "tags": ["microchip", "silicon", "circuit", "technology", "futuristic"]
    }
    score_good = compute_relevance_score(query, good_cand)
    assert score_good >= RELEVANCE_THRESHOLD

    # Hallucinated candidate 1: falling snow
    snow_drift = {
        "url": "https://www.pexels.com/video/snow-falling-winter-trees-99999/",
        "tags": ["snow", "winter", "cold", "forest", "nature"]
    }
    score_snow = compute_relevance_score(query, snow_drift)
    assert score_snow < RELEVANCE_THRESHOLD
    assert score_snow <= 0.2

    # Hallucinated candidate 2: dancing clubber for luxury fashion or tech
    club_drift = {
        "url": "https://www.pexels.com/video/woman-dancing-party-club-disco-88888/",
        "tags": ["party", "dance", "club", "disco", "nightlife"]
    }
    score_club = compute_relevance_score(query, club_drift)
    assert score_club < RELEVANCE_THRESHOLD
    assert score_club <= 0.2


@patch("pipeline.agents.art_director.generate_comfyui_shot")
def test_source_shot_asset_hallucination_triggers_comfyui(mock_generate, tmp_path):
    """Validates that when Pexels metadata drops below 0.78 relevance, ArtDirector falls back to ComfyUI."""
    mock_generate.side_effect = lambda prompt, output_path, duration, **kwargs: output_path.touch()

    shot = ShotPlan(
        segment_index=1,
        shot_index=1,
        query="cyberpunk glowing server room",
        target_duration=2.0,
        motion_type="zoom_in"
    )

    # Mock Pexels search returning unrelated snow video
    fake_pexels_resp = MagicMock()
    fake_pexels_resp.status_code = 200
    fake_pexels_resp.json.return_value = {
        "videos": [{
            "url": "https://www.pexels.com/video/falling-snow-winter-forest-123/",
            "tags": ["snow", "winter", "trees"],
            "video_files": [{"file_type": "video/mp4", "width": 1080, "link": "https://example.com/snow.mp4"}]
        }]
    }

    with patch.dict("os.environ", {"PEXELS_API_KEY": "fake_key"}), \
         patch("requests.get", return_value=fake_pexels_resp):
        out_path, source = source_shot_asset(shot, cache_dir=tmp_path, allow_pexels_fallback=True)

    assert source == "comfyui"
    mock_generate.assert_called_once()
    assert "ugly, blurry, low quality, distorted, watermark" in CURATED_NEGATIVE_PROMPT
    assert "jpeg artifacts" in CURATED_NEGATIVE_PROMPT


def test_clean_pexels_query():
    """Validates that camera directions and filler adjectives are cleanly stripped."""
    from pipeline.agents.art_director import clean_pexels_query
    q1 = "extreme close up macro shot of smartphone battery exploding in test chamber"
    assert "battery" in clean_pexels_query(q1)
    assert "extreme close up" not in clean_pexels_query(q1)

    q2 = "drone shot gigafactory solar panels glowing sunset"
    cleaned = clean_pexels_query(q2)
    assert "solar" in cleaned and "panels" in cleaned
    assert "drone shot" not in cleaned


def test_real_pexels_slug_matching_empty_tags():
    """Validates that genuine stock footage with empty tags and short URL slugs passes relevance gating."""
    from pipeline.agents.art_director import compute_relevance_score, RELEVANCE_THRESHOLD
    query = "A frustrated driver staring at charging station in rain"
    real_cand = {
        "url": "https://www.pexels.com/video/a-woman-doing-her-shopping-while-charging-her-car-4818165/",
        "tags": []
    }
    score = compute_relevance_score(query, real_cand)
    assert score >= RELEVANCE_THRESHOLD


def test_is_mock_asset_detection(tmp_path):
    """Validates detection and purging of stale mock 512x512 assets."""
    from pipeline.agents.art_director import is_mock_asset
    vid = tmp_path / "mock_cut.mp4"
    vid.write_bytes(b"mock" * 100)
    png = tmp_path / "mock_cut.png"
    png.write_bytes(b"small_png_mock" * 100)  # Under 20KB

    assert is_mock_asset(vid) is True

    # Genuine video without companion mock PNG
    real_vid = tmp_path / "real_cut.mp4"
    real_vid.write_bytes(b"real_video_data" * 1000)
    assert is_mock_asset(real_vid) is False


def test_generic_modifier_drift_rejected():
    """Validates that matches containing only generic modifiers (e.g. escalator for production lines) are rejected."""
    from pipeline.agents.art_director import compute_relevance_score, RELEVANCE_THRESHOLD
    query = "Commercial production lines in high tech gigafactory"
    # Escalator video matching only on generic "moving" / "lines"
    escalator_cand = {
        "url": "https://www.pexels.com/video/people-moving-on-escalator-in-mall-98765/",
        "tags": ["moving", "lines", "indoor", "mall"]
    }
    score = compute_relevance_score(query, escalator_cand)
    assert score < RELEVANCE_THRESHOLD
    assert score <= 0.15


def test_warehouse_forklift_forbidden_drift():
    """Validates that warehouse/forklift clips are rejected for tech/cyber queries."""
    from pipeline.agents.art_director import compute_relevance_score, RELEVANCE_THRESHOLD
    query = "cyberpunk hacker terminal red alarm server breach"
    forklift_cand = {
        "url": "https://www.pexels.com/video/warehouse-worker-driving-forklift-boxes-112233/",
        "tags": ["warehouse", "forklift", "worker", "boxes", "logistics"]
    }
    score = compute_relevance_score(query, forklift_cand)
    assert score < RELEVANCE_THRESHOLD
    assert score <= 0.05


def test_deduplication_prevents_duplicate_clip_usage(tmp_path):
    """Validates that used_paths prevents duplicate asset reuse across shots."""
    shot1 = ShotPlan(
        segment_index=1,
        shot_index=1,
        query="futuristic quantum computer glowing core",
        target_duration=2.0,
        motion_type="zoom_in"
    )
    shot2 = ShotPlan(
        segment_index=5,
        shot_index=3,
        query="futuristic quantum computer glowing core",
        target_duration=2.0,
        motion_type="zoom_out"
    )

    used_paths = set()
    with patch("pipeline.agents.art_director.generate_comfyui_shot") as mock_gen:
        mock_gen.side_effect = lambda prompt, output_path, duration, **kwargs: output_path.touch()

        p1, _ = source_shot_asset(shot1, cache_dir=tmp_path, allow_pexels_fallback=False, used_paths=used_paths)
        assert p1 in used_paths

        p2, _ = source_shot_asset(shot2, cache_dir=tmp_path, allow_pexels_fallback=False, used_paths=used_paths)
        assert p2 in used_paths
        # p1 and p2 must NOT be identical files
        assert p1.name != p2.name


