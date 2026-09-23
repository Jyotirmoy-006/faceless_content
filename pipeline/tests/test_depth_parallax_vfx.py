"""Unit tests for Depth Anything V2 Small, 2.5D Parallax Engine, and Particle VFX Compositor."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.parallax_engine import render_parallax_clip
from pipeline.core.particle_vfx import (
    ParticleSystem,
    composite_particles_on_frame,
    select_particle_style,
)
from pipeline.workers.depth_worker import preflight_vram_check


def test_depth_preflight_vram_check():
    """Validates preflight_vram_check threshold enforcement."""
    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(stdout="2048\n", returncode=0)
        assert preflight_vram_check(min_free_mb=1024) == 2048

    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(stdout="512\n", returncode=0)
        with pytest.raises(RuntimeError, match="below 1024 MB"):
            preflight_vram_check(min_free_mb=1024, max_attempts=1, backoff_seconds=0.01)


def test_particle_system_styles():
    """Verifies all defined particle styles produce valid ParticleSystems and overlays."""
    styles = ["cyber_glints", "fire_embers", "light_streaks", "subtle_dust"]
    for style in styles:
        ps = ParticleSystem(style=style, num_particles=20, width=270, height=480)
        assert ps.style == style
        assert ps.num_particles == 20

        overlay = ps.render_overlay(t=0.5)
        assert overlay.shape == (480, 270, 4)
        assert overlay.dtype == np.uint8


def test_select_particle_style_heuristics():
    """Validates heuristic keyword matching for particle styles."""
    assert select_particle_style("glowing quantum computer mainframe") == "cyber_glints"
    assert select_particle_style("ancient battlefield fire and flames") == "fire_embers"
    assert select_particle_style("cinematic galaxy anamorphic lens flare") == "light_streaks"
    assert select_particle_style("calm coffee shop interior") == "subtle_dust"


def test_composite_particles_on_frame():
    """Validates sparse alpha blending of particle overlay onto RGB frame."""
    frame = np.full((480, 270, 3), 50, dtype=np.uint8)
    ps = ParticleSystem(style="cyber_glints", num_particles=30, width=270, height=480)
    overlay = ps.render_overlay(t=1.0)

    composited = composite_particles_on_frame(frame, overlay)
    assert composited.shape == (480, 270, 3)
    assert composited.dtype == np.uint8


def test_render_parallax_clip(tmp_path):
    """Validates 2.5D depth-displacement parallax clip generation."""
    img = Image.new("RGB", (256, 448), color=(100, 150, 200))
    depth = Image.new("L", (256, 448), color=128)

    img_p = tmp_path / "test_still.png"
    depth_p = tmp_path / "test_depth.png"
    img.save(img_p)
    depth.save(depth_p)

    clip = render_parallax_clip(
        image=img_p,
        depth_map=depth_p,
        duration=1.0,
        fps=30,
        out_size=(270, 480),
        motion_type="zoom_in",
        particle_style="cyber_glints"
    )

    frame_0 = clip.get_frame(0.0)
    frame_end = clip.get_frame(0.9)

    assert frame_0.shape == (480, 270, 3)
    assert frame_end.shape == (480, 270, 3)
    assert frame_0.dtype == np.uint8
