"""Procedural 2D Particle VFX Compositing Engine.

Per Rule 1 (GPU MEMORY):
- Pure 2D vectorized NumPy / OpenCV CPU compositing.
- Zero GPU VRAM overhead or allocations.
- Selectable per-niche / per-topic styling (cyber glints, fire embers, light streaks, subtle dust).
- Fast real-time frame generation (<5ms per 1080x1920 frame).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ParticleSystem:
    """Vectorized procedural 2D particle simulation for atmospheric overlay."""

    def __init__(
        self,
        style: str = "cyber_glints",
        num_particles: int = 45,
        width: int = 1080,
        height: int = 1920,
        seed: int = 42
    ):
        self.style = style
        self.num_particles = num_particles
        self.width = width
        self.height = height
        self.seed = seed
        self._init_particles()

    def _init_particles(self) -> None:
        """Initializes randomized particle states with reproducible PRNG seed."""
        rng = np.random.RandomState(self.seed)
        self.x = rng.uniform(0, self.width, self.num_particles).astype(np.float32)
        self.y = rng.uniform(0, self.height, self.num_particles).astype(np.float32)
        self.radius = rng.uniform(2.0, 6.0, self.num_particles).astype(np.float32)
        self.speed_x = rng.uniform(-15.0, 15.0, self.num_particles).astype(np.float32)
        self.speed_y = rng.uniform(-40.0, -10.0, self.num_particles).astype(np.float32)
        self.freq = rng.uniform(1.5, 4.0, self.num_particles).astype(np.float32)
        self.phase = rng.uniform(0, 2 * math.pi, self.num_particles).astype(np.float32)

        if self.style == "fire_embers":
            self.speed_y = rng.uniform(-80.0, -25.0, self.num_particles).astype(np.float32)
            self.radius = rng.uniform(3.0, 8.0, self.num_particles).astype(np.float32)
            self.colors = np.array([
                [rng.randint(230, 255), rng.randint(120, 190), rng.randint(20, 60)]
                for _ in range(self.num_particles)
            ], dtype=np.uint8)
        elif self.style == "cyber_glints":
            self.speed_y = rng.uniform(-30.0, 30.0, self.num_particles).astype(np.float32)
            self.speed_x = rng.uniform(-30.0, 30.0, self.num_particles).astype(np.float32)
            palette = [[0, 235, 255], [0, 160, 255], [255, 200, 50], [180, 70, 255]]
            self.colors = np.array([
                palette[rng.randint(len(palette))]
                for _ in range(self.num_particles)
            ], dtype=np.uint8)
        elif self.style == "light_streaks":
            self.speed_x = rng.uniform(-60.0, 60.0, self.num_particles).astype(np.float32)
            self.speed_y = rng.uniform(-10.0, 10.0, self.num_particles).astype(np.float32)
            self.radius = rng.uniform(4.0, 12.0, self.num_particles).astype(np.float32)
            self.colors = np.array([
                [rng.randint(180, 240), rng.randint(210, 255), 255]
                for _ in range(self.num_particles)
            ], dtype=np.uint8)
        else:  # subtle_dust
            self.speed_y = rng.uniform(-15.0, -5.0, self.num_particles).astype(np.float32)
            self.radius = rng.uniform(1.5, 4.0, self.num_particles).astype(np.float32)
            self.colors = np.array([
                [220, 230, 240] for _ in range(self.num_particles)
            ], dtype=np.uint8)

    def render_overlay(self, t: float) -> np.ndarray:
        """Renders an RGBA overlay image at timestamp t (in seconds)."""
        layer = np.zeros((self.height, self.width, 4), dtype=np.uint8)

        cur_x = (self.x + self.speed_x * t + np.sin(self.freq * t + self.phase) * 20.0) % self.width
        cur_y = (self.y + self.speed_y * t) % self.height
        twinkle = 0.5 + 0.5 * np.sin(self.freq * 2.0 * t + self.phase)

        for i in range(self.num_particles):
            px = int(cur_x[i])
            py = int(cur_y[i])
            rad = max(1, int(self.radius[i]))
            alpha = float(np.clip(twinkle[i] * 0.85, 0.15, 0.95))
            color_rgb = self.colors[i]
            b, g, r = int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0])

            # Draw glowing halo
            halo_rad = rad * 3
            if 0 <= px < self.width and 0 <= py < self.height:
                if self.style == "light_streaks":
                    # Horizontal streak flare
                    streak_w = rad * 6
                    cv2.ellipse(
                        layer, (px, py), (streak_w, rad), 0, 0, 360,
                        (r, g, b, int(alpha * 120)), -1
                    )
                else:
                    cv2.circle(layer, (px, py), halo_rad, (r, g, b, int(alpha * 70)), -1)
                    cv2.circle(layer, (px, py), rad, (r, g, b, int(alpha * 240)), -1)

        return layer


def select_particle_style(prompt_or_niche: str) -> str:
    """Selects the best procedural particle style matching prompt or niche."""
    p = prompt_or_niche.lower()
    if any(k in p for k in ["cyber", "quantum", "tech", "code", "ai", "matrix", "hacker", "digital", "data", "crypto", "server"]):
        return "cyber_glints"
    elif any(k in p for k in ["fire", "ember", "war", "history", "ancient", "mystery", "flame", "battle", "burn", "gold"]):
        return "fire_embers"
    elif any(k in p for k in ["space", "cosmic", "star", "galaxy", "future", "anamorphic", "lens", "light", "cinematic"]):
        return "light_streaks"
    return "subtle_dust"


def composite_particles_on_frame(
    frame_rgb: np.ndarray,
    particle_layer_rgba: np.ndarray
) -> np.ndarray:
    """Blends an RGBA particle overlay onto an RGB video frame using fast sparse alpha masking."""
    h, w = frame_rgb.shape[:2]
    if particle_layer_rgba.shape[:2] != (h, w):
        particle_layer_rgba = cv2.resize(particle_layer_rgba, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = particle_layer_rgba[:, :, 3]
    mask = alpha > 0
    if np.any(mask):
        a = alpha[mask, None].astype(np.float32) * (0.85 / 255.0)
        p = particle_layer_rgba[mask, :3].astype(np.float32)
        orig = frame_rgb[mask].astype(np.float32)
        out = frame_rgb.copy()
        out[mask] = (orig * (1.0 - a * 0.7) + p * a).clip(0, 255).astype(np.uint8)
        return out
    return frame_rgb
