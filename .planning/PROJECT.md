# Project Specification: Faceless Automation Engine

Autonomous AI-driven studio that generates, animates, narrates, captions, and renders short-form videos with zero human latency.

## Architecture
- **Voice Actor:** Edge-TTS with dynamic SSML prosody (+8% rate, -2Hz pitch on hooks) and EBU R128 (-14.0 LUFS) normalization.
- **Art Director:** Local ComfyUI SD1.5/SDXL-Turbo keyframe generation under GPU lock, anti-drift semantic relevance gating (>= 0.78).
- **Editor:** Beat-synced micro-cuts (1.2s - 2.2s, max 2.5s) with alternating zoompan camera momentum and NVENC encoding.
- **Subtitles:** Kinetic, animated ASS captions at Y=1400 with 110% word scale pop in Vivid Yellow (&H0000FFFF&).
