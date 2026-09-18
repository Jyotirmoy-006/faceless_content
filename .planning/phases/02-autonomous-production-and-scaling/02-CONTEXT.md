# Phase 2 Context: Autonomous Production & Scaling

## Objective
Scale the single-video rendering pipeline into an autonomous, self-healing overnight production engine capable of generating, queueing, rendering, and distributing video batches with zero operator intervention.

## Existing Foundation (Phase 1 Deliverables)
- **VoiceActor:** Edge-TTS with dynamic SSML prosody (+8% rate, -2Hz pitch), 180ms/320ms punctuation breaks, Piper fallback, and EBU R128 (-14.0 LUFS / -1.5 dBTP) normalization.
- **ArtDirector:** ComfyUI SD1.5/SDXL-Turbo keyframe generation under GPU lock, 100% `visual_shots` unpacking (12–15 cuts per 30s video), anti-drift semantic gating (relevance >= 0.78).
- **Editor & Render Worker:** Beat-synced micro-cuts (1.2s – 2.2s, max 2.5s), continuous alternating pan/zoom camera motion, NVENC `-preset p4 -rc vbr -cq 23 -pix_fmt yuv420p` with explicit stream mapping (`-map 0:v:0 -map 1:a:0`).
- **Subtitles:** Kinetic, animated ASS captions at Y=1400 with 110% word scale pop in Vivid Yellow (`&H0000FFFF&`).

## Scope Fences (Phase 2 Boundaries)
- **In Scope:**
  1. Overnight batch scheduler daemon running on cron / schedule intervals.
  2. Multi-niche rotation strategy (`tech`, `finance`, `psychology`, `history`).
  3. Quota discipline: automated YouTube quota pre-flight checks (1,600 units/upload vs 10,000 daily budget).
  4. Multi-platform upload queue: YouTube Shorts + Instagram Reels container flow + webhook dispatch.
  5. Telemetry & analytics poller for published jobs (views, likes, retention).
  6. Health monitoring, automated VRAM flush, and notification webhooks (Discord / Telegram).
- **Out of Scope (Deferred to Phase 3):**
  - Advanced ML audience retention feedback loops that alter prompt templates dynamically.
  - Multi-account proxy rotators.
