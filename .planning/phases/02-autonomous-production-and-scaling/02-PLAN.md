# Phase 2 Plan: Autonomous Production & Scaling

**Milestone:** v1.0.0 — Production-Grade Faceless Automation Engine  
**Phase:** Phase 2: Autonomous Production & Scaling  
**Status:** READY TO EXECUTE  

---

## Executive Summary
Transform the verified single-video rendering pipeline into an autonomous, 24/7 self-healing production facility. The system will independently schedule multi-niche video batches overnight, enforce platform quota limits, publish across YouTube Shorts and Instagram Reels, collect audience engagement analytics, and dispatch operational notifications via Discord/Telegram webhooks.

---

## Execution Waves

### Wave 1: Autonomous Overnight Batch Scheduler & Niche Rotator (Plan 02-01)
- **Goal:** Automatically enqueue and sequentially render batches of videos (e.g. 3–8 videos) during off-peak hours without operator interaction.
- **Key Deliverables:**
  1. `pipeline/scheduler/overnight_scheduler.py`:
     - Configurable cron/interval timer (default: runs at 02:00 AM local time or every N hours).
     - Niche rotation queue: cycles through `tech` → `finance` → `psychology` → `history`.
     - Quota guard: calculates remaining daily YouTube API quota (10,000 units max) and enqueues only `min(batch_size, remaining_quota // 1600)` jobs.
  2. `pipeline/dashboard/app.py`:
     - Add endpoints `/api/scheduler/status`, `/api/scheduler/toggle`, and `/api/scheduler/trigger`.
     - Expose scheduler state on the Midnight Slate control center.
  3. `pipeline/tests/test_overnight_scheduler.py`:
     - Test niche rotation order.
     - Test quota-bounded batch calculation.
     - Test mock overnight run.

### Wave 2: Multi-Platform Publisher & Webhook Notification Engine (Plan 02-02)
- **Goal:** Unify publishing to YouTube Shorts and Instagram Reels with rich operational alerts.
- **Key Deliverables:**
  1. `pipeline/agents/publisher.py`:
     - Unified `publish_to_all_destinations(video_path, metadata)` method.
     - Exponential backoff for Instagram video container processing.
     - Graceful handling of OAuth token refreshes.
  2. `pipeline/core/notifier.py`:
     - Webhook dispatcher supporting Discord and Telegram.
     - Alerts on:
       - Video successfully rendered & published.
       - Circuit breaker trips (e.g. LLM failure, ComfyUI offline).
       - Daily morning production digest (total rendered, total published, quota remaining).
  3. `pipeline/tests/test_multiplatform_publisher.py`:
     - Test multi-destination dispatch.
     - Test webhook payload formatting and notification triggers.

### Wave 3: Post-Publishing Analytics Telemetry & Watchdog (Plan 02-03)
- **Goal:** Ingest video performance metrics and maintain 24/7 system health.
- **Key Deliverables:**
  1. `pipeline/workers/analytics_worker.py`:
     - Background telemetry poller querying YouTube Analytics (`viewCount`, `likeCount`, `commentCount`).
     - Stores engagement metrics in `jobs` database table (`views`, `likes`, `retention_rate`).
  2. `pipeline/core/watchdog.py`:
     - Health monitor checking GPU VRAM availability, ComfyUI server responsiveness, and stale queue locks.
     - Auto-recovers stranded jobs if a worker crashes.
  3. `pipeline/tests/test_analytics_watchdog.py`:
     - Test metrics parsing and database persistence.
     - Test stale lock recovery.

---

## Verification Plan

### Automated Test Suite
1. `uv run pytest pipeline/tests/test_overnight_scheduler.py -v`
2. `uv run pytest pipeline/tests/test_multiplatform_publisher.py -v`
3. `uv run pytest pipeline/tests/test_analytics_watchdog.py -v`
4. Full regression: `uv run pytest pipeline/tests/ -k "not comfyui_stages" -q`

### Acceptance Criteria
- [ ] Scheduler successfully enqueues a multi-niche batch without exceeding YouTube 10,000 unit quota.
- [ ] Sequential execution runs overnight without GPU VRAM exhaustion or memory leaks.
- [ ] Published videos report back live view/like metrics to the dashboard database.
- [ ] Webhook alerts deliver notifications upon job completion and circuit breaker events.
