# Product Requirements Document (PRD)

## Project: Faceless Video Automation Engine — Phase 2: Autonomous Production & Scaling

### Overview
Transform the single-video rendering pipeline into a 24/7 autonomous production engine that schedules multi-niche video batches overnight, enforces YouTube API quota limits, publishes across YouTube Shorts and Instagram Reels, and ingests audience performance analytics.

### Task Specifications

#### TASK-02-01: Autonomous Overnight Batch Scheduler & Niche Rotator
- **File**: `pipeline/scheduler/overnight_scheduler.py`
- **Tests**: `pipeline/tests/test_overnight_scheduler.py`
- **Requirements**:
  1. Configurable interval timer / cron execution.
  2. Niche rotation cycle: `tech` → `finance` → `psychology` → `history`.
  3. YouTube quota guard: calculates available quota units before enqueuing batch.
  4. Expose status endpoints in `pipeline/dashboard/app.py`.

#### TASK-02-02: Multi-Platform Publisher & Webhook Notification Engine
- **File**: `pipeline/agents/publisher.py`, `pipeline/core/notifier.py`
- **Tests**: `pipeline/tests/test_multiplatform_publisher.py`
- **Requirements**:
  1. Unified `publish_to_all_destinations` method supporting YouTube Shorts and Instagram Reels.
  2. Discord and Telegram webhook notification dispatcher.
  3. Alerts on job completion, circuit breaker trips, and daily digest.

#### TASK-02-03: Post-Publishing Analytics Telemetry & Watchdog
- **File**: `pipeline/workers/analytics_worker.py`, `pipeline/core/watchdog.py`
- **Tests**: `pipeline/tests/test_analytics_watchdog.py`
- **Requirements**:
  1. Background telemetry worker polling YouTube Analytics API.
  2. Stores views, likes, and retention in SQLite jobs database.
  3. Watchdog monitor detecting GPU memory leaks and recovering stranded jobs.
