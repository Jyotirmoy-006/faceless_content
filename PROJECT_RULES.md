# Architectural Rules & System Constitution

> **Mission Protocol**: Every mission in this project must open by stating which of these rules apply to the task at hand and how the implementation plan satisfies each one, before writing any code.

---

### Core System & Resource Isolation (Rules 1 – 8)

1. **GPU MEMORY**: Every GPU-bound stage (Whisper, ComfyUI, NVENC render) runs as an isolated subprocess that exits when complete. Subprocesses are coordinated with an explicit inter-process lock (`pipeline/core/gpu_lock.py`).
2. **NO SILENT CRASHES**: Every external API call (Gemini, edge-tts, Pexels, YouTube API, Instagram Graph API) is wrapped with explicit error handling, fallbacks, and bounded retries.
3. **QUOTA AWARENESS**: YouTube Data API requests are tracked through a central unit-based quota tracker (`pipeline/core/quota_tracker.py`), enforcing budget availability before any call to `videos.insert`.
4. **VALIDATE LLM OUTPUT**: All Gemini JSON responses are schema-validated using Pydantic (`pipeline/core/schema.py`) with bounded retries and fallback templates.
5. **edge-tts IS UNOFFICIAL**: Treat edge-tts as unstable; requests are chunked, jitter-rate-limited, and backed by a local Piper TTS fallback.
6. **INSTAGRAM NEEDS A PUBLIC URL**: Instagram Reels Graph API requires a crawlable temporary public URL (implemented via Backblaze B2 S3-compatible pre-signed URLs with guaranteed cleanup).
7. **SCHEDULER SAFETY**: Pipeline checks run lock files before launching to prevent concurrent or orphaned executions, inspecting process aliveness at the OS level.
8. **NO UNBOUNDED RETRIES**: All retry loops enforce hard limits and escalate to circuit breaker states (`pipeline/core/circuit_breaker.py`).

---

### Media Synthesis & Rendering Standards (Rules 9 – 11)

9. **VISUAL CONSISTENCY**: All video segments (stock footage and AI-generated Ken Burns clips) MUST be normalized to one canonical resolution, constant frame rate, pixel format, and color range before concatenation. No segment enters the final render without passing through this normalization step.
10. **CAPTIONS MUST BE HARD-BURNED**: Subtitles are burned directly into the video frame during the FFmpeg render pass. A soft subtitle stream muxed into the container does NOT satisfy this rule — social platforms do not reliably auto-display embedded subtitle tracks.
11. **AUDIO MUST SOUND CONTINUOUS**: TTS narration is assembled from as few synthesis calls as reliability allows (chunk only when a length/reliability limit actually requires it, never sentence-by-sentence by default), is loudness-normalized to one consistent target across every chunk and any fallback engine, is silence-trimmed at chunk boundaries, and is crossfaded at splice points. A flat, fixed-duration silence between every sentence is not acceptable — pause length should vary naturally with punctuation. The edge-tts API-call rate-limit delay (which exists only to avoid tripping Microsoft's servers) must never be reused as the in-audio pause length.

---

### Quality Assurance, Compliance & Resilience (Rules 12 – 14)

12. **STAGE VERIFICATION**: Every stage's output is checked before the next stage may begin. Cheap structural checks (pure code, zero API cost) always run. A semantic Gemini check only runs where structural checks cannot catch the failure mode. A stage that fails re-attempts via the same corrective-retry-then-circuit-breaker pattern already used by the Scriptwriter — never a bespoke one-off retry mechanism per stage.

13. **COMPLIANCE GATE**: No video reaches Publisher without an explicit LOW-risk verdict from the Compliance Officer (Risk & Safety agent), run separately from and after Chief Critic's quality pass.
    - **QA Proxy & Zero-Duplication**: Reuses or generates a lightweight 480p/15fps compressed QA proxy (< 1.5 MB). Strictly never uploads a duplicate copy to the Gemini File API within the session.
    - **Separate Risk Rubric**: Prompts Gemini 3.6 Flash specifically on YouTube Community Guidelines, Demonetization risk, and ComfyUI SD1.5 copyright/IP-resemblance drift. Structured RiskReport schema includes optional Instagram category for future multi-platform expansion without code rewrites.
    - **Gating Actions**:
      - `LOW`: Proceeds normally to Publisher.
      - `MEDIUM`: Holds video in `HELD_FOR_REVIEW`, emits Notifier warning with reasoning, requires manual human clear.
      - `HIGH`: Blocks entirely, logs to circuit breaker telemetry, emits Notifier error — **never auto-retries the same topic/prompt**.

14. **ACCOUNT RESILIENCE & LLM MANAGEMENT**:
    The LLM account manager must detect and quarantine a failing account rather than assume all configured accounts stay available. The system continues operating on any remaining healthy account and alerts when one goes down. This is detect-and-degrade only — never build anything that masks or evades account-level restrictions.
    - **Deprecated Models Prohibited**: Never reference retired models (e.g. `gemini-1.5-pro` retired ~Sept 2025).
    - **Multi-Tier Model Routing**:
      - High-capability (Creative Director): `gemini-3.1-pro-preview`
      - High-volume agents (Brand Designer, Chief Critic, Compliance Officer, Copywriter, Scriptwriter, Ideator): `gemini-3.6-flash`
      - Budget tasks (Strategist scheduling/metadata JSON): `gemini-3.5-flash-lite`
    - **Multi-Account Static Resilience (3 Google AI Pro accounts)**:
      - *Static assignment by Agent Role*: Deterministic static role assignment across accounts (defensible pattern, never random rotation).
      - *Per-Account Health Tracking*: Persist `consecutive_auth_failures` and `is_quarantined` state across process restarts.
      - *Error Classification*: Distinguish standard rate limits (429 with quota, handled with exponential backoff) from account-level restrictions (repeated 401/403 auth failures or zero-quota 429s) — quarantine only on the latter.
      - *Quarantine Alerting*: Immediately stop routing to quarantined accounts and alert via Notifier: `[ACCOUNT_DOWN] Account N appears restricted - operating on remaining.`
      - *Detect-and-Degrade Only*: Never build anything that masks or evades account restrictions from Google's systems; cleanly degrade remaining roles to healthy accounts.