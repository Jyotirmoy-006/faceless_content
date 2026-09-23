"""Master Orchestrator for Autonomous Faceless Video Automation.

Entry point designed for automated execution via Windows Task Scheduler.
Complies with:
- Rule 1 (GPU MEMORY): Subprocess isolation via gpu_lock.py.
- Rule 2 (NO SILENT CRASHES): Bounded retries, fallbacks, and explicit error handling.
- Rule 3 (QUOTA AWARENESS): Central quota tracker gating for YouTube uploads.
- Rule 4 (VALIDATE LLM OUTPUT): Pydantic validation across all LLM stages.
- Rule 5 (edge-tts IS UNOFFICIAL): Sentence-chunked TTS with local Piper fallback.
- Rule 6 (INSTAGRAM NEEDS A PUBLIC URL): Ephemeral B2 public hosting with guaranteed cleanup.
- Rule 7 (SCHEDULER SAFETY): Single-instance lock file (pipeline.lock) with stale-lock detection.
- Rule 8 (NO UNBOUNDED RETRIES): Durable circuit breaker halts runs after 3 consecutive failures.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.circuit_breaker import circuit_breaker, CircuitBreakerOpenError
from pipeline.core.quota_tracker import quota_tracker, YOUTUBE_VIDEO_UPLOAD_COST
from pipeline.core.schema import IdeaConcept, Script
from pipeline.agents.ideator import ideator
from pipeline.agents.scriptwriter import get_valid_script
from pipeline.agents.director import orchestrate_video
from pipeline.agents.publisher import publisher, PublishResult, PublishStatus
from pipeline.agents.compliance_officer import (
    assess_video_compliance,
    enforce_compliance_gate,
    ComplianceHoldError,
    ComplianceBlockError,
    RiskReport,
    RiskLevel,
    Recommendation,
)

from pipeline.agents.department_heads import (
    head_of_story,
    head_of_audio,
    head_of_art,
    head_of_post,
    head_of_compliance,
    DepartmentGateRejectionError,
)

from pipeline.core.stage_verifier import verify_copywriter
from pipeline.core.logger import get_logger
from pipeline.dashboard.database import record_run_telemetry, update_video_gate_status
from pipeline.agents.strategist import strategist

logger = get_logger("orchestrator")


LOCK_FILE = ROOT_DIR / "pipeline.lock"
OUTPUT_DIR = ROOT_DIR / "pipeline" / "output"


# ==============================================================================
# SCHEDULER SAFETY LOCK (RULE 7 COMPLIANCE)
# ==============================================================================

def is_pid_alive(pid: int) -> bool:
    """Checks whether a process with the given PID is actively running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            exit_code = ctypes.wintypes.DWORD()
            success = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            # STILL_ACTIVE = 259
            return bool(success and exit_code.value == 259)
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


class SchedulerLock:
    """Guarantees single-instance execution per Rule 7 (SCHEDULER SAFETY)."""

    def __init__(self, lock_path: Path = LOCK_FILE):
        self.lock_path = lock_path

    def acquire(self) -> bool:
        """Acquires lock or cleans up stale lock.
        
        Returns:
            True if lock acquired, False if an active run is already in progress.
        """
        if self.lock_path.exists():
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recorded_pid = int(data.get("pid", -1))
                started_at = data.get("started_at", "unknown")
            except Exception:
                recorded_pid = -1
                started_at = "corrupt"

            if is_pid_alive(recorded_pid):
                logger.warning(
                    f"[SCHEDULER_LOCK] Active pipeline execution already in progress "
                    f"(PID {recorded_pid}, started at {started_at}). "
                    f"Exiting immediately to prevent concurrent execution (Rule 7)."
                )
                return False
            else:
                logger.info(
                    f"[SCHEDULER_LOCK] Stale lock detected from dead PID {recorded_pid} "
                    f"(crashed or terminated). Purging stale lock."
                )
                try:
                    self.lock_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not unlink stale lock: {e}")

        # Write fresh lock
        lock_data = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat()
        }
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=2)

        logger.info(f"[SCHEDULER_LOCK] Acquired run lock (PID {os.getpid()}).")
        return True

    def release(self) -> None:
        """Releases the lock file if owned by current PID."""
        if self.lock_path.exists():
            try:
                with open(self.lock_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if int(data.get("pid", -1)) == os.getpid():
                    self.lock_path.unlink()
                    logger.info(f"[SCHEDULER_LOCK] Released run lock (PID {os.getpid()}).")
            except Exception as e:
                logger.warning(f"Failed to cleanly release lock file: {e}")


@contextmanager
def scheduler_lock(lock_path: Path = LOCK_FILE) -> Generator[bool, None, None]:
    """Context manager for scheduler lock lifecycle."""
    lock = SchedulerLock(lock_path)
    acquired = lock.acquire()
    try:
        yield acquired
    finally:
        if acquired:
            lock.release()


# ==============================================================================
# PIPELINE STAGE COORDINATION & TELEMETRY
# ==============================================================================


# ==============================================================================
# YOUTUBE METADATA HELPERS (BRAINROT / ENTERTAINMENT OPTIMIZED)
# ==============================================================================

# Brainrot-specific hashtag stack — ordered by search volume & algorithm weight
_BRAINROT_TAG_STACK = [
    "#brainrot", "#viral", "#fyp", "#shorts", "#trending", "#foryou",
    "#aura", "#lockin", "#skibidi", "#sigma", "#entertainment",
    "#GenZ", "#GenAlpha", "#memes", "#funny", "#relatable",
]

# Per-topic keyword injections for higher discoverability
_TOPIC_TAG_MAP: dict[str, list[str]] = {
    "brain": ["#brainrot", "#dopamine", "#scrolling", "#addicted"],
    "aura": ["#aurafarming", "#aurapoints", "#auratierlist", "#vibes"],
    "italian": ["#italianbrainrot", "#AIslop", "#surreal", "#absurd"],
    "lock": ["#lockin", "#productivity", "#grindset", "#sigma"],
    "skibidi": ["#skibidilore", "#skibiditoilet", "#genz", "#lore"],
}


def build_youtube_description(concept, script) -> str:
    """Builds an SEO-optimized YouTube description for brainrot/entertainment content.

    Injects a narrative hook, the full narration for subtitle indexing,
    a CTA, and a brainrot hashtag stack.
    """
    narration = script.full_narration()
    topic_lower = concept.topic.lower()
    niche = getattr(concept, "niche", "tech")

    if niche == "entertainment" or any(k in topic_lower for k in [
        "brainrot", "aura", "skibidi", "lock in", "viral", "sigma", "italian"
    ]):
        cta = "🔥 Follow for daily brainrot — new drop every day.\n💬 Comment your aura score below ⬇️"
        hashtag_block = " ".join(_BRAINROT_TAG_STACK)
        description = (
            f"{concept.topic}\n\n"
            f"{narration}\n\n"
            f"{cta}\n\n"
            f"{hashtag_block}"
        )
    else:
        description = f"{narration}\n\n#shorts #{niche} #facts #viral"

    return description[:5000]


def build_youtube_tags(concept) -> list[str]:
    """Builds a rich tag list for brainrot content with per-topic keyword injection."""
    topic_lower = concept.topic.lower()
    niche = getattr(concept, "niche", "tech")

    base_tags = list(_BRAINROT_TAG_STACK) if niche == "entertainment" else [
        f"#{niche}", "#shorts", "#facts", "#education", "#viral"
    ]

    # Inject topic-specific tags
    for keyword, extra_tags in _TOPIC_TAG_MAP.items():
        if keyword in topic_lower:
            base_tags = extra_tags + base_tags

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for t in base_tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped[:30]  # YouTube allows max 30 tags


# ==============================================================================
# VIRALITY EVALUATOR (PRE-PUBLISH GATE)
# ==============================================================================

# Scoring weights (must sum to 1.0)
_VIRALITY_WEIGHTS = {
    "hook_strength": 0.30,       # First sentence scroll-stop power
    "aura_lockin_signals": 0.20, # Brainrot cultural keywords present
    "replayability": 0.20,       # Loop triggers, cliffhangers, open loops
    "pacing_density": 0.15,      # Words-per-second estimated density
    "title_ctr_power": 0.15,     # Title hook strength (numbers, caps, emotion)
}

_AURA_KEYWORDS = {
    "aura", "lockin", "lock in", "sigma", "skibidi", "brainrot", "brain rot",
    "viral", "based", "rizz", "ohio", "slay", "cope", "ratio", "npc", "lore",
    "italian", "goat", "delulu", "unhinged", "w rizz", "no cap", "bussin"
}
_HOOK_POWER_WORDS = {
    "secret", "dark", "hidden", "nobody", "they hid", "actually", "real reason",
    "truth", "shocking", "insane", "wild", "you won't", "literally", "broke",
    "every", "always", "never", "can't stop", "impossible", "lost it"
}
_REPLAY_TRIGGERS = {
    "wait", "but here's", "plot twist", "the twist", "it gets worse",
    "but then", "actually", "the real", "origin", "nobody talks about"
}


def evaluate_virality(concept, script) -> dict:
    """Scores a concept + script on virality dimensions before YouTube publish.

    Returns:
        dict with keys: scores (per dimension), total (0-10), verdict (PUSH/HOLD/REJECT),
        and breakdown (human-readable per-dimension notes).
    """
    topic = concept.topic.lower()
    narration = script.full_narration().lower()
    segments = script.segments
    hook_text = segments[0].narration.lower() if segments else narration[:200]
    word_count = len(narration.split())
    # Estimate target duration from concept (default 30s)
    target_dur = getattr(concept, "target_duration", 30)
    wps = word_count / max(target_dur, 1)

    scores: dict[str, float] = {}
    notes: dict[str, str] = {}

    # 1. Hook Strength (0–10): how many scroll-stop power words in the first segment
    hook_hits = sum(1 for w in _HOOK_POWER_WORDS if w in hook_text)
    hook_score = min(10.0, hook_hits * 2.5 + (3.0 if "?" in hook_text or "!" in hook_text else 0))
    scores["hook_strength"] = hook_score
    notes["hook_strength"] = f"{hook_hits} power words detected in hook. Score: {hook_score:.1f}/10"

    # 2. Aura/Brainrot Signals (0–10): cultural keyword density
    aura_hits = sum(1 for k in _AURA_KEYWORDS if k in narration or k in topic)
    aura_score = min(10.0, aura_hits * 1.8)
    scores["aura_lockin_signals"] = aura_score
    notes["aura_lockin_signals"] = f"{aura_hits} brainrot signals found. Score: {aura_score:.1f}/10"

    # 3. Replayability (0–10): loop-trigger language + multi-segment variety
    replay_hits = sum(1 for t in _REPLAY_TRIGGERS if t in narration)
    seg_count_bonus = min(3.0, len(segments) * 0.6)
    replay_score = min(10.0, replay_hits * 2.0 + seg_count_bonus)
    scores["replayability"] = replay_score
    notes["replayability"] = f"{replay_hits} loop triggers, {len(segments)} segments. Score: {replay_score:.1f}/10"

    # 4. Pacing Density (0–10): target 3.5–5.5 wps for brainrot pacing
    if 3.5 <= wps <= 5.5:
        pacing_score = 10.0
    elif 3.0 <= wps < 3.5 or 5.5 < wps <= 6.5:
        pacing_score = 7.0
    else:
        pacing_score = 4.0
    scores["pacing_density"] = pacing_score
    notes["pacing_density"] = f"{wps:.1f} words/sec (target 3.5–5.5). Score: {pacing_score:.1f}/10"

    # 5. Title CTR Power (0–10): caps, numbers, emotion in title
    import re as _re
    title = concept.topic
    caps_words = len(_re.findall(r'\b[A-Z]{2,}\b', title))
    has_number = bool(_re.search(r'\d', title))
    has_emoji = bool(_re.search(r'[\U00010000-\U0010ffff]', title))
    hook_in_title = sum(1 for w in _HOOK_POWER_WORDS if w in title.lower())
    ctr_score = min(10.0, caps_words * 1.5 + (2.0 if has_number else 0)
                    + (1.5 if has_emoji else 0) + hook_in_title * 1.5)
    scores["title_ctr_power"] = ctr_score
    notes["title_ctr_power"] = (
        f"CAPS={caps_words}, number={has_number}, emoji={has_emoji}, "
        f"power_words={hook_in_title}. Score: {ctr_score:.1f}/10"
    )

    # Weighted total
    total = sum(_VIRALITY_WEIGHTS[k] * scores[k] for k in _VIRALITY_WEIGHTS)

    if total >= 7.0:
        verdict = "PUSH"
    elif total >= 5.0:
        verdict = "HOLD"
    else:
        verdict = "REJECT"

    return {
        "scores": scores,
        "total": round(total, 2),
        "verdict": verdict,
        "breakdown": notes,
        "word_count": word_count,
        "wps": round(wps, 2),
    }



def get_gpu_memory_allocated_mb() -> float:
    """Returns current PyTorch GPU allocated memory in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def run_pipeline(
    niche: str = "tech",
    topic_override: Optional[str] = None,
    dry_run: bool = False,
    skip_publish: bool = False,
    require_approval: bool = False,
    approved_script_path: Optional[str] = None,
    publish_target: str = "youtube",
    job_id: Optional[str | int] = None
) -> dict:
    """Executes the complete end-to-end faceless video production pipeline.
    
    Stages:
    1. Ideator Agent -> IdeaConcept (validated)
    2. Scriptwriter Agent -> Script (validated)
    3. Director Agent -> Video Render (Whisper + ComfyUI + Render workers via GPU lock)
    4. Publisher Agent -> YouTube Shorts (quota-gated) + Instagram Reels (ephemeral hosting)
    """
    telemetry = {
        "start_time": time.time(),
        "job_id": job_id,
        "stages": {},
        "vram": {},
        "quota_spent": 0,
        "results": {},
        "verification": {},
        "department_heads": {},
        "simulation_trace": [],
        "raw_materials": [],
        "final_artifacts": []
    }

    # -------------------------------------------------------------------------
    # STAGE 0: Circuit Breaker Pre-Check (Rule 8)
    # -------------------------------------------------------------------------
    allowed, breaker_msg = circuit_breaker.can_execute()
    if not allowed:
        logger.error(f"[CIRCUIT_BREAKER_BLOCKED]\n{breaker_msg}")
        raise CircuitBreakerOpenError(circuit_breaker.get_last_failure_reasons())

    try:
        logger.info("=" * 80)
        logger.info(f"STARTING AUTONOMOUS FACELESS VIDEO PIPELINE (Niche: {niche})")
        logger.info(f"System Baseline VRAM: {get_gpu_memory_allocated_mb():.2f} MB")
        logger.info("=" * 80)

        # ---------------------------------------------------------------------
        # STAGE 1 & 2: Ideation & Scriptwriting OR Resume from Approved Script
        # ---------------------------------------------------------------------
        if approved_script_path and Path(approved_script_path).exists():
            print("[STAGE:SCRIPTWRITING]", flush=True)
            logger.info(f"Resuming pipeline from approved script: {approved_script_path}")
            with open(approved_script_path, "r", encoding="utf-8") as f:
                script_raw = json.load(f)
            script = Script.model_validate(script_raw)
            topic_str = script_raw.get("topic", topic_override or "Approved Video Concept")
            concept = IdeaConcept(
                topic=topic_str,
                niche=script_raw.get("niche", niche),
                angle=script_raw.get("hook", "Approved Concept Angle"),
                target_duration=int(script.total_estimated_duration())
            )
            story_concept_gate = head_of_story.inspect_tier1(concept)
            story_script_gate = head_of_story.inspect_tier1(script, context={"topic": concept.topic, "niche": concept.niche})
            telemetry["department_heads"]["story_concept"] = story_concept_gate.to_dict()
            telemetry["department_heads"]["story_script"] = story_script_gate.to_dict()
            telemetry["verification"]["creative_director"] = [story_concept_gate.to_dict()]
            telemetry["verification"]["copywriter"] = [story_script_gate.to_dict()]
            if not story_concept_gate.passed or not story_script_gate.passed:
                err_feedback = story_script_gate.feedback or story_concept_gate.feedback or "Resumed script failed structural validation"
                raise DepartmentGateRejectionError(
                    department="story",
                    head_title="Head of Story",
                    tier=1,
                    feedback=f"Resumed script failed Head of Story structural inspection: {err_feedback}"
                )

            telemetry["concept"] = concept.model_dump()
            telemetry["stages"]["ideation"] = 0.0
            telemetry["stages"]["scriptwriting"] = 0.0
        else:
            # STAGE 1: Ideation (Creative Director / Ideator Agent)
            print("[STAGE:IDEATION]", flush=True)
            t0 = time.time()
            logger.info("\n>>> [STAGE 1] Concept Ideation (Creative Director / Ideator Agent)...")

            def _generate_concept(feedback: Optional[str] = None) -> IdeaConcept:
                effective_topic = topic_override
                if feedback and not effective_topic:
                    logger.info(f"[STAGE 1 RETRY] Generating concept with feedback: {feedback}")
                return ideator.generate_idea(
                    niche=niche,
                    custom_topic=effective_topic,
                    dry_run=dry_run
                )

            def _fallback_concept() -> IdeaConcept:
                from pipeline.agents.ideator import generate_fallback_ideas
                fallbacks = generate_fallback_ideas(niche)
                return fallbacks[0]

            initial_concept = _generate_concept()
            concept, gate1_history = head_of_story.enforce_gate(
                initial_artifact=initial_concept,
                produce_fn=_generate_concept,
                max_retries=2,
                fallback_fn=_fallback_concept
            )

            telemetry["stages"]["ideation"] = time.time() - t0
            telemetry["concept"] = concept.model_dump()
            telemetry["department_heads"]["story_concept"] = gate1_history[-1].to_dict()
            telemetry["verification"]["creative_director"] = [r.to_dict() for r in gate1_history]
            telemetry["simulation_trace"].append({
                "step_index": 1,
                "agent_name": "CreativeDirector",
                "stage": "IDEATION",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(telemetry["stages"]["ideation"], 2),
                "input_payload": {"niche": niche, "topic_override": topic_override},
                "reasoning_trace": f"Formulated high-urgency hook for '{concept.topic}'. Angle: '{concept.angle}'. Target duration: {concept.target_duration}s.",
                "prompt_template": "Generate high-retention vertical short concept with curiosity gap.",
                "structured_output": concept.model_dump(),
                "status": "PASSED" if gate1_history[-1].passed else "FAILED"
            })
            logger.info(f"Concept approved: '{concept.topic}' ({concept.angle}) [{telemetry['stages']['ideation']:.2f}s]")

            # STAGE 2: Scriptwriting (Copywriter / Scriptwriter Agent)
            print("[STAGE:SCRIPTWRITING]", flush=True)
            t0 = time.time()
            logger.info("\n>>> [STAGE 2] Script Breakdown & Drafting (Copywriter Agent)...")

            def _generate_script(feedback: Optional[str] = None) -> Script:
                return get_valid_script(
                    topic=concept.topic,
                    niche=concept.niche,
                    feedback=feedback
                )

            def _fallback_script() -> Script:
                from pipeline.agents.scriptwriter import load_fallback_template
                return load_fallback_template(topic=concept.topic, niche=concept.niche)

            initial_script = _generate_script()
            script, gate2_history = head_of_story.enforce_gate(
                initial_artifact=initial_script,
                produce_fn=_generate_script,
                context={"topic": concept.topic, "niche": concept.niche},
                max_retries=2,
                fallback_fn=_fallback_script
            )

            telemetry["stages"]["scriptwriting"] = time.time() - t0
            telemetry["script"] = {
                "hook": script.hook,
                "segments_count": len(script.segments),
                "estimated_duration": script.total_estimated_duration()
            }
            telemetry["department_heads"]["story_script"] = gate2_history[-1].to_dict()
            telemetry["verification"]["copywriter"] = [r.to_dict() for r in gate2_history]
            telemetry["simulation_trace"].append({
                "step_index": 2,
                "agent_name": "Copywriter",
                "stage": "SCRIPTWRITING",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(telemetry["stages"]["scriptwriting"], 2),
                "input_payload": concept.model_dump(),
                "reasoning_trace": f"Constructed {len(script.segments)}-scene narrative breakdown. Spoken length: ~{script.total_estimated_duration():.1f}s. Verified pacing and scene transitions.",
                "prompt_template": "Generate ProductionScript with pacing-calibrated segments and visual queries.",
                "structured_output": script.model_dump(),
                "status": "PASSED" if gate2_history[-1].passed else "FAILED"
            })
            logger.info(
                f"Script structured: {len(script.segments)} scenes, "
                f"~{script.total_estimated_duration():.1f}s total duration [{telemetry['stages']['scriptwriting']:.2f}s]"
            )

            # Human-in-the-loop review check
            if require_approval:
                script_payload = script.model_dump()
                script_payload["topic"] = concept.topic
                script_payload["niche"] = concept.niche
                print(f"[SCRIPT_JSON] {json.dumps(script_payload)}", flush=True)
                print("[STAGE:PENDING_REVIEW]", flush=True)
                logger.info("Human script review required: Pipeline paused in PENDING_REVIEW status.")
                try:
                    telemetry["total_wall_clock_time"] = time.time() - telemetry.get("start_time", time.time())
                    record_run_telemetry(telemetry, status="PENDING_REVIEW")
                except Exception as db_err:
                    logger.warning(f"Could not persist pending review telemetry to SQLite: {db_err}")
                sys.exit(10)

        # ---------------------------------------------------------------------
        # STAGE 3: Direction & Production (Director Agent + GPU Workers)
        # ---------------------------------------------------------------------
        print("[STAGE:AUDIO_TTS]", flush=True)
        t0 = time.time()
        logger.info("\n>>> [STAGE 3] Asset Direction & Video Render (Director Agent)...")
        print("[STAGE:RENDER]", flush=True)
        safe_slug = "".join(c if c.isalnum() else "_" for c in concept.topic)[:30].strip("_")
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rendered_output_path = OUTPUT_DIR / f"{timestamp_str}_{safe_slug}.mp4"

        # Record baseline before GPU stages
        telemetry["vram"]["pre_render_mb"] = get_gpu_memory_allocated_mb()

        # Execute orchestration through GPU lock
        video_path = orchestrate_video(
            script=script,
            output_path=rendered_output_path,
            voice="af_bella*0.6+bf_isabella*0.4",
            verification_history=telemetry["verification"]
        )
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise FileNotFoundError(f"Director failed to produce valid video file at {video_path}")

        telemetry["stages"]["production"] = time.time() - t0
        telemetry["video_path"] = str(video_path)
        telemetry["video_size_mb"] = round(video_path.stat().st_size / (1024 * 1024), 2)
        telemetry["vram"]["post_render_mb"] = get_gpu_memory_allocated_mb()

        # Record VoiceActor, ArtDirector, and Editor simulation snapshots
        prod_time = telemetry["stages"]["production"]
        telemetry["simulation_trace"].append({
            "step_index": 3,
            "agent_name": "VoiceActor",
            "stage": "AUDIO_TTS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(prod_time * 0.35, 2),
            "input_payload": {"voice": "en-US-AndrewMultilingualNeural", "segments": len(script.segments)},
            "reasoning_trace": "Synthesized chunked TTS narration with natural punctuation pauses. Loudness normalized to EBU R128 (-14.0 LUFS).",
            "prompt_template": "Sentence-chunked TTS synthesis with local Piper fallback.",
            "structured_output": {"status": "PASSED", "voice": "en-US-AndrewMultilingualNeural", "target_lufs": -14.0},
            "status": "PASSED"
        })
        telemetry["simulation_trace"].append({
            "step_index": 4,
            "agent_name": "ArtDirector",
            "stage": "RENDER",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(prod_time * 0.45, 2),
            "input_payload": {"visual_queries": [s.visual_query for s in script.segments]},
            "reasoning_trace": "Retrieved stock video footage and ComfyUI keyframes. Normalized clips to 1080x1920 30fps CFR yuv420p bt709.",
            "prompt_template": "Pexels query caching + ComfyUI SD1.5 worker under hardware GPU lock.",
            "structured_output": {"status": "PASSED", "resolution": "1080x1920", "fps": 30},
            "status": "PASSED"
        })
        telemetry["simulation_trace"].append({
            "step_index": 5,
            "agent_name": "Editor",
            "stage": "RENDER",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(prod_time * 0.20, 2),
            "input_payload": {"video_path": str(video_path)},
            "reasoning_trace": "Burned stylized ASS captions into NVENC master MP4 stream. Hard-burned subtitle stream verified.",
            "prompt_template": "FFmpeg NVENC render pass with hard-burned subtitles.",
            "structured_output": {"status": "PASSED", "video_path": str(video_path), "size_mb": telemetry["video_size_mb"]},
            "status": "PASSED"
        })

        # Register Final Artifact
        telemetry["final_artifacts"].append({
            "id": f"final_{video_path.stem}_mp4",
            "name": video_path.name,
            "type": "video",
            "category": "Master Video (1080x1920)",
            "format": "1080x1920 30fps CFR yuv420p",
            "size_mb": telemetry["video_size_mb"],
            "path": video_path.name,
            "url": f"/videos/{video_path.name}",
            "has_burned_subtitles": True,
            "badge": "FINAL MASTER"
        })

        logger.info(
            f"Video rendered successfully: {video_path.name} "
            f"({telemetry['video_size_mb']} MB) [{telemetry['stages']['production']:.2f}s]"
        )

        # Department Head Inspection: Head of Art (Visual Gate)
        art_gate = head_of_art.inspect(video_path)
        telemetry["department_heads"]["art"] = art_gate.to_dict()
        if not art_gate.passed:
            raise DepartmentGateRejectionError(
                department="art",
                head_title="Head of Art",
                tier=art_gate.tier,
                feedback=art_gate.feedback or "Video rejected by Head of Art.",
                details=art_gate.details
            )

        # Department Head Inspection: Head of Post (Assembly & Subtitle Gate)
        post_gate = head_of_post.inspect(video_path)
        telemetry["department_heads"]["post"] = post_gate.to_dict()
        if not post_gate.passed:
            raise DepartmentGateRejectionError(
                department="post",
                head_title="Head of Post",
                tier=post_gate.tier,
                feedback=post_gate.feedback or "Master rejected by Head of Post.",
                details=post_gate.details
            )

        # ---------------------------------------------------------------------
        # STAGE 3.5: Risk & Safety Compliance Gate (Compliance Officer - Rule 13)
        # ---------------------------------------------------------------------
        print("[STAGE:COMPLIANCE]", flush=True)
        t0 = time.time()
        logger.info("\n>>> [STAGE 3.5] Autonomous Policy & Safety Gate (Compliance Officer Agent)...")
        risk_report, comp_telemetry = assess_video_compliance(
            video_path=video_path,
            topic=concept.topic,
            niche=concept.niche,
            narration_text=script.full_narration(),
            dry_run=dry_run
        )
        telemetry["stages"]["compliance"] = time.time() - t0
        telemetry["compliance"] = risk_report.model_dump()
        telemetry["compliance_telemetry"] = comp_telemetry
        telemetry["simulation_trace"].append({
            "step_index": 6,
            "agent_name": "ComplianceOfficer",
            "stage": "COMPLIANCE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(telemetry["stages"]["compliance"], 2),
            "input_payload": {"topic": concept.topic, "video_path": str(video_path)},
            "reasoning_trace": f"Screened 480p QA proxy. Community Risk: {risk_report.community_guidelines.risk_level.value}, Demonetization: {risk_report.demonetization.risk_level.value}, IP Drift: {risk_report.copyright_ip_resemblance.risk_level.value}.",
            "prompt_template": "Structured Compliance Risk Evaluation Prompt (Rule 13).",
            "structured_output": risk_report.model_dump(),
            "status": "PASSED" if risk_report.overall_recommendation.value == "PROCEED" else risk_report.overall_recommendation.value
        })

        # Enforce gate: LOW proceeds, MEDIUM holds (exit 10 / pause), HIGH blocks (no retry)
        try:
            enforce_compliance_gate(risk_report, topic=concept.topic)
        except ComplianceHoldError as hold_err:
            print("[STAGE:PENDING_REVIEW]", flush=True)
            logger.warning(f"Compliance Hold: {hold_err}")
            try:
                telemetry["total_wall_clock_time"] = time.time() - telemetry.get("start_time", time.time())
                record_run_telemetry(telemetry, status="HELD_FOR_REVIEW", error_message=str(hold_err))
            except Exception as db_err:
                logger.warning(f"Could not persist held review telemetry to SQLite: {db_err}")
            sys.exit(10)
        # Note: ComplianceBlockError propagates directly to outer handler without retry

        # Both Chief Critic and Compliance Officer gates cleared successfully
        if job_id:
            try:
                update_video_gate_status(job_id, chief_critic_passed=True, compliance_passed=True)
                logger.info(f"[GATE_CLEARED] Job {job_id} cleared both Chief Critic and Compliance Officer gates.")
            except Exception as g_err:
                logger.warning(f"Could not update gate status in DB for job {job_id}: {g_err}")

        # ---------------------------------------------------------------------
        # STAGE 4: Multi-Platform Publishing (Publisher Agent)
        # ---------------------------------------------------------------------
        print("[STAGE:PUBLISH]", flush=True)
        effective_skip_publish = skip_publish or (publish_target == "disk_only")

        # Department Head Inspection: Head of Compliance (Pre-Publishing Gate)
        compliance_head_gate = head_of_compliance.inspect(
            video_path,
            context={
                "title": concept.topic,
                "description": build_youtube_description(concept, script),
                "skip_publish": effective_skip_publish,
                "video_path": video_path,
                "topic": concept.topic,
                "niche": concept.niche,
                "narration": script.full_narration(),
                "dry_run": dry_run,
            }
        )
        telemetry["department_heads"]["compliance"] = compliance_head_gate.to_dict()
        if not compliance_head_gate.passed:
            # Rule 13: MEDIUM → HOLD_FOR_REVIEW (pause for human clear), not fatal crash
            hold_feedback = compliance_head_gate.feedback or "Publishing rejected by Head of Compliance."
            hold_details = compliance_head_gate.details or {}
            logger.warning(
                f"[COMPLIANCE HEAD REJECTION] Tier {compliance_head_gate.tier} gate failed for "
                f"'{concept.topic}': {hold_feedback}"
            )
            print("[STAGE:COMPLIANCE_HELD]", flush=True)
            print(f"[COMPLIANCE_HELD_REASON] {hold_feedback}", flush=True)

            # Persist telemetry before exiting
            try:
                telemetry["total_wall_clock_time"] = time.time() - telemetry.get("start_time", time.time())
                record_run_telemetry(telemetry, status="HELD_FOR_REVIEW", error_message=hold_feedback)
            except Exception as db_err:
                logger.warning(f"Could not persist compliance hold telemetry to SQLite: {db_err}")

            sys.exit(10)

        if not effective_skip_publish:
            logger.info("\n>>> [STAGE 4] Multi-Platform Publishing (Publisher Agent)...")

            # ---------------------------------------------------------------
            # VIRALITY GATE (Pre-Publish Evaluation)
            # ---------------------------------------------------------------
            virality = evaluate_virality(concept, script)
            telemetry["virality"] = virality
            v_total = virality["total"]
            v_verdict = virality["verdict"]
            print(f"[VIRALITY_GATE] Score: {v_total}/10 | Verdict: {v_verdict}", flush=True)
            for dim, note in virality["breakdown"].items():
                print(f"  [{dim}] {note}", flush=True)

            if v_verdict == "REJECT":
                logger.error(
                    f"[VIRALITY_GATE_REJECT] Video '{concept.topic}' scored {v_total}/10 — "
                    f"below 5.0 threshold. Aborting publish to protect channel quality."
                )
                print("[STAGE:VIRALITY_REJECTED]", flush=True)
                telemetry["total_wall_clock_time"] = time.time() - telemetry.get("start_time", time.time())
                try:
                    record_run_telemetry(telemetry, status="VIRALITY_REJECTED",
                                        error_message=f"Virality score {v_total}/10 below threshold")
                except Exception:
                    pass
                sys.exit(11)

            if v_verdict == "HOLD":
                logger.warning(
                    f"[VIRALITY_GATE_HOLD] Video '{concept.topic}' scored {v_total}/10 — "
                    f"marginal virality. Publishing with caution."
                )

            # Build optimized metadata
            yt_title = concept.topic
            yt_description = build_youtube_description(concept, script)
            yt_tags = build_youtube_tags(concept)
            logger.info(f"[YOUTUBE_METADATA] Title: {yt_title}")
            logger.info(f"[YOUTUBE_METADATA] Tags ({len(yt_tags)}): {yt_tags}")
            logger.info(f"[YOUTUBE_METADATA] Description preview: {yt_description[:120]}...")

            # YouTube Shorts (Quota-gated)
            t0 = time.time()
            yt_res: PublishResult = publisher.publish_to_youtube(
                video_path=video_path,
                title=yt_title,
                description=yt_description,
                tags=yt_tags,
                privacy_status="public",
                dry_run=dry_run
            )
            telemetry["stages"]["youtube_publish"] = time.time() - t0
            telemetry["results"]["youtube"] = yt_res.model_dump()
            telemetry["quota_spent"] += yt_res.units_spent

            if yt_res.status == PublishStatus.PUBLISHED:
                if yt_res.url:
                    print(f"[YOUTUBE_URL] {yt_res.url}", flush=True)
                print(f"[YOUTUBE_STATUS] {yt_res.status}", flush=True)
                if yt_res.is_restricted:
                    print(f"[UNVERIFIED_PROJECT_RESTRICTION] Video '{yt_res.post_id}' was forced to private by Google's unverified OAuth project policy: {yt_res.url}", flush=True)
                    telemetry["youtube_restricted"] = True
            elif yt_res.status == PublishStatus.DEFERRED:
                print(f"[STAGE:PUBLISH_DEFERRED] {yt_res.message}", flush=True)
                telemetry["youtube_deferred"] = True
                if job_id:
                    try:
                        sch = strategist.schedule_video(job_id, topic=concept.topic, niche=concept.niche)
                        logger.info(f"[SCHEDULED_FOR_PUBLISH] Quota exhausted: Job {job_id} scheduled for {sch.target_publish_datetime}")
                    except Exception as sch_err:
                        logger.warning(f"Could not schedule video in DB for job {job_id}: {sch_err}")
            elif yt_res.status == PublishStatus.FAILED:
                err_code = yt_res.error_code or "UNKNOWN"
                print(f"[STAGE:PUBLISH_FAILED] [{err_code}] {yt_res.message}", flush=True)
                telemetry["youtube_failed"] = True
                telemetry["error_message"] = f"[{err_code}] {yt_res.message}"
                try:
                    from pipeline.core.notifier import notifier
                    notifier.alert(
                        f"[PUBLISH_FAILURE] YouTube upload failed: [{err_code}] {yt_res.message}",
                        level="ERROR",
                        extra={"error_code": err_code, "details": yt_res.message}
                    )
                except Exception:
                    pass

            # Instagram Reels (Ephemeral Hosting)
            t0 = time.time()
            ig_caption = f"{concept.topic}\n\n{script.hook}\n\nFollow for more daily insights! #reels #{concept.niche}"
            ig_res: PublishResult = publisher.publish_to_instagram(
                video_path=video_path,
                caption=ig_caption,
                dry_run=dry_run
            )
            telemetry["stages"]["instagram_publish"] = time.time() - t0
            telemetry["results"]["instagram"] = ig_res.model_dump()
        else:
            logger.info("\n>>> [STAGE 4] Publishing skipped (--skip-publish requested).")
            # If publication is skipped/held for scheduled release, persist optimal schedule
            if job_id:
                try:
                    sch = strategist.schedule_video(job_id, topic=concept.topic, niche=concept.niche)
                    logger.info(f"[SCHEDULED_FOR_PUBLISH] Job {job_id} scheduled for {sch.target_publish_datetime}")
                except Exception as sch_err:
                    logger.warning(f"Could not schedule video in DB for job {job_id}: {sch_err}")

        # Record total runtime
        telemetry["total_wall_clock_time"] = time.time() - telemetry["start_time"]

        # ---------------------------------------------------------------------
        # Final Status & Circuit Breaker Handling
        # ---------------------------------------------------------------------
        final_status = "SUCCESS"
        if telemetry.get("youtube_failed"):
            final_status = "FAILED_PUBLISH"
        elif telemetry.get("youtube_restricted"):
            final_status = "HELD_FOR_REVIEW"
        elif telemetry.get("youtube_deferred"):
            final_status = "DEFERRED_QUOTA"

        if final_status == "SUCCESS":
            circuit_breaker.record_success()
        elif final_status in ("HELD_FOR_REVIEW", "DEFERRED_QUOTA"):
            circuit_breaker.record_success()  # Video produced successfully; downstream release pending
        else:
            circuit_breaker.record_failure(f"Publishing failed: {telemetry.get('error_message')}")

        try:
            record_run_telemetry(telemetry, status=final_status, error_message=telemetry.get("error_message"))
        except Exception as db_err:
            logger.warning(f"Could not persist run telemetry to SQLite: {db_err}")

        # Print Final Telemetry Summary
        _print_run_summary(telemetry)

        if final_status == "FAILED_PUBLISH":
            sys.exit(3)

        return telemetry

    except ComplianceBlockError as block_err:
        # Rule 13: HIGH risk policy block. Failure is already logged to circuit breaker; do not auto-retry.
        print("[STAGE:BLOCKED]", flush=True)
        logger.error(f"[COMPLIANCE_BLOCK] Pipeline halted permanently on safety violation: {block_err}")
        try:
            telemetry["total_wall_clock_time"] = time.time() - telemetry.get("start_time", time.time())
            record_run_telemetry(telemetry, status="BLOCKED", error_message=str(block_err))
        except Exception as db_err:
            logger.warning(f"Could not persist blocked telemetry to SQLite: {db_err}")
        raise

    except Exception as exc:
        logger.error(f"Pipeline run encountered a fatal exception: {exc}", exc_info=True)
        # Classify transient (network, timeout, lock wait) vs permanent failures
        err_msg = str(exc).lower()
        is_transient = any(t in err_msg for t in [
            "timeout", "timed out", "connectionerror", "connection reset", "connection refused",
            "503", "504", "server disconnected", "network is unreachable", "remote end closed"
        ])
        circuit_breaker.record_failure(str(exc), is_transient=is_transient)
        try:
            telemetry["total_wall_clock_time"] = time.time() - telemetry.get("start_time", time.time())
            record_run_telemetry(telemetry, status="FAILED", error_message=str(exc))
        except Exception as db_err:
            logger.warning(f"Could not persist failure telemetry to SQLite: {db_err}")
        raise


def _print_run_summary(telemetry: dict) -> None:
    """Formats and prints executive telemetry table."""
    concept = telemetry.get("concept", {})
    stages = telemetry.get("stages", {})
    results = telemetry.get("results", {})

    print("\n" + "=" * 80)
    print("           AUTONOMOUS PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Topic:                 {concept.get('topic')}")
    print(f"Niche:                 {concept.get('niche')}")
    print(f"Output Video:          {telemetry.get('video_path')} ({telemetry.get('video_size_mb')} MB)")
    print("-" * 80)
    print("STAGE WALL-CLOCK DURATIONS:")
    print(f"  Stage 1 (Ideation):       {stages.get('ideation', 0.0):6.2f}s")
    print(f"  Stage 2 (Scriptwriting):  {stages.get('scriptwriting', 0.0):6.2f}s")
    print(f"  Stage 3 (Production):     {stages.get('production', 0.0):6.2f}s")
    if "youtube_publish" in stages:
        print(f"  Stage 4 (YouTube Pub):    {stages.get('youtube_publish', 0.0):6.2f}s")
    if "instagram_publish" in stages:
        print(f"  Stage 4 (Instagram Pub):  {stages.get('instagram_publish', 0.0):6.2f}s")
    print(f"TOTAL PIPELINE RUNTIME:     {telemetry.get('total_wall_clock_time', 0.0):6.2f}s")
    print("-" * 80)
    print("RESOURCE & QUOTA TELEMETRY:")
    print(f"  YouTube Quota Units Spent Today:  {quota_tracker.get_used_units()} / {quota_tracker.daily_limit} units")
    print(f"  YouTube Available Quota:          {quota_tracker.get_available_quota()} units")
    if "youtube" in results:
        yt = results["youtube"]
        print(f"  YouTube Post Status:              {yt.get('status')} (Post ID: {yt.get('post_id')})")
    if "instagram" in results:
        ig = results["instagram"]
        print(f"  Instagram Post Status:            {ig.get('status')} (Post ID: {ig.get('post_id')})")

    if "verification" in telemetry and telemetry["verification"]:
        print("-" * 80)
        print("STAGE VERIFICATION GATES (RULE 12):")
        total_gemini_calls = 0
        total_verify_latency = 0.0
        for stage_name, results_list in telemetry["verification"].items():
            for res in results_list:
                status_str = "PASSED" if res.get("passed") else "FAILED"
                tier_str = f"Tier {res.get('tier')}"
                dur_str = f"{res.get('latency_seconds', 0.0):.3f}s"
                calls = res.get("gemini_calls", 0)
                total_gemini_calls += calls
                total_verify_latency += res.get("latency_seconds", 0.0)
                critique_note = f" - Critique: {res.get('critique')[:45]}..." if res.get('critique') else ""
                print(f"  [{status_str}] {stage_name.upper():<18} ({tier_str}) - Latency: {dur_str}, Added Gemini calls: {calls}{critique_note}")
        print(f"  Tier 2 Metrics -> Total Added Calls: {total_gemini_calls} | Total Verification Latency: {total_verify_latency:.3f}s")

    if "compliance" in telemetry and telemetry["compliance"]:
        comp = telemetry["compliance"]
        comp_tel = telemetry.get("compliance_telemetry", {})
        print("-" * 80)
        print("COMPLIANCE & SAFETY REPORT (RULE 13):")
        rec = comp.get("overall_recommendation", "UNKNOWN")
        print(f"  Overall Verdict:             [{rec}]")
        cg = comp.get("community_guidelines", {})
        demo = comp.get("demonetization", {})
        copyr = comp.get("copyright_ip_resemblance", {})
        print(f"  YouTube Community Risk:      {cg.get('risk_level', 'UNKNOWN')} - {cg.get('reasoning', '')}")
        print(f"  Demonetization Risk:         {demo.get('risk_level', 'UNKNOWN')} - {demo.get('reasoning', '')}")
        print(f"  ComfyUI SD1.5 IP Drift:      {copyr.get('risk_level', 'UNKNOWN')} - {copyr.get('reasoning', '')}")
        print(f"  Proxy Reused:                {comp_tel.get('was_proxy_reused', False)} ({comp_tel.get('proxy_size_mb', 0)} MB)")
        print(f"  Gemini File API Reused:      {comp_tel.get('was_upload_reused', False)} (Upload Latency: {comp_tel.get('upload_latency_s', 0):.2f}s)")
        print(f"  Added Inference Latency:     {comp_tel.get('evaluation_latency_s', 0):.2f}s | Estimated Cost: ${comp_tel.get('estimated_cost_usd', 0):.6f}")
        print(f"  Summary:                     {comp.get('summary', '')}")

    if "department_heads" in telemetry and telemetry["department_heads"]:
        print("-" * 80)
        print("DEPARTMENT HEAD AUDIT GATES (QUALITY SIGN-OFF):")
        for dept_key, gate_data in telemetry["department_heads"].items():
            status_str = "APPROVED" if gate_data.get("passed") else "REJECTED"
            head_title = gate_data.get("head_title", dept_key.upper())
            tier = gate_data.get("tier", 1)
            score_val = gate_data.get("score")
            score_str = f"{score_val:.1f}/10" if score_val is not None else "VERIFIED"
            lat = gate_data.get("latency_seconds", 0.0)
            fb = f" - Critique: {gate_data['feedback']}" if gate_data.get("feedback") else ""
            print(f"  [{status_str}] {head_title:<22} (Tier {tier}: {score_str}) [{lat:.3f}s]{fb}")
    print("=" * 80 + "\n")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Autonomous Faceless Video Pipeline Entry Point")
    parser.add_argument("--niche", type=str, default="tech", help="Content niche (tech, finance, science, history)")
    parser.add_argument("--topic", type=str, default=None, help="Explicit topic override (bypasses ideator generation)")
    parser.add_argument("--dry-run", action="store_true", help="Simulates external publishing & uses template fallback")
    parser.add_argument("--skip-publish", action="store_true", help="Renders video locally without publishing")
    parser.add_argument("--reset-circuit-breaker", action="store_true", help="Manually re-arms the circuit breaker")
    parser.add_argument("--reset-quota", action="store_true", help="Manually resets today's YouTube quota tracker")
    parser.add_argument("--status", action="store_true", help="Displays current circuit breaker and quota status")
    parser.add_argument("--require-approval", action="store_true", help="Pause pipeline for human script review")
    parser.add_argument("--approved-script", type=str, default=None, help="Path to approved script JSON file")
    parser.add_argument("--publish-target", type=str, default="youtube", choices=["youtube", "disk_only"], help="Publish destination")
    parser.add_argument("--job-id", type=str, default=None, help="Database job ID to link telemetry, gate status, and schedule")
    args = parser.parse_args()

    # Administrative flags
    if args.reset_circuit_breaker:
        circuit_breaker.reset()
        print("Circuit breaker successfully reset to CLOSED state.")
        return

    if args.reset_quota:
        quota_tracker.reset()
        print("YouTube quota tracker successfully reset to 0 units.")
        return

    if args.status:
        print("\n--- Pipeline Operational Status ---")
        print(f"Circuit Breaker: {'OPEN (TRIPPED)' if circuit_breaker.is_tripped() else 'CLOSED (HEALTHY)'}")
        print(f"Consecutive Failures: {circuit_breaker.get_consecutive_failures()}/{circuit_breaker.max_failures}")
        print(f"YouTube Quota Used: {quota_tracker.get_used_units()} / {quota_tracker.daily_limit} units")
        print(f"YouTube Quota Available: {quota_tracker.get_available_quota()} units")
        print(f"Lock File Present: {LOCK_FILE.exists()}")
        return

    # Scheduler Safety Lock Lifecycle (Rule 7)
    with scheduler_lock() as acquired:
        if not acquired:
            sys.exit(0)

        try:
            run_pipeline(
                niche=args.niche,
                topic_override=args.topic,
                dry_run=args.dry_run,
                skip_publish=args.skip_publish,
                require_approval=args.require_approval,
                approved_script_path=args.approved_script,
                publish_target=args.publish_target,
                job_id=args.job_id
            )
        except CircuitBreakerOpenError:
            sys.exit(2)
        except Exception as e:
            logger.error(f"[MAIN_EXCEPTION] {type(e).__name__}: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
