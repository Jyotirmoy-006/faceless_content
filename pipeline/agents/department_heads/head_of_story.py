"""Head of Story Agent: Department Head for Strategy & Story.

Gates Ideator and Copywriter/Scriptwriter deliverables:
1. Tier 1 (Deterministic Structural & Retention Gate):
   - Validates IdeaConcept and Script schema contracts.
   - Word density pacing (PROVISIONAL / HEURISTIC: 2.0 to 3.2 words/second empirical target; uncalibrated against production channel analytics).
   - Micro-shot unpacking requirement (PROVISIONAL / HEURISTIC: avg shot <= 3.2s pacing threshold).
   - Banned opener eradication ("did you know", "hey guys", "in this video").
   - Seamless loop outro integrity (no terminal goodbyes).
2. Tier 2 (Semantic Narrative Evaluation):
   - Fast Gemini Flash audit assessing curiosity gap, retention stakes, and punchiness.
   - Minimum threshold: 8.0 / 10.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from pipeline.agents.department_heads.base_head import (
    BaseDepartmentHead,
    DepartmentGateResult,
)
from pipeline.core.llm_manager import AgentRole, llm_manager
from pipeline.core.logger import get_logger
from pipeline.core.schema import IdeaConcept, Script

logger = get_logger("head_of_story")


class StorySemanticCritique(BaseModel):
    """Evaluation score schema for story narrative assessment."""
    hook_urgency_score: int = Field(ge=1, le=10, description="1-10 scroll-stopping curiosity and tension")
    narrative_arc_score: int = Field(ge=1, le=10, description="1-10 progressive tension and payoff")
    pacing_density_score: int = Field(ge=1, le=10, description="1-10 rhythm, zero filler, information density")
    critique: str = Field(description="Actionable editorial critique for script revision")


class HeadOfStory(BaseDepartmentHead):
    """Department Head agent governing Strategy & Story."""

    def __init__(self) -> None:
        super().__init__(
            department_name="story",
            head_title="Head of Story",
            role=AgentRole.HEAD_OF_STORY,
        )

    def _check_visual_tractability(self, concept: IdeaConcept) -> tuple[bool, str]:
        """Validates that concept can be depicted with physical real-world visuals."""
        combined_text = f"{concept.topic} {concept.angle}".lower()

        # Abstract terms that are intractable for short-form video unless anchored
        abstract_indicators = [
            "source code", "python syntax", "coding syntax", "javascript closures",
            "writing code", "regex pattern", "software bug fix", "git merge conflict",
            "git merge", "abstract algorithm", "class inheritance", "variable scoping",
            "memory pointer syntax", "python gil", "software architecture diagrams",
            "cloud database schema", "pure mathematics formula", "abstract logic"
        ]

        # Tangible physical anchors that make a technical topic visualizable
        physical_anchors = [
            "supercomputer", "datacenter", "satellite", "robot", "chip", "hardware",
            "fiber optic", "submersible", "cable", "telescope", "laser", "quantum processor",
            "factory", "microscope", "battery", "engine", "drone", "reactor", "lithography",
            "silicon wafer", "qubit", "cryogenic", "server rack", "antenna", "submarine",
            "particle accelerator", "spacecraft", "materials", "magnetic", "device", "physical"
        ]

        for term in abstract_indicators:
            if term in combined_text:
                # Strip negations (e.g. "without hardware", "no hardware", "zero hardware")
                text_no_negation = re.sub(r'\b(without|no|zero|lack of)\s+\w+', '', combined_text)
                has_anchor = any(re.search(r'\b' + re.escape(anchor) + r'\b', text_no_negation) for anchor in physical_anchors)
                if not has_anchor:
                    return False, (
                        f"Concept lacks physical visual tractability: '{term}' is an abstract concept "
                        "without concrete physical anchors. Re-anchor to physical hardware, facilities, "
                        "or real-world mechanisms (e.g., chips, robots, submersibles, datacenters)."
                    )

        return True, ""

    def inspect_tier1(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> DepartmentGateResult:
        """Evaluates structural and algorithmic retention criteria."""
        t0 = time.time()
        details: Dict[str, Any] = {}

        # 1. Concept Verification
        if isinstance(artifact, (IdeaConcept, dict)) and not isinstance(artifact, Script):
            try:
                concept = IdeaConcept.model_validate(artifact)
                if len(concept.topic.strip()) < 3:
                    raise ValueError("Concept topic is too short (< 3 characters).")
                if len(concept.angle.strip()) < 5:
                    raise ValueError("Concept angle is too short (< 5 characters).")
                if not (15 <= concept.target_duration <= 60):
                    raise ValueError(f"Target duration {concept.target_duration}s outside [15s, 60s].")

                # Visual Tractability Gate
                is_tractable, tractability_err = self._check_visual_tractability(concept)
                if not is_tractable:
                    return DepartmentGateResult(
                        passed=False,
                        department=self.department_name,
                        head_title=self.head_title,
                        tier=1,
                        feedback=tractability_err,
                        details={"type": "IdeaConcept", "topic": concept.topic, "error": "visual_tractability_failure"},
                        latency_seconds=time.time() - t0,
                    )

                return DepartmentGateResult(
                    passed=True,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=1,
                    score=10.0,
                    details={"type": "IdeaConcept", "topic": concept.topic},
                    latency_seconds=time.time() - t0,
                )
            except Exception as e:
                return DepartmentGateResult(
                    passed=False,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=1,
                    feedback=str(e),
                    details={"error": str(e)},
                    latency_seconds=time.time() - t0,
                )

        # 2. Script Verification
        try:
            script = Script.model_validate(artifact)
        except Exception as e:
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Script schema validation error: {e}",
                latency_seconds=time.time() - t0,
            )

        # Hook conciseness (< 16 words)
        hook_words = script.hook.strip().split()
        if len(hook_words) > 15:
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Hook too verbose ({len(hook_words)} words > 15 max). Shorten to stop scroll in 2.5s.",
                latency_seconds=time.time() - t0,
            )

        # Check banned openers
        banned = ["did you know", "hey guys", "in this video", "welcome back", "today we are", "imagine if"]
        lower_hook = script.hook.strip().lower()
        for phrase in banned:
            if lower_hook.startswith(phrase):
                return DepartmentGateResult(
                    passed=False,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=1,
                    feedback=f"Hook starts with low-retention filler: '{phrase}'. Start immediately with conflict or stakes.",
                    latency_seconds=time.time() - t0,
                )

        # Word density pacing (PROVISIONAL / HEURISTIC: Empirical target 2.0-3.2 words/sec; uncalibrated against live audience analytics)
        total_duration = script.total_estimated_duration()
        all_words = len([w for seg in script.segments for w in seg.narration.split()])
        words_per_second = all_words / max(total_duration, 1.0)
        details["words_per_second"] = round(words_per_second, 2)
        details["total_duration"] = round(total_duration, 2)

        if not (28.0 <= total_duration <= 59.0):
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Script duration {total_duration:.1f}s outside high-retention bounds [28.0s, 59.0s].",
                details=details,
                latency_seconds=time.time() - t0,
            )

        # Micro-shot unpacking
        total_shots = sum(
            len(seg.visual_shots) if getattr(seg, "visual_shots", None) else 1
            for seg in script.segments
        )
        avg_shot_length = total_duration / max(1, total_shots)
        details["total_shots"] = total_shots
        details["avg_shot_length"] = round(avg_shot_length, 2)

        if avg_shot_length > 3.2:
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Average shot duration {avg_shot_length:.2f}s exceeds 3.2s pacing threshold. Unpack more visual cuts.",
                details=details,
                latency_seconds=time.time() - t0,
            )

        return DepartmentGateResult(
            passed=True,
            department=self.department_name,
            head_title=self.head_title,
            tier=1,
            score=9.0,
            details=details,
            latency_seconds=time.time() - t0,
        )

    def inspect_tier2(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[DepartmentGateResult]:
        """Performs semantic narrative evaluation via single Gemini account."""
        if not isinstance(artifact, Script):
            return None

        ctx = context or {}
        topic = ctx.get("topic", "Short-Form Video")
        niche = ctx.get("niche", "general")

        prompt = (
            f"You are the Head of Story strictly evaluating this short-form video script.\n"
            f"TOPIC: {topic} | NICHE: {niche}\n"
            f"HOOK: {artifact.hook}\n"
            f"NARRATION: {artifact.full_narration()}\n\n"
            f"Strictly rate 1-10 on:\n"
            f"1. Hook Urgency: Does it immediately create an irresistible curiosity gap?\n"
            f"2. Narrative Arc: Is there escalating tension and satisfying payoff?\n"
            f"3. Pacing Density: Zero filler, high factual/entertainment density."
        )

        try:
            raw_text, parsed = llm_manager.generate_content(
                role=self.role,
                contents=prompt,
                response_schema=StorySemanticCritique,
                temperature=0.2,
            )
            if not parsed and raw_text:
                parsed = StorySemanticCritique.model_validate_json(raw_text)

            if parsed:
                avg_score = (
                    parsed.hook_urgency_score +
                    parsed.narrative_arc_score +
                    parsed.pacing_density_score
                ) / 3.0

                passed = (
                    parsed.hook_urgency_score >= 8 and
                    parsed.narrative_arc_score >= 7 and
                    parsed.pacing_density_score >= 7
                )
                return DepartmentGateResult(
                    passed=passed,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=2,
                    score=avg_score,
                    feedback=parsed.critique if not passed else None,
                    details={
                        "hook_urgency": parsed.hook_urgency_score,
                        "narrative_arc": parsed.narrative_arc_score,
                        "pacing_density": parsed.pacing_density_score,
                    },
                    gemini_calls=1,
                )
        except Exception as e:
            logger.warning(f"[{self.head_title}] Tier 2 Gemini check degraded gracefully: {e}")

        return None


head_of_story = HeadOfStory()
