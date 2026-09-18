"""Base Department Head Agent Interface & Quality Gate Protocol.

Defines the core contract for studio Department Head agents:
1. Two-Tier Quality Gating:
   - Tier 1: Deterministic structural/sensor verification (zero API cost).
   - Tier 2: Semantic evaluation using Gemini Flash with rate-limiting.
2. Corrective Feedback Loop: Returns actionable critiques to worker agents.
3. Strict Gate Enforcement: Blocks pipeline progression until uncompromised approval.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline.core.llm_manager import AgentRole
from pipeline.core.logger import get_logger

logger = get_logger("department_head")


class DepartmentGateRejectionError(Exception):
    """Raised when a Department Head rejects a deliverable after retries are exhausted."""

    def __init__(
        self,
        department: str,
        head_title: str,
        tier: int,
        feedback: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        self.department = department
        self.head_title = head_title
        self.tier = tier
        self.feedback = feedback
        self.details = details or {}
        super().__init__(
            f"[{department.upper()} HEAD REJECTION] {head_title} rejected deliverable "
            f"at Tier {tier}: {feedback}"
        )


@dataclass
class DepartmentGateResult:
    """Quantitative decision artifact produced by a Department Head."""

    passed: bool
    department: str
    head_title: str
    tier: int
    score: Optional[float] = None
    feedback: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gemini_calls: int = 0
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes decision for dashboard telemetry and event streaming."""
        return {
            "passed": self.passed,
            "department": self.department,
            "head_title": self.head_title,
            "tier": self.tier,
            "score": round(self.score, 2) if self.score is not None else None,
            "feedback": self.feedback,
            "details": self.details,
            "timestamp": self.timestamp,
            "gemini_calls": self.gemini_calls,
            "latency_seconds": round(self.latency_seconds, 3),
        }


class BaseDepartmentHead(ABC):
    """Abstract base class for studio Department Head agents."""

    def __init__(
        self,
        department_name: str,
        head_title: str,
        role: AgentRole
    ) -> None:
        self.department_name = department_name
        self.head_title = head_title
        self.role = role

    @abstractmethod
    def inspect_tier1(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> DepartmentGateResult:
        """Executes pure deterministic code/structural checks (Tier 1: 0 API cost)."""
        pass

    def inspect_tier2(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[DepartmentGateResult]:
        """Executes semantic Gemini evaluation (Tier 2: rate-limited, bounded tokens).

        Default implementation returns None if head relies strictly on Tier 1.
        """
        return None

    def inspect(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> DepartmentGateResult:
        """Executes the full two-tier quality gate inspection.

        If Tier 1 fails, Tier 2 is skipped completely to preserve Gemini API quota.
        """
        t0 = time.time()
        logger.info(
            f"[{self.department_name.upper()}] {self.head_title} initiating deliverable inspection..."
        )

        # 1. Tier 1 Deterministic Inspection
        tier1_res = self.inspect_tier1(artifact, context)
        if not tier1_res.passed:
            tier1_res.latency_seconds = time.time() - t0
            logger.warning(
                f"[{self.department_name.upper()}] {self.head_title} REJECTED deliverable at Tier 1: "
                f"{tier1_res.feedback}"
            )
            return tier1_res

        # 2. Tier 2 Semantic Inspection (only when Tier 1 passes)
        tier2_res = self.inspect_tier2(artifact, context)
        if tier2_res is not None:
            tier2_res.latency_seconds += (time.time() - t0)
            if not tier2_res.passed:
                logger.warning(
                    f"[{self.department_name.upper()}] {self.head_title} REJECTED deliverable at Tier 2: "
                    f"{tier2_res.feedback}"
                )
            else:
                logger.info(
                    f"[{self.department_name.upper()}] {self.head_title} APPROVED deliverable at Tier 2 "
                    f"(Score: {tier2_res.score or 'N/A'})."
                )
            return tier2_res

        # Tier 1 alone approved deliverable
        tier1_res.latency_seconds = time.time() - t0
        logger.info(
            f"[{self.department_name.upper()}] {self.head_title} APPROVED deliverable at Tier 1."
        )
        return tier1_res

    def enforce_gate(
        self,
        initial_artifact: Any,
        produce_fn: Callable[[Optional[str]], Any],
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        fallback_fn: Optional[Callable[[], Any]] = None
    ) -> Tuple[Any, List[DepartmentGateResult]]:
        """Enforces quality gate with bounded corrective retries.

        If deliverable is rejected, injects structured critique to worker agent.
        """
        history: List[DepartmentGateResult] = []
        artifact = initial_artifact
        feedback: Optional[str] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(
                    f"[{self.department_name.upper()}] {self.head_title} ordering corrective rework "
                    f"(Attempt {attempt + 1}/{max_retries + 1}). Critique: {feedback}"
                )
                artifact = produce_fn(feedback)

            decision = self.inspect(artifact, context)
            history.append(decision)

            if decision.passed:
                return artifact, history

            feedback = decision.feedback or "Deliverable did not meet department quality standards."

        # Retries exhausted -> evaluate fallback if available
        if fallback_fn is not None:
            logger.warning(
                f"[{self.department_name.upper()}] Retries exhausted. Engaging deterministic emergency fallback."
            )
            fallback_artifact = fallback_fn()
            fallback_decision = self.inspect(fallback_artifact, context)
            history.append(fallback_decision)
            if fallback_decision.passed:
                return fallback_artifact, history

        # Uncompromising rejection
        last_decision = history[-1]
        raise DepartmentGateRejectionError(
            department=self.department_name,
            head_title=self.head_title,
            tier=last_decision.tier,
            feedback=last_decision.feedback or "Deliverable failed quality gates.",
            details=last_decision.details
        )
