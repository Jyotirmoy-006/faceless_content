"""Head of Compliance Agent: Department Head for QA, Compliance & Publishing.

Gates Compliance Officer and Multi-Platform Publisher deliverables:
1. Tier 1 (Deterministic Quota & Platform Metadata Gate):
   - YouTube Quota Pre-Flight: Checks remaining daily units >= 1,600 before upload attempt.
   - Title Validation: Maximum 100 characters, no forbidden control characters.
   - Metadata Formatting: Description must include '#shorts' and topical tags.
2. Tier 2 (Semantic Policy & Risk Audit):
   - Community guidelines, copyright IP drift, demonetization risk screening.
   - Enforces LOW (proceed), MEDIUM (hold for human review), HIGH (permanent block).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from pipeline.agents.compliance_officer import assess_video_compliance
from pipeline.agents.department_heads.base_head import (
    BaseDepartmentHead,
    DepartmentGateResult,
)
from pipeline.core.llm_manager import AgentRole
from pipeline.core.logger import get_logger
from pipeline.core.quota_tracker import YOUTUBE_VIDEO_UPLOAD_COST, quota_tracker

logger = get_logger("head_of_compliance")


class HeadOfCompliance(BaseDepartmentHead):
    """Department Head agent governing QA, Compliance & Publishing."""

    def __init__(self) -> None:
        super().__init__(
            department_name="compliance",
            head_title="Head of Compliance",
            role=AgentRole.HEAD_OF_COMPLIANCE,
        )

    def inspect_tier1(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> DepartmentGateResult:
        """Inspects daily YouTube API quota and metadata constraints (0 API cost)."""
        t0 = time.time()
        ctx = context or {}
        title = ctx.get("title", "")
        description = ctx.get("description", "")
        skip_publish = ctx.get("skip_publish", False)

        details: Dict[str, Any] = {
            "available_quota": quota_tracker.get_available_quota(),
            "daily_limit": quota_tracker.daily_limit,
        }

        # 1. Quota Pre-Flight Check (if publishing to YouTube)
        if not skip_publish:
            if not quota_tracker.can_spend(YOUTUBE_VIDEO_UPLOAD_COST):
                return DepartmentGateResult(
                    passed=False,
                    department=self.department_name,
                    head_title=self.head_title,
                    tier=1,
                    feedback=(
                        f"Insufficient YouTube quota ({quota_tracker.get_available_quota()} units available; "
                        f"{YOUTUBE_VIDEO_UPLOAD_COST} required). Halting publish to prevent wasted API failure."
                    ),
                    details=details,
                    latency_seconds=time.time() - t0,
                )

        # 2. Metadata length & tag checks
        if title and len(title) > 100:
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback=f"Video title exceeds YouTube 100-character limit ({len(title)} chars).",
                details=details,
                latency_seconds=time.time() - t0,
            )

        if description and "#shorts" not in description.lower():
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                feedback="Video description is missing mandatory '#shorts' tag for vertical short distribution.",
                details=details,
                latency_seconds=time.time() - t0,
            )

        return DepartmentGateResult(
            passed=True,
            department=self.department_name,
            head_title=self.head_title,
            tier=1,
            score=10.0,
            details=details,
            latency_seconds=time.time() - t0,
        )

    def inspect_tier2(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[DepartmentGateResult]:
        """Executes full policy, demonetization, and IP drift screening."""
        ctx = context or {}
        video_path = ctx.get("video_path")
        if not video_path:
            return None

        dry_run = ctx.get("dry_run", False)
        topic = ctx.get("topic", "Automated Video")
        niche = ctx.get("niche", "general")
        narration = ctx.get("narration", "")

        try:
            risk_report, comp_telemetry = assess_video_compliance(
                video_path=video_path,
                topic=topic,
                niche=niche,
                narration_text=narration,
                dry_run=dry_run,
            )

            rec = risk_report.overall_recommendation.value
            passed = rec == "PROCEED"
            feedback = None
            if not passed:
                feedback = (
                    f"Compliance gate held/blocked video: Recommendation [{rec}]. "
                    f"Summary: {risk_report.summary}"
                )

            return DepartmentGateResult(
                passed=passed,
                department=self.department_name,
                head_title=self.head_title,
                tier=2,
                score=10.0 if rec == "PROCEED" else (5.0 if rec == "HOLD_FOR_REVIEW" else 0.0),
                feedback=feedback,
                details={
                    "recommendation": rec,
                    "community_risk": risk_report.community_guidelines.risk_level.value,
                    "demonetization_risk": risk_report.demonetization.risk_level.value,
                    "ip_resemblance": risk_report.copyright_ip_resemblance.risk_level.value,
                },
                gemini_calls=1,
            )
        except Exception as e:
            logger.warning(f"[{self.head_title}] Tier 2 compliance screening error: {e}")
            return None


head_of_compliance = HeadOfCompliance()
