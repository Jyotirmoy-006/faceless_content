"""Head of Visual Relevance Agent: Department Head for Cross-Modal Semantic Gating.

Gates ArtDirector asset sourcing to guarantee semantic correspondence:
1. Tier 1 (Deterministic Keyword & Metadata Overlap):
   - Computes token overlap between visual_query / narration and candidate clip tags/title.
   - Detects forbidden drift topics (vacation, party, beach, cooking, etc.).
   - Instant pass on high overlap (>= 0.35) to conserve API quota.
   - Instant rejection on zero overlap or forbidden drift.
2. Tier 2 (Single-Frame Gemini Vision Gate):
   - Triggered when keyword overlap is ambiguous or borderline.
   - Extracts a single mid-point frame via FFmpeg and queries Gemini Flash.
   - Rejects mismatched imagery and triggers ComfyUI rerouting.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from pipeline.agents.department_heads.base_head import (
    BaseDepartmentHead,
    DepartmentGateResult,
)
from pipeline.agents.pexels_sourcer import (
    FORBIDDEN_DRIFT,
    GENERIC_MODIFIERS,
    STOPWORDS,
    extract_content_tokens,
)
from pipeline.core.llm_manager import AgentRole, llm_manager
from pipeline.core.logger import get_logger

logger = get_logger("head_of_visual_relevance")


class VisualRelevanceCritique(BaseModel):
    """Pydantic contract for Tier 2 visual relevance semantic evaluation."""

    is_relevant: bool = Field(
        description="True if the visual content accurately represents narration and query."
    )
    relevance_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Semantic alignment score from 0.0 (unrelated) to 10.0 (perfect match)."
    )
    reason: str = Field(
        description="Concise rationale explaining why the visual frame matches or fails."
    )


class HeadOfVisualRelevance(BaseDepartmentHead):
    """Department Head agent governing cross-modal semantic relevance."""

    def __init__(self) -> None:
        super().__init__(
            department_name="visual_relevance",
            head_title="Head of Visual Relevance",
            role=AgentRole.HEAD_OF_VISUAL_RELEVANCE,
        )

    def _extract_metadata_tokens(
        self,
        artifact_path: Path,
        clip_metadata: Optional[Dict[str, Any]] = None
    ) -> Set[str]:
        """Extracts content tokens from clip tags, URL slugs, or filename."""
        tokens: Set[str] = set()
        meta = clip_metadata or {}

        # 1. Tags list from metadata
        tags_raw = meta.get("tags", [])
        if isinstance(tags_raw, list):
            for t in tags_raw:
                tokens.update(extract_content_tokens(str(t)))
        elif isinstance(tags_raw, str):
            tokens.update(extract_content_tokens(tags_raw))

        # 2. URL slug
        url = meta.get("url", "")
        if "/video/" in url:
            slug_parts = url.split("/video/")[-1].split("-")
            clean_slug = " ".join([p for p in slug_parts if not p.isdigit()])
            tokens.update(extract_content_tokens(clean_slug))

        # 3. Description or title if present
        desc = meta.get("description", "") or meta.get("title", "")
        if desc:
            tokens.update(extract_content_tokens(desc))

        return tokens

    def _extract_midpoint_frame(self, video_path: Path, output_image: Path) -> bool:
        """Extracts a single representative mid-point frame from the video via FFmpeg."""
        try:
            # Probe duration
            cmd_dur = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path)
            ]
            res_dur = subprocess.run(cmd_dur, capture_output=True, text=True, check=True)
            dur_data = json.loads(res_dur.stdout)
            dur = float(dur_data.get("format", {}).get("duration", 2.0))
            midpoint = max(0.2, dur / 2.0)

            cmd_frame = [
                "ffmpeg", "-v", "error", "-y",
                "-ss", f"{midpoint:.2f}",
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",
                str(output_image)
            ]
            subprocess.run(cmd_frame, capture_output=True, check=True)
            return output_image.exists() and output_image.stat().st_size > 0
        except Exception as err:
            logger.warning(f"Could not extract midpoint frame from {video_path.name}: {err}")
            return False

    def inspect_tier1(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> DepartmentGateResult:
        """Evaluates deterministic token overlap between query/narration and clip tags."""
        t0 = time.time()
        ctx = context or {}
        p = Path(artifact)

        query = str(ctx.get("visual_query") or "").strip()
        narration = str(ctx.get("narration") or "").strip()
        clip_meta = ctx.get("clip_metadata") or {}

        if not query and not narration:
            # No context passed — pass conditionally
            return DepartmentGateResult(
                passed=True,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                score=7.0,
                details={"reason": "No query or narration context provided to evaluate."},
                latency_seconds=time.time() - t0,
            )

        # Extract tokens
        query_tokens = set(extract_content_tokens(query))
        narration_tokens = set(extract_content_tokens(narration))
        target_tokens = query_tokens.union(narration_tokens)

        if not target_tokens:
            target_tokens = set(query.lower().split())

        clip_tokens = self._extract_metadata_tokens(p, clip_meta)

        # 1. Drift check: instant rejection if clip contains forbidden topics
        drift_hits = FORBIDDEN_DRIFT.intersection(clip_tokens)
        if drift_hits and not FORBIDDEN_DRIFT.intersection(target_tokens):
            msg = f"Forbidden visual drift detected: clip metadata contains {sorted(list(drift_hits))}."
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                score=1.0,
                feedback=msg,
                details={"drift_hits": list(drift_hits)},
                latency_seconds=time.time() - t0,
            )

        # 2. Keyword overlap calculation
        overlap = query_tokens.intersection(clip_tokens)
        ratio = len(overlap) / float(len(query_tokens)) if query_tokens else 0.0

        domain_overlap = [t for t in overlap if t not in GENERIC_MODIFIERS]
        is_strong_match = (ratio >= 0.40 and len(domain_overlap) >= 1) or (len(domain_overlap) >= 2)
        is_total_mismatch = (len(overlap) == 0 or len(domain_overlap) == 0)

        details = {
            "query_tokens": list(query_tokens),
            "clip_tokens": list(clip_tokens)[:10],
            "overlap_tokens": list(overlap),
            "domain_overlap": domain_overlap,
            "overlap_ratio": round(ratio, 3),
        }

        if is_strong_match:
            score = min(10.0, 6.0 + ratio * 4.0)
            return DepartmentGateResult(
                passed=True,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                score=score,
                details=details,
                latency_seconds=time.time() - t0,
            )

        if is_total_mismatch:
            feedback = (
                f"Cross-modal mismatch: clip metadata {list(clip_tokens)[:6]} shares "
                f"zero semantic overlap with query '{query}'."
            )
            return DepartmentGateResult(
                passed=False,
                department=self.department_name,
                head_title=self.head_title,
                tier=1,
                score=1.5,
                feedback=feedback,
                details=details,
                latency_seconds=time.time() - t0,
            )

        # Ambiguous / borderline match: defer to Tier 2 single-frame inspection
        details["needs_tier2"] = True
        return DepartmentGateResult(
            passed=True,
            department=self.department_name,
            head_title=self.head_title,
            tier=1,
            score=5.0,
            details=details,
            latency_seconds=time.time() - t0,
        )

    def inspect_tier2(
        self,
        artifact: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[DepartmentGateResult]:
        """Evaluates single-frame visual relevance using Gemini Flash vision."""
        ctx = context or {}
        p = Path(artifact)
        t0 = time.time()

        query = str(ctx.get("visual_query") or "").strip()
        narration = str(ctx.get("narration") or "").strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = Path(tmpdir) / "midpoint.jpg"
            if not self._extract_midpoint_frame(p, frame_path):
                return None

            try:
                from google.genai import types
                frame_bytes = frame_path.read_bytes()
                image_part = types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")

                prompt = (
                    f"You are the Head of Visual Relevance evaluating a video shot.\n"
                    f"NARRATION: '{narration}'\n"
                    f"VISUAL QUERY: '{query}'\n\n"
                    f"Does this visual frame accurately match or plausibly represent the narration and query?\n"
                    f"Reject if the frame is completely unrelated, generic stock drift, or jarringly disconnected."
                )

                raw_text, parsed = llm_manager.generate_content(
                    role=self.role,
                    contents=[image_part, prompt],
                    response_schema=VisualRelevanceCritique,
                    temperature=0.1,
                )
                if not parsed and raw_text:
                    parsed = VisualRelevanceCritique.model_validate_json(raw_text)

                if parsed and isinstance(parsed, VisualRelevanceCritique):
                    passed = parsed.is_relevant and parsed.relevance_score >= 6.0
                    return DepartmentGateResult(
                        passed=passed,
                        department=self.department_name,
                        head_title=self.head_title,
                        tier=2,
                        score=parsed.relevance_score,
                        feedback=parsed.reason if not passed else None,
                        details={"tier2_reason": parsed.reason},
                        gemini_calls=1,
                        latency_seconds=time.time() - t0,
                    )
            except Exception as err:
                logger.warning(f"Tier 2 visual relevance check encountered error: {err}")
                return None

        return None


head_of_visual_relevance = HeadOfVisualRelevance()
