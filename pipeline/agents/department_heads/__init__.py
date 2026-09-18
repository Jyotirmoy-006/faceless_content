"""Autonomous Department Head Agents Package.

Exports the 5 studio Department Head gatekeepers and registry:
- HeadOfStory (Strategy & Story Department)
- HeadOfAudio (Sound & Audio Department)
- HeadOfArt (Visual & Animation Department)
- HeadOfPost (Assembly & Post-Production Department)
- HeadOfCompliance (QA, Compliance & Publishing Department)
"""

from __future__ import annotations

from typing import Dict

from pipeline.agents.department_heads.base_head import (
    BaseDepartmentHead,
    DepartmentGateRejectionError,
    DepartmentGateResult,
)
from pipeline.agents.department_heads.head_of_art import HeadOfArt, head_of_art
from pipeline.agents.department_heads.head_of_audio import HeadOfAudio, head_of_audio
from pipeline.agents.department_heads.head_of_compliance import (
    HeadOfCompliance,
    head_of_compliance,
)
from pipeline.agents.department_heads.head_of_post import HeadOfPost, head_of_post
from pipeline.agents.department_heads.head_of_story import HeadOfStory, head_of_story

DEPARTMENT_HEADS: Dict[str, BaseDepartmentHead] = {
    "story": head_of_story,
    "audio": head_of_audio,
    "art": head_of_art,
    "post": head_of_post,
    "compliance": head_of_compliance,
}

__all__ = [
    "BaseDepartmentHead",
    "DepartmentGateResult",
    "DepartmentGateRejectionError",
    "HeadOfStory",
    "head_of_story",
    "HeadOfAudio",
    "head_of_audio",
    "HeadOfArt",
    "head_of_art",
    "HeadOfPost",
    "head_of_post",
    "HeadOfCompliance",
    "head_of_compliance",
    "DEPARTMENT_HEADS",
]
