"""Ideator Agent: Generates viral short-form video concepts with schema validation.

Per Rule 2 (NO SILENT CRASHES):
- All Gemini API calls are wrapped with explicit error handling and bounded retries.
- Deterministic fallback ideas are loaded if LLM generation fails or network drops.

Per Rule 4 (VALIDATE LLM OUTPUT):
- Uses Gemini native structured output against IdeaConcept Pydantic schema.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import ValidationError

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.schema import IdeaConcept
from pipeline.core.llm_manager import llm_manager, AgentRole

load_dotenv()

logger = logging.getLogger("ideator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [ideator] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# High-retention fallback idea catalog
FALLBACK_IDEAS: dict[str, list[dict[str, Any]]] = {
    "tech": [
        {
            "topic": "Why Solid-State Batteries Will Change Everything",
            "niche": "tech",
            "angle": "How EV charging in under 5 minutes will make gas cars completely obsolete",
            "target_duration": 30
        },
        {
            "topic": "The Secret Tech Inside Modern Submarines",
            "niche": "tech",
            "angle": "How anechoic tiles make 10,000-ton steel monsters completely invisible to sonar",
            "target_duration": 30
        },
        {
            "topic": "Why Quantum Computers Won't Replace Your Phone",
            "niche": "tech",
            "angle": "The bizarre physics that require absolute zero temperatures to calculate",
            "target_duration": 30
        }
    ],
    "finance": [
        {
            "topic": "The Trillion-Dollar Nickel Heist",
            "niche": "finance",
            "angle": "How a trading giant bought stones disguised as metals in global ports",
            "target_duration": 30
        },
        {
            "topic": "The Psychology of Luxury Pricing",
            "niche": "finance",
            "angle": "Why raising the price of a handbag actually makes customers demand it more",
            "target_duration": 30
        }
    ],
    "history": [
        {
            "topic": "The Roman Concrete That Heals Itself",
            "niche": "history",
            "angle": "How 2,000-year-old structures survive underwater using volcanic ash and quicklime",
            "target_duration": 30
        },
        {
            "topic": "The Lost City Beneath the Sahara Desert",
            "niche": "history",
            "angle": "Satellite scans reveal rivers and civilizations under the endless sand",
            "target_duration": 30
        }
    ],
    "science": [
        {
            "topic": "What Actually Happens at the Center of a Black Hole",
            "niche": "science",
            "angle": "Why time literally stops and space becomes a one-way street",
            "target_duration": 30
        },
        {
            "topic": "The Deepest Trench on Planet Earth",
            "niche": "science",
            "angle": "Creatures thriving under pressure equivalent to 50 jumbo jets resting on you",
            "target_duration": 30
        }
    ],
    "psychology": [
        {
            "topic": "The Illusion of Free Choice in Supermarkets",
            "niche": "psychology",
            "angle": "How store layouts and eye-level shelves manipulate 80% of your purchases",
            "target_duration": 30
        }
    ]
}


class IdeatorAgent:
    """Generates high-engagement short-form video topics and angles."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._client = None
        if self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def generate_idea(
        self,
        niche: str = "tech",
        custom_topic: Optional[str] = None,
        max_retries: int = 2,
        dry_run: bool = False,
        excluded_topics: Optional[list[str]] = None,
        trend_candidates: Optional[list[str]] = None
    ) -> IdeaConcept:
        """Generates a structured video concept for short-form video production.
        
        Args:
            niche: Content category (tech, finance, science, history, psychology).
            custom_topic: Optional user-specified topic override.
            max_retries: Maximum LLM re-prompt attempts on validation error.
            dry_run: If True, uses fallback template without calling Gemini API.
            excluded_topics: Optional list of recent topics to exclude for SEO deduplication.
            trend_candidates: Optional list of trending candidates to guide ideation.
            
        Returns:
            Validated IdeaConcept Pydantic model.
        """
        clean_niche = niche.lower().strip()
        if custom_topic:
            return IdeaConcept(
                topic=custom_topic,
                niche=clean_niche,
                angle=f"The hidden truth and most fascinating revelation about {custom_topic}",
                target_duration=30
            )

        if excluded_topics is None:
            try:
                from pipeline.dashboard.database import get_recent_topics
                excluded_topics = get_recent_topics(niche=clean_niche, limit=200)
            except Exception as ex:
                logger.debug(f"[ideator] Could not load recent topics from database: {ex}")
                excluded_topics = []

        # 1. Fetch live YouTube trend candidates for inspiration (Rule 3: 1 Data API unit)
        if trend_candidates is None:
            try:
                from pipeline.agents.strategist import strategist
                trend_candidates = strategist.get_trending_topics(niche=clean_niche, dry_run=dry_run)
            except Exception as ex:
                logger.debug(f"[ideator] Could not load live trend signals: {ex}")
                trend_candidates = []

        client = self._client
        model_name = llm_manager.resolve_model(AgentRole.COPYWRITER)
        if client is None:
            try:
                client, model_name, _ = llm_manager.get_client_and_model(AgentRole.COPYWRITER)
            except Exception as e:
                logger.info(f"[ideator] No healthy LLM account available ({e}). Using fallback concept.")
                return self._get_fallback_idea(clean_niche, excluded_topics)

        if dry_run or not client:
            logger.info(f"[ideator] Using curated fallback concept for niche '{clean_niche}' (dry_run={dry_run}).")
            return self._get_fallback_idea(clean_niche, excluded_topics)

        prompt = (
            f"You are a viral YouTube Shorts concept strategist. "
            f"Generate a single, highly engaging, click-worthy video concept in the '{clean_niche}' niche. "
            f"The angle must create curiosity and deliver an astonishing fact or perspective. "
            f"Prioritize claims that are verifiably true or clearly framed as opinion/perspective — do not fabricate statistics or overstate certainty. "
            f"Target duration should be 30 to 45 seconds.\n\n"
            f"CRITICAL VISUAL TRACTABILITY REQUIREMENT:\n"
            f"Every core idea in your concept MUST be capable of concrete, physical visual representation. "
            f"DO NOT generate concepts centered on abstract source code, coding syntax, intangible software architectures, or abstract formulas without physical anchors. "
            f"The concept MUST feature tangible physical anchors (e.g., massive server racks, glowing fiber optic cables under the ocean, robotic hardware, silicon wafer lithography, deep-sea submersibles, mega-structures) that allow the video generator to depict real objects, physical mechanisms, and real-world scale comparisons."
        )

        if trend_candidates:
            trends_list = "\n".join(f"- {t}" for t in trend_candidates[:5])
            prompt += (
                f"\n\nLIVE TREND INSPIRATION (Use these current on-platform signals for inspiration; craft an original angle):\n"
                f"{trends_list}"
            )

        if excluded_topics:
            exclusion_list = "\n".join(f"- {t}" for t in excluded_topics[:40])
            prompt += (
                f"\n\nCRITICAL DEDUPLICATION REQUIREMENT:\n"
                f"You MUST NOT generate any topic on or substantially similar to these previously published topics:\n"
                f"{exclusion_list}\n"
                f"Your topic must be completely novel and distinctive to prevent audience and SEO cannibalization."
            )

        from google.genai import types

        for attempt in range(1, max_retries + 2):
            logger.info(f"Generating concept idea for niche '{clean_niche}' (Attempt {attempt}/{max_retries + 1})...")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=IdeaConcept,
                        temperature=0.7,
                    )
                )

                raw_text = response.text.strip()
                parsed = json.loads(raw_text)
                idea = IdeaConcept.model_validate(parsed)

                if excluded_topics:
                    if self._is_duplicate_topic(idea.topic, excluded_topics):
                        logger.warning(f"Generated topic '{idea.topic}' duplicates an existing topic in channel history. Retrying...")
                        prompt += f"\n\nRejection: Topic '{idea.topic}' is too similar to past channel content. Produce an entirely different concept."
                        continue

                logger.info(f"Concept validated successfully: '{idea.topic}'")
                return idea

            except (ValidationError, json.JSONDecodeError) as err:
                logger.warning(f"Validation failure on attempt {attempt}: {err}")
                if attempt > max_retries:
                    break
                prompt += f"\n\nPrevious attempt failed with error: {err}. Please return valid schema matching fields strictly."
            except Exception as e:
                logger.warning(f"Gemini API error on attempt {attempt}: {e}")
                if attempt > max_retries:
                    break

        logger.warning(f"[FALLBACK] Ideation failed after {max_retries + 1} attempts. Loading fallback concept.")
        return self._get_fallback_idea(clean_niche, excluded_topics)

    def _get_fallback_idea(self, niche: str, excluded_topics: Optional[list[str]] = None) -> IdeaConcept:
        """Selects a curated fallback idea for the specified niche, filtering out duplicates."""
        catalog = FALLBACK_IDEAS.get(niche) or FALLBACK_IDEAS.get("tech", [])
        if excluded_topics:
            available = [item for item in catalog if not self._is_duplicate_topic(item["topic"], excluded_topics)]
            if available:
                chosen = random.choice(available)
                return IdeaConcept.model_validate(chosen)
        chosen = random.choice(catalog)
        return IdeaConcept.model_validate(chosen)

    def _is_duplicate_topic(self, candidate: str, existing_list: list[str]) -> bool:
        """Determines if candidate topic matches or substantially overlaps with past topics."""
        cand_clean = re.sub(r"[^a-z0-9 ]", "", candidate.lower()).strip()
        stop = {"why", "how", "the", "what", "will", "is", "are", "and", "in", "of", "to", "a", "an"}
        cand_words = set(cand_clean.split()) - stop
        for ex in existing_list:
            ex_clean = re.sub(r"[^a-z0-9 ]", "", ex.lower()).strip()
            if cand_clean == ex_clean:
                return True
            ex_words = set(ex_clean.split()) - stop
            if cand_words and ex_words:
                jaccard = len(cand_words & ex_words) / float(len(cand_words | ex_words))
                if jaccard >= 0.65:
                    return True
        return False


# Global default instance
ideator = IdeatorAgent()
