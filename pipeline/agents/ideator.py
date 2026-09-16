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
        dry_run: bool = False
    ) -> IdeaConcept:
        """Generates a structured video concept for short-form video production.
        
        Args:
            niche: Content category (tech, finance, science, history, psychology).
            custom_topic: Optional user-specified topic override.
            max_retries: Maximum LLM re-prompt attempts on validation error.
            dry_run: If True, uses fallback template without calling Gemini API.
            
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

        if dry_run or not self._client:
            logger.info(f"[ideator] Using curated fallback concept for niche '{clean_niche}' (dry_run={dry_run}).")
            return self._get_fallback_idea(clean_niche)

        prompt = (
            f"You are a viral YouTube Shorts and Instagram Reels concept strategist. "
            f"Generate a single, highly engaging, click-worthy video concept in the '{clean_niche}' niche. "
            f"The angle must create curiosity and deliver an astonishing fact or perspective. "
            f"Target duration should be 30 seconds."
        )

        from google.genai import types

        for attempt in range(1, max_retries + 2):
            logger.info(f"Generating concept idea for niche '{clean_niche}' (Attempt {attempt}/{max_retries + 1})...")
            try:
                response = self._client.models.generate_content(
                    model="gemini-3.6-flash",
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
        return self._get_fallback_idea(clean_niche)

    def _get_fallback_idea(self, niche: str) -> IdeaConcept:
        """Selects a curated fallback idea for the specified niche."""
        catalog = FALLBACK_IDEAS.get(niche) or FALLBACK_IDEAS.get("tech", [])
        chosen = random.choice(catalog)
        return IdeaConcept.model_validate(chosen)


# Global default instance
ideator = IdeatorAgent()
