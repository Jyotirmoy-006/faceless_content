"""Scriptwriter Agent with schema-enforced Gemini structured output.

Per Rule 4 (VALIDATE LLM OUTPUT):
- Calls Gemini using native structured output (response_schema=Script).
- Schema-validated using Pydantic.
- On ValidationError: bounded retry (max 2) with corrective prompt containing the error.
- On persistent failure: loads niche/generic fallback template and logs at WARNING level.
- Downstream callers are guaranteed never to receive an invalid script or unhandled crash.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import dotenv
from pydantic import ValidationError

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.schema import Script, ScriptSegment
from pipeline.core.llm_manager import llm_manager, AgentRole

# Configure module logger
logger = logging.getLogger("scriptwriter")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

FALLBACK_TEMPLATES_DIR = Path(__file__).resolve().parent / "fallback_templates"
PRIMARY_MODEL = llm_manager.resolve_model(AgentRole.COPYWRITER)
BACKUP_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]


def load_fallback_template(topic: str, niche: Optional[str] = None) -> Script:
    """Loads a deterministic fallback template matching the niche, or generic.json."""
    selected_template = FALLBACK_TEMPLATES_DIR / "generic.json"

    if niche:
        candidate = FALLBACK_TEMPLATES_DIR / f"{niche.lower().strip()}.json"
        if candidate.exists():
            selected_template = candidate

    try:
        with open(selected_template, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Adapt topic if safe
        if topic and topic.strip():
            data["topic"] = topic.strip()
        return Script.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to load fallback template from {selected_template}: {e}. Generating in-memory safety script.")
        # Ultimate fail-safe: guaranteed valid Script with visual_shots satisfying Head of Story
        return Script(
            topic=topic or "Curiosities of the World",
            niche=niche or "general",
            target_duration=30,
            hook="Here is a wild truth that nobody was ready for until right now.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Throughout history, extraordinary hidden discoveries have completely altered human reality.",
                    visual_query="mysterious glowing astronomical phenomena high detail cinematic",
                    visual_shots=[
                        "mysterious glowing astronomical galaxy cinematic",
                        "macro glowing physics particle stream",
                        "dramatic zoom into deep space cosmic void"
                    ],
                    duration_seconds=6.0
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="Wait, modern researchers looked closer and uncovered secrets hidden in plain sight.",
                    visual_query="advanced laboratory microscopic visualization glowing particles 4k",
                    visual_shots=[
                        "advanced laboratory microscopic visualization",
                        "close up scientist shocked looking into holographic monitor",
                        "rapid data visualization scrolling glowing code"
                    ],
                    duration_seconds=6.0
                ),
                ScriptSegment(
                    segment_index=3,
                    narration="Actually, the real lore goes deeper than anyone ever predicted.",
                    visual_query="holographic futuristic blueprint glowing neon light",
                    visual_shots=[
                        "holographic futuristic blueprint glowing",
                        "over the shoulder view deciphering ancient code",
                        "extreme macro laser pulse illuminating artifact"
                    ],
                    duration_seconds=6.0
                ),
                ScriptSegment(
                    segment_index=4,
                    narration="Lock in, because understanding this shifts how you perceive everything around you.",
                    visual_query="person watching futuristic city sunrise contemplation",
                    visual_shots=[
                        "silhouette person watching futuristic city sunrise",
                        "dramatic sunset golden hour horizon reflection",
                        "rapid zoom out high tech metropolis"
                    ],
                    duration_seconds=6.0
                )
            ],
            cta="Follow for more daily discoveries."
        )


def _call_gemini_structured(
    client,
    contents: str,
    system_instruction: Optional[str] = None,
    model_name: str = PRIMARY_MODEL
) -> tuple:
    """Calls Gemini with native structured output mode."""
    from google.genai import types

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Script,
        temperature=0.7,
        system_instruction=system_instruction
    )

    models_to_try = [model_name] + [m for m in BACKUP_MODELS if m != model_name]
    last_error = None

    for m in models_to_try:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=contents,
                config=config
            )
            raw_text = resp.text
            parsed_obj = getattr(resp, "parsed", None)
            return raw_text, parsed_obj
        except Exception as err:
            last_error = err
            logger.debug(f"Model {m} failed: {err}. Trying next candidate.")

    raise last_error


def get_valid_script(
    topic: str,
    niche: Optional[str] = "general",
    max_retries: int = 2,
    client = None,
    feedback: Optional[str] = None
) -> Script:
    """Fetches and validates a video script from Gemini with bounded retries and fallback.

    Args:
        topic: Subject or title of the video.
        niche: Category (e.g. 'tech', 'finance', 'history', 'psychology', 'general').
        max_retries: Maximum number of corrective retries on ValidationError (Rule 8).
        client: Optional pre-configured google.genai.Client instance.
        feedback: Optional feedback from previous verification failure for corrective prompt.

    Returns:
        Script: Guaranteed valid Pydantic Script instance.
    """
    dotenv.load_dotenv()
    model_name = PRIMARY_MODEL
    if client is None:
        try:
            client, model_name, active_account = llm_manager.get_client_and_model(AgentRole.COPYWRITER)
        except Exception as e:
            logger.warning(
                f"[LLM_MANAGER] Unable to resolve healthy account ({e}). Loading fallback template for topic: '{topic}'."
            )
            return load_fallback_template(topic, niche)

    system_prompt = (
        "You are an elite short-form viral retention engineer and scriptwriter (YouTube Shorts, TikTok). "
        "Create punchy, high-retention video scripts that strictly adhere to the provided JSON schema.\n"
        "TONE & NARRATOR PERSONA (ENERGETIC TONE LOCK):\n"
        "- High-Energy & Upbeat: Speak with intense curiosity, vivid momentum, and confident urgency. "
        "Never sound like a dull academic lecture or monotone documentary narrator.\n"
        "- Contemporary & Rhythmic: Use punchy, snappy, conversational phrasing with strong cadence and rhythmic drive. "
        "Avoid archaic transitions ('furthermore', 'in addition', 'moreover', 'as we can see').\n"
        "CORE RETENTION CONSTRAINTS:\n"
        "1. HOOK (0.0s - 3.0s): Must stop the scroll in under 12 spoken words. Deliver immediate personal stakes "
        "('Your passwords', 'Your bank account') or high-stakes visceral threats. NEVER start with 'Imagine', 'In this video', 'Did you know', or rhetorical questions.\n"
        "2. NO REPETITION IN SEGMENT 1: Segment 1 is the 'HOOK' beat, but its narration MUST NOT repeat the hook sentence! "
        "Segment 1 narration must immediately dive into the context, mechanism, or proof.\n"
        "3. 5 NARRATIVE BEATS: Structure across segments using beats: 'HOOK', 'TENSION', 'ESCALATION', 'REVELATION', and 'LOOP'.\n"
        "4. TOPIC-ANCHORED VISUAL SHOTS: In each segment, provide 'visual_shots' with 2-3 micro-shot queries "
        "DEEPLY ANCHORED in the topic aesthetic. NEVER use generic everyday queries like 'keys', 'hands typing', or 'city' without high-tech context (e.g. use 'cyber hacker typing decryption code on dark terminal', 'quantum processor chip with glowing qubit circuits', 'digital padlock dissolving into binary dust').\n"
        "5. DUAL VISUAL QUERIES & SOURCE: Provide 'pexels_query' and 'sd_prompt'. Set 'asset_source' to 'comfyui' for futuristic, quantum, or conceptual imagery to guarantee 100% thematic relevance.\n"
        "6. SEAMLESS CYCLICAL LOOP OUTRO: The 'loop_outro' field must be a 2 to 4 word UNPUNCTUATED connector clause (e.g. 'and that is why', 'which proves why', 'leaving us with') "
        "that grammatically connects directly to the first word of the hook for infinite replay loops without any trailing period.\n"
        "7. ACTIVE VERBS & TANGIBLE STAKES: Use active dynamic verbs ('simulate', 'permeate', 'collapse', 'annihilate') instead of passive phrasing like 'walk through'. "
        "Never insert intra-phrase commas into compound adjectives (write 'military-grade encryption', NEVER 'military grade,').\n"
        "8. ZERO FLUFF & 180 WPM BUDGET: Total spoken narration across all segments must be 75 to 95 words. High information density."
    )

    prompt = (
        f"Write an ultra-high-retention viral short-form video script on the topic: '{topic}'.\n"
        f"Category/Niche: {niche or 'general'}.\n"
        f"Target duration: 30 to 35 seconds total across 4 to 5 narrative segments.\n"
        "Requirements:\n"
        "- Persona: Upbeat, electrifying, fast-paced delivery with infectious curiosity.\n"
        "- The 'hook' must be bold, intense, personally threatening, and under 12 words.\n"
        "- Segment 1 narration MUST NOT repeat the hook sentence; start immediately with new facts or tension.\n"
        "- Assign each segment a 'beat' ('HOOK', 'TENSION', 'ESCALATION', 'REVELATION', 'LOOP').\n"
        "- Each segment's 'duration_seconds' must be strictly between 4.0 and 10.0 seconds.\n"
        "- Provide both 'pexels_query' (natural stock search) and 'sd_prompt' (cinematic SD1.5 prompt) for each segment.\n"
        "- In each segment, provide 'visual_shots' with 2 to 3 micro-shot visual queries.\n"
        "- Total spoken words across all segments must be between 75 and 95 words (approx 30s at 180 WPM).\n"
        "- Ground all concepts in physical analogies (no abstract mathematical named theorems without physical analogies).\n"
        "- Craft 'loop_outro' as a 2 to 4 word connector without a period that flows grammatically into the first word of your hook.\n"
        "Ensure all schema fields are strictly provided and valid."
    )

    if feedback:
        prompt += (
            f"\n\nIMPORTANT REVISION FEEDBACK: Your previous draft failed verification:\n"
            f"{feedback}\n"
            f"Please strictly correct these issues and adhere to all requirements."
        )

    # Brainrot / Entertainment niche overlay — active when niche is 'entertainment'
    # Injects cultural vocabulary and structural constraints that score high on the
    # virality evaluator (aura signals, loop triggers, scroll-stop hook power words).
    if niche and niche.lower() == "entertainment":
        system_prompt += (
            "\nBRAINROT ENTERTAINMENT MODE — MANDATORY OVERRIDES:\n"
            "- You are writing for Gen Z / Gen Alpha brainrot culture. Every segment MUST use "
            "at least one of these cultural signal words naturally: "
            "'aura', 'lock in', 'sigma', 'skibidi', 'brainrot', 'lore', 'unhinged', 'based', "
            "'NPC', 'ratio', 'actually', 'nobody', 'literally', 'wild', 'insane', 'dark', "
            "'hidden', 'secret', 'they hid', 'wait'.\n"
            "- The HOOK must contain at least 2 power-stop words from: "
            "'secret', 'dark', 'hidden', 'nobody', 'literally', 'insane', 'wild', 'actually', 'truth'.\n"
            "- EVERY segment must contain at least one LOOP TRIGGER phrase: "
            "'wait', 'but here\'s the thing', 'plot twist', 'it gets worse', "
            "'but then', 'the real truth', 'nobody talks about this', 'actually', 'origin'.\n"
            "- Pacing: Write at 4.5 to 5.5 words per second (high-density, fast-fire delivery).\n"
            "- Use self-aware meta language — the video can reference its own virality, "
            "the algorithm, the scroll, or the viewer\'s brain directly.\n"
            "- Title must contain CAPS, an emoji, or a number for high CTR."
        )
        prompt += (
            "\n\nBRAINROT CULTURE REQUIREMENTS:\n"
            "- Weave in Gen Z/Gen Alpha vocabulary: aura, lock in, sigma, skibidi, brainrot, lore, NPC, ratio.\n"
            "- Start the hook with a scroll-stopping power phrase using words like: "
            "'nobody', 'literally', 'actually', 'dark', 'hidden', 'secret', 'insane', 'wild'.\n"
            "- Every segment must have at least one loop trigger: 'wait', 'but here\'s the thing', "
            "'plot twist', 'it gets worse', 'nobody talks about this', 'actually', 'origin'.\n"
            "- Target pacing: 4.5 to 5.5 words per second (fast, punchy, high-dopamine delivery).\n"
            "- Make the viewer feel like they\'re losing aura by NOT knowing this."
        )

    last_failure_reason = "Unknown error"
    current_contents = prompt

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Generating script for topic '{topic}' (Attempt {attempt + 1}/{max_retries + 1})...")
            raw_text, parsed_obj = _call_gemini_structured(
                client=client,
                contents=current_contents,
                system_instruction=system_prompt,
                model_name=model_name
            )

            # 1. Check if native SDK parsed it into the Pydantic model directly
            if isinstance(parsed_obj, Script):
                logger.info(f"Script validated successfully on attempt {attempt + 1}.")
                return parsed_obj

            # 2. If raw JSON text was returned, validate via Pydantic model_validate_json
            if raw_text:
                script = Script.model_validate_json(raw_text)
                logger.info(f"Script validated successfully via Pydantic on attempt {attempt + 1}.")
                return script

            raise ValueError("Gemini returned empty response text.")

        except (ValidationError, json.JSONDecodeError, ValueError) as val_err:
            last_failure_reason = f"{val_err.__class__.__name__}: {val_err}"
            logger.info(f"Validation failure on attempt {attempt + 1}: {last_failure_reason}")

            if attempt < max_retries:
                # Corrective re-prompting per Rule 4
                current_contents = (
                    f"Your previous output for topic '{topic}' failed schema validation with error:\n"
                    f"{last_failure_reason}\n\n"
                    "Correct ONLY the invalid fields and output a strictly valid JSON object matching the schema."
                )
            else:
                break
        except Exception as api_err:
            last_failure_reason = f"{api_err.__class__.__name__}: {api_err}"
            logger.info(f"API/Network error on attempt {attempt + 1}: {last_failure_reason}")
            if attempt >= max_retries:
                break

    # If all attempts exhausted, trigger Rule 4 & Rule 2 fallback
    logger.warning(
        f"LLM script validation/generation failed for topic '{topic}': {last_failure_reason}. "
        f"Loaded fallback template for niche '{niche}'."
    )
    return load_fallback_template(topic, niche)


def main():
    parser = argparse.ArgumentParser(description="Scriptwriter Agent CLI")
    parser.add_argument("--topic", type=str, required=True, help="Topic for the video script")
    parser.add_argument("--niche", type=str, default="general", help="Content category (tech, finance, history, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file path")
    args = parser.parse_args()

    script = get_valid_script(topic=args.topic, niche=args.niche)
    pretty_json = script.model_dump_json(indent=2)
    print(pretty_json)

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pretty_json)
        print(f"\nSaved script to {out_path}")


if __name__ == "__main__":
    main()
