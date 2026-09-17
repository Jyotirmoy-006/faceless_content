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
        # Ultimate fail-safe: guaranteed valid Script
        return Script(
            topic=topic or "Curiosities of the World",
            niche=niche or "general",
            target_duration=30,
            hook="Here is something fascinating you probably didn't know until now.",
            segments=[
                ScriptSegment(
                    segment_index=1,
                    narration="Throughout history, extraordinary discoveries have completely altered human understanding.",
                    visual_query="mysterious glowing astronomical phenomena high detail cinematic",
                    duration_seconds=10.0
                ),
                ScriptSegment(
                    segment_index=2,
                    narration="Modern science continues to uncover new layers to the world around us.",
                    visual_query="advanced laboratory microscopic visualization glowing particles 4k",
                    duration_seconds=10.0
                ),
                ScriptSegment(
                    segment_index=3,
                    narration="Stay curious, because every breakthrough begins with a single question.",
                    visual_query="person watching futuristic city sunrise contemplation",
                    duration_seconds=10.0
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
        "You are an elite short-form viral video scriptwriter (YouTube Shorts, Instagram Reels). "
        "Create punchy, high-retention video scripts that strictly adhere to the provided JSON schema. "
        "The hook MUST be delivered in the first 3 seconds to immediately stop the scroll. "
        "Each segment must contain spoken voiceover text and a detailed visual_query specifying "
        "the exact stock footage or generative image prompt needed to illustrate that scene."
    )

    prompt = (
        f"Write a high-retention short-form video script on the topic: '{topic}'.\n"
        f"Category/Niche: {niche or 'general'}.\n"
        f"Target duration: 30 to 45 seconds total across 3 to 5 narrative segments.\n"
        "Ensure all schema fields (hook, segments with segment_index, narration, visual_query, duration_seconds) are fully provided."
    )
    if feedback:
        prompt += (
            f"\n\nIMPORTANT REVISION FEEDBACK: Your previous draft failed verification:\n"
            f"{feedback}\n"
            f"Please strictly correct these issues and adhere to all requirements."
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
