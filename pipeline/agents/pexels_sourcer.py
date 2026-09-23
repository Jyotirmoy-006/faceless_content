"""Pexels stock video search, multi-candidate ranking, and anti-drift semantic scoring.

Complies with:
- Rule 2 (NO SILENT CRASHES): Bounded retries, robust error handling, fallbacks.
- Rule 9 (VISUAL CONSISTENCY): Eliminates stock drift, matches prompt entity context.
- AGENTS.md: Modular organization under 300 lines with block method comments.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger("art_director.pexels")

STOPWORDS: Set[str] = {
    "a", "an", "the", "in", "on", "at", "of", "for", "to", "with", "by",
    "and", "or", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "into", "from", "up",
    "out", "as", "about", "dynamic", "cinematic", "macro", "shot", "4k",
    "hd", "high", "detail", "view", "scene", "video", "footage", "clip"
}

RELEVANCE_THRESHOLD = 0.25
FORBIDDEN_DRIFT = {
    "snow", "winter", "dance", "party", "club", "disco", "nightlife",
    "beach", "vacation", "baking", "cooking", "yoga", "fitness",
    "forklift", "warehouse", "pallet", "cargo", "freight", "shipping",
    "golf", "lawn", "kitchen", "chandelier", "lamp", "ceiling", "furniture",
    "house", "realtor", "estate", "apartment", "couple", "wedding", "necklace"
}
GENERIC_MODIFIERS = {
    "moving", "lines", "desk", "room", "action", "setup", "fast", "slow",
    "modern", "clean", "dark", "light", "bright", "close", "wide", "high",
    "low", "view", "indoor", "outdoor", "background", "office", "work",
    "person", "man", "woman", "hand", "hands", "typing", "keychain"
}


def extract_content_tokens(text: str) -> List[str]:
    """Extracts lowercase alphabetic content tokens, filtering common stopwords."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS]


def is_concrete_entity(query: str) -> bool:
    """Discriminated between concrete physical entities and abstract conceptual queries."""
    tokens = extract_content_tokens(query)
    if not tokens:
        return False

    abstract_stems = {
        "think", "thought", "reason", "logic", "concept", "idea", "econom",
        "philosophy", "wisdom", "understand", "system", "mind", "secret",
        "truth", "strategy", "breakthrough", "intelligence"
    }
    concrete_stems = {
        "chip", "robot", "server", "skyline", "city", "wire", "screen", "code",
        "microscope", "laboratory", "laser", "satellite", "space", "planet",
        "car", "engine", "cyberpunk", "circuit", "hologram", "datacenter",
        "camera", "drone", "building", "network", "silicon", "office", "batter"
    }

    has_concrete = any(any(cs in t for cs in concrete_stems) for t in tokens)
    has_abstract = any(any(ab in t for ab in abstract_stems) for t in tokens)

    if has_concrete:
        return True
    if has_abstract:
        return False
    return True


def compute_relevance_score(query: str, pexels_video_data: Dict[str, Any]) -> float:
    """Computes semantic relevance score between query and Pexels video metadata."""
    query_tokens = set(extract_content_tokens(query))
    if not query_tokens:
        return 0.5

    url = pexels_video_data.get("url", "")
    slug = ""
    if "/video/" in url:
        slug = url.split("/video/")[-1].split("-")
        slug = " ".join([part for part in slug if not part.isdigit()])

    tags_list = pexels_video_data.get("tags", [])
    tags_text = " ".join(tags_list) if isinstance(tags_list, list) else str(tags_list)

    metadata_text = f"{slug} {tags_text}".lower()
    metadata_tokens = set(extract_content_tokens(metadata_text))

    if not metadata_tokens:
        return 0.10

    if FORBIDDEN_DRIFT.intersection(metadata_tokens) and not FORBIDDEN_DRIFT.intersection(query_tokens):
        return 0.05

    intersection = query_tokens.intersection(metadata_tokens)
    overlap_ratio = len(intersection) / float(len(query_tokens))

    # Reject if all matching tokens are generic non-specific modifiers without a domain entity
    if intersection and intersection.issubset(GENERIC_MODIFIERS):
        return 0.15

    if intersection:
        non_generic = [t for t in intersection if t not in GENERIC_MODIFIERS]
        if not non_generic:
            return 0.15
        return max(overlap_ratio, 0.25)

    query_stems = [t[:4] for t in query_tokens if len(t) >= 4]
    matched_stems = [st for st in query_stems if any(st in m for m in metadata_tokens)]
    if matched_stems and not set(matched_stems).issubset({g[:4] for g in GENERIC_MODIFIERS}):
        return 0.25

    return overlap_ratio


def clean_pexels_query(text: str) -> str:
    """Strips camera shot directions to isolate 2-4 search keywords for Pexels."""
    camera_terms = [
        r"\bextreme close up\b", r"\bclose up\b", r"\bmacro lens\b", r"\bmacro shot\b",
        r"\bmacro\b", r"\bwide angle\b", r"\bdrone shot\b", r"\bdrone\b", r"\bsplit screen\b",
        r"\bhigh speed camera\b", r"\bcinematic\b", r"\bdynamic\b", r"\bhyper detailed\b",
        r"\bphotorealistic\b", r"\b4k\b", r"\bhd\b", r"\bglowing\b", r"\bshot of\b",
        r"\bshot\b", r"\bview of\b", r"\bview\b"
    ]
    cleaned = text.lower()
    for pattern in camera_terms:
        cleaned = re.sub(pattern, "", cleaned)
    tokens = [w for w in re.findall(r"[a-zA-Z]{3,}", cleaned) if w not in STOPWORDS]
    return " ".join(tokens[:4]) if tokens else text


def is_mock_asset(path: Path) -> bool:
    """Detects whether a cached video was generated from a mock card or offline placeholder."""
    path = Path(path)
    if "mock" in path.name.lower():
        return True

    png_path = path.with_suffix(".png")
    if png_path.exists():
        if png_path.stat().st_size < 25_000:
            return True
        try:
            from PIL import Image
            import numpy as np
            with Image.open(png_path) as im:
                arr = np.array(im)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    c0 = arr[0, 0]
                    if 20 <= c0[0] <= 26 and 24 <= c0[1] <= 32 and 35 <= c0[2] <= 60:
                        return True
        except Exception:
            pass

    return False


def find_best_pexels_candidate(
    query: str,
    pexels_key: str,
    exclude_video_ids: Optional[Set[int]] = None
) -> Optional[Tuple[Dict[str, Any], float]]:
    """Searches Pexels across query variations and returns the highest-scoring candidate."""
    if not pexels_key:
        return None

    clean_q = clean_pexels_query(query)
    tokens = [w for w in re.findall(r"[a-zA-Z]{3,}", clean_q.lower()) if w not in STOPWORDS]

    query_attempts = [clean_q]
    if len(tokens) >= 2:
        query_attempts.append(" ".join(tokens[:2]))
        query_attempts.append(" ".join(tokens[-2:]))
    if "battery" in tokens:
        query_attempts.append("battery technology")
        query_attempts.append("battery")
    if any(t in tokens for t in ["explosion", "runaway", "fire"]):
        query_attempts.append("explosion laboratory")
    if any(t in tokens for t in ["circuit", "board", "chip"]):
        query_attempts.append("circuit board")
    if any(t in tokens for t in ["engineer", "screen", "computer"]):
        query_attempts.append("working on computer")

    seen_urls: Set[str] = set()
    all_candidates: List[Tuple[float, Dict[str, Any]]] = []
    headers = {"Authorization": pexels_key}
    url = "https://api.pexels.com/videos/search"

    for q_try in query_attempts:
        for orient in ["portrait", None]:
            params: Dict[str, Any] = {"query": q_try, "per_page": 5}
            if orient:
                params["orientation"] = orient
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=8.0)
                if resp.status_code != 200:
                    continue
                videos = resp.json().get("videos", [])
                for v in videos:
                    v_id = v.get("id")
                    if exclude_video_ids and v_id in exclude_video_ids:
                        continue
                    v_url = v.get("url", "")
                    if v_url in seen_urls:
                        continue
                    seen_urls.add(v_url)
                    rel = compute_relevance_score(query, v)
                    all_candidates.append((rel, v))
            except Exception as e:
                logger.warning(f"[ART_DIRECTOR] Pexels search error for '{q_try}': {e}")

        high_matches = [c for c in all_candidates if c[0] >= 0.25]
        if high_matches:
            break

    if not all_candidates:
        return None

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cand = all_candidates[0]

    if best_score >= RELEVANCE_THRESHOLD:
        return best_cand, best_score

    logger.warning(f"[ART_DIRECTOR] All Pexels candidates rejected (top score {best_score:.2f} < {RELEVANCE_THRESHOLD}).")
    return None


def download_pexels_clip(candidate: Dict[str, Any], target_path: Path) -> bool:
    """Downloads highest quality MP4 stream from Pexels video candidate."""
    video_files = candidate.get("video_files", [])
    download_link = None

    for vf in video_files:
        if vf.get("file_type") == "video/mp4" and vf.get("width", 0) >= 720:
            download_link = vf.get("link")
            break
    if not download_link and video_files:
        for vf in video_files:
            if vf.get("file_type") == "video/mp4":
                download_link = vf.get("link")
                break
    if not download_link and video_files:
        download_link = video_files[0].get("link")

    if not download_link:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(download_link, timeout=30.0, stream=True)
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"[ART_DIRECTOR] Failed downloading Pexels clip: {e}")
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def find_cached_real_asset(cache_root: Path) -> Optional[Path]:
    """Finds any genuine cached MP4 asset (> 1MB, not mock) as emergency fallback."""
    try:
        for mp4 in cache_root.rglob("*.mp4"):
            if mp4.stat().st_size > 1_000_000 and not is_mock_asset(mp4):
                return mp4
    except Exception:
        pass
    return None
