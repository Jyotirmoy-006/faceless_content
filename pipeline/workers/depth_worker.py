"""Depth estimation subprocess worker using Depth Anything V2 Small.

Per Rule 1 (GPU MEMORY):
- Subprocess isolation: runs as an independent GPU worker under `gpu_lock.py`.
- Releases all PyTorch CUDA allocations upon process exit.
- Preflight VRAM check ensures safe headroom on 4GB RTX 3050.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def ts() -> str:
    """Returns current timestamp formatted as YYYY-MM-DD HH:MM:SS.mmm."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def check_free_vram_mb() -> int:
    """Queries current free VRAM in MiB using nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5.0
        )
        return int(res.stdout.strip().split("\n")[0])
    except Exception as e:
        print(f"[{ts()}] [DEPTH_WORKER] Warning: failed to query nvidia-smi ({e}), assuming 4096 MB", file=sys.stderr)
        return 4096


def preflight_vram_check(min_free_mb: int = 1024, max_attempts: int = 5, backoff_seconds: float = 1.5) -> int:
    """Verifies sufficient free VRAM exists before loading depth model."""
    for attempt in range(1, max_attempts + 1):
        free_mb = check_free_vram_mb()
        if free_mb >= min_free_mb:
            print(f"[{ts()}] [DEPTH_WORKER] [STAGE: PREFLIGHT_VRAM_OK] Free VRAM: {free_mb} MB (threshold: {min_free_mb} MB)")
            return free_mb
        print(
            f"[{ts()}] [DEPTH_WORKER] [STAGE: PREFLIGHT_VRAM_LOW] Free VRAM ({free_mb} MB) below {min_free_mb} MB. "
            f"Attempt {attempt}/{max_attempts}. Backing off for {backoff_seconds:.1f}s...",
            file=sys.stderr
        )
        time.sleep(backoff_seconds)
        backoff_seconds *= 1.5

    raise RuntimeError(
        f"Preflight VRAM check failed: Free VRAM ({check_free_vram_mb()} MB) below {min_free_mb} MB after {max_attempts} attempts."
    )


def estimate_depth(
    image_path: Path,
    output_path: Path,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: Optional[str] = None
) -> Path:
    """Estimates monocular depth map from an input image using Depth Anything V2.

    Outputs a normalized 8-bit grayscale PNG where brighter values represent
    objects closer to the virtual camera (disparity map).
    """
    import torch
    from transformers import pipeline

    start_t = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[{ts()}] [DEPTH_WORKER] [STAGE: DEPTH_LOAD_START] Loading depth model '{model_name}' on {device}...", flush=True)
    pipe = pipeline(task="depth-estimation", model=model_name, device=device)
    print(f"[{ts()}] [DEPTH_WORKER] [STAGE: DEPTH_LOAD_COMPLETE] Model loaded into memory.", flush=True)

    # Load source image
    input_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = input_img.size
    print(f"[{ts()}] [DEPTH_WORKER] [STAGE: DEPTH_INFERENCE_START] Running inference on image ({orig_w}x{orig_h})...", flush=True)

    result = pipe(input_img)
    depth_map = result["depth"]

    # Ensure depth map matches source image dimensions
    if depth_map.size != (orig_w, orig_h):
        depth_map = depth_map.resize((orig_w, orig_h), Image.Resampling.BICUBIC)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_map.save(output_path)
    print(f"[{ts()}] [DEPTH_WORKER] [STAGE: DEPTH_MAP_SAVED] Depth map saved to {output_path}", flush=True)

    # Explicit memory cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_t
    print(f"[{ts()}] [DEPTH_WORKER] [STAGE: DEPTH_COMPLETE] Depth estimation complete ({elapsed:.2f}s elapsed)", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Standalone Depth Anything V2 estimation worker")
    parser.add_argument("--input", type=str, required=True, help="Path to input source image")
    parser.add_argument("--output", type=str, required=True, help="Path to save output depth map PNG")
    parser.add_argument("--model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('cuda' or 'cpu')")
    parser.add_argument("--min-free-vram", type=int, default=1024, help="Preflight VRAM threshold in MB")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"[{ts()}] [DEPTH_WORKER] ERROR: Input image not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve()

    # Preflight check
    preflight_vram_check(min_free_mb=args.min_free_vram)

    try:
        estimate_depth(
            image_path=input_path,
            output_path=output_path,
            model_name=args.model,
            device=args.device
        )
        sys.exit(0)
    except Exception as e:
        print(f"[{ts()}] [DEPTH_WORKER] ERROR: Depth estimation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
