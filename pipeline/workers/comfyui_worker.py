"""ComfyUI image generation & Ken Burns animation subprocess worker.

Per Rule 1 (GPU MEMORY):
- Subprocess isolation: When image generation and rendering finish, the process
  terminates, clearing MoviePy and buffer allocations.

COMFYUI SERVER LONGEVITY & RESTART STRATEGY NOTE:
-------------------------------------------------
Long-running ComfyUI server processes accumulate PyTorch CUDA cache fragmentation
and retained model weights in VRAM over extended continuous operation.
Mitigation Architecture:
1. Client-side Cleanup (/free): After every generation batch, this worker calls
   ComfyUI's POST /free endpoint with `{"unload_models": true, "free_memory": true}`.
   This instructs ComfyUI to purge cached weights from the 4GB RTX 3050 VRAM.
2. Startup Flag: The local ComfyUI server MUST be launched with `--lowvram` or
   `--medvram`, forcing dynamic CPU-GPU weight swapping between execution steps.
3. Automated Recycle Policy: The pipeline orchestrator must track lifetime generation
   cycles. Every 25-50 image generations (or if baseline idle VRAM exceeds 2.5 GB),
   the ComfyUI server process must be gracefully terminated and relaunched by the
   scheduler before starting the next video pipeline.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

from moviepy import VideoClip


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
        print(f"[{ts()}] [COMFYUI_WORKER] Warning: failed to query nvidia-smi ({e}), assuming 4096 MB", file=sys.stderr)
        return 4096


def preflight_vram_check(min_free_mb: int = 1024, max_attempts: int = 5, backoff_seconds: float = 2.0) -> int:
    """Preflight check: verifies sufficient free VRAM exists before starting SD1.5 generation.
    If below safety threshold, retries with backoff. Raises RuntimeError if still insufficient.
    """
    for attempt in range(1, max_attempts + 1):
        free_mb = check_free_vram_mb()
        if free_mb >= min_free_mb:
            print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: PREFLIGHT_VRAM_OK] Free VRAM: {free_mb} MB (threshold: {min_free_mb} MB)")
            return free_mb
        print(
            f"[{ts()}] [COMFYUI_WORKER] [STAGE: PREFLIGHT_VRAM_LOW] Free VRAM ({free_mb} MB) below safety threshold ({min_free_mb} MB). "
            f"Attempt {attempt}/{max_attempts}. Backing off for {backoff_seconds:.1f}s...",
            file=sys.stderr
        )
        time.sleep(backoff_seconds)
        backoff_seconds *= 1.5

    raise RuntimeError(
        f"Preflight VRAM check failed: Free VRAM ({check_free_vram_mb()} MB) is below {min_free_mb} MB after {max_attempts} attempts."
    )


def generate_mock_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """Generates a styled placeholder image when ComfyUI server is offline."""
    img = Image.new("RGB", (width, height), color=(20, 24, 35))
    draw = ImageDraw.Draw(img)

    # Draw gradient or pattern
    for y in range(height):
        ratio = y / height
        r = int(25 + ratio * 35)
        g = int(30 + ratio * 45)
        b = int(55 + ratio * 75)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Inner decorative border
    draw.rectangle(
        [(24, 24), (width - 24, height - 24)],
        outline=(100, 140, 220),
        width=2
    )

    # Center card
    card_margin = 48
    draw.rounded_rectangle(
        [(card_margin, card_margin), (width - card_margin, height - card_margin)],
        radius=12,
        fill=(15, 20, 30, 200),
        outline=(60, 80, 120),
        width=1
    )

    # Text overlay
    header = "[ MOCK ASSET ]"
    lines = [
        header,
        "ComfyUI Server Offline (Fallback)",
        "",
        f"Prompt: {prompt[:80]}...",
        f"Resolution: {width}x{height}",
        "Hardware: RTX 3050 (4GB)"
    ]

    y_text = card_margin + 36
    for line in lines:
        draw.text((card_margin + 24, y_text), line, fill=(230, 235, 245))
        y_text += 28

    return img


def call_comfyui_api(
    server_url: str,
    prompt_text: str,
    width: int = 512,
    height: int = 512,
    timeout: float = 60.0,
    negative_prompt: str = "text, watermark, low quality, distorted, cartoon, blurry, flat"
) -> Optional[Image.Image]:
    """Sends a generation request to the ComfyUI API and retrieves the result."""
    prompt_endpoint = f"{server_url.rstrip('/')}/prompt"
    history_endpoint = f"{server_url.rstrip('/')}/history"
    view_endpoint = f"{server_url.rstrip('/')}/view"
    free_endpoint = f"{server_url.rstrip('/')}/free"

    # Standard SD 1.5 / SDXL workflow definition
    ckpt_name = "DreamShaper_8_pruned.safetensors"
    try:
        obj_resp = requests.get(f"{server_url.rstrip('/')}/object_info/CheckpointLoaderSimple", timeout=5.0)
        if obj_resp.status_code == 200:
            available_ckpts = obj_resp.json().get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            if available_ckpts:
                if "DreamShaper_8_pruned.safetensors" in available_ckpts:
                    ckpt_name = "DreamShaper_8_pruned.safetensors"
                elif "v1-5-pruned-emaonly.safetensors" in available_ckpts:
                    ckpt_name = "v1-5-pruned-emaonly.safetensors"
                else:
                    ckpt_name = available_ckpts[0]
    except Exception:
        pass

    workflow = {
        "3": {
            "inputs": {
                "seed": int(time.time()),
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": ckpt_name
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": prompt_text,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "faceless_automation",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

    # 1. Queue generation
    resp = requests.post(prompt_endpoint, json={"prompt": workflow}, timeout=10.0)
    resp.raise_for_status()
    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        raise ValueError(f"No prompt_id returned by ComfyUI: {resp.text}")

    print(f"[COMFYUI_WORKER] Prompt queued successfully (ID: {prompt_id}). Polling for result...")

    # 2. Poll for completion with timeout (Rule 8: bounded retries)
    start_poll = time.time()
    image_info = None
    while time.time() - start_poll < timeout:
        h_resp = requests.get(f"{history_endpoint}/{prompt_id}", timeout=5.0)
        if h_resp.status_code == 200:
            history = h_resp.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    if "images" in node_output and len(node_output["images"]) > 0:
                        image_info = node_output["images"][0]
                        break
                if image_info:
                    break
        time.sleep(1.0)

    if not image_info:
        raise TimeoutError(f"ComfyUI generation timed out after {timeout:.1f}s for prompt {prompt_id}")

    # 3. Download the generated image
    params = urllib.parse.urlencode({
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output")
    })
    img_resp = requests.get(f"{view_endpoint}?{params}", timeout=15.0)
    img_resp.raise_for_status()

    import io
    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


def apply_ken_burns_effect(
    image: Image.Image,
    duration: float = 4.0,
    fps: int = 30,
    zoom_factor: float = 1.15,
    out_size: Tuple[int, int] = (1080, 1920)
) -> VideoClip:
    """Applies a smooth Ken Burns pan/zoom animation to a PIL image using MoviePy.

    Rule 9 Compliance:
    - Normalizes to canonical 1080x1920 (9:16) resolution with proper center-scale+crop (never stretch).
    - Uses Lanczos interpolation for initial high-resolution canvas scaling.
    - Uses continuous subpixel floating-point coordinates and Bicubic resampling
      to avoid visible stepping or jitter across frames.
    """
    orig_w, orig_h = image.size
    target_w, target_h = out_size

    # Scale original image so that even at maximum zoom-in/out and pan offsets,
    # the canvas completely covers the 9:16 target viewport without black edges.
    min_scale = max(target_w / orig_w, target_h / orig_h) * zoom_factor
    base_w = int(math.ceil(orig_w * min_scale))
    base_h = int(math.ceil(orig_h * min_scale))
    base_image = image.resize((base_w, base_h), Image.Resampling.LANCZOS)

    # Frame generator function
    def make_frame(t: float) -> np.ndarray:
        progress = min(max(t / max(duration, 0.001), 0.0), 1.0)
        # Smooth ease-in-out curve
        ease = 0.5 - 0.5 * math.cos(progress * math.pi)

        # Smooth zoom from 1.0 to zoom_factor
        current_zoom = 1.0 + (zoom_factor - 1.0) * ease

        # Maintain exact target aspect ratio (9:16) at all zoom levels
        crop_w = (target_w * zoom_factor) / current_zoom
        crop_h = (target_h * zoom_factor) / current_zoom

        # Smooth pan from top-left bias to bottom-right bias
        max_pan_x = max(0.0, float(base_w - crop_w))
        max_pan_y = max(0.0, float(base_h - crop_h))

        left = (max_pan_x * 0.3) + (max_pan_x * 0.4) * ease
        top = (max_pan_y * 0.3) + (max_pan_y * 0.4) * ease
        right = left + crop_w
        bottom = top + crop_h

        cropped = base_image.crop((left, top, right, bottom))
        resized = cropped.resize((target_w, target_h), Image.Resampling.BICUBIC)
        return np.array(resized)

    return VideoClip(make_frame, duration=duration)


def main():
    parser = argparse.ArgumentParser(description="Standalone ComfyUI image generator with Ken Burns pan/zoom")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, required=True, help="Path for rendered output video clip (.mp4)")
    parser.add_argument("--image-output", type=str, default=None, help="Optional path to save still image")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--width", type=int, default=512, help="Image generation width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image generation height (default: 512)")
    parser.add_argument("--out-width", type=int, default=1080, help="Output video width (default: 1080)")
    parser.add_argument("--out-height", type=int, default=1920, help="Output video height (default: 1920)")
    parser.add_argument("--duration", type=float, default=4.0, help="Ken Burns clip duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Video clip frame rate")
    parser.add_argument("--zoom-factor", type=float, default=1.15, help="Zoom scale factor (e.g. 1.15)")
    parser.add_argument("--negative-prompt", type=str, default="text, watermark, low quality, distorted, cartoon, blurry, flat", help="Curated negative prompt")
    parser.add_argument("--min-free-vram", type=int, default=1024, help="Preflight free VRAM safety threshold in MB")
    parser.add_argument("--mock-on-error", action="store_true", default=False, help="Fallback to mock image if server offline (testing/dev only)")
    parser.add_argument("--disable-mock", action="store_true", default=False, help="Strict mode: fail immediately if ComfyUI generation fails")
    args = parser.parse_args()
    if args.disable_mock:
        args.mock_on_error = False

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Preflight VRAM safety check before acquiring resources or starting generation
    preflight_vram_check(min_free_mb=args.min_free_vram)

    pil_image = None
    start_time = time.time()

    print(f"[{ts()}] [COMFYUI_WORKER] Connecting to ComfyUI at {args.server}...", flush=True)
    try:
        pil_image = call_comfyui_api(
            server_url=args.server,
            prompt_text=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height
        )
        print(f"[{ts()}] [COMFYUI_WORKER] Successfully generated image from ComfyUI.", flush=True)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, TimeoutError) as e:
        if args.mock_on_error:
            print(
                f"[{ts()}] [COMFYUI_WORKER] WARNING: ComfyUI server unavailable ({e.__class__.__name__}: {e}). "
                f"Engaging Rule 2 fallback: generating styled mock asset.",
                file=sys.stderr, flush=True
            )
            pil_image = generate_mock_image(args.prompt, args.width, args.height)
        else:
            print(f"[{ts()}] [COMFYUI_WORKER] ERROR: Failed to generate image: {e}", file=sys.stderr, flush=True)
            sys.exit(1)
    except Exception as e:
        if args.mock_on_error:
            print(f"[{ts()}] [COMFYUI_WORKER] WARNING: Unexpected error ({e}). Using mock asset fallback.", file=sys.stderr, flush=True)
            pil_image = generate_mock_image(args.prompt, args.width, args.height)
        else:
            raise

    # 2. Save still image to disk
    still_img_path = Path(args.image_output).resolve() if args.image_output else output_path.with_suffix(".png")
    still_img_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(still_img_path)
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: STILL_IMAGE_SAVED] Saved still image to {still_img_path}", flush=True)

    # 3. Purge ComfyUI VRAM via /free IMMEDIATELY after still image is saved, before Ken Burns / NVENC
    free_endpoint = f"{args.server.rstrip('/')}/free"
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: FREE_CALLED] Requesting ComfyUI /free (unload_models=True, free_memory=True)...", flush=True)
    try:
        free_resp = requests.post(
            free_endpoint,
            json={"unload_models": True, "free_memory": True},
            timeout=10.0
        )
        free_resp.raise_for_status()
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: FREE_RESPONSE] ComfyUI /free succeeded (status {free_resp.status_code})", flush=True)
    except Exception as free_err:
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: FREE_RESPONSE] Warning: /free request failed: {free_err}", file=sys.stderr, flush=True)

    # 4. Apply Ken Burns animation
    print(f"[{ts()}] [COMFYUI_WORKER] Applying Ken Burns pan/zoom (duration: {args.duration}s, zoom: {args.zoom_factor}x, out: {args.out_width}x{args.out_height})...", flush=True)
    clip = apply_ken_burns_effect(
        image=pil_image,
        duration=args.duration,
        fps=args.fps,
        zoom_factor=args.zoom_factor,
        out_size=(args.out_width, args.out_height)
    )

    # 5. Render clip using NVENC (Rule 1 compliance)
    codec = "h264_nvenc"
    instagram_ffmpeg_params = [
        "-g", "48",
        "-keyint_min", "48",
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-movflags", "+faststart"
    ]
    try:
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_START] Rendering Ken Burns video with {codec} to {output_path}...", flush=True)
        clip.write_videofile(
            str(output_path),
            fps=args.fps,
            codec=codec,
            preset="fast",
            logger=None,
            ffmpeg_params=instagram_ffmpeg_params
        )
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_COMPLETE] NVENC encode complete.", flush=True)
    except Exception as render_err:
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_FAILED] NVENC render fallback to libx264 due to: {render_err}", file=sys.stderr, flush=True)
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_START] Rendering Ken Burns video with libx264 to {output_path}...", flush=True)
        clip.write_videofile(
            str(output_path),
            fps=args.fps,
            codec="libx264",
            preset="fast",
            logger=None,
            ffmpeg_params=instagram_ffmpeg_params
        )
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_COMPLETE] libx264 encode complete.", flush=True)

    elapsed = time.time() - start_time
    print(f"[{ts()}] [COMFYUI_WORKER] Ken Burns clip created successfully at {output_path} ({elapsed:.2f}s elapsed)", flush=True)

    # Explicit subprocess exit guarantees OS-level cleanup
    sys.exit(0)


if __name__ == "__main__":
    main()
