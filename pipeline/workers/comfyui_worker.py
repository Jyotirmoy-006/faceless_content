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
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

from moviepy import VideoClip


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
    timeout: float = 60.0
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
                "text": "ugly, blurry, low quality, distorted, watermark",
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

    # 4. Client-side memory purge (/free) to prevent VRAM accumulation
    try:
        requests.post(free_endpoint, json={"unload_models": True, "free_memory": True}, timeout=5.0)
        print("[COMFYUI_WORKER] Triggered ComfyUI /free endpoint to purge VRAM.")
    except Exception as free_err:
        print(f"[COMFYUI_WORKER] Warning: /free request failed: {free_err}", file=sys.stderr)

    import io
    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


def apply_ken_burns_effect(
    image: Image.Image,
    duration: float = 4.0,
    fps: int = 30,
    zoom_factor: float = 1.15,
    out_size: Tuple[int, int] = (512, 512)
) -> VideoClip:
    """Applies a smooth Ken Burns pan/zoom animation to a PIL image using MoviePy."""
    orig_w, orig_h = image.size
    target_w, target_h = out_size

    # Pre-scale image if smaller than target to ensure high-quality crops
    min_scale = max(target_w / orig_w, target_h / orig_h)
    base_w = int(orig_w * min_scale)
    base_h = int(orig_h * min_scale)
    base_image = image.resize((base_w, base_h), Image.Resampling.LANCZOS)

    # Frame generator function
    def make_frame(t: float) -> np.ndarray:
        progress = min(max(t / max(duration, 0.001), 0.0), 1.0)
        # Smooth ease-in-out curve
        ease = 0.5 - 0.5 * math.cos(progress * math.pi)

        current_zoom = 1.0 + (zoom_factor - 1.0) * ease
        crop_w = base_w / current_zoom
        crop_h = base_h / current_zoom

        # Slight pan from top-left offset to bottom-right offset
        max_pan_x = max(0, base_w - crop_w)
        max_pan_y = max(0, base_h - crop_h)

        left = max_pan_x * 0.3 + (max_pan_x * 0.4) * ease
        top = max_pan_y * 0.3 + (max_pan_y * 0.4) * ease
        right = left + crop_w
        bottom = top + crop_h

        cropped = base_image.crop((left, top, right, bottom))
        resized = cropped.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.array(resized)

    return VideoClip(make_frame, duration=duration)


def main():
    parser = argparse.ArgumentParser(description="Standalone ComfyUI image generator with Ken Burns pan/zoom")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, required=True, help="Path for rendered output video clip (.mp4)")
    parser.add_argument("--image-output", type=str, default=None, help="Optional path to save still image")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--width", type=int, default=512, help="Image width (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--duration", type=float, default=4.0, help="Ken Burns clip duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Video clip frame rate")
    parser.add_argument("--zoom-factor", type=float, default=1.15, help="Zoom scale factor (e.g. 1.15)")
    parser.add_argument("--mock-on-error", action="store_true", default=True, help="Fallback to mock image if server offline")
    parser.add_argument("--disable-mock", action="store_true", default=False, help="Strict mode: fail immediately if ComfyUI generation fails")
    args = parser.parse_args()
    if args.disable_mock:
        args.mock_on_error = False

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image = None
    start_time = time.time()

    print(f"[COMFYUI_WORKER] Connecting to ComfyUI at {args.server}...")
    try:
        pil_image = call_comfyui_api(
            server_url=args.server,
            prompt_text=args.prompt,
            width=args.width,
            height=args.height
        )
        print("[COMFYUI_WORKER] Successfully generated image from ComfyUI.")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, TimeoutError) as e:
        if args.mock_on_error:
            print(
                f"[COMFYUI_WORKER] WARNING: ComfyUI server unavailable ({e.__class__.__name__}: {e}). "
                f"Engaging Rule 2 fallback: generating styled mock asset.",
                file=sys.stderr
            )
            pil_image = generate_mock_image(args.prompt, args.width, args.height)
        else:
            print(f"[COMFYUI_WORKER] ERROR: Failed to generate image: {e}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        if args.mock_on_error:
            print(f"[COMFYUI_WORKER] WARNING: Unexpected error ({e}). Using mock asset fallback.", file=sys.stderr)
            pil_image = generate_mock_image(args.prompt, args.width, args.height)
        else:
            raise

    # Save intermediate still image if requested
    if args.image_output:
        img_out = Path(args.image_output).resolve()
        img_out.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(img_out)
        print(f"[COMFYUI_WORKER] Saved intermediate still image to {img_out}")

    # Apply Ken Burns animation
    print(f"[COMFYUI_WORKER] Applying Ken Burns pan/zoom (duration: {args.duration}s, zoom: {args.zoom_factor}x)...")
    clip = apply_ken_burns_effect(
        image=pil_image,
        duration=args.duration,
        fps=args.fps,
        zoom_factor=args.zoom_factor,
        out_size=(args.width, args.height)
    )

    # Render clip using NVENC if available or libx264
    codec = "h264_nvenc"
    instagram_ffmpeg_params = [
        "-g", "48",
        "-keyint_min", "48",
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart"
    ]
    try:
        print(f"[COMFYUI_WORKER] Rendering Ken Burns video with {codec} to {output_path}...")
        clip.write_videofile(
            str(output_path),
            fps=args.fps,
            codec=codec,
            preset="fast",
            logger=None,
            ffmpeg_params=instagram_ffmpeg_params
        )
    except Exception as render_err:
        print(f"[COMFYUI_WORKER] NVENC render fallback to libx264 due to: {render_err}", file=sys.stderr)
        clip.write_videofile(
            str(output_path),
            fps=args.fps,
            codec="libx264",
            preset="fast",
            logger=None,
            ffmpeg_params=instagram_ffmpeg_params
        )

    elapsed = time.time() - start_time
    print(f"[COMFYUI_WORKER] Ken Burns clip created successfully at {output_path} ({elapsed:.2f}s elapsed)")

    # Explicit subprocess exit guarantees OS-level cleanup
    sys.exit(0)


if __name__ == "__main__":
    main()
