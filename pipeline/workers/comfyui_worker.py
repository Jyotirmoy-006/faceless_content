"""ComfyUI SD1.5 image generation & True 2.5D Parallax + Particle VFX subprocess worker.

Per Rule 1 (GPU MEMORY):
- Subprocess isolation: Releases all CUDA memory and MoviePy/FFmpeg buffers upon exit.
- Unloads ComfyUI weights via `/free` immediately after image generation.
- Executes depth estimation under GPU lock before parallax rendering.
- Old flat Ken Burns 2D scaling is completely superseded by depth-aware parallax.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.parallax_engine import render_parallax_clip
from pipeline.core.particle_vfx import select_particle_style


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
    """Verifies sufficient free VRAM exists before starting SD1.5 generation."""
    for attempt in range(1, max_attempts + 1):
        free_mb = check_free_vram_mb()
        if free_mb >= min_free_mb:
            print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: PREFLIGHT_VRAM_OK] Free VRAM: {free_mb} MB (threshold: {min_free_mb} MB)")
            return free_mb
        print(
            f"[{ts()}] [COMFYUI_WORKER] [STAGE: PREFLIGHT_VRAM_LOW] Free VRAM ({free_mb} MB) below {min_free_mb} MB. "
            f"Attempt {attempt}/{max_attempts}. Backing off for {backoff_seconds:.1f}s...",
            file=sys.stderr
        )
        time.sleep(backoff_seconds)
        backoff_seconds *= 1.5

    raise RuntimeError(
        f"Preflight VRAM check failed: Free VRAM ({check_free_vram_mb()} MB) is below {min_free_mb} MB after {max_attempts} attempts."
    )


def generate_mock_image(prompt: str, width: int = 512, height: int = 512) -> Image.Image:
    """Generates an aesthetic abstract background with zero text or placeholder cards."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2)
    glow = np.clip(1.0 - r * 0.75, 0.0, 1.0)

    # Clean atmospheric midnight palette: dark cinematic blue with soft neon cyan/purple core
    r_channel = (15 + glow * 45).astype(np.uint8)
    g_channel = (20 + glow * 75).astype(np.uint8)
    b_channel = (45 + glow * 135).astype(np.uint8)

    rgb_arr = np.dstack((r_channel, g_channel, b_channel))
    return Image.fromarray(rgb_arr)


def call_comfyui_api(
    server_url: str,
    prompt_text: str,
    width: int = 512,
    height: int = 896,
    timeout: float = 60.0,
    negative_prompt: str = "ugly, blurry, low quality, distorted, watermark, extra limbs, deformed hands, bad anatomy, disfigured, poorly drawn face, mutation, duplicate, text, signature, oversaturated, jpeg artifacts"
) -> Image.Image:
    """Sends a generation request to the ComfyUI API and retrieves the generated image."""
    prompt_endpoint = f"{server_url.rstrip('/')}/prompt"
    history_endpoint = f"{server_url.rstrip('/')}/history"
    view_endpoint = f"{server_url.rstrip('/')}/view"

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
                "seed": int(time.time() * 1000) % 100000000,
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
        "4": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
        "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"text": prompt_text, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"text": negative_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "faceless_automation", "images": ["8", 0]}, "class_type": "SaveImage"}
    }

    resp = requests.post(prompt_endpoint, json={"prompt": workflow}, timeout=10.0)
    resp.raise_for_status()
    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        raise ValueError(f"No prompt_id returned by ComfyUI: {resp.text}")

    print(f"[{ts()}] [COMFYUI_WORKER] Prompt queued (ID: {prompt_id}). Polling for result...")
    start_poll = time.time()
    image_info = None
    while time.time() - start_poll < timeout:
        h_resp = requests.get(f"{history_endpoint}/{prompt_id}", timeout=5.0)
        if h_resp.status_code == 200:
            history = h_resp.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for _, node_output in outputs.items():
                    if "images" in node_output and len(node_output["images"]) > 0:
                        image_info = node_output["images"][0]
                        break
                if image_info:
                    break
        time.sleep(0.5)

    if not image_info:
        raise TimeoutError(f"ComfyUI generation timed out after {timeout:.1f}s for prompt {prompt_id}")

    params = urllib.parse.urlencode({
        "filename": image_info["filename"],
        "subfolder": image_info.get("subfolder", ""),
        "type": image_info.get("type", "output")
    })
    img_resp = requests.get(f"{view_endpoint}?{params}", timeout=15.0)
    img_resp.raise_for_status()
    return Image.open(io.BytesIO(img_resp.content)).convert("RGB")


def purge_comfyui_vram(server_url: str) -> None:
    """Instructs ComfyUI to purge model weights and free GPU memory."""
    free_endpoint = f"{server_url.rstrip('/')}/free"
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: FREE_CALLED] Requesting ComfyUI /free...", flush=True)
    try:
        free_resp = requests.post(
            free_endpoint,
            json={"unload_models": True, "free_memory": True},
            timeout=10.0
        )
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: FREE_RESPONSE] ComfyUI /free status {free_resp.status_code}", flush=True)
    except Exception as free_err:
        print(f"[{ts()}] [COMFYUI_WORKER] Warning: /free failed ({free_err})", file=sys.stderr, flush=True)


def generate_depth_map(image_path: Path, depth_path: Path) -> Path:
    """Executes depth_worker subprocess to produce monocular depth map."""
    depth_worker_script = ROOT_DIR / "pipeline" / "workers" / "depth_worker.py"
    cmd = [
        sys.executable,
        str(depth_worker_script),
        "--input", str(image_path),
        "--output", str(depth_path)
    ]
    print(f"[{ts()}] [COMFYUI_WORKER] Launching depth estimation subprocess...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[{ts()}] [COMFYUI_WORKER] Depth estimation failed ({proc.stderr}). Generating fallback synthetic depth map.", file=sys.stderr)
        # Synthetic center-weighted depth map fallback
        img = Image.open(image_path)
        w, h = img.size
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - w / 2)**2 + (Y - h / 2)**2)
        max_dist = np.sqrt((w / 2)**2 + (h / 2)**2)
        synth_depth = ((1.0 - dist_from_center / max_dist) * 255.0).astype(np.uint8)
        Image.fromarray(synth_depth).save(depth_path)
    return depth_path


def main():
    parser = argparse.ArgumentParser(description="ComfyUI image generator with True 2.5D Parallax & Particle VFX")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--output", type=str, required=True, help="Path for rendered output video clip (.mp4)")
    parser.add_argument("--image-output", type=str, default=None, help="Optional path to save still image")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8188", help="ComfyUI server URL")
    parser.add_argument("--width", type=int, default=512, help="Image generation width")
    parser.add_argument("--height", type=int, default=896, help="Image generation height")
    parser.add_argument("--out-width", type=int, default=1080, help="Output video width")
    parser.add_argument("--out-height", type=int, default=1920, help="Output video height")
    parser.add_argument("--duration", type=float, default=3.0, help="Clip duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate (Rule 9 CFR)")
    parser.add_argument("--motion-type", type=str, default="zoom_in", choices=["zoom_in", "zoom_out", "pan_left", "pan_right"])
    parser.add_argument("--particle-style", type=str, default="auto", help="Particle VFX style ('auto', 'cyber_glints', 'fire_embers', 'light_streaks', 'subtle_dust', 'none')")
    parser.add_argument("--negative-prompt", type=str, default="ugly, blurry, low quality, distorted, watermark, extra limbs, deformed hands, bad anatomy, text", help="Negative prompt")
    parser.add_argument("--min-free-vram", type=int, default=1024, help="Preflight VRAM threshold in MB")
    parser.add_argument("--mock-on-error", action="store_true", default=False, help="Fallback to mock image if server offline")
    parser.add_argument("--disable-mock", action="store_true", default=False, help="Strict mode: fail if generation fails")
    args = parser.parse_args()
    if args.disable_mock:
        args.mock_on_error = False

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    still_img_path = Path(args.image_output).resolve() if args.image_output else output_path.with_suffix(".png")
    depth_img_path = output_path.with_name(f"{output_path.stem}_depth.png")

    # 1. Preflight VRAM safety check
    preflight_vram_check(min_free_mb=args.min_free_vram)
    start_time = time.time()

    # 2. SD1.5 Generation via ComfyUI API
    pil_image = None
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: SD15_START] Dispatching SD1.5 generation request to ComfyUI ({args.width}x{args.height})...", flush=True)
    try:
        pil_image = call_comfyui_api(
            server_url=args.server,
            prompt_text=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height
        )
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: SD15_COMPLETE] Successfully generated image from ComfyUI.", flush=True)
    except Exception as e:
        if args.mock_on_error:
            print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: SD15_FALLBACK] Server unavailable ({e}). Using mock asset.", file=sys.stderr, flush=True)
            pil_image = generate_mock_image(args.prompt, args.width, args.height)
        else:
            print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: SD15_FAILED] ERROR: Failed generation: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    still_img_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(still_img_path)
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: STILL_IMAGE_SAVED] Saved still to {still_img_path}", flush=True)

    # 3. Purge ComfyUI VRAM via /free before starting depth & parallax
    purge_comfyui_vram(args.server)

    # 4. Generate Depth Map via isolated depth_worker
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: DEPTH_START] Generating depth map for {still_img_path.name}...", flush=True)
    generate_depth_map(still_img_path, depth_img_path)
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: DEPTH_COMPLETE] Depth map ready at {depth_img_path.name}", flush=True)

    # 5. Determine particle style
    style = select_particle_style(args.prompt) if args.particle_style == "auto" else args.particle_style

    # 6. Apply True 2.5D Parallax Motion + Particle Compositing
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: PARALLAX_START] Rendering True Parallax ({args.motion_type}, {args.duration}s, style={style}, out={args.out_width}x{args.out_height})...", flush=True)
    clip = render_parallax_clip(
        image=still_img_path,
        depth_map=depth_img_path,
        duration=args.duration,
        fps=args.fps,
        out_size=(args.out_width, args.out_height),
        motion_type=args.motion_type,
        particle_style=style
    )
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: PARALLAX_COMPLETE] Parallax frame graph constructed.", flush=True)

    # 7. Render clip with Instagram-compliant NVENC encoding (Rule 9)
    codec = "h264_nvenc"
    instagram_ffmpeg_params = [
        "-vf", "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709",
        "-g", "48", "-keyint_min", "48", "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-color_range", "tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart"
    ]
    try:
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_START] Encoding with {codec}...", flush=True)
        clip.write_videofile(
            str(output_path),
            fps=args.fps,
            codec=codec,
            preset="fast",
            logger=None,
            ffmpeg_params=instagram_ffmpeg_params
        )
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_COMPLETE] NVENC encode successful.", flush=True)
    except Exception as render_err:
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_FAILED] NVENC failed ({render_err}). Retrying with libx264...", file=sys.stderr, flush=True)
        clip.write_videofile(
            str(output_path),
            fps=args.fps,
            codec="libx264",
            preset="fast",
            logger=None,
            ffmpeg_params=instagram_ffmpeg_params
        )
        print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: NVENC_COMPLETE] libx264 encode successful.", flush=True)

    elapsed = time.time() - start_time
    print(f"[{ts()}] [COMFYUI_WORKER] [STAGE: SHOT_COMPLETE] Parallax + Particle shot created at {output_path} ({elapsed:.2f}s total elapsed)", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
