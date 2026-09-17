"""Automated setup script for local ComfyUI instance with PyTorch 2.6 compatibility.

Ensures that any fresh clone of this repository can set up a functioning ComfyUI
server without tribal knowledge or manual source patching.

Usage:
    python comfyui_setup/setup_comfyui.py [--target-dir ComfyUI_server]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PATCH_FILE = ROOT_DIR / "comfyui_setup" / "comfyui_pytorch26_compat.patch"
DEFAULT_CHECKPOINT = ROOT_DIR / "pipeline" / "models" / "DreamShaper_8_pruned.safetensors"


def run(cmd, cwd=None, check=True):
    print(f"[RUN] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed with code {res.returncode}:\n{res.stderr}", file=sys.stderr)
        sys.exit(res.returncode)
    return res


def setup(target_dir: Path):
    print("=" * 80)
    print("SETTING UP LOCAL COMFYUI SERVER ENVIRONMENT")
    print("=" * 80)

    # 1. Clone ComfyUI if not already present
    if not target_dir.exists():
        print(f"[1/4] Cloning ComfyUI into {target_dir}...")
        run(["git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", str(target_dir)])
    else:
        print(f"[1/4] ComfyUI directory already exists at {target_dir}.")

    # 2. Install ComfyUI dependencies (excluding incompatible comfy-kitchen)
    print("[2/5] Installing ComfyUI dependencies (excluding comfy-kitchen)...")
    req_file = target_dir / "requirements.txt"
    if req_file.exists():
        raw_lines = req_file.read_text(encoding="utf-8").splitlines()
        filtered = [l for l in raw_lines if not l.strip().startswith("comfy-kitchen")]
        temp_req = target_dir / "requirements_filtered.txt"
        temp_req.write_text("\n".join(filtered), encoding="utf-8")
        try:
            run([sys.executable, "-m", "pip", "install", "-r", str(temp_req)])
        finally:
            temp_req.unlink(missing_ok=True)

    # 3. Verify comfy-kitchen is NOT installed transitively
    print("[3/5] Verifying whether comfy-kitchen was transitively installed...")
    check_ck = run([sys.executable, "-c", "import comfy_kitchen"], check=False)
    if check_ck.returncode == 0:
        print("  -> WARNING: comfy-kitchen was transitively installed! Removing it...")
        run([sys.executable, "-m", "pip", "uninstall", "-y", "comfy-kitchen"], check=False)
    else:
        print("  -> CONFIRMED: comfy-kitchen is absent (NOT installed transitively by any dependency).")

    # 4. Apply compatibility patch
    print(f"[4/5] Applying PyTorch 2.6 compatibility patch ({PATCH_FILE.name})...")
    # Check if patch is already applied
    check_patch = run(["git", "apply", "--check", str(PATCH_FILE)], cwd=target_dir, check=False)
    if check_patch.returncode == 0:
        run(["git", "apply", str(PATCH_FILE)], cwd=target_dir)
        print("  -> Patch applied successfully.")
    else:
        # Check if already applied (reverse applies cleanly)
        check_rev = run(["git", "apply", "--reverse", "--check", str(PATCH_FILE)], cwd=target_dir, check=False)
        if check_rev.returncode == 0:
            print("  -> Patch already applied.")
        else:
            print("  -> Notice: git apply could not check patch. Attempting manual verification...")
            att_file = target_dir / "comfy" / "ldm" / "modules" / "attention.py"
            if att_file.exists() and "from comfy import model_management" in att_file.read_text(encoding="utf-8"):
                print("  -> Verified: patch is already active in attention.py.")
            else:
                print("  -> Warning: Failed to apply patch automatically. Check comfyui_setup/COMFYUI_SETUP.md", file=sys.stderr)

    # 5. Link or set up SD1.5 model checkpoint
    print("[5/5] Setting up SD1.5 model checkpoint...")
    ckpt_dir = target_dir / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = ckpt_dir / "DreamShaper_8_pruned.safetensors"

    if not target_ckpt.exists():
        if DEFAULT_CHECKPOINT.exists():
            print(f"  -> Creating hardlink from {DEFAULT_CHECKPOINT} to {target_ckpt}...")
            try:
                os.link(str(DEFAULT_CHECKPOINT), str(target_ckpt))
            except Exception:
                shutil.copy2(str(DEFAULT_CHECKPOINT), str(target_ckpt))
        else:
            print(f"  -> NOTICE: Model not found at {DEFAULT_CHECKPOINT}.")
            print(f"     Place 'DreamShaper_8_pruned.safetensors' into {ckpt_dir}/")
    else:
        print(f"  -> Verified: Model exists at {target_ckpt} ({target_ckpt.stat().st_size / (1024**3):.2f} GB).")

    print("\n" + "=" * 80)
    print("COMFYUI SETUP COMPLETE")
    print(f"Start server using:")
    print(f"  {sys.executable} {target_dir / 'main.py'} --listen 127.0.0.1 --port 8188 --lowvram")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated ComfyUI setup for faceless_automation")
    parser.add_argument("--target-dir", type=str, default=str(ROOT_DIR / "ComfyUI_server"), help="Path to install ComfyUI")
    args = parser.parse_args()
    setup(Path(args.target_dir).resolve())
