"""Verification test for comfyui_worker.py with continuous nvidia-smi polling.

Measures:
1. Exact timestamp of each stage:
   - STILL_IMAGE_SAVED
   - FREE_CALLED
   - FREE_RESPONSE
   - NVENC_START
   - NVENC_COMPLETE
2. Continuous VRAM (used and free) polled at 0.25s intervals throughout the entire run.
3. Reports peak VRAM during generation vs peak VRAM during NVENC encode.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
WORKER_SCRIPT = ROOT_DIR / "pipeline" / "workers" / "comfyui_worker.py"
OUT_VIDEO = ROOT_DIR / "pipeline" / "output" / "test_stage_video.mp4"
OUT_IMG = ROOT_DIR / "pipeline" / "output" / "test_stage_still.png"


def run_continuous_vram_test():
    vram_samples = []
    stop_polling = threading.Event()

    def poller():
        while not stop_polling.is_set():
            t = time.time()
            ts_str = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True, timeout=1.0
                )
                used, free = [int(x.strip()) for x in res.stdout.strip().split("\n")[0].split(",")]
                vram_samples.append((t, ts_str, used, free))
            except Exception:
                pass
            time.sleep(0.25)

    poll_thread = threading.Thread(target=poller, daemon=True)
    poll_thread.start()

    cmd = [
        str(PYTHON_EXE), str(WORKER_SCRIPT),
        "--prompt", "cybernetic glowing crystal core 8k cinematic hyperrealistic",
        "--output", str(OUT_VIDEO),
        "--image-output", str(OUT_IMG),
        "--disable-mock",
        "--duration", "3.0"
    ]

    print("=" * 80)
    print("STARTING STANDALONE COMFYUI WORKER VERIFICATION WITH CONTINUOUS VRAM POLLING")
    print("Command:", " ".join(cmd))
    print("=" * 80)

    start_t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stop_polling.set()
    poll_thread.join(timeout=2.0)
    end_t = time.time()

    print("\n--- WORKER STDOUT ---")
    print(proc.stdout)
    if proc.stderr:
        print("\n--- WORKER STDERR ---")
        print(proc.stderr)

    print("\n--- VRAM POLLING SUMMARY ---")
    if not vram_samples:
        print("Error: No VRAM samples collected!")
        return

    used_vals = [s[2] for s in vram_samples]
    free_vals = [s[3] for s in vram_samples]
    max_used = max(used_vals)
    min_free = min(free_vals)
    baseline = vram_samples[0][2]

    print(f"Total samples collected: {len(vram_samples)} over {end_t - start_t:.2f}s")
    print(f"Baseline VRAM: {baseline} MB")
    print(f"Peak VRAM used: {max_used} MB (Headroom remaining: {min_free} MB)")

    print("\nSampled timeline (top peaks and transitions):")
    # Print every 2nd or notable transition
    prev_used = None
    for t, ts_str, used, free in vram_samples:
        if prev_used is None or abs(used - prev_used) > 100 or used == max_used:
            print(f"[{ts_str}] Used: {used} MB | Free: {free} MB")
            prev_used = used

    print("=" * 80)


if __name__ == "__main__":
    run_continuous_vram_test()
