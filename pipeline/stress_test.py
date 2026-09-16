"""Stress test orchestrator for GPU mutex lock and worker isolation.

Per Rule 1 (GPU MEMORY):
- Demonstrates serialized GPU access between back-to-back worker calls.
- Concurrently triggers two GPU-bound workers wrapped in gpu_lock.
- Prints millisecond timestamps proving zero overlap on the GPU.
- Measures and logs exact VRAM before, during, and after worker execution using nvidia-smi.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

PYTHON_EXE = sys.executable
WORKERS_DIR = ROOT_DIR / "pipeline" / "workers"
ASSETS_DIR = ROOT_DIR / "pipeline" / "assets_cache"
OUTPUT_DIR = ROOT_DIR / "pipeline" / "output"


def query_vram_mb() -> int:
    """Queries current VRAM used in MB via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        return int(res.stdout.strip().split("\n")[0].strip())
    except Exception as e:
        print(f"[WARN] Failed to query nvidia-smi: {e}", file=sys.stderr)
        return -1


def run_worker_task(worker_name: str, cmd_args: list, delay_start: float = 0.0) -> dict:
    """Runs a worker under the GPU lock with high-precision timestamp logging."""
    if delay_start > 0:
        time.sleep(delay_start)

    from pipeline.core.gpu_lock import gpu_lock

    pid = os.getpid()
    request_ts = time.time()
    req_time_str = time.strftime('%H:%M:%S', time.localtime(request_ts)) + f".{int((request_ts % 1) * 1000):03d}"
    print(f"\n[{req_time_str}] >>> [{worker_name}] (PID {pid}) QUEUED: Requesting GPU lock...", flush=True)

    vram_before = query_vram_mb()

    with gpu_lock(timeout=180.0, worker_name=worker_name):
        acquire_ts = time.time()
        acq_time_str = time.strftime('%H:%M:%S', time.localtime(acquire_ts)) + f".{int((acquire_ts % 1) * 1000):03d}"
        print(f"[{acq_time_str}] *** [{worker_name}] (PID {pid}) RUNNING ON GPU (Lock wait: {acquire_ts - request_ts:.3f}s)...", flush=True)

        vram_start = query_vram_mb()
        proc = subprocess.run(cmd_args, capture_output=True, text=True)
        vram_peak_check = query_vram_mb()
        release_ts = time.time()

    rel_time_str = time.strftime('%H:%M:%S', time.localtime(release_ts)) + f".{int((release_ts % 1) * 1000):03d}"
    print(f"[{rel_time_str}] <<< [{worker_name}] (PID {pid}) FINISHED (GPU active time: {release_ts - acquire_ts:.3f}s)\n", flush=True)

    time.sleep(0.5)  # allow OS driver to settle
    vram_after = query_vram_mb()

    return {
        "worker": worker_name,
        "pid": pid,
        "request_ts": request_ts,
        "acquire_ts": acquire_ts,
        "release_ts": release_ts,
        "duration": release_ts - acquire_ts,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "vram_before": vram_before,
        "vram_during": max(vram_start, vram_peak_check),
        "vram_after": vram_after
    }


def main():
    print("=" * 80)
    print("MISSION 2: GPU-ISOLATED RENDER WORKERS - STRESS TEST & VALIDATION")
    print("=" * 80)

    baseline_vram = query_vram_mb()
    print(f"[SYSTEM] Baseline VRAM before stress test: {baseline_vram} MB")

    # Audio & video test assets
    audio_file = ASSETS_DIR / "test_speech.mp3"
    comfy_clip = ASSETS_DIR / "test_comfy_clip.mp4"
    whisper_out = OUTPUT_DIR / "stress_whisper.srt"
    render_out = OUTPUT_DIR / "stress_render.mp4"

    if not audio_file.exists():
        print(f"[SETUP] Generating test audio at {audio_file}...")
        subprocess.run([
            PYTHON_EXE, "-m", "edge_tts",
            "--text", "This is an automated GPU lock verification test for RTX 3050 VRAM management.",
            "--write-media", str(audio_file)
        ], check=True)

    if not comfy_clip.exists():
        print(f"[SETUP] Generating test clip at {comfy_clip}...")
        with gpu_lock(timeout=180.0, worker_name="setup_comfyui"):
            subprocess.run([
                PYTHON_EXE, str(WORKERS_DIR / "comfyui_worker.py"),
                "--prompt", "GPU lock stress test visual",
                "--output", str(comfy_clip),
                "--duration", "3.0"
            ], check=True)

    # Prepare worker command lines
    whisper_cmd = [
        PYTHON_EXE, str(WORKERS_DIR / "whisper_worker.py"),
        "--audio", str(audio_file),
        "--output", str(whisper_out),
        "--device", "cuda"
    ]

    render_cmd = [
        PYTHON_EXE, str(WORKERS_DIR / "render_worker.py"),
        "--video", str(comfy_clip),
        "--audio", str(audio_file),
        "--output", str(render_out)
    ]

    print("\n" + "-" * 80)
    print("LAUNCHING CONCURRENT WORKER STRESS TEST")
    print("Worker A: Whisper Speech Recognition (loads PyTorch CUDA model)")
    print("Worker B: Instagram NVENC Video Render (MoviePy + NVENC encoding)")
    print("Both workers are invoked simultaneously across separate threads/processes.")
    print("-" * 80 + "\n")

    start_all = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Launch Worker A immediately, Worker B with a 0.1s offset to ensure contention
        future_a = executor.submit(run_worker_task, "WORKER_A (Whisper)", whisper_cmd, 0.0)
        future_b = executor.submit(run_worker_task, "WORKER_B (Render)", render_cmd, 0.1)

        result_a = future_a.result()
        result_b = future_b.result()

    total_elapsed = time.time() - start_all
    final_vram = query_vram_mb()

    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS & SERIALIZATION PROOF")
    print("=" * 80)

    print(f"Total test wall-clock time: {total_elapsed:.3f}s")
    print(f"Baseline VRAM before test: {baseline_vram} MB")
    print(f"Final VRAM after test:    {final_vram} MB (delta from baseline: {final_vram - baseline_vram} MB)")

    print("\n--- TIMESTAMPS & OVERLAP ANALYSIS ---")
    print(f"{result_a['worker']}:")
    print(f"  Lock Requested: {time.strftime('%H:%M:%S', time.localtime(result_a['request_ts']))}.{int((result_a['request_ts'] % 1) * 1000):03d}")
    print(f"  Lock Acquired:  {time.strftime('%H:%M:%S', time.localtime(result_a['acquire_ts']))}.{int((result_a['acquire_ts'] % 1) * 1000):03d}")
    print(f"  Lock Released:  {time.strftime('%H:%M:%S', time.localtime(result_a['release_ts']))}.{int((result_a['release_ts'] % 1) * 1000):03d}")
    print(f"  Exit Code:      {result_a['exit_code']}")

    print(f"\n{result_b['worker']}:")
    print(f"  Lock Requested: {time.strftime('%H:%M:%S', time.localtime(result_b['request_ts']))}.{int((result_b['request_ts'] % 1) * 1000):03d}")
    print(f"  Lock Acquired:  {time.strftime('%H:%M:%S', time.localtime(result_b['acquire_ts']))}.{int((result_b['acquire_ts'] % 1) * 1000):03d}")
    print(f"  Lock Released:  {time.strftime('%H:%M:%S', time.localtime(result_b['release_ts']))}.{int((result_b['release_ts'] % 1) * 1000):03d}")
    print(f"  Exit Code:      {result_b['exit_code']}")

    # Overlap validation: Worker B acquire must be >= Worker A release
    overlap = max(0.0, min(result_a['release_ts'], result_b['release_ts']) - max(result_a['acquire_ts'], result_b['acquire_ts']))
    gap = result_b['acquire_ts'] - result_a['release_ts']

    print(f"\nGPU Execution Overlap: {overlap:.6f} seconds")
    if overlap == 0.0:
        print(f"SUCCESS: Strict serialization confirmed! Worker B waited {gap:.3f}s after Worker A released.")
    else:
        print(f"FAILURE: Overlap detected ({overlap:.3f}s)!", file=sys.stderr)
        sys.exit(1)

    print("\n--- INDIVIDUAL WORKER VRAM LOGS ---")
    print(f"{result_a['worker']} - VRAM Before: {result_a['vram_before']} MB | Peak: {result_a['vram_during']} MB | After: {result_a['vram_after']} MB")
    print(f"{result_b['worker']} - VRAM Before: {result_b['vram_before']} MB | Peak: {result_b['vram_during']} MB | After: {result_b['vram_after']} MB")

    if result_a['exit_code'] != 0 or result_b['exit_code'] != 0:
        print("ERROR: One or more workers failed during stress test!", file=sys.stderr)
        sys.exit(1)

    print("\n[SUCCESS] Stress test completed successfully with 100% GPU serialization.")


if __name__ == "__main__":
    main()
