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


import threading

def query_vram_stats() -> tuple:
    """Queries current VRAM used and free in MB via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0
        )
        parts = [int(x.strip()) for x in res.stdout.strip().split("\n")[0].split(",")]
        return parts[0], parts[1]
    except Exception as e:
        print(f"[WARN] Failed to query nvidia-smi: {e}", file=sys.stderr)
        return -1, -1


def query_vram_mb() -> int:
    """Queries current VRAM used in MB via nvidia-smi."""
    used, _ = query_vram_stats()
    return used


# Global shared poller samples: list of (timestamp, time_str, used_mb, free_mb)
_vram_samples = []
_stop_polling = threading.Event()


def vram_poller_thread(interval: float = 0.25):
    """Continuously polls nvidia-smi at 0.2-0.3s intervals for the entire stress test."""
    while not _stop_polling.is_set():
        t = time.time()
        time_str = time.strftime('%H:%M:%S', time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
        used, free = query_vram_stats()
        if used >= 0:
            _vram_samples.append((t, time_str, used, free))
        time.sleep(interval)


def run_worker_task(worker_name: str, cmd_args: list, delay_start: float = 0.0) -> dict:
    """Runs a worker under the GPU lock with continuous VRAM tracking and preflight checks."""
    if delay_start > 0:
        time.sleep(delay_start)

    from pipeline.core.gpu_lock import gpu_lock, preflight_vram_check

    pid = os.getpid()
    request_ts = time.time()
    req_time_str = time.strftime('%H:%M:%S', time.localtime(request_ts)) + f".{int((request_ts % 1) * 1000):03d}"
    print(f"\n[{req_time_str}] >>> [{worker_name}] (PID {pid}) QUEUED: Requesting GPU lock...", flush=True)

    # Preflight VRAM check immediately before acquiring GPU lock for ComfyUI
    if "comfyui" in worker_name.lower():
        print(f"[{req_time_str}] [{worker_name}] Running preflight VRAM check before acquiring GPU lock...", flush=True)
        preflight_vram_check(min_free_mb=1024, worker_name=worker_name)

    vram_before = query_vram_mb()

    with gpu_lock(timeout=180.0, worker_name=worker_name):
        acquire_ts = time.time()
        acq_time_str = time.strftime('%H:%M:%S', time.localtime(acquire_ts)) + f".{int((acquire_ts % 1) * 1000):03d}"
        print(f"[{acq_time_str}] *** [{worker_name}] (PID {pid}) RUNNING ON GPU (Lock wait: {acquire_ts - request_ts:.3f}s)...", flush=True)

        proc = subprocess.run(cmd_args, capture_output=True, text=True)
        release_ts = time.time()

    rel_time_str = time.strftime('%H:%M:%S', time.localtime(release_ts)) + f".{int((release_ts % 1) * 1000):03d}"
    print(f"[{rel_time_str}] <<< [{worker_name}] (PID {pid}) FINISHED (GPU active time: {release_ts - acquire_ts:.3f}s)\n", flush=True)

    time.sleep(0.5)  # allow OS driver to settle
    vram_after = query_vram_mb()

    # Extract continuous VRAM samples during this worker's lock window
    worker_samples = [s for s in _vram_samples if acquire_ts <= s[0] <= release_ts]
    if worker_samples:
        peak_used = max(s[2] for s in worker_samples)
        min_free = min(s[3] for s in worker_samples)
    else:
        peak_used = vram_before
        min_free = -1

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
        "vram_during": peak_used,
        "vram_min_free": min_free,
        "vram_after": vram_after,
        "samples_count": len(worker_samples)
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
    comfy_out = OUTPUT_DIR / "stress_comfy.mp4"
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
                "--duration", "3.0",
                "--mock-on-error"
            ], check=True)

    # Prepare worker command lines (3 distinct GPU-bound workers)
    whisper_cmd = [
        PYTHON_EXE, str(WORKERS_DIR / "whisper_worker.py"),
        "--audio", str(audio_file),
        "--output", str(whisper_out),
        "--device", "cuda"
    ]

    comfy_cmd = [
        PYTHON_EXE, str(WORKERS_DIR / "comfyui_worker.py"),
        "--prompt", "cybernetic mechanical crystal core 8k high quality cinematic",
        "--output", str(comfy_out),
        "--disable-mock",
        "--duration", "3.0"
    ]

    render_cmd = [
        PYTHON_EXE, str(WORKERS_DIR / "render_worker.py"),
        "--video", str(comfy_clip),
        "--audio", str(audio_file),
        "--output", str(render_out)
    ]

    print("\n" + "-" * 80)
    print("LAUNCHING 3-WAY CONCURRENT WORKER STRESS TEST")
    print("Worker A: Whisper Speech Recognition (PyTorch CUDA model)")
    print("Worker B: Real ComfyUI SD1.5 Generation (DreamShaper 8 UNet + Ken Burns NVENC)")
    print("Worker C: MoviePy Final Video Render (NVENC h264 encoding)")
    print("All 3 workers are invoked concurrently across separate threads/processes.")
    print("-" * 80 + "\n")

    start_all = time.time()

    # Launch continuous background VRAM poller thread at 0.25s intervals
    _stop_polling.clear()
    poller = threading.Thread(target=vram_poller_thread, args=(0.25,), daemon=True)
    poller.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Launch Worker A at 0.0s, Worker B at 0.05s, Worker C at 0.10s to force intense lock contention
        future_a = executor.submit(run_worker_task, "WORKER_A (Whisper)", whisper_cmd, 0.0)
        future_b = executor.submit(run_worker_task, "WORKER_B (ComfyUI-Real)", comfy_cmd, 0.05)
        future_c = executor.submit(run_worker_task, "WORKER_C (Render)", render_cmd, 0.10)

        result_a = future_a.result()
        result_b = future_b.result()
        result_c = future_c.result()

    _stop_polling.set()
    poller.join(timeout=2.0)

    total_elapsed = time.time() - start_all
    final_vram = query_vram_mb()

    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS & SERIALIZATION PROOF")
    print("=" * 80)

    print(f"Total test wall-clock time: {total_elapsed:.3f}s")
    print(f"Baseline VRAM before test: {baseline_vram} MB")
    print(f"Final VRAM after test:    {final_vram} MB (delta from baseline: {final_vram - baseline_vram} MB)")

    results = [result_a, result_b, result_c]
    # Sort results by acquire timestamp to verify temporal sequence
    sorted_results = sorted(results, key=lambda r: r['acquire_ts'])

    print("\n--- RAW TIMESTAMPS & TIMELINE ---")
    for r in results:
        print(f"{r['worker']}:")
        print(f"  Lock Requested: {r['request_ts']:.6f} ({time.strftime('%H:%M:%S', time.localtime(r['request_ts']))}.{int((r['request_ts'] % 1) * 1000):03d})")
        print(f"  Lock Acquired:  {r['acquire_ts']:.6f} ({time.strftime('%H:%M:%S', time.localtime(r['acquire_ts']))}.{int((r['acquire_ts'] % 1) * 1000):03d})")
        print(f"  Lock Released:  {r['release_ts']:.6f} ({time.strftime('%H:%M:%S', time.localtime(r['release_ts']))}.{int((r['release_ts'] % 1) * 1000):03d})")
        print(f"  Duration Held:  {r['duration']:.3f}s")
        print(f"  Exit Code:      {r['exit_code']}")

    print("\n--- CHRONOLOGICAL EXECUTION ORDER & OVERLAP ANALYSIS ---")
    has_overlap = False
    for i in range(len(sorted_results) - 1):
        curr = sorted_results[i]
        nxt = sorted_results[i + 1]
        pair_overlap = max(0.0, min(curr['release_ts'], nxt['release_ts']) - max(curr['acquire_ts'], nxt['acquire_ts']))
        pair_gap = nxt['acquire_ts'] - curr['release_ts']
        print(f"Interval {i+1} -> {i+2}: [{curr['worker']}] -> [{nxt['worker']}]")
        print(f"  Prior Release:  {curr['release_ts']:.6f}")
        print(f"  Next Acquire:   {nxt['acquire_ts']:.6f}")
        print(f"  Serialized Gap: {pair_gap:.6f}s")
        print(f"  Overlap:        {pair_overlap:.6f}s")
        if pair_overlap > 0.0:
            has_overlap = True

    if has_overlap:
        print("\nFAILURE: GPU overlap detected across concurrent workers!", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nSUCCESS: 100% GPU serialization confirmed across Whisper, Real ComfyUI, and NVENC Render!")

    print("\n--- CONTINUOUS VRAM POLLING & PEAK ANALYSIS ---")
    total_samples = len(_vram_samples)
    global_peak_used = max(s[2] for s in _vram_samples) if _vram_samples else baseline_vram
    global_min_free = min(s[3] for s in _vram_samples) if _vram_samples else -1
    print(f"Total continuous VRAM samples: {total_samples} (polled every 0.25s)")
    print(f"Overall Global Peak VRAM:       {global_peak_used} MB")
    print(f"Overall Minimum Free VRAM:      {global_min_free} MB")

    print("\n--- PER-WORKER VRAM METRICS (LOCK-BOUND WINDOWS) ---")
    for r in results:
        print(f"{r['worker']}:")
        print(f"  VRAM Before Lock:     {r['vram_before']} MB")
        print(f"  Real Peak Under Lock: {r['vram_during']} MB")
        print(f"  Min Free Under Lock:  {r['vram_min_free']} MB")
        print(f"  VRAM After Lock:      {r['vram_after']} MB")
        print(f"  Continuous Samples:   {r['samples_count']}")

    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS (PROJECT_RULES.md & MISSION):")
    print("=" * 80)

    comfy_res = next(r for r in results if "comfyui" in r['worker'].lower())
    print(f"1. Real ComfyUI Worker Peak VRAM: {comfy_res['vram_during']} MB ({comfy_res['vram_during'] / 1024:.2f} GB)")
    if comfy_res['vram_during'] >= 3000:
        print("   -> CONFIRMED: ComfyUI peak VRAM approaches the ~3.7GB standalone figure under continuous polling.")
    else:
        print(f"   -> WARNING: ComfyUI peak VRAM ({comfy_res['vram_during']} MB) did not reach 3GB!", file=sys.stderr)

    if global_min_free > 0:
        print(f"2. Positive VRAM Headroom: CONFIRMED ({global_min_free} MB free at minimum point, no OOM occurred).")
    else:
        print(f"2. Positive VRAM Headroom: FAILED (free VRAM hit {global_min_free} MB)!", file=sys.stderr)
        sys.exit(1)

    for r in results:
        if r['exit_code'] != 0:
            print(f"ERROR: {r['worker']} failed with exit code {r['exit_code']}!", file=sys.stderr)
            sys.exit(1)

    print("3. All 3 workers completed with exit code 0.")
    print("=" * 80)


if __name__ == "__main__":
    main()
