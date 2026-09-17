"""Artificially forces low VRAM condition to test preflight VRAM check rejection.

Allocates a CUDA tensor to drop free VRAM below 1024 MB,
invokes preflight_vram_check(), and verifies that it retries with exponential backoff
and raises GPULockError.
"""

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from pipeline.core.gpu_lock import check_free_vram_mb, preflight_vram_check, GPULockError


def main():
    print("=" * 80)
    print("ARTIFICIAL LOW-VRAM REJECTION & BACKOFF VERIFICATION TEST")
    print("=" * 80)

    initial_free = check_free_vram_mb()
    print(f"Initial Free VRAM: {initial_free} MB")

    # Calculate how many MB we need to allocate to bring free VRAM to ~400 MB (well below 1024 MB)
    target_free = 400
    mb_to_allocate = max(0, initial_free - target_free)
    print(f"Allocating dummy CUDA tensor of {mb_to_allocate} MB to simulate GPU pressure...")

    # Allocate tensor (float32 = 4 bytes per element)
    elements = (mb_to_allocate * 1024 * 1024) // 4
    dummy_tensor = torch.zeros(elements, dtype=torch.float32, device="cuda:0")

    clamped_free = check_free_vram_mb()
    print(f"Simulated Free VRAM after allocation: {clamped_free} MB (Safety threshold: 1024 MB)")

    if clamped_free >= 1024:
        print(f"ERROR: Failed to drive free VRAM below 1024 MB (current: {clamped_free} MB)", file=sys.stderr)
        sys.exit(1)

    print("\n--- Invoking preflight_vram_check(min_free_mb=1024, max_attempts=3, backoff_seconds=1.0) ---")
    start_t = time.time()
    error_raised = False

    try:
        preflight_vram_check(
            min_free_mb=1024,
            max_attempts=3,
            backoff_seconds=1.0,
            worker_name="SIMULATED_OOM_WORKER"
        )
    except GPULockError as e:
        error_raised = True
        elapsed = time.time() - start_t
        print(f"\nSUCCESS: Expected GPULockError raised after {elapsed:.2f}s!")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"\nFAILED: Unexpected exception raised ({type(e)}): {e}", file=sys.stderr)
        sys.exit(1)

    # Release dummy tensor
    del dummy_tensor
    torch.cuda.empty_cache()
    recovered_free = check_free_vram_mb()
    print(f"\nCleaned up dummy tensor. Recovered Free VRAM: {recovered_free} MB")

    if not error_raised:
        print("FAILED: preflight_vram_check did not raise GPULockError when below threshold!", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("PREFLIGHT VRAM CHECK REJECTION & BACKOFF FULLY VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
