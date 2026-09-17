"""Cross-process GPU mutex lock for RTX 3050 (4GB VRAM).

Per Rule 1 (GPU MEMORY):
- No two GPU-bound subprocesses may run concurrently.
- Enforced with an explicit OS-level file lock (`filelock.FileLock`).
- Automatically handles process crashes (kernel releases lock upon process death).
- Enforces strict acquisition timeout to prevent indefinite hangs (Rule 8).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from filelock import FileLock, Timeout

# Default lock file in project core directory (gitignored via *.gpu.lock)
DEFAULT_LOCK_DIR = Path(__file__).resolve().parent
DEFAULT_LOCK_PATH = DEFAULT_LOCK_DIR / ".gpu.lock"
DEFAULT_META_PATH = DEFAULT_LOCK_DIR / ".gpu.lock.meta"


class GPULockError(Exception):
    """Raised when GPU lock cannot be acquired or an error occurs."""
    pass


def check_free_vram_mb() -> int:
    """Queries current free VRAM in MiB using nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5.0
        )
        return int(res.stdout.strip().split("\n")[0].strip())
    except Exception as e:
        print(f"[GPU_LOCK] Warning: failed to query nvidia-smi ({e}), assuming 4096 MB", file=sys.stderr)
        return 4096


def preflight_vram_check(
    min_free_mb: int = 1024,
    max_attempts: int = 5,
    backoff_seconds: float = 2.0,
    worker_name: str = "GPU_WORKER"
) -> int:
    """Preflight VRAM safety check: ensures sufficient free VRAM exists before starting GPU workload.

    If free VRAM is below safety threshold, retries with exponential backoff.
    Raises GPULockError if free VRAM remains insufficient after all attempts.
    """
    current_backoff = backoff_seconds
    for attempt in range(1, max_attempts + 1):
        free_mb = check_free_vram_mb()
        if free_mb >= min_free_mb:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PREFLIGHT_VRAM_OK] {worker_name}: "
                f"Free VRAM {free_mb} MB >= threshold {min_free_mb} MB.",
                flush=True
            )
            return free_mb
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [PREFLIGHT_VRAM_LOW] {worker_name}: "
            f"Free VRAM ({free_mb} MB) below safety threshold ({min_free_mb} MB). "
            f"Attempt {attempt}/{max_attempts}. Backing off for {current_backoff:.1f}s...",
            file=sys.stderr,
            flush=True
        )
        time.sleep(current_backoff)
        current_backoff *= 1.5

    raise GPULockError(
        f"Preflight VRAM safety check failed for {worker_name}: "
        f"Free VRAM ({check_free_vram_mb()} MB) remained below {min_free_mb} MB after {max_attempts} attempts."
    )


@contextmanager
def gpu_lock(
    timeout: float = 180.0,
    lock_path: Optional[Path] = None,
    worker_name: str = "unknown",
    preflight_vram_mb: Optional[int] = None
):
    """Context manager for acquiring the GPU lock across processes.

    Args:
        timeout: Maximum seconds to wait before timing out (Rule 8 compliance).
        lock_path: Path to the lock file. Defaults to DEFAULT_LOCK_PATH.
        worker_name: Descriptive name of the worker holding the lock.
        preflight_vram_mb: Optional VRAM threshold in MB to verify before acquiring lock.


    Yields:
        Path: Path to the lock file.

    Raises:
        GPULockError: If lock acquisition times out.
    """
    l_path = Path(lock_path) if lock_path else DEFAULT_LOCK_PATH
    m_path = l_path.with_suffix(".lock.meta")
    l_path.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(l_path), timeout=timeout)
    pid = os.getpid()
    acquired = False
    acquire_start = time.time()

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GPU_LOCK] PID {pid} ({worker_name}) requesting GPU lock...", flush=True)

    if preflight_vram_mb is not None:
        preflight_vram_check(min_free_mb=preflight_vram_mb, worker_name=worker_name)

    try:
        lock.acquire(timeout=timeout)
        acquired = True
        elapsed = time.time() - acquire_start
        acquired_time = time.time()

        # Write metadata for debugging and monitoring
        try:
            with open(m_path, "w") as f:
                json.dump({
                    "pid": pid,
                    "worker": worker_name,
                    "acquired_at": acquired_time,
                    "acquired_str": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(acquired_time))
                }, f)
        except Exception:
            pass

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GPU_LOCK] PID {pid} ({worker_name}) "
            f"ACQUIRED GPU lock after {elapsed:.2f}s.",
            flush=True
        )

        yield l_path

    except Timeout:
        raise GPULockError(
            f"Failed to acquire GPU lock within {timeout:.1f}s for PID {pid} ({worker_name}). "
            f"Another GPU-bound process is active."
        )
    finally:
        if acquired:
            release_time = time.time()
            hold_duration = release_time - acquired_time
            try:
                if m_path.exists():
                    m_path.unlink()
            except Exception:
                pass

            lock.release()
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [GPU_LOCK] PID {pid} ({worker_name}) "
                f"RELEASED GPU lock after holding for {hold_duration:.2f}s.",
                flush=True
            )


def get_lock_status(lock_path: Optional[Path] = None) -> dict:
    """Returns the current state and metadata of the GPU lock."""
    l_path = Path(lock_path) if lock_path else DEFAULT_LOCK_PATH
    m_path = l_path.with_suffix(".lock.meta")

    if not l_path.exists():
        return {"locked": False}

    meta = {}
    if m_path.exists():
        try:
            with open(m_path, "r") as f:
                meta = json.load(f)
        except Exception:
            pass

    test_lock = FileLock(str(l_path), timeout=0.01)
    is_locked = False
    try:
        test_lock.acquire(timeout=0.01)
        test_lock.release()
    except Timeout:
        is_locked = True

    return {
        "locked": is_locked,
        "lock_file": str(l_path),
        **meta
    }


def main():
    parser = argparse.ArgumentParser(description="GPU Lock CLI utility for RTX 3050 process serialization")
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # Status subcommand
    subparsers.add_parser("status", help="Inspect GPU lock state")

    # Run subcommand: executes command while holding the GPU lock
    run_parser = subparsers.add_parser("run", help="Run a command under the GPU lock")
    run_parser.add_argument("--timeout", type=float, default=300.0, help="Lock acquisition timeout in seconds")
    run_parser.add_argument("--name", type=str, default="cli_run", help="Worker name label")
    run_parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command and arguments to execute")

    args = parser.parse_args()

    if args.action == "status":
        status = get_lock_status()
        print(json.dumps(status, indent=2))
        sys.exit(0)

    elif args.action == "run":
        if not args.cmd:
            print("Error: No command specified to run.", file=sys.stderr)
            sys.exit(1)

        # If remainder starts with '--', strip it
        cmd = args.cmd
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]

        try:
            with gpu_lock(timeout=args.timeout, worker_name=args.name):
                proc = subprocess.run(cmd)
                sys.exit(proc.returncode)
        except GPULockError as e:
            print(f"[GPU_LOCK] ERROR: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
