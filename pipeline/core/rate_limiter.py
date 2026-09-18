"""Single-Account Global Rate Limiter for Gemini API.

Enforces strict quota discipline when multiple worker agents and department heads
share a single Google Gemini API key:
1. Hard Request-Per-Minute (RPM) Cap: Enforces maximum requests within a 60-second sliding window.
2. Inter-Request Cooldown: Minimum spacing (default 800ms) between successive calls to avoid bursting.
3. Thread-Safe Locking: Prevents race conditions during concurrent agent or subagent execution.
4. Transparent Observability: Real-time telemetry for rate-limit pressure and queue delays.
"""

from __future__ import annotations

import collections
import threading
import time
from contextlib import contextmanager
from typing import Any, Deque, Dict, Generator, Optional

from pipeline.core.logger import get_logger

logger = get_logger("rate_limiter")


class SingleAccountRateLimiter:
    """Thread-safe Token/Window rate limiter for a single Gemini API account."""

    def __init__(
        self,
        max_rpm: int = 10,
        min_delay_seconds: float = 0.8
    ) -> None:
        """Initializes the rate limiter.

        Args:
            max_rpm: Maximum allowable requests per 60-second rolling window.
            min_delay_seconds: Minimum cooldown spacing between consecutive requests.
        """
        self.max_rpm = max(1, max_rpm)
        self.min_delay_seconds = max(0.0, min_delay_seconds)
        self._lock = threading.Lock()
        self._timestamps: Deque[float] = collections.deque()
        self._last_call_time: float = 0.0
        self._total_acquired: int = 0
        self._total_wait_seconds: float = 0.0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquires a rate-limit slot, blocking if necessary until quota is available.

        Args:
            timeout: Optional max seconds to wait. If exceeded, returns False.

        Returns:
            True if slot was successfully acquired within timeout, False otherwise.
        """
        start_wait = time.time()

        while True:
            with self._lock:
                now = time.time()

                # Clean up timestamps older than 60 seconds
                cutoff = now - 60.0
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                # Calculate cooldown delay
                cooldown_needed = 0.0
                if self._last_call_time > 0:
                    elapsed_since_last = now - self._last_call_time
                    if elapsed_since_last < self.min_delay_seconds:
                        cooldown_needed = self.min_delay_seconds - elapsed_since_last

                # Calculate window cap delay
                window_needed = 0.0
                if len(self._timestamps) >= self.max_rpm:
                    oldest = self._timestamps[0]
                    window_needed = max(0.0, 60.0 - (now - oldest))

                wait_time = max(cooldown_needed, window_needed)

                # If no wait needed, claim slot immediately
                if wait_time <= 0.001:
                    actual_now = time.time()
                    self._timestamps.append(actual_now)
                    self._last_call_time = actual_now
                    self._total_acquired += 1
                    total_waited = actual_now - start_wait
                    self._total_wait_seconds += total_waited
                    if total_waited > 0.1:
                        logger.debug(
                            f"[RATE_LIMITER] Slot acquired after {total_waited:.3f}s wait. "
                            f"(Current window: {len(self._timestamps)}/{self.max_rpm} RPM)"
                        )
                    return True

            # Check timeout
            if timeout is not None:
                elapsed_total = time.time() - start_wait
                if elapsed_total + wait_time > timeout:
                    logger.warning(
                        f"[RATE_LIMITER] Request timed out waiting for slot ({elapsed_total:.2f}s elapsed)."
                    )
                    return False

            # Sleep outside lock to allow other threads to evaluate
            sleep_chunk = min(wait_time, 0.2)
            time.sleep(sleep_chunk)

    @contextmanager
    def limit(self, timeout: Optional[float] = None) -> Generator[None, None, None]:
        """Context manager for acquiring a rate-limited execution slot."""
        acquired = self.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("Timed out waiting for single-account Gemini rate-limit slot.")
        try:
            yield
        finally:
            pass

    def get_status(self) -> Dict[str, Any]:
        """Returns current rate limiter telemetry."""
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            valid_calls = sum(1 for ts in self._timestamps if ts > cutoff)
            return {
                "max_rpm": self.max_rpm,
                "current_rpm": valid_calls,
                "min_delay_seconds": self.min_delay_seconds,
                "seconds_since_last_call": round(now - self._last_call_time, 3) if self._last_call_time else None,
                "total_requests_acquired": self._total_acquired,
                "total_wait_seconds": round(self._total_wait_seconds, 3),
            }

    def reset(self) -> None:
        """Resets rate limiter state (useful for test isolation)."""
        with self._lock:
            self._timestamps.clear()
            self._last_call_time = 0.0
            self._total_acquired = 0
            self._total_wait_seconds = 0.0


# Shared singleton instance for all agents & department heads
global_rate_limiter = SingleAccountRateLimiter(max_rpm=10, min_delay_seconds=0.8)
