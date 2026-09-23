"""Tests for Single-Account Global Rate Limiter.

Verifies:
1. Minimum cooldown delay between consecutive requests.
2. Hard Request-Per-Minute (RPM) sliding window enforcement.
3. Thread-safe concurrency without race conditions or quota overflow.
4. Context manager and timeout error handling.
"""

from __future__ import annotations

import concurrent.futures
import time
from unittest import TestCase

from pipeline.core.rate_limiter import SingleAccountRateLimiter, AccountRateLimiterPool


class TestSingleAccountRateLimiter(TestCase):
    """Test suite for single-account Gemini rate limiting."""

    def test_cooldown_enforcement(self) -> None:
        """Verifies that consecutive requests respect the minimum spacing delay."""
        limiter = SingleAccountRateLimiter(max_rpm=60, min_delay_seconds=0.15)
        limiter.reset()

        t0 = time.time()
        self.assertTrue(limiter.acquire())
        elapsed1 = time.time() - t0

        t1 = time.time()
        self.assertTrue(limiter.acquire())
        elapsed2 = time.time() - t1

        self.assertLess(elapsed1, 0.05)
        # Second call must have waited at least ~0.14s
        self.assertGreaterEqual(elapsed2, 0.12)

    def test_rpm_sliding_window_cap(self) -> None:
        """Verifies that RPM cap throttles requests within the rolling window."""
        # Setup small 3 RPM limiter with minimal cooldown
        limiter = SingleAccountRateLimiter(max_rpm=3, min_delay_seconds=0.01)
        limiter.reset()

        # 3 calls should succeed immediately
        for _ in range(3):
            self.assertTrue(limiter.acquire(timeout=0.1))

        status = limiter.get_status()
        self.assertEqual(status["current_rpm"], 3)

        # 4th call with short timeout must time out because window is full
        self.assertFalse(limiter.acquire(timeout=0.05))

    def test_thread_safe_concurrency(self) -> None:
        """Verifies that concurrent requests from multiple agents acquire slots safely."""
        limiter = SingleAccountRateLimiter(max_rpm=20, min_delay_seconds=0.02)
        limiter.reset()

        acquired_times: list[float] = []

        def worker() -> float:
            limiter.acquire(timeout=5.0)
            return time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker) for _ in range(5)]
            for f in concurrent.futures.as_completed(futures):
                acquired_times.append(f.result())

        acquired_times.sort()
        self.assertEqual(len(acquired_times), 5)

        # Check that intervals between sorted acquisitions are >= min_delay_seconds (with slight float tolerance)
        for i in range(1, len(acquired_times)):
            interval = acquired_times[i] - acquired_times[i - 1]
            self.assertGreaterEqual(interval, 0.015)

    def test_context_manager_and_status(self) -> None:
        """Verifies the limit() context manager and telemetry dictionary."""
        limiter = SingleAccountRateLimiter(max_rpm=10, min_delay_seconds=0.05)
        limiter.reset()

        with limiter.limit(timeout=1.0):
            pass

        status = limiter.get_status()
        self.assertEqual(status["total_requests_acquired"], 1)
        self.assertEqual(status["current_rpm"], 1)
        self.assertIsNotNone(status["seconds_since_last_call"])

    def test_account_rate_limiter_pool_isolation(self) -> None:
        """Verifies that separate accounts have independent rate limits and do not block each other."""
        pool = AccountRateLimiterPool(max_rpm=2, min_delay_seconds=0.2)
        pool.reset_all()

        # Account 1 acquires 2 slots (filling its 2 RPM capacity)
        self.assertTrue(pool.acquire(account_id=1, timeout=0.1))
        self.assertTrue(pool.acquire(account_id=1, timeout=0.3))
        # Account 1 third call should be blocked by window cap
        self.assertFalse(pool.acquire(account_id=1, timeout=0.05))

        # Account 2 has completely independent capacity and must succeed immediately
        t0 = time.monotonic()
        self.assertTrue(pool.acquire(account_id=2, timeout=0.1))
        self.assertLess(time.monotonic() - t0, 0.15)
