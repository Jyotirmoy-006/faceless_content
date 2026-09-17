"""Durable Circuit Breaker for Autonomous Pipeline Safety.

Per Rule 8 (NO UNBOUNDED RETRIES) & Rule 7 (SCHEDULER SAFETY):
- Tracks consecutive full-pipeline failures across independent process invocations.
- Persisted to a durable state file (`pipeline/core/circuit_breaker_state.json`).
- If 3 consecutive pipeline runs fail, the breaker trips (status = OPEN).
- When tripped, subsequent scheduled runs refuse to execute normally, logging an
  alert describing the last 3 failure reasons to console and `pipeline/output/alerts.log`.
- Requires a manual reset (e.g. deleting state file or `--reset-circuit-breaker`) to re-arm.
- Durability: State survives across process crashes and reboots.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATE_FILE = Path(os.environ.get("CIRCUIT_BREAKER_STATE_FILE", Path(__file__).resolve().parent / "circuit_breaker_state.json"))
DEFAULT_ALERT_FILE = ROOT_DIR / "pipeline" / "output" / "alerts.log"

logger = logging.getLogger("circuit_breaker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [circuit_breaker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CircuitBreakerOpenError(Exception):
    """Raised when an operation is blocked due to an open circuit breaker."""
    def __init__(self, failure_reasons: list[str]):
        self.failure_reasons = failure_reasons
        msg = (
            f"Circuit breaker is OPEN after {len(failure_reasons)} consecutive failures. "
            f"Automated execution halted to protect resources and quotas. "
            f"Run 'python main.py --reset-circuit-breaker' to re-arm."
        )
        super().__init__(msg)


class CircuitBreaker:
    """Manages persistent circuit breaker state across separate process runs."""

    def __init__(
        self,
        state_file: Path | str = DEFAULT_STATE_FILE,
        alert_file: Path | str = DEFAULT_ALERT_FILE,
        max_failures: int = 3
    ) -> None:
        self.state_file = Path(state_file)
        self.alert_file = Path(alert_file)
        self.max_failures = max_failures
        self._ensure_state()

    def _ensure_state(self) -> dict[str, Any]:
        """Loads state from disk or initializes fresh state if missing."""
        if not self.state_file.exists():
            state = {
                "consecutive_failures": 0,
                "max_failures": self.max_failures,
                "is_open": False,
                "tripped_at": None,
                "history": []
            }
            self._save_state(state)
            return state

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Corrupted circuit breaker state file ({e}), re-initializing.")
            state = {
                "consecutive_failures": 0,
                "max_failures": self.max_failures,
                "is_open": False,
                "tripped_at": None,
                "history": []
            }
            self._save_state(state)
            return state

    def _save_state(self, state: dict[str, Any]) -> None:
        """Atomically saves state dictionary to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def get_state(self) -> dict[str, Any]:
        """Returns the current circuit breaker state dictionary."""
        return self._ensure_state()

    def is_tripped(self) -> bool:
        """Returns True if the circuit breaker is currently OPEN."""
        state = self._ensure_state()
        return bool(state.get("is_open", False))

    def can_execute(self) -> tuple[bool, str]:
        """Checks if pipeline run is permitted to proceed.
        
        Returns:
            Tuple of (is_allowed, reason_message)
        """
        state = self._ensure_state()
        if state.get("is_open", False):
            failures = self.get_last_failure_reasons(self.max_failures)
            formatted_reasons = "\n".join(f"  - [{i+1}] {r}" for i, r in enumerate(failures))
            alert_msg = (
                f"CIRCUIT BREAKER IS OPEN!\n"
                f"Execution blocked because {state.get('consecutive_failures', self.max_failures)} consecutive "
                f"runs failed. Last {len(failures)} failure reasons:\n{formatted_reasons}\n"
                f"To re-arm, run: python main.py --reset-circuit-breaker"
            )
            return False, alert_msg
        return True, "Circuit breaker is closed. Execution permitted."

    def record_success(self) -> None:
        """Resets consecutive failures to 0 and closes breaker on successful run."""
        state = self._ensure_state()
        if state.get("consecutive_failures", 0) > 0 or state.get("transient_failures", 0) > 0 or state.get("is_open", False):
            logger.info("Pipeline run succeeded. Resetting consecutive failure counter.")
        state["consecutive_failures"] = 0
        state["transient_failures"] = 0
        state["is_open"] = False
        state["tripped_at"] = None
        self._save_state(state)

    def record_failure(
        self,
        reason: str,
        is_transient: bool = False,
        max_transient_failures: int = 6
    ) -> None:
        """Records a pipeline failure with smart categorization.
        
        Args:
            reason: Description of the failure.
            is_transient: If True, indicates a temporary glitch (network timeout, lock wait).
                          Transient failures are tracked separately and do not increment the
                          strict permanent 3-strike counter unless prolonged (>= max_transient_failures).
            max_transient_failures: Threshold for consecutive transient failures before tripping.
        """
        state = self._ensure_state()

        if is_transient:
            trans_count = int(state.get("transient_failures", 0)) + 1
            state["transient_failures"] = trans_count
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failure_number": trans_count,
                "category": "TRANSIENT",
                "reason": str(reason).strip()
            }
            if "history" not in state:
                state["history"] = []
            state["history"].append(entry)

            logger.warning(
                f"[CIRCUIT_BREAKER] Transient pipeline failure recorded ({trans_count}/{max_transient_failures}): "
                f"{reason[:120]} (Permanent failure count remains {state.get('consecutive_failures', 0)})."
            )

            if trans_count >= max_transient_failures:
                state["is_open"] = True
                state["tripped_at"] = entry["timestamp"]
                self._save_state(state)
                self._emit_alert(state)
            else:
                self._save_state(state)
            return

        failures = int(state.get("consecutive_failures", 0)) + 1
        state["consecutive_failures"] = failures

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "failure_number": failures,
            "category": "PERMANENT",
            "reason": str(reason).strip()
        }
        if "history" not in state:
            state["history"] = []
        state["history"].append(entry)

        logger.warning(f"Pipeline failure recorded ({failures}/{self.max_failures}): {reason[:120]}...")

        if failures >= self.max_failures:
            state["is_open"] = True
            state["tripped_at"] = entry["timestamp"]
            self._save_state(state)
            self._emit_alert(state)
        else:
            self._save_state(state)

    def _emit_alert(self, state: dict[str, Any]) -> None:
        """Logs and records diagnostic alert when breaker trips."""
        history = state.get("history", [])[-self.max_failures:]
        reasons = [h.get("reason", "Unknown") for h in history]
        
        banner = (
            "\n" + "=" * 80 + "\n"
            "CRITICAL ALERT: PIPELINE CIRCUIT BREAKER TRIPPED (STATUS: OPEN)\n"
            f"Reached {len(reasons)} consecutive full-pipeline failures.\n"
            "Automated execution has been HALTED to prevent resource exhaustion and quota waste.\n"
            "Failure Diagnostics:\n"
        )
        for idx, r in enumerate(reasons, 1):
            banner += f"  [{idx}] {r}\n"
        banner += (
            "Manual action required to re-arm:\n"
            "  Run: python main.py --reset-circuit-breaker\n"
            + "=" * 80 + "\n"
        )
        
        logger.error(banner)

        # Write to alert file
        try:
            self.alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alert_file, "a", encoding="utf-8") as f:
                f.write(banner + "\n")
        except Exception as e:
            logger.error(f"Failed to append alert to {self.alert_file}: {e}")

    def reset(self) -> None:
        """Manually re-arms the circuit breaker."""
        state = self._ensure_state()
        state["consecutive_failures"] = 0
        state["transient_failures"] = 0
        state["is_open"] = False
        state["tripped_at"] = None
        self._save_state(state)
        logger.info("Circuit breaker has been manually RESET to CLOSED status. Ready for scheduled execution.")

    def get_last_failure_reasons(self, count: int = 3) -> list[str]:
        """Retrieves the most recent failure reasons."""
        state = self._ensure_state()
        history = state.get("history", [])
        return [h.get("reason", "Unknown") if isinstance(h, dict) else str(h) for h in history[-count:]]

    def get_consecutive_failures(self) -> int:
        """Returns the current number of consecutive failures."""
        state = self._ensure_state()
        return int(state.get("consecutive_failures", 0))


# Global default circuit breaker
circuit_breaker = CircuitBreaker()


if __name__ == "__main__":
    cb = CircuitBreaker()
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        cb.reset()
        print("Circuit breaker successfully reset.")
    else:
        allowed, msg = cb.can_execute()
        print(f"Status: {'OPEN (TRIPPED)' if cb.is_tripped() else 'CLOSED (HEALTHY)'}")
        print(f"Consecutive Failures: {cb.get_consecutive_failures()}/{cb.max_failures}")
        print(f"Can Execute: {allowed}")
        if not allowed:
            print(f"Alert Message:\n{msg}")
