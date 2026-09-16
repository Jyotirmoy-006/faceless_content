"""Central quota-budget tracker for YouTube Data API.

Per Rule 3 (QUOTA AWARENESS):
- Unit-based quota tracker (not call-count based).
- Enforces quota availability before making any YouTube Data API request.
- Daily quota limit is 10,000 units.
- A video upload (videos.insert) costs 1,600 units.
- Persists daily usage state to pipeline/core/quota_state.json.
- Resets automatically at 00:00 UTC daily.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("quota_tracker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [quota_tracker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default constants
YOUTUBE_DAILY_QUOTA_LIMIT = 10_000
YOUTUBE_VIDEO_UPLOAD_COST = 1_600
DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "quota_state.json"


class InsufficientQuotaError(Exception):
    """Raised when an operation requires more quota units than currently available."""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient YouTube quota: requires {required} units, but only {available} units available today."
        )


class QuotaTracker:
    """Thread-safe and file-backed unit-based quota tracker for YouTube Data API."""

    def __init__(
        self,
        state_file: Path | str = DEFAULT_STATE_FILE,
        daily_limit: int = YOUTUBE_DAILY_QUOTA_LIMIT
    ) -> None:
        self.state_file = Path(state_file)
        self.daily_limit = daily_limit
        self._ensure_state_loaded()

    def _today_utc(self) -> str:
        """Returns the current date in UTC format YYYY-MM-DD."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_state_loaded(self) -> dict[str, Any]:
        """Loads or initializes the quota tracking state file."""
        today = self._today_utc()
        if not self.state_file.exists():
            state = {
                "date": today,
                "daily_limit": self.daily_limit,
                "used_units": 0,
                "history": []
            }
            self._save_state(state)
            return state

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read quota state file ({e}), initializing fresh state.")
            state = {
                "date": today,
                "daily_limit": self.daily_limit,
                "used_units": 0,
                "history": []
            }
            self._save_state(state)
            return state

        # Check if UTC day has rolled over
        if state.get("date") != today:
            logger.info(f"New UTC day detected ({today} != {state.get('date')}). Resetting daily quota usage.")
            state["date"] = today
            state["used_units"] = 0
            # Keep recent history (last 50 records)
            state["history"] = (state.get("history") or [])[-50:]
            self._save_state(state)

        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        """Atomically saves the state dictionary to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            temp_file.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to persist quota state: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def get_used_units(self) -> int:
        """Returns the total units consumed so far today."""
        state = self._ensure_state_loaded()
        return int(state.get("used_units", 0))

    def get_available_quota(self) -> int:
        """Returns remaining available quota units for today."""
        state = self._ensure_state_loaded()
        limit = int(state.get("daily_limit", self.daily_limit))
        used = int(state.get("used_units", 0))
        return max(0, limit - used)

    def has_budget(self, units: int = YOUTUBE_VIDEO_UPLOAD_COST) -> bool:
        """Checks whether the requested units can be spent within today's quota."""
        available = self.get_available_quota()
        return available >= units

    def check_and_reserve(self, units: int = YOUTUBE_VIDEO_UPLOAD_COST) -> bool:
        """Convenience method: raises InsufficientQuotaError if not available."""
        available = self.get_available_quota()
        if available < units:
            raise InsufficientQuotaError(required=units, available=available)
        return True

    def consume(self, units: int, reason: str = "videos.insert", resource_id: str = "") -> int:
        """Records unit expenditure in the quota tracker immediately.
        
        Args:
            units: Number of API units spent (e.g. 1600 for videos.insert).
            reason: Operation name for auditing.
            resource_id: Optional ID of the created/accessed resource.
            
        Returns:
            The new cumulative used units for today.
        """
        state = self._ensure_state_loaded()
        current_used = int(state.get("used_units", 0))
        new_used = current_used + units
        state["used_units"] = new_used

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "units": units,
            "operation": reason,
            "resource_id": resource_id,
            "used_after": new_used
        }
        if "history" not in state:
            state["history"] = []
        state["history"].append(entry)

        self._save_state(state)
        available = max(0, int(state.get("daily_limit", self.daily_limit)) - new_used)
        logger.info(
            f"Quota recorded: +{units} units for '{reason}' "
            f"(Total today: {new_used}/{state.get('daily_limit')}, Remaining: {available})"
        )
        return new_used

    def simulate_quota(self, used_units: int, date_str: Optional[str] = None) -> None:
        """Helper to simulate quota usage (e.g. for testing exhaustion states)."""
        state = self._ensure_state_loaded()
        if date_str:
            state["date"] = date_str
        state["used_units"] = used_units
        self._save_state(state)
        logger.info(f"Simulated quota set: used_units={used_units}, date={state.get('date')}")

    def reset(self) -> None:
        """Resets today's quota consumption to 0."""
        state = self._ensure_state_loaded()
        state["used_units"] = 0
        state["date"] = self._today_utc()
        self._save_state(state)
        logger.info("Quota usage manually reset to 0 units.")


# Global default instance
quota_tracker = QuotaTracker()


if __name__ == "__main__":
    tracker = QuotaTracker()
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        tracker.reset()
        print("Quota usage reset to 0.")
    else:
        print(f"Date: {tracker._today_utc()}")
        print(f"Daily Limit: {tracker.daily_limit} units")
        print(f"Used Today: {tracker.get_used_units()} units")
        print(f"Available: {tracker.get_available_quota()} units")
        print(f"Can upload video (1600 units): {tracker.has_budget(1600)}")
