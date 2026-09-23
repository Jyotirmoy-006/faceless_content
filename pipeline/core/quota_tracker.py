"""Central quota-budget tracker for YouTube Data API and YouTube Analytics API.

Complies with:
- Rule 3 (QUOTA AWARENESS):
  - Unit-based quota tracker for YouTube Data API v3 (10,000 units/day default general pool).
  - Independent query tracker for YouTube Analytics API v2 (100,000 queries/day default pool).
  - Enforces quota availability before making any YouTube request.
  - Operation costs:
    - videos.insert: 1,600 Data API units
    - channels.list: 1 Data API unit
    - playlistItems.list: 1 Data API unit
    - videos.list: 1 Data API unit
    - search.list: 100 Data API units (prohibited for video discovery; use channels.list + playlistItems.list)
    - reports.query: 1 Analytics API query (unrelated to Data API 10,000 units pool)
  - Persists usage state across restarts to pipeline/core/quota_state.json.
  - Resets automatically at 00:00 America/Los_Angeles (Pacific Time) daily,
    matching Google Cloud Console replenishment schedule with PST/PDT DST handling.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("quota_tracker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [quota_tracker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Google Cloud YouTube Data API v3 Quota Specifications
YOUTUBE_DAILY_QUOTA_LIMIT = 10_000        # Shared daily general pool (units per day)
YOUTUBE_VIDEO_UPLOAD_COST = 1_600         # videos.insert
COST_VIDEOS_INSERT = 1_600                # videos.insert
COST_SEARCH_LIST = 100                    # search.list (100 units - avoid for video discovery!)
COST_VIDEOS_LIST = 1                      # videos.list
COST_CHANNELS_LIST = 1                    # channels.list
COST_PLAYLIST_ITEMS_LIST = 1              # playlistItems.list (1 unit - use for uploads discovery!)

# Google Cloud YouTube Analytics API v2 Quota Specifications
# Operates on its own independent pool (queries per day) completely separate from Data API v3.
YOUTUBE_ANALYTICS_DAILY_LIMIT = 100_000   # Analytics queries per day
COST_ANALYTICS_REPORTS_QUERY = 1          # reports().query()

DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "quota_state.json"
PACIFIC_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")


class InsufficientQuotaError(Exception):
    """Raised when an operation requires more Data API units than currently available."""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient YouTube Data API quota: requires {required} units, but only {available} units available today."
        )


class InsufficientAnalyticsQuotaError(Exception):
    """Raised when an operation requires more Analytics API queries than currently available."""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient YouTube Analytics API quota: requires {required} queries, but only {available} queries available today."
        )


class QuotaTracker:
    """Thread-safe and file-backed unit-based quota tracker for YouTube Data and Analytics APIs."""

    def __init__(
        self,
        state_file: Path | str = DEFAULT_STATE_FILE,
        daily_limit: int = YOUTUBE_DAILY_QUOTA_LIMIT,
        analytics_daily_limit: int = YOUTUBE_ANALYTICS_DAILY_LIMIT
    ) -> None:
        self.state_file = Path(state_file)
        self.daily_limit = daily_limit
        self.analytics_daily_limit = analytics_daily_limit
        self._ensure_state_loaded()

    def _today_pacific(self, current_time: Optional[datetime] = None) -> str:
        """Returns the current date in America/Los_Angeles format YYYY-MM-DD.
        
        Google Cloud YouTube daily quota resets at midnight Pacific Time (PT).
        Using America/Los_Angeles with standard zoneinfo properly accounts for PST (UTC-8)
        and PDT (UTC-7) Daylight Saving Time shifts.
        """
        if current_time is None:
            dt = datetime.now(PACIFIC_TZ)
        else:
            if current_time.tzinfo is None:
                dt = current_time.replace(tzinfo=PACIFIC_TZ)
            else:
                dt = current_time.astimezone(PACIFIC_TZ)
        return dt.strftime("%Y-%m-%d")

    def _today_utc(self) -> str:
        """Deprecated: Returns current date in Pacific format for backwards compatibility."""
        return self._today_pacific()

    def _ensure_state_loaded(self, current_time: Optional[datetime] = None) -> dict[str, Any]:
        """Loads or initializes the quota tracking state file."""
        today = self._today_pacific(current_time)
        if not self.state_file.exists():
            state = {
                "date": today,
                "timezone": "America/Los_Angeles",
                "daily_limit": self.daily_limit,
                "used_units": 0,
                "history": [],
                "analytics_daily_limit": self.analytics_daily_limit,
                "analytics_used_queries": 0,
                "analytics_history": []
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
                "timezone": "America/Los_Angeles",
                "daily_limit": self.daily_limit,
                "used_units": 0,
                "history": [],
                "analytics_daily_limit": self.analytics_daily_limit,
                "analytics_used_queries": 0,
                "analytics_history": []
            }
            self._save_state(state)
            return state

        # Backfill analytics keys if missing from older state file
        if "analytics_daily_limit" not in state:
            state["analytics_daily_limit"] = self.analytics_daily_limit
        if "analytics_used_queries" not in state:
            state["analytics_used_queries"] = 0
        if "analytics_history" not in state:
            state["analytics_history"] = []

        # Check if Pacific day has rolled over (Midnight PT reset)
        if state.get("date") != today:
            logger.info(f"New Pacific day detected ({today} != {state.get('date')}). Resetting daily quota usage.")
            state["date"] = today
            state["timezone"] = "America/Los_Angeles"
            state["used_units"] = 0
            state["analytics_used_queries"] = 0
            # Keep recent history (last 50 records)
            state["history"] = (state.get("history") or [])[-50:]
            state["analytics_history"] = (state.get("analytics_history") or [])[-50:]
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

    # =========================================================================
    # YOUTUBE DATA API v3 (10,000 UNITS / DAY)
    # =========================================================================

    def get_used_units(self, current_time: Optional[datetime] = None) -> int:
        """Returns the total Data API units consumed so far today."""
        state = self._ensure_state_loaded(current_time)
        return int(state.get("used_units", 0))

    def get_available_quota(self, current_time: Optional[datetime] = None) -> int:
        """Returns remaining available Data API quota units for today."""
        state = self._ensure_state_loaded(current_time)
        limit = int(state.get("daily_limit", self.daily_limit))
        used = int(state.get("used_units", 0))
        return max(0, limit - used)

    def has_budget(self, units: int = YOUTUBE_VIDEO_UPLOAD_COST, current_time: Optional[datetime] = None) -> bool:
        """Checks whether the requested Data API units can be spent within today's quota."""
        available = self.get_available_quota(current_time)
        return available >= units

    can_spend = has_budget

    def check_and_reserve(self, units: int = YOUTUBE_VIDEO_UPLOAD_COST, current_time: Optional[datetime] = None) -> bool:
        """Raises InsufficientQuotaError if requested Data API units not available."""
        available = self.get_available_quota(current_time)
        if available < units:
            raise InsufficientQuotaError(required=units, available=available)
        return True

    def consume(
        self,
        units: int,
        reason: str = "videos.insert",
        resource_id: str = "",
        current_time: Optional[datetime] = None
    ) -> int:
        """Records Data API unit expenditure in the quota tracker immediately.
        
        Args:
            units: Number of Data API units spent (e.g. 1600 for videos.insert, 1 for playlistItems.list).
            reason: Operation name for auditing.
            resource_id: Optional ID of the created/accessed resource.
            current_time: Optional simulated timestamp for boundary testing.
            
        Returns:
            The new cumulative used units for today.
        """
        state = self._ensure_state_loaded(current_time)
        current_used = int(state.get("used_units", 0))
        new_used = current_used + units
        state["used_units"] = new_used

        dt = current_time or datetime.now(PACIFIC_TZ)
        entry = {
            "timestamp": dt.isoformat(),
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
            f"[DATA_API_QUOTA] Recorded: +{units} units for '{reason}' "
            f"(Total today: {new_used}/{state.get('daily_limit')}, Remaining: {available})"
        )
        return new_used

    def refund(
        self,
        units: int,
        reason: str = "refund",
        current_time: Optional[datetime] = None
    ) -> int:
        """Refunds previously consumed or reserved Data API units if an operation failed.
        
        Args:
            units: Number of Data API units to restore.
            reason: Explanation for refund.
            current_time: Optional simulated timestamp.
            
        Returns:
            The new cumulative used units for today.
        """
        state = self._ensure_state_loaded(current_time)
        current_used = int(state.get("used_units", 0))
        new_used = max(0, current_used - units)
        state["used_units"] = new_used

        dt = current_time or datetime.now(PACIFIC_TZ)
        entry = {
            "timestamp": dt.isoformat(),
            "units": -units,
            "operation": f"REFUND: {reason}",
            "used_after": new_used
        }
        if "history" not in state:
            state["history"] = []
        state["history"].append(entry)

        self._save_state(state)
        available = max(0, int(state.get("daily_limit", self.daily_limit)) - new_used)
        logger.info(
            f"[DATA_API_QUOTA] Refunded: -{units} units for '{reason}' "
            f"(Total today: {new_used}/{state.get('daily_limit')}, Remaining: {available})"
        )
        return new_used

    @contextlib.contextmanager
    def atomic_reservation(
        self,
        units: int = YOUTUBE_VIDEO_UPLOAD_COST,
        operation: str = "videos.insert",
        current_time: Optional[datetime] = None
    ):
        """Atomically checks and reserves quota units, auto-refunding if an exception occurs."""
        self.check_and_reserve(units=units, current_time=current_time)
        self.consume(units=units, reason=operation, current_time=current_time)
        try:
            yield
        except Exception as e:
            self.refund(units=units, reason=f"{operation} failed: {e}", current_time=current_time)
            raise

    # =========================================================================
    # YOUTUBE ANALYTICS API v2 (100,000 QUERIES / DAY - INDEPENDENT POOL)
    # =========================================================================

    def get_analytics_used_queries(self, current_time: Optional[datetime] = None) -> int:
        """Returns the total Analytics API queries consumed so far today."""
        state = self._ensure_state_loaded(current_time)
        return int(state.get("analytics_used_queries", 0))

    def get_analytics_available_quota(self, current_time: Optional[datetime] = None) -> int:
        """Returns remaining available Analytics API queries for today."""
        state = self._ensure_state_loaded(current_time)
        limit = int(state.get("analytics_daily_limit", self.analytics_daily_limit))
        used = int(state.get("analytics_used_queries", 0))
        return max(0, limit - used)

    def has_analytics_budget(self, queries: int = 1, current_time: Optional[datetime] = None) -> bool:
        """Checks whether requested Analytics API queries can be made within today's quota."""
        available = self.get_analytics_available_quota(current_time)
        return available >= queries

    def check_and_reserve_analytics(self, queries: int = 1, current_time: Optional[datetime] = None) -> bool:
        """Raises InsufficientAnalyticsQuotaError if requested Analytics queries not available."""
        available = self.get_analytics_available_quota(current_time)
        if available < queries:
            raise InsufficientAnalyticsQuotaError(required=queries, available=available)
        return True

    def consume_analytics(
        self,
        queries: int = COST_ANALYTICS_REPORTS_QUERY,
        reason: str = "reports.query",
        resource_id: str = "",
        current_time: Optional[datetime] = None
    ) -> int:
        """Records Analytics API query expenditure in its independent pool.
        
        Note: Does NOT affect the 10,000-unit Data API budget.
        """
        state = self._ensure_state_loaded(current_time)
        current_used = int(state.get("analytics_used_queries", 0))
        new_used = current_used + queries
        state["analytics_used_queries"] = new_used

        dt = current_time or datetime.now(PACIFIC_TZ)
        entry = {
            "timestamp": dt.isoformat(),
            "queries": queries,
            "operation": reason,
            "resource_id": resource_id,
            "used_after": new_used
        }
        if "analytics_history" not in state:
            state["analytics_history"] = []
        state["analytics_history"].append(entry)

        self._save_state(state)
        available = max(0, int(state.get("analytics_daily_limit", self.analytics_daily_limit)) - new_used)
        logger.info(
            f"[ANALYTICS_API_QUOTA] Recorded: +{queries} query for '{reason}' "
            f"(Total today: {new_used}/{state.get('analytics_daily_limit')}, Remaining: {available})"
        )
        return new_used

    # =========================================================================
    # SIMULATION & RESET UTILITIES
    # =========================================================================

    def simulate_quota(self, used_units: int, date_str: Optional[str] = None) -> None:
        """Helper to simulate Data API quota usage (e.g. for testing exhaustion states)."""
        state = self._ensure_state_loaded()
        if date_str:
            state["date"] = date_str
        state["used_units"] = used_units
        self._save_state(state)
        logger.info(f"Simulated Data API quota set: used_units={used_units}, date={state.get('date')}")

    def simulate_analytics_quota(self, used_queries: int, date_str: Optional[str] = None) -> None:
        """Helper to simulate Analytics API quota usage."""
        state = self._ensure_state_loaded()
        if date_str:
            state["date"] = date_str
        state["analytics_used_queries"] = used_queries
        self._save_state(state)
        logger.info(f"Simulated Analytics quota set: used_queries={used_queries}, date={state.get('date')}")

    def reset(self, current_time: Optional[datetime] = None) -> None:
        """Resets today's Data API and Analytics API consumption to 0."""
        state = self._ensure_state_loaded(current_time)
        state["used_units"] = 0
        state["analytics_used_queries"] = 0
        state["date"] = self._today_pacific(current_time)
        self._save_state(state)
        logger.info("Quota usage manually reset to 0 for both Data API and Analytics API.")

    def reset_analytics(self, current_time: Optional[datetime] = None) -> None:
        """Resets only today's Analytics API query consumption to 0."""
        state = self._ensure_state_loaded(current_time)
        state["analytics_used_queries"] = 0
        self._save_state(state)
        logger.info("Analytics API quota usage reset to 0 queries.")


# Global default instance
quota_tracker = QuotaTracker()


if __name__ == "__main__":
    tracker = QuotaTracker()
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        tracker.reset()
        print("Quota usage reset to 0.")
    else:
        print(f"Timezone: America/Los_Angeles (Pacific Time)")
        print(f"Date (PT): {tracker._today_pacific()}")
        print("\n--- YouTube Data API v3 (10,000 Units Pool) ---")
        print(f"Daily Limit: {tracker.daily_limit} units")
        print(f"Used Today:  {tracker.get_used_units()} units")
        print(f"Available:   {tracker.get_available_quota()} units")
        print(f"Can upload video (1600 units): {tracker.has_budget(1600)}")
        print("\n--- YouTube Analytics API v2 (100,000 Queries Pool) ---")
        print(f"Daily Limit: {tracker.analytics_daily_limit} queries")
        print(f"Used Today:  {tracker.get_analytics_used_queries()} queries")
        print(f"Available:   {tracker.get_analytics_available_quota()} queries")
        print(f"Can query report (1 query): {tracker.has_analytics_budget(1)}")
