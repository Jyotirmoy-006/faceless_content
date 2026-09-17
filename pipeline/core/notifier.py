"""Central Notifier for Pipeline Alerts & Observability.

Complies with:
- Rule 14 (ACCOUNT RESILIENCE): Emits explicit notifications when accounts are quarantined:
  "[ACCOUNT_DOWN] Account N appears restricted - operating on remaining."
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from pipeline.core.logger import get_logger

logger = get_logger("notifier")


class Notifier:
    """Manages system-level alerts and notifications with multi-channel persistence."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []

    def alert(self, message: str, level: str = "ERROR", extra: Optional[dict] = None) -> str:
        """Emits a general system alert, persisting to DB & file, and forwarding to webhook if configured."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "message": message,
            "extra": extra or {},
        }
        self._history.append(record)
        if level.upper() == "ERROR":
            logger.error(f"[NOTIFIER_ALERT] {message}")
        elif level.upper() == "WARNING":
            logger.warning(f"[NOTIFIER_ALERT] {message}")
        else:
            logger.info(f"[NOTIFIER_ALERT] {message}")

        # 1. Persist to SQLite alerts table
        try:
            from pipeline.dashboard.database import record_alert
            record_alert(message=message, level=level, extra=extra)
        except Exception as e:
            logger.debug(f"[notifier] Database alert persistence bypassed: {e}")

        # 2. Persist to output/alerts.jsonl
        try:
            alerts_file = Path("pipeline/output/alerts.jsonl")
            alerts_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alerts_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"[notifier] File alert persistence bypassed: {e}")

        # 3. Dispatch to webhook if configured
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL") or os.getenv("ALERT_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")
        if webhook_url:
            self._dispatch_webhook(webhook_url, record)

        return message

    def _dispatch_webhook(self, webhook_url: str, record: dict[str, Any]) -> None:
        """Dispatches structured alert notification to an external webhook endpoint."""
        try:
            color = 15158332 if record["level"] == "ERROR" else (16776960 if record["level"] == "WARNING" else 3066993)
            payload = {
                "content": f"**[{record['level']}]** {record['message']}",
                "embeds": [{
                    "title": f"Autonomous Studio Alert — {record['level']}",
                    "description": record["message"],
                    "timestamp": record["timestamp"],
                    "color": color
                }]
            }
            requests.post(webhook_url, json=payload, timeout=5.0)
        except Exception as e:
            logger.warning(f"[notifier] External webhook delivery failed: {e}")

    def account_down(self, account_id: int | str, reason: str = "") -> str:
        """Emits mandatory Rule 14 notification when a Google AI account is quarantined.
        
        Strict format:
        [ACCOUNT_DOWN] Account N appears restricted - operating on remaining.
        """
        message = f"[ACCOUNT_DOWN] Account {account_id} appears restricted - operating on remaining."
        self.alert(message, level="ERROR", extra={"account_id": account_id, "reason": reason})
        return message

    def get_alerts(self) -> list[dict[str, Any]]:
        """Returns chronological history of all emitted alerts."""
        return list(self._history)

    def clear_alerts(self) -> None:
        """Clears in-memory alert history."""
        self._history.clear()


# Global default notifier singleton
notifier = Notifier()

