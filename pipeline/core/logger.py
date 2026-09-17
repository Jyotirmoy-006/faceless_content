"""Centralized Logging Configuration for Autonomous Faceless Video Studio (Mission H).

Enforces strict timestamp formatting across all loggers and workers:
Format: [YYYY-MM-DD HH:MM:SS] [AGENT/WORKER] Message
"""

import logging
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
LOG_FILE_ROOT = ROOT_DIR / "pipeline.log"
LOG_FILE_SUB = ROOT_DIR / "pipeline" / "logs" / "pipeline.log"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "[%(asctime)s] [%(name)s] %(message)s"


class StrictTimestampFormatter(logging.Formatter):
    """Enforces strict [YYYY-MM-DD HH:MM:SS] timestamp formatting."""
    def __init__(self, fmt: str = LOG_FORMAT, datefmt: str = DATE_FORMAT):
        super().__init__(fmt=fmt, datefmt=datefmt)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger enforcing standard timestamp and tag formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = StrictTimestampFormatter()

        # Stream Handler (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # File Handlers
        for path in [LOG_FILE_ROOT, LOG_FILE_SUB]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(str(path), encoding="utf-8")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass

        logger.propagate = False

    return logger
