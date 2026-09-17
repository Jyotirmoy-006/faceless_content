"""Launcher for Autonomous Faceless Command Center Web Dashboard (Mission G).

Usage:
    python run_dashboard.py [--host 127.0.0.1] [--port 5000] [--interval 5.0]
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.dashboard.app import app, init_db, start_queue_worker


def main():
    parser = argparse.ArgumentParser(description="Autonomous Faceless Command Center Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", default=5000, type=int, help="Port number (default: 5000)")
    parser.add_argument("--interval", default=5.0, type=float, help="Queue worker poll interval (seconds, default: 5.0)")
    parser.add_argument("--mock", action="store_true", help="Run with mock_main.py for fast testing")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    import os
    if args.mock:
        os.environ["MAIN_SCRIPT"] = "mock_main.py"

    init_db()
    start_queue_worker(poll_interval=args.interval)

    print("=" * 70)
    print(f"COMMAND CENTER ACTIVE -> http://{args.host}:{args.port}")
    print(f"Sequential Queue Worker Active -> Polling interval: {args.interval}s")
    print("=" * 70)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
