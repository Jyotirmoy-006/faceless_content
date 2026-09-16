"""Upload a rendered video to YouTube Shorts."""

import sys
import logging
from pathlib import Path

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.agents.publisher import publisher, PublishStatus
from pipeline.core.quota_tracker import quota_tracker

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    video_path = Path("pipeline/output/20260916_180043_Why_Your_Smart_Bulb_Might_Be_L.mp4")
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    title = "Why Your Smart Bulb Might Leak Your Passwords"
    description = (
        "Smart bulb security vulnerabilities explained: hackers can capture microscopic "
        "light fluctuations from smart bulbs to read keystrokes from hundreds of feet away.\n\n"
        "#shorts #tech #cybersecurity #technology #privacy"
    )
    tags = ["#tech", "#shorts", "#cybersecurity", "#privacy", "#technology"]

    print("=" * 70)
    print("STARTING YOUTUBE SHORTS UPLOAD")
    print(f"Video: {video_path} ({video_path.stat().st_size / (1024 * 1024):.2f} MB)")
    print(f"Title: {title}")
    print(f"Quota Available: {quota_tracker.get_available_quota()} units")
    print("=" * 70)

    result = publisher.publish_to_youtube(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        privacy_status="public"
    )

    print("\n" + "=" * 70)
    print("YOUTUBE UPLOAD RESULT")
    print("=" * 70)
    print(f"Status:        {result.status}")
    print(f"Video ID:      {result.post_id}")
    print(f"Watch URL:     {result.url}")
    print(f"Visibility:    {result.visibility} (Restricted: {result.is_restricted})")
    print(f"Quota Spent:   {result.units_spent} units")
    print(f"Message:       {result.message}")
    print("=" * 70)

if __name__ == "__main__":
    main()
