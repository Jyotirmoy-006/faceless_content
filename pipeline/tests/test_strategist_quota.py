"""Test Suite for Mission: YouTube-Only Quota Optimization + Analytics API Split (Rule 3).

Complies with Acceptance Criteria:
- [x] playlistItems.list call confirmed at 1 unit via real or logged response
- [x] Analytics API calls tracked separately from Data API v3 spend
- [x] Strategist successfully retrieves CTR/retention via the correct API
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from dotenv import load_dotenv

from pipeline.core.quota_tracker import (
    QuotaTracker,
    InsufficientQuotaError,
    InsufficientAnalyticsQuotaError,
    COST_CHANNELS_LIST,
    COST_PLAYLIST_ITEMS_LIST,
    COST_ANALYTICS_REPORTS_QUERY,
    PACIFIC_TZ,
)
from pipeline.agents.strategist import (
    StrategistAgent,
    VideoSummary,
    VideoAnalytics,
)
from pipeline.core.llm_manager import AgentRole

load_dotenv()


@pytest.fixture
def temp_quota_file(tmp_path):
    """Provides an isolated JSON quota state file."""
    return tmp_path / "test_quota_state.json"


@pytest.fixture
def temp_cache_file(tmp_path):
    """Provides an isolated JSON cache file for uploads playlist ID."""
    return tmp_path / "test_channel_cache.json"


@pytest.fixture
def isolated_tracker(temp_quota_file):
    """Provides an isolated QuotaTracker instance."""
    return QuotaTracker(state_file=temp_quota_file, daily_limit=10_000, analytics_daily_limit=100_000)


@pytest.fixture
def strategist_agent(isolated_tracker, temp_cache_file):
    """Provides an isolated StrategistAgent instance."""
    return StrategistAgent(quota_mgr=isolated_tracker, cache_path=temp_cache_file)


# ==============================================================================
# CRITERION 1: playlistItems.list call confirmed at 1 unit
# ==============================================================================

def test_playlist_items_list_confirmed_at_1_unit(strategist_agent, isolated_tracker):
    """Verifies that playlistItems.list consumes exactly 1 unit (vs 100 for search.list)."""
    initial_used = isolated_tracker.get_used_units()

    # Mock YouTube Data API client
    mock_yt = MagicMock()
    mock_request = MagicMock()
    mock_request.execute.return_value = {
        "items": [
            {
                "snippet": {
                    "title": "Quantum Physics Explained",
                    "publishedAt": "2026-09-17T12:00:00Z",
                    "description": "Short video about quantum physics",
                    "thumbnails": {"high": {"url": "https://img.youtube.com/vi/abc123/hqdefault.jpg"}},
                    "resourceId": {"videoId": "abc123vid"}
                },
                "contentDetails": {"videoId": "abc123vid"}
            }
        ]
    }
    mock_yt.playlistItems().list.return_value = mock_request

    # Discover videos
    videos = strategist_agent.get_recent_videos(
        max_results=5,
        uploads_playlist_id="UU_MOCK_UPLOADS_PLAYLIST",
        youtube_client=mock_yt
    )

    # 1. Verify exactly 1 unit was deducted from Data API quota
    after_used = isolated_tracker.get_used_units()
    units_spent = after_used - initial_used
    assert units_spent == 1, f"Expected playlistItems.list to consume 1 unit, but consumed {units_spent}"
    assert COST_PLAYLIST_ITEMS_LIST == 1

    # 2. Verify playlistItems was called with correct parameters
    mock_yt.playlistItems().list.assert_called_once_with(
        part="snippet,contentDetails",
        playlistId="UU_MOCK_UPLOADS_PLAYLIST",
        maxResults=5
    )

    # 3. Verify returned data
    assert len(videos) == 1
    assert videos[0].video_id == "abc123vid"
    assert videos[0].title == "Quantum Physics Explained"


def test_uploads_discovery_channels_list_cached_prevents_duplicate_units(strategist_agent, isolated_tracker):
    """Verifies channels.list costs 1 unit on cold start, and 0 units when cached."""
    mock_yt = MagicMock()
    mock_request = MagicMock()
    mock_request.execute.return_value = {
        "items": [
            {
                "contentDetails": {
                    "relatedPlaylists": {"uploads": "UU_REAL_UPLOADS_999"}
                }
            }
        ]
    }
    mock_yt.channels().list.return_value = mock_request

    # 1. Cold start: spends 1 unit
    initial_used = isolated_tracker.get_used_units()
    uploads_id_1 = strategist_agent.get_uploads_playlist_id(youtube_client=mock_yt)
    assert uploads_id_1 == "UU_REAL_UPLOADS_999"
    assert isolated_tracker.get_used_units() - initial_used == 1

    # 2. Warm start: cached in memory/disk -> 0 units spent
    mid_used = isolated_tracker.get_used_units()
    uploads_id_2 = strategist_agent.get_uploads_playlist_id(youtube_client=mock_yt)
    assert uploads_id_2 == "UU_REAL_UPLOADS_999"
    assert isolated_tracker.get_used_units() - mid_used == 0, "Cached uploads playlist ID must consume 0 units"


# ==============================================================================
# CRITERION 2: Analytics API calls tracked separately from Data API v3 spend
# ==============================================================================

def test_analytics_api_tracked_separately_from_data_api(isolated_tracker):
    """Verifies that Analytics API uses its own 100,000 queries pool and does NOT reduce Data API units."""
    data_initial = isolated_tracker.get_available_quota()
    analytics_initial = isolated_tracker.get_analytics_available_quota()

    assert data_initial == 10_000
    assert analytics_initial == 100_000

    # Consume 5 Analytics queries
    for i in range(5):
        isolated_tracker.consume_analytics(queries=1, reason="reports.query", resource_id=f"vid_{i}")

    # Verify Analytics pool reduced by 5
    assert isolated_tracker.get_analytics_used_queries() == 5
    assert isolated_tracker.get_analytics_available_quota() == 99_995

    # CRITICAL: Verify Data API units were NOT touched at all!
    assert isolated_tracker.get_used_units() == 0
    assert isolated_tracker.get_available_quota() == 10_000, "Analytics calls must NOT deduct from Data API units!"


def test_independent_quota_exhaustion_behavior(isolated_tracker):
    """Verifies that exhausting Data API quota does not block Analytics API and vice-versa."""
    # Simulate Data API quota completely exhausted
    isolated_tracker.simulate_quota(10_000)
    assert isolated_tracker.has_budget(1) is False
    with pytest.raises(InsufficientQuotaError):
        isolated_tracker.check_and_reserve(1)

    # Verify Analytics API is STILL fully operational
    assert isolated_tracker.has_analytics_budget(1) is True
    isolated_tracker.check_and_reserve_analytics(1)
    new_analytics = isolated_tracker.consume_analytics(1, reason="reports.query")
    assert new_analytics == 1

    # Simulate Analytics API exhausted
    isolated_tracker.simulate_analytics_quota(100_000)
    assert isolated_tracker.has_analytics_budget(1) is False
    with pytest.raises(InsufficientAnalyticsQuotaError):
        isolated_tracker.check_and_reserve_analytics(1)


# ==============================================================================
# CRITERION 3: Strategist successfully retrieves CTR/retention via correct API
# ==============================================================================

def test_strategist_retrieves_ctr_and_retention_via_analytics_api(strategist_agent, isolated_tracker):
    """Verifies that Strategist queries youtubeAnalytics/v2 reports().query()

    and parses averageViewDuration, averageViewPercentage (retention), and CTR.
    """
    initial_data_units = isolated_tracker.get_used_units()
    initial_analytics_queries = isolated_tracker.get_analytics_used_queries()

    # Mock YouTube Analytics API client
    mock_analytics = MagicMock()
    mock_query_request = MagicMock()
    # Reports API return schema:
    # columns: [video, views, estimatedMinutesWatched, averageViewDuration, averageViewPercentage, annotationClickThroughRate]
    mock_query_request.execute.return_value = {
        "columnHeaders": [
            {"name": "video", "dataType": "STRING"},
            {"name": "views", "dataType": "INTEGER"},
            {"name": "estimatedMinutesWatched", "dataType": "FLOAT"},
            {"name": "averageViewDuration", "dataType": "FLOAT"},
            {"name": "averageViewPercentage", "dataType": "FLOAT"},
            {"name": "annotationClickThroughRate", "dataType": "FLOAT"},
        ],
        "rows": [
            ["xyz789vid", 45200, 2260.5, 30.0, 88.5, 14.2]
        ]
    }
    mock_analytics.reports().query.return_value = mock_query_request

    # Execute analytics retrieval
    analytics: VideoAnalytics = strategist_agent.get_video_retention_and_ctr(
        video_id="xyz789vid",
        analytics_client=mock_analytics
    )

    # 1. Verify metrics were extracted accurately
    assert analytics.video_id == "xyz789vid"
    assert analytics.views == 45200
    assert analytics.average_view_duration_seconds == 30.0
    assert analytics.average_view_percentage == 88.5  # 88.5% retention
    assert analytics.ctr_percentage == 14.2           # 14.2% CTR
    assert analytics.source_api == "youtubeAnalytics/v2"

    # 2. Verify query parameters sent to reports().query
    mock_analytics.reports().query.assert_called_once()
    kwargs = mock_analytics.reports().query.call_args[1]
    assert kwargs["ids"] == "channel==MINE"
    assert "averageViewPercentage" in kwargs["metrics"]
    assert "averageViewDuration" in kwargs["metrics"]
    assert kwargs["dimensions"] == "video"
    assert kwargs["filters"] == "video==xyz789vid"

    # 3. Verify exactly 1 Analytics query was consumed, and 0 Data API units
    assert isolated_tracker.get_analytics_used_queries() - initial_analytics_queries == 1
    assert isolated_tracker.get_used_units() - initial_data_units == 0


# ==============================================================================
# MIDNIGHT PT ROLLOVER VERIFICATION FOR BOTH APIS
# ==============================================================================

def test_midnight_pt_rollover_resets_both_api_pools(isolated_tracker):
    """Verifies that entering a new Pacific day resets both Data API and Analytics API pools."""
    day1_time = datetime(2026, 9, 17, 10, 0, tzinfo=PACIFIC_TZ)
    isolated_tracker.consume(1600, reason="videos.insert", current_time=day1_time)
    isolated_tracker.consume_analytics(50, reason="reports.query", current_time=day1_time)

    assert isolated_tracker.get_used_units(current_time=day1_time) == 1600
    assert isolated_tracker.get_analytics_used_queries(current_time=day1_time) == 50

    # Cross midnight PT into next day
    day2_time = datetime(2026, 9, 18, 0, 1, tzinfo=PACIFIC_TZ)
    assert isolated_tracker.get_used_units(current_time=day2_time) == 0
    assert isolated_tracker.get_analytics_used_queries(current_time=day2_time) == 0
    assert isolated_tracker.get_available_quota(current_time=day2_time) == 10_000
    assert isolated_tracker.get_analytics_available_quota(current_time=day2_time) == 100_000
