"""Strategist Agent: YouTube-Only Quota-Optimized Video Discovery & Analytics (Rule 3).

Complies with:
- Rule 3 (QUOTA AWARENESS):
  - Strict 1-unit video discovery: Replaces search.list (100 units) with:
    channels.list (1 unit, get uploads playlist ID) + playlistItems.list (1 unit, list recent videos).
  - Analytics API Split:
    Retrieves CTR, retention (averageViewDuration, averageViewPercentage), and watch time
    via the separate YouTube Analytics API (youtubeAnalytics/v2, reports().query()),
    tracked independently from the Data API v3 10,000-unit budget.
- Rule 2 (NO SILENT CRASHES): Explicit error handling, auth scope validation, and clean fallback structures.
- Rule 14 (LLM MANAGEMENT & MODEL ROUTING): Uses gemini-3.5-flash-lite (AgentRole.STRATEGIST) for scheduling JSON.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import dotenv
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.logger import get_logger
from pipeline.core.llm_manager import llm_manager, AgentRole
from pipeline.core.quota_tracker import (
    quota_tracker,
    QuotaTracker,
    COST_VIDEOS_LIST,
    COST_CHANNELS_LIST,
    COST_PLAYLIST_ITEMS_LIST,
    COST_ANALYTICS_REPORTS_QUERY,
)

logger = get_logger("strategist")

# Cache file for channel uploads playlist ID to save even the 1 unit channels.list call
CACHE_FILE = ROOT_DIR / "pipeline" / "assets_cache" / "channel_metadata_cache.json"


class VideoSummary(BaseModel):
    """Normalized video metadata extracted from playlistItems.list."""
    video_id: str
    title: str
    published_at: str
    description: str = ""
    thumbnail_url: str = ""


class VideoAnalytics(BaseModel):
    """Retention, watch time, and engagement metrics extracted from youtubeAnalytics/v2."""
    video_id: str
    views: int = 0
    estimated_minutes_watched: float = 0.0
    average_view_duration_seconds: float = 0.0
    average_view_percentage: float = 0.0  # Retention percentage
    ctr_percentage: float = 0.0           # Click-through rate percentage
    source_api: str = "youtubeAnalytics/v2"


class OptimalSchedule(BaseModel):
    """Optimal publication schedule contract produced by the Strategist (Rule 14 & T-1hr Watcher)."""
    target_publish_datetime: str = Field(description="ISO 8601 UTC target publication datetime")
    best_publishing_hour_utc: int = Field(default=15, description="Best publishing hour in UTC (0-23)")
    recommended_tags: list[str] = Field(default_factory=list, description="Recommended discovery tags")
    primary_hashtags: list[str] = Field(default_factory=list, description="Primary hashtags")
    estimated_retention_benchmark: Optional[float] = Field(default=None, description="Grounded retention benchmark if historical performance exists")
    pacing_recommendation: str = Field(default="", description="Deprecated field retained for backward compatibility")
    summary: str = Field(default="", description="Concise strategy summary")



class StrategistAgent:
    """Coordinates quota-minimized video discovery and deep performance telemetry."""

    def __init__(
        self,
        quota_mgr: Optional[QuotaTracker] = None,
        cache_path: Path = CACHE_FILE
    ) -> None:
        dotenv.load_dotenv()
        self.quota_tracker = quota_mgr or quota_tracker
        self.cache_path = cache_path
        self._uploads_playlist_id: Optional[str] = self._load_cached_uploads_id()

    def _load_cached_uploads_id(self) -> Optional[str]:
        """Loads cached channel uploads playlist ID if available."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("uploads_playlist_id")
            except Exception as e:
                logger.debug(f"[STRATEGIST] Could not read cache file {self.cache_path}: {e}")
        return None

    def _save_cached_uploads_id(self, uploads_id: str) -> None:
        """Caches channel uploads playlist ID to disk."""
        self._uploads_playlist_id = uploads_id
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump({"uploads_playlist_id": uploads_id, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
        except Exception as e:
            logger.debug(f"[STRATEGIST] Could not write cache file {self.cache_path}: {e}")

    # =========================================================================
    # GOOGLE API CLIENT INITIALIZATION
    # =========================================================================

    def _get_credentials(self) -> Any:
        """Loads Google OAuth credentials from youtube_token.json or client_secrets.json."""
        token_path = ROOT_DIR / "youtube_token.json"
        raw_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "client_secrets.json")
        client_secrets_path = Path(raw_secret)
        if not client_secrets_path.is_absolute():
            client_secrets_path = ROOT_DIR / client_secrets_path

        creds = None
        if token_path.exists():
            from google.oauth2.credentials import Credentials
            try:
                creds = Credentials.from_authorized_user_file(str(token_path))
            except Exception as e:
                logger.warning(f"[STRATEGIST] Failed to load {token_path}: {e}")

        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            try:
                creds.refresh(Request())
                with open(token_path, "w", encoding="utf-8") as tf:
                    tf.write(creds.to_json())
            except Exception as e:
                logger.warning(f"[STRATEGIST] Token refresh failed: {e}")

        return creds

    def get_youtube_data_client(self, credentials: Optional[Any] = None) -> Any:
        """Returns initialized YouTube Data API v3 client."""
        creds = credentials or self._get_credentials()
        from googleapiclient.discovery import build
        if creds:
            return build("youtube", "v3", credentials=creds)
        # Fallback to API Key for read-only endpoints if available
        api_key = os.getenv("YOUTUBE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return build("youtube", "v3", developerKey=api_key)

    def get_youtube_analytics_client(self, credentials: Optional[Any] = None) -> Any:
        """Returns initialized YouTube Analytics API v2 client."""
        creds = credentials or self._get_credentials()
        from googleapiclient.discovery import build
        return build("youtubeAnalytics", "v2", credentials=creds)

    # =========================================================================
    # 1. QUOTA-OPTIMIZED VIDEO DISCOVERY: channels.list (1) + playlistItems.list (1)
    # =========================================================================

    def get_uploads_playlist_id(
        self,
        channel_id: Optional[str] = None,
        youtube_client: Optional[Any] = None,
        dry_run: bool = False
    ) -> str:
        """Retrieves uploads playlist ID via channels.list (1 Data API unit).
        
        Cached locally to eliminate even the 1-unit overhead on subsequent invocations.
        Replaces search.list (100 units).
        """
        if self._uploads_playlist_id and not channel_id:
            logger.debug(f"[STRATEGIST] Using cached uploads playlist ID: {self._uploads_playlist_id} (0 units spent)")
            return self._uploads_playlist_id

        if dry_run:
            mock_id = f"UU{channel_id or 'MINE_UPLOADS_123'}"
            self._save_cached_uploads_id(mock_id)
            return mock_id

        # 1. Check budget for channels.list (1 unit)
        self.quota_tracker.check_and_reserve(COST_CHANNELS_LIST)

        client = youtube_client or self.get_youtube_data_client()
        request_kwargs: dict[str, Any] = {"part": "contentDetails"}
        if channel_id:
            request_kwargs["id"] = channel_id
        else:
            request_kwargs["mine"] = True

        response = client.channels().list(**request_kwargs).execute()

        # 2. Consume 1 Data API unit
        self.quota_tracker.consume(
            units=COST_CHANNELS_LIST,
            reason="channels.list",
            resource_id=channel_id or "mine"
        )

        items = response.get("items", [])
        if not items:
            raise ValueError(f"No channel found for query: {request_kwargs}")

        uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        self._save_cached_uploads_id(uploads_id)
        logger.info(f"[STRATEGIST_DISCOVERY] channels.list executed (1 Data API unit) -> Uploads Playlist: {uploads_id}")
        return uploads_id

    def get_recent_videos(
        self,
        max_results: int = 10,
        uploads_playlist_id: Optional[str] = None,
        youtube_client: Optional[Any] = None,
        dry_run: bool = False
    ) -> list[VideoSummary]:
        """Discovers recent videos via playlistItems.list (1 Data API unit).
        
        Rule 3 Compliance:
        - NEVER uses search.list (100 units).
        - Uses playlistItems.list (1 unit) to enumerate uploads in chronological order.
        """
        if dry_run:
            logger.info(f"[STRATEGIST_DISCOVERY] (DRY RUN) playlistItems.list simulated (1 Data API unit).")
            self.quota_tracker.consume(COST_PLAYLIST_ITEMS_LIST, "playlistItems.list (dry_run)", "mock_playlist")
            return [
                VideoSummary(
                    video_id=f"MOCK_VID_{i}",
                    title=f"Sample Video {i}: Breakthroughs in Science",
                    published_at=datetime.now(timezone.utc).isoformat(),
                    description="Educational short video.",
                    thumbnail_url="https://example.com/thumb.jpg"
                )
                for i in range(1, max_results + 1)
            ]

        # Resolve uploads playlist ID (1 unit cold start, 0 units warm cache)
        playlist_id = uploads_playlist_id or self.get_uploads_playlist_id(youtube_client=youtube_client)

        # 1. Check Data API budget for playlistItems.list (1 unit)
        self.quota_tracker.check_and_reserve(COST_PLAYLIST_ITEMS_LIST)

        client = youtube_client or self.get_youtube_data_client()
        response = client.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=playlist_id,
            maxResults=min(max_results, 50)
        ).execute()

        # 2. Consume exactly 1 Data API unit
        self.quota_tracker.consume(
            units=COST_PLAYLIST_ITEMS_LIST,
            reason="playlistItems.list",
            resource_id=playlist_id
        )

        videos: list[VideoSummary] = []
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            vid_id = content_details.get("videoId") or snippet.get("resourceId", {}).get("videoId")
            if vid_id:
                thumbnails = snippet.get("thumbnails", {})
                best_thumb = (
                    thumbnails.get("maxres") or
                    thumbnails.get("high") or
                    thumbnails.get("medium") or
                    thumbnails.get("default", {})
                ).get("url", "")

                videos.append(
                    VideoSummary(
                        video_id=vid_id,
                        title=snippet.get("title", ""),
                        published_at=snippet.get("publishedAt", ""),
                        description=snippet.get("description", ""),
                        thumbnail_url=best_thumb
                    )
                )

        logger.info(
            f"[STRATEGIST_DISCOVERY] playlistItems.list executed successfully: "
            f"retrieved {len(videos)} videos (1 Data API unit spent vs 100 units for search.list)."
        )
        return videos

    def get_trending_topics(
        self,
        niche: str = "tech",
        region_code: str = "US",
        max_results: int = 10,
        youtube_client: Optional[Any] = None,
        dry_run: bool = False
    ) -> list[str]:
        """Discovers real trending video topics on YouTube via videos.list(chart='mostPopular') (1 Data API unit).
        
        Rule 3 Compliance:
        - Only consumes 1 Data API unit (vs 100 units for search.list).
        - Supplies live on-platform trend signals directly into Ideator as inspiration.
        """
        category_map = {
            "tech": "28",        # Science & Technology
            "science": "28",     # Science & Technology
            "education": "27",   # Education
            "finance": "25",     # News & Politics / Business
            "history": "27",     # Education
            "psychology": "27",  # Education
            "general": "24",     # Entertainment
        }
        cat_id = category_map.get(niche.lower().strip(), "28")

        fallback_trends = {
            "tech": [
                "Quantum computing breaking cryptographic standards",
                "Solid-state battery commercialization breakthroughs",
                "Undersea fiber optic cable submersibles and sabotage detection",
                "Next-generation nuclear fusion magnetic confinement coils",
                "Autonomous humanoid robotics motor torque developments"
            ],
            "science": [
                "Deep sea hydrothermal vent extremophiles discovery",
                "James Webb space telescope gravitational lensing anomaly",
                "CRISPR gene drive deployment in invasive species control",
                "Superconducting materials at elevated pressures"
            ],
            "finance": [
                "Global central bank digital currency settlement architecture",
                "High-frequency algorithmic flash crash safeguards",
                "Commodity port physical warehousing verification scandals"
            ],
            "history": [
                "Self-healing Roman concrete maritime structural durability",
                "LiDAR subterranean scans revealing ancient river networks",
                "Decryption of charred Herculaneum papyrus scrolls via X-ray CT"
            ],
            "psychology": [
                "Supermarket aisle behavioral architecture manipulation",
                "Dopamine prediction error loops in short-form media",
                "Choice architecture and default effect heuristics"
            ]
        }

        if dry_run:
            logger.info(f"[STRATEGIST_TRENDS] (DRY RUN) videos.list(chart='mostPopular') simulated (1 Data API unit).")
            self.quota_tracker.consume(COST_VIDEOS_LIST, "videos.list(mostPopular) [dry_run]", f"chart:{niche}")
            return fallback_trends.get(niche.lower().strip(), fallback_trends["tech"])[:max_results]

        try:
            # 1. Check Data API budget for videos.list (1 unit)
            self.quota_tracker.check_and_reserve(COST_VIDEOS_LIST)

            client = youtube_client or self.get_youtube_data_client()
            req_params: dict[str, Any] = {
                "part": "snippet",
                "chart": "mostPopular",
                "regionCode": region_code,
                "maxResults": min(max_results, 25)
            }
            if cat_id:
                req_params["videoCategoryId"] = cat_id

            response = client.videos().list(**req_params).execute()

            # 2. Consume 1 Data API unit
            self.quota_tracker.consume(
                units=COST_VIDEOS_LIST,
                reason="videos.list(mostPopular)",
                resource_id=f"chart:{niche}:{region_code}"
            )

            trend_titles = []
            for item in response.get("items", []):
                title = item.get("snippet", {}).get("title", "").strip()
                if title:
                    trend_titles.append(title)

            if trend_titles:
                logger.info(
                    f"[STRATEGIST_TRENDS] Retrieved {len(trend_titles)} live trending topics from YouTube "
                    f"(niche={niche}, category={cat_id}, 1 Data API unit spent)."
                )
                return trend_titles

        except Exception as e:
            logger.warning(
                f"[STRATEGIST_TRENDS] videos.list(mostPopular) failed or unavailable ({e}). "
                "Returning grounded fallback trend candidates."
            )

        return fallback_trends.get(niche.lower().strip(), fallback_trends["tech"])[:max_results]

    # =========================================================================
    # 2. RETENTION & CTR: youtubeAnalytics/v2 (INDEPENDENT POOL)
    # =========================================================================

    def get_video_retention_and_ctr(
        self,
        video_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        analytics_client: Optional[Any] = None,
        dry_run: bool = False
    ) -> VideoAnalytics:
        """Retrieves CTR, retention %, and watch duration via YouTube Analytics API v2.
        
        Important:
        - CTR and average-view-duration are NOT available in Data API v3.
        - Calls youtubeAnalytics/v2 reports().query() with independent 100,000 queries quota.
        - Does NOT consume from the 10,000-unit Data API pool.
        """
        # Date window defaults to last 30 days
        now_dt = datetime.now(timezone.utc)
        if not end_date:
            end_date = now_dt.strftime("%Y-%m-%d")
        if not start_date:
            start_date = (now_dt - timedelta(days=30)).strftime("%Y-%m-%d")

        if dry_run:
            logger.info(f"[STRATEGIST_ANALYTICS] (DRY RUN) reports().query simulated (1 Analytics query).")
            self.quota_tracker.consume_analytics(
                queries=COST_ANALYTICS_REPORTS_QUERY,
                reason="reports.query (dry_run)",
                resource_id=video_id
            )
            return VideoAnalytics(
                video_id=video_id,
                views=12500,
                estimated_minutes_watched=625.0,
                average_view_duration_seconds=30.0,
                average_view_percentage=85.5,  # 85.5% short-form retention
                ctr_percentage=12.4,           # 12.4% CTR
                source_api="youtubeAnalytics/v2"
            )

        # 1. Check independent Analytics API budget (1 query)
        self.quota_tracker.check_and_reserve_analytics(COST_ANALYTICS_REPORTS_QUERY)

        try:
            client = analytics_client or self.get_youtube_analytics_client()
            response = client.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,annotationClickThroughRate",
                dimensions="video",
                filters=f"video=={video_id}"
            ).execute()

            # 2. Record 1 query into independent Analytics quota pool
            self.quota_tracker.consume_analytics(
                queries=COST_ANALYTICS_REPORTS_QUERY,
                reason="reports.query",
                resource_id=video_id
            )

            rows = response.get("rows", [])
            if rows:
                # Row mapping: [video_id, views, estimatedMinutesWatched, averageViewDuration, averageViewPercentage, annotationClickThroughRate]
                row = rows[0]
                views = int(row[1]) if len(row) > 1 else 0
                est_minutes = float(row[2]) if len(row) > 2 else 0.0
                avg_duration = float(row[3]) if len(row) > 3 else 0.0
                avg_pct = float(row[4]) if len(row) > 4 else 0.0
                ctr = float(row[5]) if len(row) > 5 else 0.0

                analytics = VideoAnalytics(
                    video_id=video_id,
                    views=views,
                    estimated_minutes_watched=est_minutes,
                    average_view_duration_seconds=avg_duration,
                    average_view_percentage=avg_pct,
                    ctr_percentage=ctr,
                    source_api="youtubeAnalytics/v2"
                )
            else:
                logger.info(f"[STRATEGIST_ANALYTICS] No analytics data returned yet for video '{video_id}' (new upload).")
                analytics = VideoAnalytics(video_id=video_id, source_api="youtubeAnalytics/v2")

            logger.info(
                f"[STRATEGIST_ANALYTICS] youtubeAnalytics/v2 retrieved retention: "
                f"avgDuration={analytics.average_view_duration_seconds:.1f}s, "
                f"retention={analytics.average_view_percentage:.1f}%, "
                f"CTR={analytics.ctr_percentage:.1f}% (Data API units untouched)."
            )
            return analytics

        except Exception as e:
            logger.warning(
                f"[STRATEGIST_ANALYTICS] Analytics API call failed ({e}). "
                "Returning zeroed analytics structure per Rule 2 safe degradation."
            )
            return VideoAnalytics(video_id=video_id, source_api="youtubeAnalytics/v2 (error_fallback)")

    # =========================================================================
    # 3. METADATA & SCHEDULING STRATEGY (BUDGET TIER: gemini-3.5-flash-lite)
    # =========================================================================

    def _get_historical_channel_context(self) -> tuple[str, Optional[float]]:
        """Retrieves historical channel performance to ground scheduling and benchmarks."""
        try:
            recent_vids = self.get_recent_videos(max_results=3, dry_run=False)
            if recent_vids:
                retentions = []
                views_list = []
                for v in recent_vids:
                    an = self.get_video_analytics(v.video_id, dry_run=False)
                    if an.views > 0:
                        views_list.append(an.views)
                    if an.average_view_percentage > 0:
                        retentions.append(an.average_view_percentage)
                if retentions:
                    avg_ret = sum(retentions) / len(retentions)
                    avg_views = sum(views_list) / len(views_list) if views_list else 0
                    ctx_str = (
                        f"HISTORICAL CHANNEL PERFORMANCE (from YouTube Analytics API v2):\n"
                        f"- Average Retention across recent uploads: {avg_ret:.1f}%\n"
                        f"- Average Views: {int(avg_views)}"
                    )
                    return ctx_str, avg_ret
        except Exception as e:
            logger.debug(f"[STRATEGIST] Live YouTube analytics unavailable: {e}")

        # Local telemetry check
        try:
            from pipeline.dashboard.database import get_db_connection, get_db_path
            with get_db_connection(get_db_path()) as conn:
                rows = conn.execute(
                    "SELECT duration_seconds FROM telemetry WHERE status='COMPLETED' ORDER BY id DESC LIMIT 5"
                ).fetchall()
                if rows:
                    return f"HISTORICAL TELEMETRY: {len(rows)} studio-rendered videos on record.", None
        except Exception:
            pass

        return "HISTORICAL CHANNEL PERFORMANCE: No historical analytics data available yet (new channel / cold start).", None

    def generate_scheduling_and_tags(
        self,
        topic: str,
        niche: str = "tech",
        target_audience: str = "general curiosity"
    ) -> dict[str, Any]:
        """Generates optimal release timing and hashtag strategy grounded in channel context.
        
        Complies with Rule 14: Routed to gemini-3.5-flash-lite (AgentRole.STRATEGIST).
        """
        hist_context, grounded_retention = self._get_historical_channel_context()

        retention_instruction = (
            f"Ground 'estimated_retention_benchmark' in the historical channel average ({grounded_retention:.1f}%).\n"
            if grounded_retention is not None
            else "Do not fabricate an ungrounded retention benchmark; set 'estimated_retention_benchmark' to null.\n"
        )

        prompt = (
            f"You are a YouTube Shorts publishing strategist.\n"
            f"Topic: '{topic}'\n"
            f"Niche: '{niche}'\n"
            f"Target Audience: '{target_audience}'\n"
            f"{hist_context}\n\n"
            "STRATEGY INSTRUCTIONS:\n"
            "- Consider the target audience's likely timezone and typical short-form consumption peak hours before converting your recommendation to best_publishing_hour_utc.\n"
            f"- {retention_instruction}"
            "Return a JSON object with:\n"
            "1. 'best_publishing_hour_utc': integer (0 to 23)\n"
            "2. 'recommended_tags': list of 5-8 search tags\n"
            "3. 'primary_hashtags': list of 3 hashtags\n"
            "4. 'estimated_retention_benchmark': float percentage or null if ungrounded"
        )

        try:
            raw_text, _ = llm_manager.generate_content(
                role=AgentRole.STRATEGIST,
                contents=prompt,
                temperature=0.4
            )
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception as e:
            logger.warning(f"[STRATEGIST] LLM scheduling strategy generation failed: {e}. Using deterministic defaults.")
            return {
                "best_publishing_hour_utc": 15,  # 3 PM UTC
                "recommended_tags": [f"{niche}", "shorts", "technology", "facts", "curiosity"],
                "primary_hashtags": ["#shorts", f"#{niche}", "#viral"],
                "estimated_retention_benchmark": grounded_retention
            }

    def compute_optimal_schedule(
        self,
        topic: str,
        niche: str = "tech",
        target_audience: str = "general curiosity",
        target_datetime: Optional[datetime | str] = None,
    ) -> OptimalSchedule:
        """Computes optimal schedule metrics and determines target publication datetime."""
        raw_meta = self.generate_scheduling_and_tags(topic=topic, niche=niche, target_audience=target_audience)
        best_hour = int(raw_meta.get("best_publishing_hour_utc", 15))

        if target_datetime is not None:
            if isinstance(target_datetime, str):
                target_iso = target_datetime
            else:
                if target_datetime.tzinfo is None:
                    target_datetime = target_datetime.replace(tzinfo=timezone.utc)
                target_iso = target_datetime.isoformat()
        else:
            now = datetime.now(timezone.utc)
            candidate = now.replace(hour=best_hour, minute=0, second=0, microsecond=0)
            if candidate <= now + timedelta(hours=1):
                candidate += timedelta(days=1)
            target_iso = candidate.isoformat()

        retention_raw = raw_meta.get("estimated_retention_benchmark")
        retention_val: Optional[float] = None
        if retention_raw is not None:
            try:
                if isinstance(retention_raw, str):
                    retention_val = float(retention_raw.replace("%", "").strip())
                else:
                    retention_val = float(retention_raw)
            except Exception:
                retention_val = None

        summary_text = (
            f"Scheduled for {target_iso} (Hour {best_hour}:00 UTC). "
            f"Target tags: {', '.join(raw_meta.get('recommended_tags', [])[:4])}. "
        )
        if retention_val is not None:
            summary_text += f"Retention target: {retention_val:.1f}%."
        else:
            summary_text += "Retention target: baseline pending channel history."

        return OptimalSchedule(
            target_publish_datetime=target_iso,
            best_publishing_hour_utc=best_hour,
            recommended_tags=raw_meta.get("recommended_tags", []),
            primary_hashtags=raw_meta.get("primary_hashtags", []),
            estimated_retention_benchmark=retention_val,
            pacing_recommendation="",
            summary=summary_text
        )

    def schedule_video(
        self,
        job_id_or_pk: int | str,
        topic: Optional[str] = None,
        niche: str = "tech",
        target_datetime: Optional[datetime | str] = None,
        db_path: Path | str | None = None
    ) -> OptimalSchedule:
        """Computes optimal schedule and persists it directly to SQLite DB per approved video."""
        from pipeline.dashboard.database import persist_video_schedule, get_job

        clean_topic = topic
        if not clean_topic:
            job = get_job(str(job_id_or_pk), db_path=db_path)
            clean_topic = job.get("topic") if job else "Untitled Video"

        schedule = self.compute_optimal_schedule(
            topic=clean_topic or "Untitled Video",
            niche=niche,
            target_datetime=target_datetime
        )

        persist_video_schedule(
            job_id_or_pk=job_id_or_pk,
            target_publish_datetime=schedule.target_publish_datetime,
            summary=schedule.summary,
            db_path=db_path
        )
        logger.info(
            f"[STRATEGIST] Persisted schedule for job {job_id_or_pk}: {schedule.target_publish_datetime} (pre_notice_sent=0)"
        )
        return schedule


# Global default strategist instance
strategist = StrategistAgent()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strategist Agent CLI")
    parser.add_argument("--status", action="store_true", help="Print quota and uploads playlist ID")
    parser.add_argument("--recent", action="store_true", help="List recent videos via playlistItems.list (1 unit)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate API calls")
    args = parser.parse_args()

    agent = StrategistAgent()
    if args.status or len(sys.argv) == 1:
        print("\n--- YouTube Distribution Quota Status ---")
        print(f"Data API v3 Daily Limit:      {agent.quota_tracker.daily_limit} units")
        print(f"Data API v3 Units Spent Today: {agent.quota_tracker.get_used_units()} units")
        print(f"Data API v3 Units Available:   {agent.quota_tracker.get_available_quota()} units")
        print(f"Analytics API Daily Limit:    {agent.quota_tracker.analytics_daily_limit} queries")
        print(f"Analytics API Queries Spent:  {agent.quota_tracker.get_analytics_used_queries()} queries")
        print(f"Cached Uploads Playlist ID:   {agent._uploads_playlist_id or 'None (uncached)'}")
        print("-" * 50 + "\n")

    if args.recent:
        print("Fetching recent videos via playlistItems.list (1 Data API unit)...")
        vids = agent.get_recent_videos(max_results=5, dry_run=args.dry_run)
        for v in vids:
            print(f"- [{v.video_id}] {v.title} ({v.published_at})")
