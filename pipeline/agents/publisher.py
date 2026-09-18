"""Publisher Agent.

Handles quota-gated YouTube Shorts publishing and Instagram-hosting-gated Reels publishing.
Subject to:
- Rule 2 (No Silent Crashes: explicit error handling, retries, and return states)
- Rule 3 (Quota Awareness: unit-based YouTube budget check prior to call; 1600 units spent on success)
- Rule 6 (Instagram Needs a Public URL: short-lived public hosting via B2 with guaranteed cleanup)
- Rule 8 (Bounded Retries: 180s timeout, paced non-tight-loop polling)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from pipeline.core.b2_hosting import b2_bridge, B2HostingBridge
from pipeline.core.quota_tracker import quota_tracker, QuotaTracker, YOUTUBE_VIDEO_UPLOAD_COST

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv()

logger = logging.getLogger("publisher")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [publisher] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Standard YouTube API scopes for full channel management and verification
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/userinfo.email"
]


class PublishStatus(str, Enum):
    PUBLISHED = "PUBLISHED"
    DEFERRED = "DEFERRED"
    FAILED = "FAILED"


class PublishResult(BaseModel):
    """Structured publication outcome returned by publisher agent."""
    platform: str
    status: PublishStatus
    post_id: Optional[str] = None
    url: Optional[str] = None
    message: str = ""
    visibility: Optional[str] = None
    is_restricted: bool = False
    error_code: Optional[str] = None
    channel_info: Optional[dict[str, Any]] = None
    units_spent: int = 0


def classify_youtube_error(exc: Exception) -> tuple[str, str]:
    """Classifies YouTube API and OAuth exceptions into actionable machine error codes and descriptions."""
    err_str = str(exc)
    lower_err = err_str.lower()
    
    if "token expired" in lower_err or "invalid_grant" in lower_err or "revoked" in lower_err or "refresh" in lower_err and "fail" in lower_err:
        return "YOUTUBE_OAUTH_EXPIRED", f"YouTube OAuth token is expired or revoked ({err_str}). Re-authentication required via client_secrets.json."
    if "insufficient authentication scopes" in lower_err or "insufficientpermissions" in lower_err:
        return "YOUTUBE_INSUFFICIENT_SCOPES", f"OAuth token lacks required permissions ({err_str}). Re-authentication with expanded scopes required."
    if "quotaexceeded" in lower_err or "uploadlimitexceeded" in lower_err or "daily limit" in lower_err:
        return "YOUTUBE_QUOTA_EXHAUSTED", f"YouTube API quota limit exceeded for today ({err_str}). Publishing must be deferred."
    if "unverified" in lower_err or "testing" in lower_err:
        return "UNVERIFIED_PROJECT_RESTRICTION", f"Google Cloud Project is unverified or in testing mode ({err_str})."
    if "timed out" in lower_err or "timeout" in lower_err or "connection" in lower_err:
        return "YOUTUBE_NETWORK_ERROR", f"Network error during YouTube upload ({err_str}). Check internet connectivity."
    
    return "YOUTUBE_API_ERROR", f"YouTube API encountered an error: {err_str}"


def get_authenticated_channel_info(creds: Any) -> dict[str, Any]:
    """Inspects authenticated Google account and YouTube channel metadata."""
    info = {"authenticated": False, "email": None, "channel_id": None, "channel_title": None, "custom_url": None}
    if not creds:
        return info

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", credentials=creds)
        ch_res = youtube.channels().list(part="snippet", mine=True).execute()
        items = ch_res.get("items", [])
        if items:
            ch = items[0]
            info["authenticated"] = True
            info["channel_id"] = ch.get("id")
            info["channel_title"] = ch.get("snippet", {}).get("title")
            info["custom_url"] = ch.get("snippet", {}).get("customUrl")
    except Exception as e:
        logger.debug(f"[YOUTUBE_CHANNEL_CHECK] Could not query channel metadata: {e}")

    try:
        from googleapiclient.discovery import build
        oauth2_client = build("oauth2", "v2", credentials=creds)
        user_info = oauth2_client.userinfo().get().execute()
        info["email"] = user_info.get("email")
    except Exception as e:
        logger.debug(f"[YOUTUBE_USER_CHECK] Could not query userinfo email: {e}")

    return info


class PublisherAgent:
    """Multi-platform publishing coordinator for YouTube Shorts and Instagram Reels."""

    def __init__(
        self,
        quota_mgr: Optional[QuotaTracker] = None,
        hosting_bridge: Optional[B2HostingBridge] = None,
    ) -> None:
        self.quota_tracker = quota_mgr or quota_tracker
        self.hosting_bridge = hosting_bridge or b2_bridge

    # =========================================================================
    # YOUTUBE SHORTS PUBLISHING (RULE 3: QUOTA AWARE)
    # =========================================================================

    def publish_to_youtube(
        self,
        video_path: Path | str,
        title: str,
        description: str = "",
        tags: Optional[list[str]] = None,
        privacy_status: str = "public",
        dry_run: bool = False,
    ) -> PublishResult:
        """Publishes a video to YouTube Shorts with unit-based quota gating.
        
        Before invoking videos.insert:
        - Checks quota_tracker.has_budget(1600).
        - If insufficient, returns PublishStatus.DEFERRED cleanly without failing.
        - On successful upload, records 1600 units spent.
        - Inspects returned visibility; detects and classifies unverified OAuth app restrictions.
        """
        path = Path(video_path)
        if not path.exists():
            return PublishResult(
                platform="youtube",
                status=PublishStatus.FAILED,
                error_code="VIDEO_FILE_NOT_FOUND",
                message=f"Video file does not exist: {path}"
            )

        # 1. Check Quota Budget (Rule 3)
        available_quota = self.quota_tracker.get_available_quota()
        if not self.quota_tracker.has_budget(YOUTUBE_VIDEO_UPLOAD_COST):
            msg = (
                f"YouTube quota budget exhausted for today: requires {YOUTUBE_VIDEO_UPLOAD_COST} units, "
                f"but only {available_quota} units available. Deferring upload to next quota reset window."
            )
            logger.warning(f"[QUOTA_DEFERRED] {msg}")
            try:
                from pipeline.core.notifier import notifier
                notifier.alert(f"[YOUTUBE_QUOTA_DEFERRED] {msg}", level="WARNING")
            except Exception:
                pass

            return PublishResult(
                platform="youtube",
                status=PublishStatus.DEFERRED,
                error_code="QUOTA_EXHAUSTED",
                message=msg,
                units_spent=0
            )

        logger.info(
            f"Quota check passed: {available_quota} units available "
            f"(requires {YOUTUBE_VIDEO_UPLOAD_COST} units for videos.insert)."
        )

        if dry_run:
            logger.info(f"[DRY_RUN] Simulating YouTube upload (target visibility: {privacy_status}). Quota will NOT be spent.")
            return PublishResult(
                platform="youtube",
                status=PublishStatus.PUBLISHED,
                post_id="DRY_RUN_YT_123",
                url="https://youtube.com/shorts/DRY_RUN_YT_123",
                message="Dry run YouTube upload succeeded.",
                visibility=privacy_status,
                is_restricted=False,
                units_spent=0
            )

        # 2. Perform Real or OAuth Upload with atomic quota reservation (GAP 14)
        try:
            with self.quota_tracker.atomic_reservation(
                units=YOUTUBE_VIDEO_UPLOAD_COST,
                operation="videos.insert"
            ):
                exec_res = self._execute_youtube_upload(
                    video_path=path,
                    title=title,
                    description=description,
                    tags=tags or ["#shorts"],
                    privacy_status=privacy_status
                )
                if isinstance(exec_res, tuple) and len(exec_res) == 2:
                    upload_response, channel_meta = exec_res
                else:
                    upload_response, channel_meta = exec_res, {}
                video_id = upload_response.get("id", "UNKNOWN_ID")
                actual_privacy = upload_response.get("status", {}).get("privacyStatus", privacy_status)
                
                # 3. Detect & Surface Unverified Project Restriction
                is_restricted = False
                error_code = None
                if privacy_status.lower() == "public" and actual_privacy.lower() in ("private", "unlisted"):
                    is_restricted = True
                    error_code = "UNVERIFIED_PROJECT_RESTRICTION"
                    warning_msg = (
                        f"[UNVERIFIED_PROJECT_RESTRICTION] Video '{video_id}' was forced to '{actual_privacy}' (Locked Private) "
                        f"instead of requested '{privacy_status}'. Google enforces private-only uploads for OAuth apps in "
                        f"'Testing' mode or without full verification. The video is safely stored on the channel but cannot "
                        f"be viewed publicly until approved or opened in YouTube Studio."
                    )
                    logger.warning(warning_msg)
                    try:
                        from pipeline.core.notifier import notifier
                        notifier.alert(
                            f"[UNVERIFIED_PROJECT_RESTRICTION] Video https://youtube.com/shorts/{video_id} locked to '{actual_privacy}' by Google policy. Configure test users in Google Cloud Console.",
                            level="WARNING",
                            extra={"video_id": video_id, "actual_privacy": actual_privacy}
                        )
                    except Exception:
                        pass

            video_url = f"https://youtube.com/shorts/{video_id}"
            logger.info(f"[YOUTUBE_UPLOAD_SUCCESS] Watch URL: {video_url} (Requested: {privacy_status}, Actual: {actual_privacy})")
            if channel_meta.get("channel_title") or channel_meta.get("email"):
                logger.info(f"[YOUTUBE_CHANNEL_INFO] Uploaded to: {channel_meta.get('channel_title')} ({channel_meta.get('email')})")

            return PublishResult(
                platform="youtube",
                status=PublishStatus.PUBLISHED,
                post_id=video_id,
                url=video_url,
                message="Video successfully uploaded to YouTube Shorts.",
                visibility=actual_privacy,
                is_restricted=is_restricted,
                error_code=error_code,
                channel_info=channel_meta,
                units_spent=YOUTUBE_VIDEO_UPLOAD_COST
            )
        except Exception as e:
            err_code, err_desc = classify_youtube_error(e)
            logger.error(f"[YOUTUBE_UPLOAD_FAILED] Error Code: [{err_code}] - {err_desc}", exc_info=True)
            try:
                from pipeline.core.notifier import notifier
                notifier.alert(
                    f"[{err_code}] YouTube upload failed: {err_desc}",
                    level="ERROR",
                    extra={"error_code": err_code, "raw_error": str(e)}
                )
            except Exception:
                pass

            return PublishResult(
                platform="youtube",
                status=PublishStatus.FAILED,
                error_code=err_code,
                message=err_desc,
                units_spent=0
            )

    def _execute_youtube_upload(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str],
        privacy_status: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Handles Google API Client initialization and resumable video upload."""
        # Attempt to load token / client secrets
        token_path = ROOT_DIR / "youtube_token.json"
        raw_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "client_secrets.json")
        client_secrets_path = Path(raw_secret)
        if not client_secrets_path.is_absolute():
            client_secrets_path = ROOT_DIR / client_secrets_path

        creds = None
        if token_path.exists():
            from google.oauth2.credentials import Credentials
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), YOUTUBE_SCOPES)
            except Exception as e:
                logger.warning(f"Failed to load credentials from {token_path}: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    from google.auth.transport.requests import Request
                    creds.refresh(Request())
                    with open(token_path, "w", encoding="utf-8") as token_file:
                        token_file.write(creds.to_json())
                    logger.info("[YOUTUBE_OAUTH] Token refreshed successfully.")
                except Exception as ref_err:
                    err_msg = f"[YOUTUBE_OAUTH_EXPIRED] Token refresh failed: {ref_err}"
                    logger.error(err_msg)
                    raise RuntimeError(err_msg) from ref_err
            elif client_secrets_path.exists():
                from google_auth_oauthlib.flow import InstalledAppFlow
                logger.info(f"Initiating YouTube OAuth consent flow using {client_secrets_path.name} with expanded scopes...")
                flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), YOUTUBE_SCOPES)
                creds = flow.run_local_server(port=0, open_browser=True)
                with open(token_path, "w", encoding="utf-8") as token_file:
                    token_file.write(creds.to_json())
                logger.info(f"YouTube OAuth token generated and saved to {token_path.name}")
            else:
                raise FileNotFoundError(
                    f"YouTube client secrets file not found at {client_secrets_path} and no valid {token_path} present."
                )

        channel_info = get_authenticated_channel_info(creds)
        if channel_info.get("channel_title"):
            logger.info(f"[YOUTUBE_TARGET_CHANNEL] Channel: {channel_info['channel_title']} (ID: {channel_info.get('channel_id')}, Account: {channel_info.get('email')})")

        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags,
                "categoryId": "28"  # Science & Technology
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False
            }
        }

        media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        
        response = None
        upload_start = time.monotonic()
        max_upload_time = 300.0  # 5-minute hard bound
        chunk_count = 0
        max_chunks = 100

        while response is None:
            chunk_count += 1
            if (time.monotonic() - upload_start) > max_upload_time:
                raise TimeoutError(f"YouTube resumable upload timed out after {max_upload_time}s.")
            if chunk_count > max_chunks:
                raise RuntimeError(f"YouTube resumable upload exceeded maximum chunk attempts ({max_chunks}).")

            status, response = request.next_chunk()
            if status:
                logger.info(f"YouTube upload progress: {int(status.progress() * 100)}%")

        return response, channel_info

    # =========================================================================
    # INSTAGRAM REELS PUBLISHING (RULE 6: EPHEMERAL PUBLIC HOSTING)
    # =========================================================================

    def publish_to_instagram(
        self,
        video_path: Path | str,
        caption: str = "",
        access_token: Optional[str] = None,
        account_id: Optional[str] = None,
        timeout_seconds: float = 180.0,
        dry_run: bool = False,
    ) -> PublishResult:
        """Publishes a video to Instagram Reels via the two-phase Graph API flow.
        
        1. Hosts .mp4 on Backblaze B2, obtaining a short-lived pre-signed HTTPS URL.
        2. Creates Reels media container via POST /{account_id}/media with video_url.
        3. Polls container status with backoff until FINISHED (max 180s, no tight loop).
        4. Publishes media container via POST /{account_id}/media_publish.
        5. Purges temporary video from B2 in guaranteed finally block.
        """
        path = Path(video_path)
        if not path.exists():
            return PublishResult(
                platform="instagram",
                status=PublishStatus.FAILED,
                message=f"Video file does not exist: {path}"
            )

        token = access_token or os.getenv("IG_ACCESS_TOKEN")
        acct_id = account_id or os.getenv("IG_ACCOUNT_ID")

        if dry_run:
            logger.info("[DRY_RUN] Simulating Instagram Reels publishing.")
            return PublishResult(
                platform="instagram",
                status=PublishStatus.PUBLISHED,
                post_id="DRY_RUN_IG_456",
                url="https://instagram.com/reel/DRY_RUN_IG_456",
                message="Dry run Instagram Reels publish succeeded.",
                is_restricted=False
            )

        if not token or not acct_id:
            logger.warning("Instagram credentials missing (IG_ACCESS_TOKEN or IG_ACCOUNT_ID).")
            return PublishResult(
                platform="instagram",
                status=PublishStatus.FAILED,
                message="Missing IG_ACCESS_TOKEN or IG_ACCOUNT_ID in environment."
            )

        if not self.hosting_bridge.is_configured():
            return PublishResult(
                platform="instagram",
                status=PublishStatus.FAILED,
                message="Ephemeral hosting bridge (Backblaze B2) is not configured in .env."
            )

        logger.info(f"Initiating ephemeral public hosting for Instagram: {path.name}")

        # Context manager guarantees cleanup on both success and failure (Rule 6)
        try:
            with self.hosting_bridge.temporary_public_url(path, expires_in_seconds=900) as (public_url, object_key):
                logger.info(f"Ephemeral public URL established. Sending container request to Meta Graph API...")

                # Phase 1: Create Media Container
                container_id = self._create_instagram_container(
                    account_id=acct_id,
                    access_token=token,
                    video_url=public_url,
                    caption=caption
                )
                logger.info(f"Instagram media container created: {container_id}. Polling processing status...")

                # Phase 2: Poll Container Status (Rule 8: Bounded loop, no tight loop)
                self._poll_container_status(
                    container_id=container_id,
                    access_token=token,
                    timeout_seconds=timeout_seconds
                )
                logger.info(f"Instagram media container {container_id} is FINISHED. Publishing...")

                # Phase 3: Publish Container
                media_id = self._publish_instagram_container(
                    account_id=acct_id,
                    access_token=token,
                    container_id=container_id
                )
                logger.info(f"Instagram Reel successfully published! Post ID: {media_id}")

                return PublishResult(
                    platform="instagram",
                    status=PublishStatus.PUBLISHED,
                    post_id=media_id,
                    url=f"https://www.instagram.com/reel/{media_id}/",
                    message="Reel published successfully to Instagram.",
                    is_restricted=False
                )
        except Exception as e:
            logger.error(f"Instagram Reels publish failed: {e}", exc_info=True)
            err_str = str(e)
            is_token_issue = any(k in err_str.lower() for k in ["oauthexception", "190", "expired", "session has expired", "error validating access token", "invalid oauth"])
            if is_token_issue:
                try:
                    from pipeline.core.notifier import notifier
                    notifier.alert(
                        f"[INSTAGRAM_TOKEN_EXPIRED] Instagram Graph API access token has expired or is invalid: {err_str}. "
                        f"Reels publishing is halted. Please generate a new 60-day token in Meta Developer Portal.",
                        level="ERROR",
                        extra={"platform": "instagram", "error": err_str}
                    )
                except Exception:
                    pass
            return PublishResult(
                platform="instagram",
                status=PublishStatus.FAILED,
                message=f"Instagram publishing encountered an error: {str(e)}"
            )

    def check_instagram_token_health(self, access_token: Optional[str] = None) -> Dict[str, Any]:
        """Proactively checks the validity and expiration window of the configured Instagram token."""
        token = access_token or os.getenv("IG_ACCESS_TOKEN")
        if not token:
            return {"valid": False, "reason": "No IG_ACCESS_TOKEN configured in environment."}
        
        try:
            endpoint = "https://graph.facebook.com/v19.0/debug_token"
            params = {"input_token": token, "access_token": token}
            res = requests.get(endpoint, params=params, timeout=10)
            data = res.json()
            if res.status_code != 200 or "data" not in data:
                me_res = requests.get("https://graph.facebook.com/v19.0/me", params={"access_token": token}, timeout=10)
                if me_res.status_code == 200:
                    return {"valid": True, "details": me_res.json()}
                err_msg = me_res.json().get("error", {}).get("message", "Token invalid")
                from pipeline.core.notifier import notifier
                notifier.alert(
                    f"[INSTAGRAM_TOKEN_EXPIRED] Instagram token health check failed: {err_msg}",
                    level="ERROR",
                    extra={"platform": "instagram", "error": err_msg}
                )
                return {"valid": False, "error": err_msg}

            token_data = data.get("data", {})
            is_valid = token_data.get("is_valid", False)
            expires_at = token_data.get("data_access_expires_at") or token_data.get("expires_at")
            
            if not is_valid:
                from pipeline.core.notifier import notifier
                notifier.alert(
                    "[INSTAGRAM_TOKEN_EXPIRED] Meta reports Instagram access token is no longer valid. Immediate manual refresh required.",
                    level="ERROR",
                    extra={"token_data": token_data}
                )
                return {"valid": False, "reason": "Token marked invalid by Meta"}

            days_remaining = None
            if expires_at and isinstance(expires_at, (int, float)) and expires_at > 0:
                now_ts = datetime.now(timezone.utc).timestamp()
                diff_sec = expires_at - now_ts
                days_remaining = round(diff_sec / 86400, 1)
                if days_remaining <= 7:
                    from pipeline.core.notifier import notifier
                    notifier.alert(
                        f"[INSTAGRAM_TOKEN_EXPIRING] Instagram access token expires in {days_remaining} days. "
                        f"Please refresh the 60-day token soon to avoid publishing interruptions.",
                        level="WARNING",
                        extra={"days_remaining": days_remaining}
                    )
            
            return {
                "valid": True,
                "days_remaining": days_remaining,
                "scopes": token_data.get("scopes", [])
            }
        except Exception as e:
            logger.warning(f"Failed to check Instagram token health: {e}")
            return {"valid": False, "error": str(e)}

    def _create_instagram_container(
        self,
        account_id: str,
        access_token: str,
        video_url: str,
        caption: str
    ) -> str:
        """Phase 1: Requests creation of a Reels media container."""
        endpoint = f"https://graph.facebook.com/v19.0/{account_id}/media"
        payload = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "access_token": access_token
        }
        res = requests.post(endpoint, data=payload, timeout=30)
        data = res.json()
        if res.status_code != 200 or "id" not in data:
            error_msg = data.get("error", {}).get("message", res.text)
            raise RuntimeError(f"Failed to create Instagram container: {error_msg}")
        return data["id"]

    def _poll_container_status(
        self,
        container_id: str,
        access_token: str,
        timeout_seconds: float = 180.0,
        initial_poll_interval: float = 5.0,
        max_poll_interval: float = 10.0,
    ) -> None:
        """Phase 2: Polls container status with paced backoff up to timeout_seconds.
        
        Status codes:
        - FINISHED: Container ingested and ready for publishing.
        - IN_PROGRESS: Ingestion actively underway.
        - ERROR / EXPIRED: Ingestion failed.
        """
        endpoint = f"https://graph.facebook.com/v19.0/{container_id}"
        params = {"fields": "status_code,status", "access_token": access_token}
        
        start_time = time.monotonic()
        poll_interval = initial_poll_interval
        attempts = 0

        while (time.monotonic() - start_time) < timeout_seconds:
            attempts += 1
            # Sane non-tight loop pacing (Rule 8)
            time.sleep(poll_interval)

            try:
                res = requests.get(endpoint, params=params, timeout=15)
                data = res.json()
            except Exception as e:
                logger.warning(f"Transient error polling container status ({e}); retrying...")
                continue

            status_code = data.get("status_code", "").upper()
            logger.info(
                f"[POLL_STATUS] Container {container_id} attempt {attempts}: "
                f"status_code='{status_code}' (Elapsed: {time.monotonic() - start_time:.1f}s)"
            )

            if status_code == "FINISHED":
                return
            elif status_code in ("ERROR", "EXPIRED"):
                raise RuntimeError(
                    f"Instagram container {container_id} failed with status '{status_code}': "
                    f"{data.get('status', 'No error details provided by Meta')}"
                )
            
            # Step up poll interval gently to prevent aggressive polling
            poll_interval = min(poll_interval + 1.5, max_poll_interval)

        raise TimeoutError(
            f"Timed out after {timeout_seconds}s waiting for Instagram container {container_id} to report FINISHED."
        )

    def _publish_instagram_container(
        self,
        account_id: str,
        access_token: str,
        container_id: str
    ) -> str:
        """Phase 3: Publishes the validated media container."""
        endpoint = f"https://graph.facebook.com/v19.0/{account_id}/media_publish"
        payload = {
            "creation_id": container_id,
            "access_token": access_token
        }
        res = requests.post(endpoint, data=payload, timeout=30)
        data = res.json()
        if res.status_code != 200 or "id" not in data:
            error_msg = data.get("error", {}).get("message", res.text)
            raise RuntimeError(f"Failed to publish Instagram media container: {error_msg}")
        return data["id"]


# Global default publisher instance
publisher = PublisherAgent()
