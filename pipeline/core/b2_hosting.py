"""Ephemeral public hosting bridge using Backblaze B2 (S3-compatible).

Per Rule 6 (INSTAGRAM NEEDS A PUBLIC URL):
- Instagram Reels Graph API requires a crawlable temporary public URL to download video.
- Cloudflare Tunnel alternative evaluation:
  - Quick Tunnels (*.trycloudflare.com) suffer frequent connection resets, require a running daemon
    (cloudflared.exe) + local web server, and are frequently throttled/blocked by Meta crawlers.
- Backblaze B2 Justification:
  - Free Tier: 10 GB storage + free egress / S3 API compatibility.
  - Reliability: High-availability cloud infrastructure with trusted TLS certificates.
  - Crawler Accessibility: 100% globally accessible by Meta's crawler IPs without firewall or port-forwarding issues.
  - Security & Ephemerality: Generates pre-signed HTTPS URLs with a 15-minute expiration window.
  - Guaranteed Cleanup: Context manager automatically deletes the hosted object upon publication or failure.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("b2_hosting")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] [b2_hosting] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class B2HostingBridge:
    """Manages short-lived public hosting of video assets on Backblaze B2.
    
    Architectural Justification (Rule 6):
    - Cloudflare Tunnel (Alternative): Quick Tunnels (*.trycloudflare.com) suffer frequent connection drops,
      require a long-running daemon (cloudflared.exe) and local HTTP server, and are prone to blocking by Meta's crawlers.
    - Backblaze B2 (Selected): Zero cost (10GB permanent free tier, free egress), Reliability with enterprise-grade
      high availability, trusted TLS certificates, pre-signed temporary HTTPS URLs directly accessible by Meta's crawlers,
      and guaranteed ephemeral cleanup via context managers.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        key_id: Optional[str] = None,
        application_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ) -> None:
        self.endpoint_url = endpoint_url or os.getenv("B2_ENDPOINT_URL", "https://s3.eu-central-003.backblazeb2.com")
        self.key_id = key_id or os.getenv("B2_KEY_ID")
        self.application_key = application_key or os.getenv("B2_APPLICATION_KEY")
        self.bucket_name = bucket_name or os.getenv("B2_BUCKET_NAME", "faceless-video-bridge-1")
        if self.bucket_name:
            self.bucket_name = self.bucket_name.strip("\"'")

        self._s3 = None
        if self.key_id and self.application_key:
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.key_id,
                aws_secret_access_key=self.application_key,
                config=Config(signature_version="s3v4")
            )

    def is_configured(self) -> bool:
        """Returns True if Backblaze B2 credentials are fully populated."""
        return bool(self._s3 and self.bucket_name)

    def upload_temporary_media(
        self,
        file_path: Path | str,
        expires_in_seconds: int = 900
    ) -> tuple[str, str]:
        """Uploads a video to B2 and returns a pre-signed HTTPS public URL and object key.
        
        Args:
            file_path: Local path to the .mp4 video.
            expires_in_seconds: Lifetime of pre-signed URL in seconds (default 15 minutes).
            
        Returns:
            Tuple of (presigned_public_url, object_key).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot host missing file: {path}")

        if not self.is_configured():
            raise RuntimeError(
                "Backblaze B2 is not configured in .env (B2_KEY_ID, B2_APPLICATION_KEY, B2_BUCKET_NAME)."
            )

        unique_id = uuid.uuid4().hex[:8]
        safe_name = path.name.replace(" ", "_")
        object_key = f"temp_reels/{unique_id}_{safe_name}"

        logger.info(f"Uploading temporary video asset to B2: {object_key} ({path.stat().st_size / 1024 / 1024:.2f} MB)...")
        try:
            with open(path, "rb") as f:
                self._s3.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=f,
                    ContentType="video/mp4"
                )

            presigned_url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expires_in_seconds
            )
            logger.info(f"Generated pre-signed public HTTPS URL (expires in {expires_in_seconds}s): {presigned_url[:60]}...")
            return presigned_url, object_key
        except ClientError as e:
            logger.error(f"Failed to upload temporary video to B2: {e}")
            raise

    def delete_media(self, object_key: str) -> bool:
        """Deletes the temporary media object from B2 to prevent lingering public files.
        
        Args:
            object_key: The storage key of the object to delete.
            
        Returns:
            True if deleted successfully, False otherwise.
        """
        if not self.is_configured():
            logger.warning("Cannot delete B2 object: client not configured.")
            return False

        logger.info(f"Deleting temporary public asset from B2: {object_key}...")
        try:
            self._s3.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Temporary asset successfully purged from B2: {object_key}")
            return True
        except ClientError as e:
            logger.error(f"Error purging temporary asset from B2 ({object_key}): {e}")
            return False

    def check_exists(self, object_key: str) -> bool:
        """Checks whether an object currently exists in the bucket."""
        if not self.is_configured():
            return False
        try:
            self._s3.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    @contextmanager
    def temporary_public_url(
        self,
        file_path: Path | str,
        expires_in_seconds: int = 900
    ) -> Generator[tuple[str, str], None, None]:
        """Context manager guaranteeing cleanup of the temporary public asset.
        
        Yields:
            (presigned_public_url, object_key)
        """
        public_url, object_key = self.upload_temporary_media(file_path, expires_in_seconds)
        try:
            yield public_url, object_key
        finally:
            self.delete_media(object_key)


# Global default bridge instance
b2_bridge = B2HostingBridge()
