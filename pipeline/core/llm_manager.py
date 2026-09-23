"""Central LLM Manager: Model Routing & Multi-Account Resilience.

Complies with:
- Rule 14 (LLM MANAGEMENT & MULTI-ACCOUNT RESILIENCE):
  - Deprecated Models Prohibited: strictly forbids retired models (e.g. 1.5-era Pro/Flash).
  - Multi-Tier Model Routing:
    - High-capability (Creative Director): gemini-3.1-pro-preview
    - High-volume agents (Brand Designer, Chief Critic, Copywriter, Scriptwriter, Ideator): gemini-3.6-flash
    - Budget tasks (Strategist scheduling/metadata JSON): gemini-3.5-flash-lite
  - Multi-Account Resilience (3 Google AI Pro accounts):
    - Round-robin static assignment by AGENT ROLE (not random rotation).
    - Per-account health tracking: consecutive_auth_failures, is_quarantined flag, persisted across restarts.
    - Error classification: distinguishes standard rate limits from account-level restrictions.
    - Quarantine alerting via Notifier: "[ACCOUNT_DOWN] Account N appears restricted - operating on remaining."
    - Detect-and-degrade ONLY: never masks or evades Google's systems.
- Rule 2 (NO SILENT CRASHES): Explicit error handling, transparent degradation, and structured returns.
- Rule 4 (VALIDATE LLM OUTPUT): Preserves Pydantic schema validation across all tiers.
- Rule 8 (NO UNBOUNDED RETRIES): Bounded backoff and immediate quarantine on restriction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Type

import dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.core.logger import get_logger
from pipeline.core.notifier import notifier
from pipeline.core.rate_limiter import account_rate_limiter_pool, global_rate_limiter

logger = get_logger("llm_manager")

# Default state file location
DEFAULT_STATE_FILE = Path(__file__).resolve().parent / "llm_account_state.json"

# Prohibited legacy model substrings
RETIRED_PATTERNS = ["1.5-pro", "1.5-flash", "gemini-1.5"]


class AgentRole(str, Enum):
    """Agent roles defining capability tier and model assignment."""
    CREATIVE_DIRECTOR = "creative_director"
    BRAND_DESIGNER = "brand_designer"
    CHIEF_CRITIC = "chief_critic"
    COPYWRITER = "copywriter"
    STRATEGIST = "strategist"
    COMPLIANCE_OFFICER = "compliance_officer"
    SAFETY_OFFICER = "safety_officer"
    # Pipeline aliases
    SCRIPTWRITER = "scriptwriter"
    IDEATOR = "ideator"
    # Autonomous Department Head Gatekeepers
    HEAD_OF_STORY = "head_of_story"
    HEAD_OF_AUDIO = "head_of_audio"
    HEAD_OF_ART = "head_of_art"
    HEAD_OF_POST = "head_of_post"
    HEAD_OF_COMPLIANCE = "head_of_compliance"
    HEAD_OF_VISUAL_RELEVANCE = "head_of_visual_relevance"


# Strict Rule 14 Model Routing Mapping
# Tier 1 (Pro): High-capability creative tasks — low RPD (~25/day)
# Tier 2 (Flash): High-volume agents needing good quality — medium RPD (~1500/day)
# Tier 3 (Flash-Lite): Budget/trivial tasks — high RPD (~1500/day)
ROLE_MODEL_MAPPING: dict[AgentRole, str] = {
    # Tier 1: Pro
    AgentRole.CREATIVE_DIRECTOR:      "gemini-3.1-pro-preview",
    # Tier 2: Flash (high-volume agents per Rule 14)
    AgentRole.CHIEF_CRITIC:           "gemini-3.6-flash",
    AgentRole.COPYWRITER:             "gemini-3.6-flash",
    AgentRole.COMPLIANCE_OFFICER:     "gemini-3.6-flash",
    AgentRole.SAFETY_OFFICER:         "gemini-3.6-flash",
    AgentRole.SCRIPTWRITER:           "gemini-3.6-flash",
    AgentRole.IDEATOR:                "gemini-3.6-flash",
    AgentRole.HEAD_OF_STORY:          "gemini-3.6-flash",
    AgentRole.HEAD_OF_AUDIO:          "gemini-3.6-flash",
    AgentRole.HEAD_OF_ART:            "gemini-3.6-flash",
    AgentRole.HEAD_OF_POST:           "gemini-3.6-flash",
    AgentRole.HEAD_OF_COMPLIANCE:     "gemini-3.6-flash",
    AgentRole.HEAD_OF_VISUAL_RELEVANCE: "gemini-3.6-flash",
    # Tier 3: Flash-Lite (budget tasks)
    AgentRole.BRAND_DESIGNER:         "gemini-3.5-flash-lite",
    AgentRole.STRATEGIST:             "gemini-3.5-flash-lite",
}

# Canonical ordered list for static round-robin role assignment
CANONICAL_ROLES_ORDER: list[AgentRole] = [
    AgentRole.CREATIVE_DIRECTOR,
    AgentRole.BRAND_DESIGNER,
    AgentRole.CHIEF_CRITIC,
    AgentRole.COPYWRITER,
    AgentRole.STRATEGIST,
]

# Per-model daily call limits (RPD per account, Google AI free tier)
# These are conservative estimates — check Google AI Studio for live values.
MODEL_DAILY_LIMITS: dict[str, int] = {
    "gemini-3.1-pro-preview": 25,
    "gemini-3.6-flash": 1500,
    "gemini-3.5-flash-lite": 1500,
}

# Auto-downgrade fallback chain: preferred → fallback → emergency → halt
MODEL_FALLBACK_CHAIN: list[str] = [
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
]


class ErrorClassification(str, Enum):
    """Categorization of LLM API errors."""
    NORMAL_RATE_LIMIT = "normal_rate_limit"
    ACCOUNT_RESTRICTION = "account_restriction"
    OTHER_ERROR = "other_error"


class AllAccountsQuarantinedError(RuntimeError):
    """Raised when all configured Google AI accounts are quarantined."""
    pass


class AllModelsExhaustedError(RuntimeError):
    """Raised when all models in the fallback chain have exhausted their daily RPD quota."""
    pass


MODEL_QUOTA_STATE_FILE = Path(__file__).resolve().parent / "gemini_model_quota_state.json"
PACIFIC_TZ = __import__("zoneinfo").ZoneInfo("America/Los_Angeles")


class ModelQuotaTracker:
    """Tracks per-model daily API call counts with midnight PT auto-reset.
    
    Each model has an independent daily RPD counter. Persisted to disk so
    counts survive process restarts within the same day.
    """

    def __init__(self, state_path: Path = MODEL_QUOTA_STATE_FILE) -> None:
        self.state_path = state_path
        self._counts: dict[str, int] = {}  # model_name -> calls_today
        self._date_key: str = ""  # YYYY-MM-DD in Pacific Time
        self._load()

    def _today_key(self) -> str:
        """Returns today's date string in Pacific Time (matches Google's reset schedule)."""
        return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")

    def _load(self) -> None:
        """Loads persisted quota state, auto-resetting if the date has rolled over."""
        today = self._today_key()
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saved_date = data.get("date_key", "")
                if saved_date == today:
                    self._counts = data.get("counts", {})
                    self._date_key = today
                    return
            except Exception:
                pass
        # New day or corrupt/missing file — start fresh
        self._counts = {}
        self._date_key = today

    def _save(self) -> None:
        """Persists current counts atomically."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.state_path.parent, prefix="model_quota_", suffix=".tmp"
            )
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump({"date_key": self._date_key, "counts": self._counts}, f, indent=2)
            os.replace(temp_path, self.state_path)
        except Exception as e:
            logger.warning(f"[MODEL_QUOTA] Could not persist quota state: {e}")

    def _check_reset(self) -> None:
        """Auto-resets counters if the Pacific Time date has rolled past midnight."""
        today = self._today_key()
        if today != self._date_key:
            logger.info(f"[MODEL_QUOTA] New day detected ({self._date_key} → {today}). Resetting all model counters.")
            self._counts = {}
            self._date_key = today
            self._save()

    def record_call(self, model_name: str) -> None:
        """Increments today's call count for a model after a successful API call."""
        self._check_reset()
        self._counts[model_name] = self._counts.get(model_name, 0) + 1
        self._save()

    def get_used(self, model_name: str) -> int:
        """Returns how many calls have been made to this model today."""
        self._check_reset()
        return self._counts.get(model_name, 0)

    def get_remaining(self, model_name: str) -> int:
        """Returns how many daily calls remain for this model."""
        limit = MODEL_DAILY_LIMITS.get(model_name, 1500)
        return max(0, limit - self.get_used(model_name))

    def is_exhausted(self, model_name: str) -> bool:
        """Returns True if this model's daily RPD quota is fully spent."""
        return self.get_remaining(model_name) <= 0

    def get_status(self) -> dict[str, Any]:
        """Returns a snapshot of all model quotas for telemetry/dashboard."""
        self._check_reset()
        status = {}
        for model, limit in MODEL_DAILY_LIMITS.items():
            used = self._counts.get(model, 0)
            status[model] = {
                "used": used,
                "limit": limit,
                "remaining": max(0, limit - used),
                "exhausted": used >= limit,
            }
        return {"date": self._date_key, "models": status}


# Global model quota tracker instance
model_quota_tracker = ModelQuotaTracker()


@dataclass
class LLMAccount:
    """Represents a Google AI Pro account and its persisted health state."""
    account_id: int
    api_key: str
    consecutive_auth_failures: int = 0
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None
    quarantined_at: Optional[str] = None
    total_requests: int = 0
    total_errors: int = 0
    last_used_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Obfuscate API key in serialized state for safety
        d["api_key_masked"] = f"...{self.api_key[-6:]}" if len(self.api_key) >= 6 else "***"
        del d["api_key"]
        return d


class LLMManager:
    """Central manager for model routing and multi-account resilience."""

    def __init__(
        self,
        accounts: Optional[list[LLMAccount]] = None,
        state_path: Path = DEFAULT_STATE_FILE,
        notifier_instance: Optional[Any] = None,
        auto_load_env: bool = True
    ):
        self.state_path = state_path
        self.notifier = notifier_instance or notifier
        self._clients: dict[int, Any] = {}

        if auto_load_env:
            dotenv.load_dotenv()

        if accounts is not None:
            self.accounts = accounts
        else:
            self.accounts = self._discover_accounts_from_env()

        # Load persisted health/quarantine state
        self._load_state()

    def _discover_accounts_from_env(self) -> list[LLMAccount]:
        """Discovers up to 3 Google AI accounts from environment variables."""
        discovered: list[LLMAccount] = []
        
        # Primary key
        k1 = os.getenv("GEMINI_API_KEY")
        # Secondary key
        k2 = os.getenv("GEMINI_API_KEY_2")
        # Tertiary key
        k3 = os.getenv("GEMINI_API_KEY_3")

        keys = [k for k in [k1, k2, k3] if k and k.strip()]
        
        # If user only has 1 key configured, create 3 account slots using the primary key
        # (enabling multi-account resilience testing/simulation in single-key setups)
        if len(keys) == 1:
            discovered = [
                LLMAccount(account_id=1, api_key=keys[0]),
                LLMAccount(account_id=2, api_key=keys[0]),
                LLMAccount(account_id=3, api_key=keys[0]),
            ]
        elif len(keys) > 1:
            for idx, k in enumerate(keys[:3], start=1):
                discovered.append(LLMAccount(account_id=idx, api_key=k.strip()))
            # Pad up to 3 if only 2 supplied
            while len(discovered) < 3:
                discovered.append(
                    LLMAccount(account_id=len(discovered) + 1, api_key=keys[0].strip())
                )
        else:
            # Fallback placeholder accounts if no key configured
            discovered = [
                LLMAccount(account_id=1, api_key="placeholder_key_1"),
                LLMAccount(account_id=2, api_key="placeholder_key_2"),
                LLMAccount(account_id=3, api_key="placeholder_key_3"),
            ]

        return discovered

    def _load_state(self) -> None:
        """Loads persisted account health and quarantine state from disk."""
        if not self.state_path.exists():
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                saved = json.load(f)

            for acc in self.accounts:
                acc_key = str(acc.account_id)
                if acc_key in saved:
                    st = saved[acc_key]
                    acc.consecutive_auth_failures = int(st.get("consecutive_auth_failures", 0))
                    acc.is_quarantined = bool(st.get("is_quarantined", False))
                    acc.quarantine_reason = st.get("quarantine_reason")
                    acc.quarantined_at = st.get("quarantined_at")
                    acc.total_requests = int(st.get("total_requests", 0))
                    acc.total_errors = int(st.get("total_errors", 0))
                    acc.last_used_at = st.get("last_used_at")

            logger.info(
                f"[LLM_MANAGER] Loaded account health state from {self.state_path.name}. "
                f"Active accounts: {len(self.get_healthy_accounts())}/{len(self.accounts)}"
            )
        except Exception as e:
            logger.warning(f"[LLM_MANAGER] Could not load state from {self.state_path}: {e}")

    def save_state(self) -> None:
        """Persists account health and quarantine state to disk atomically."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {str(acc.account_id): acc.to_dict() for acc in self.accounts}

            # Write atomically via tempfile
            temp_fd, temp_file = tempfile.mkstemp(
                dir=self.state_path.parent, prefix="llm_state_", suffix=".tmp"
            )
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file, self.state_path)
        except Exception as e:
            logger.error(f"[LLM_MANAGER] Failed to persist account state: {e}")

    def get_healthy_accounts(self) -> list[LLMAccount]:
        """Returns list of non-quarantined accounts."""
        return [acc for acc in self.accounts if not acc.is_quarantined]

    def resolve_model(self, role: AgentRole | str) -> str:
        """Resolves target Gemini model name for an agent role per Rule 14.
        
        Checks per-model daily quota and auto-downgrades through the fallback
        chain (Pro → Flash → Lite) if the preferred model is exhausted.
        """
        if isinstance(role, str):
            try:
                role = AgentRole(role.lower())
            except ValueError:
                role = AgentRole.COPYWRITER

        preferred_model = ROLE_MODEL_MAPPING.get(role, "gemini-3.6-flash")

        # Strict Rule 14 Verification: Never allow retired 1.5-era models
        if any(retired in preferred_model for retired in RETIRED_PATTERNS):
            raise ValueError(
                f"Rule 14 Violation: Model '{preferred_model}' is retired and prohibited. "
                "Creative Director must route to 'gemini-3.1-pro-preview'."
            )

        # Quota-aware auto-downgrade: walk fallback chain if preferred model is exhausted
        if not model_quota_tracker.is_exhausted(preferred_model):
            return preferred_model

        # Find the preferred model's position in the fallback chain
        try:
            start_idx = MODEL_FALLBACK_CHAIN.index(preferred_model)
        except ValueError:
            start_idx = 0

        # Walk downward through remaining fallback tiers
        for fallback in MODEL_FALLBACK_CHAIN[start_idx + 1:]:
            remaining = model_quota_tracker.get_remaining(fallback)
            if remaining > 0:
                logger.warning(
                    f"[MODEL_DOWNGRADE] Role '{getattr(role, 'value', role)}' preferred "
                    f"'{preferred_model}' (exhausted: {model_quota_tracker.get_remaining(preferred_model)}/"
                    f"{MODEL_DAILY_LIMITS.get(preferred_model, '?')} remaining) "
                    f"→ downgraded to '{fallback}' ({remaining}/{MODEL_DAILY_LIMITS.get(fallback, '?')} remaining)"
                )
                return fallback

        # All models exhausted
        raise AllModelsExhaustedError(
            f"All models in fallback chain are exhausted for today. "
            f"Quota status: {model_quota_tracker.get_status()}"
        )

    def get_assigned_account(self, role: AgentRole | str) -> LLMAccount:
        """Returns the statically assigned account for an agent role.
        
        Rule 14: Static round-robin by AGENT ROLE (not random rotation).
        If the statically assigned account is quarantined, gracefully degrades
        to the remaining healthy accounts.
        """
        healthy = self.get_healthy_accounts()
        if not healthy:
            raise AllAccountsQuarantinedError(
                "All Google AI Pro accounts are quarantined due to account-level restrictions. "
                "Operating halted per Rule 14 detect-and-degrade policy."
            )

        if isinstance(role, str):
            try:
                role = AgentRole(role.lower())
            except ValueError:
                role = AgentRole.COPYWRITER

        # Map role aliases to canonical roles for assignment
        canonical_role = role
        if role in (AgentRole.SCRIPTWRITER, AgentRole.IDEATOR):
            canonical_role = AgentRole.COPYWRITER
        elif role in (
            AgentRole.COMPLIANCE_OFFICER,
            AgentRole.SAFETY_OFFICER,
            AgentRole.HEAD_OF_STORY,
            AgentRole.HEAD_OF_AUDIO,
            AgentRole.HEAD_OF_ART,
            AgentRole.HEAD_OF_POST,
            AgentRole.HEAD_OF_COMPLIANCE,
        ):
            canonical_role = AgentRole.CHIEF_CRITIC

        try:
            role_idx = CANONICAL_ROLES_ORDER.index(canonical_role)
        except ValueError:
            role_idx = 0

        # 1. Primary static assignment across all configured accounts
        static_account = self.accounts[role_idx % len(self.accounts)]
        if not static_account.is_quarantined:
            return static_account

        # 2. Detect-and-degrade: If assigned account is quarantined, route deterministically
        # across the remaining healthy accounts
        degraded_account = healthy[role_idx % len(healthy)]
        logger.info(
            f"[DEGRADED_ROUTING] Role '{role.value}' statically assigned to Account {static_account.account_id} "
            f"(quarantined) -> operating on healthy Account {degraded_account.account_id}."
        )
        return degraded_account

    def quarantine_account(self, account_id: int, reason: str) -> None:
        """Quarantines an account due to account-level restrictions (Rule 14)."""
        target = next((acc for acc in self.accounts if acc.account_id == account_id), None)
        if not target:
            return

        target.is_quarantined = True
        target.quarantine_reason = reason
        target.quarantined_at = datetime.now(timezone.utc).isoformat()
        self.save_state()

        # Emit required Rule 14 notification
        self.notifier.account_down(account_id=account_id, reason=reason)
        logger.error(
            f"[QUARANTINE] Account {account_id} has been quarantined. Reason: {reason}. "
            f"Remaining healthy accounts: {len(self.get_healthy_accounts())}/{len(self.accounts)}"
        )

    def reset_quarantine(self, account_id: Optional[int] = None) -> None:
        """Re-enables quarantined account(s) when restrictions are resolved."""
        for acc in self.accounts:
            if account_id is None or acc.account_id == account_id:
                acc.is_quarantined = False
                acc.consecutive_auth_failures = 0
                acc.quarantine_reason = None
                acc.quarantined_at = None
        self.save_state()
        logger.info(f"[LLM_MANAGER] Quarantine reset for {'all accounts' if account_id is None else f'account {account_id}'}.")

    def classify_error(self, error: Exception) -> ErrorClassification:
        """Classifies an error as standard rate limit vs. account-level restriction.
        
        Rule 14:
        - Normal rate limits (429 with active quota): expected, retry with backoff.
        - Account-level restrictions: 401/403, 429 with limit 0, or explicit suspension/restriction.
        """
        err_str = str(error).lower()
        err_repr = repr(error).lower()
        combined = f"{err_str} {err_repr}"

        # 1. Check for fatal auth failures (401 / 403)
        if any(tok in combined for tok in ["401", "unauthorized", "api_key_invalid", "consumer_invalid"]):
            return ErrorClassification.ACCOUNT_RESTRICTION

        if any(tok in combined for tok in ["403", "permission_denied", "account suspended", "billing disabled", "consumer suspended"]):
            return ErrorClassification.ACCOUNT_RESTRICTION

        # 2. Check for 429 with zero quota (account-level restriction / disabled tier)
        # e.g., "limit: 0" or "quota exceeded for metric ... limit: 0"
        if "limit: 0" in combined or "limit:0" in combined:
            return ErrorClassification.ACCOUNT_RESTRICTION

        if "account restricted" in combined or "account appears restricted" in combined:
            return ErrorClassification.ACCOUNT_RESTRICTION

        # 3. Standard 429 rate limit (active quota exhausted temporarily)
        if "429" in combined or "resource_exhausted" in combined or "rate limit" in combined:
            return ErrorClassification.NORMAL_RATE_LIMIT

        return ErrorClassification.OTHER_ERROR

    def get_client(self, account: LLMAccount) -> Any:
        """Retrieves or creates a google.genai.Client for an account."""
        if account.account_id in self._clients:
            return self._clients[account.account_id]

        from google import genai
        client = genai.Client(api_key=account.api_key)
        self._clients[account.account_id] = client
        return client

    def get_client_and_model(self, role: AgentRole | str) -> tuple[Any, str, LLMAccount]:
        """Returns the client, resolved model name, and assigned account for a role."""
        account = self.get_assigned_account(role)
        model = self.resolve_model(role)
        client = self.get_client(account)
        return client, model, account

    def generate_content(
        self,
        role: AgentRole | str,
        contents: Any,
        system_instruction: Optional[str] = None,
        response_schema: Optional[Type] = None,
        temperature: float = 0.7,
        max_retries: int = 2
    ) -> tuple[str, Any]:
        """Unified LLM generation with model routing, schema validation, and resilience.
        
        Args:
            role: Agent role determining model tier and static account assignment.
            contents: Prompt string or content payload.
            system_instruction: Optional system instruction.
            response_schema: Optional Pydantic schema for structured output (Rule 4).
            temperature: Sampling temperature.
            max_retries: Max retries on normal rate limit.
            
        Returns:
            tuple[raw_text, parsed_object]
        """
        model_name = self.resolve_model(role)

        # Outer loop handles failover across healthy accounts if an account is quarantined
        while True:
            account = self.get_assigned_account(role)
            client = self.get_client(account)
            account.total_requests += 1
            account.last_used_at = datetime.now(timezone.utc).isoformat()

            from google.genai import types
            config_kwargs: dict[str, Any] = {"temperature": temperature}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if response_schema:
                config_kwargs["response_mime_type"] = "application/json"
                config_kwargs["response_schema"] = response_schema

            config = types.GenerateContentConfig(**config_kwargs)

            # Inner loop handles normal rate-limit backoff on the current account
            success = False
            for attempt in range(max_retries + 1):
                try:
                    # Rate Limiter: strictly throttle calls per unique API key without leaking credentials
                    rate_key = (
                        hashlib.sha256(account.api_key.encode()).hexdigest()[:16]
                        if account.api_key
                        else f"account-{account.account_id}"
                    )
                    account_rate_limiter_pool.acquire(rate_key)

                    logger.debug(
                        f"[LLM_CALL] Role: '{getattr(role, 'value', role)}' | "
                        f"Model: '{model_name}' | Account: {account.account_id} | Attempt {attempt + 1}"
                    )
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config
                    )

                    # Success: reset consecutive failures and record quota usage
                    account.consecutive_auth_failures = 0
                    self.save_state()
                    model_quota_tracker.record_call(model_name)

                    raw_text = response.text or ""
                    parsed = getattr(response, "parsed", None)
                    return raw_text, parsed

                except Exception as exc:
                    classification = self.classify_error(exc)
                    logger.warning(
                        f"[LLM_ERROR] Account {account.account_id} encounter: {exc} "
                        f"(Classification: {classification.value})"
                    )

                    if classification == ErrorClassification.ACCOUNT_RESTRICTION:
                        account.consecutive_auth_failures += 1
                        account.total_errors += 1
                        # Rule 14: Immediately quarantine account on restriction
                        self.quarantine_account(
                            account_id=account.account_id,
                            reason=f"Account-level restriction detected: {exc}"
                        )
                        # Break inner loop and retry request on the next healthy account
                        break

                    elif classification == ErrorClassification.NORMAL_RATE_LIMIT:
                        account.total_errors += 1
                        if attempt < max_retries:
                            backoff = 2.0 ** attempt
                            logger.info(
                                f"[RATE_LIMIT_BACKOFF] Account {account.account_id} rate limited. "
                                f"Backing off {backoff:.1f}s before retry (Attempt {attempt + 1}/{max_retries})."
                            )
                            time.sleep(backoff)
                            continue
                        else:
                            # If retries exhausted on this account, raise or fall back
                            raise exc
                    else:
                        account.total_errors += 1
                        self.save_state()
                        raise exc


# Global default manager instance
llm_manager = LLMManager()
