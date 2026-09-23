"""Test Suite for Mission: LLM Manager — Model Routing & Multi-Account Resilience (Rule 14).

Complies with Acceptance Criteria:
- [x] models.list() confirms no gemini-1.5-pro references remain
- [x] Simulated account-restricted response triggers quarantine + alert
- [x] Full pipeline run completes with only 1 of 3 accounts healthy
- [x] Quarantine state persists across process restart
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from dotenv import load_dotenv

from pipeline.core.llm_manager import (
    AgentRole,
    AllAccountsQuarantinedError,
    ErrorClassification,
    LLMAccount,
    LLMManager,
    ROLE_MODEL_MAPPING,
)
from pipeline.core.notifier import Notifier

load_dotenv()


@pytest.fixture
def temp_state_file(tmp_path):
    """Provides an isolated JSON state file for tests."""
    return tmp_path / "test_llm_state.json"


@pytest.fixture
def test_notifier():
    """Provides an isolated Notifier instance."""
    return Notifier()


@pytest.fixture
def mock_accounts():
    """Provides 3 mock LLMAccount instances."""
    return [
        LLMAccount(account_id=1, api_key="test_key_account_1"),
        LLMAccount(account_id=2, api_key="test_key_account_2"),
        LLMAccount(account_id=3, api_key="test_key_account_3"),
    ]


# ==============================================================================
# CRITERION 1: models.list() confirms no gemini-1.5-pro references remain
# ==============================================================================

def test_models_list_confirms_no_gemini_15_pro_references():
    """Verifies that no live Gemini model in models.list() contains 'gemini-1.5-pro'."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set in environment.")

    from google import genai
    client = genai.Client(api_key=api_key)

    # Fetch live models from Google GenAI API
    models = [m.name for m in client.models.list()]
    
    # 1. Ensure retired gemini-1.5-pro is completely absent
    matching_15 = [m for m in models if "1.5-pro" in m or "gemini-1.5" in m]
    assert len(matching_15) == 0, f"Found retired 1.5 models in models.list(): {matching_15}"

    # 2. Ensure live target models are present
    target_pro = "gemini-3.1-pro-preview"
    target_flash = "gemini-3.6-flash"
    target_lite = "gemini-3.5-flash-lite"

    found_models = [m.replace("models/", "") for m in models]
    assert target_pro in found_models, f"Expected {target_pro} to be live."
    assert target_flash in found_models, f"Expected {target_flash} to be live."
    assert target_lite in found_models, f"Expected {target_lite} to be live."


def test_codebase_has_no_gemini_15_pro_references():
    """Scans all python files in the repository to guarantee no hardcoded 'gemini-1.5-pro'."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    py_files = list(root_dir.glob("**/*.py"))
    target_retired_str = "gemini" + "-1.5" + "-pro"

    offending = []
    for py_path in py_files:
        if ".venv" in str(py_path) or py_path.name == "test_llm_manager.py":
            continue
        try:
            content = py_path.read_text(encoding="utf-8")
            if target_retired_str in content:
                offending.append(str(py_path))
        except Exception:
            pass

    assert len(offending) == 0, f"Found retired model references in files: {offending}"


# ==============================================================================
# MODEL ROUTING TESTS
# ==============================================================================

def test_model_routing_by_agent_role(temp_state_file, test_notifier, mock_accounts):
    """Verifies strict Rule 14 model routing across agent capability tiers."""
    mgr = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    # High-capability: Creative Director only (Pro tier ~25 RPD)
    assert mgr.resolve_model(AgentRole.CREATIVE_DIRECTOR) == "gemini-3.1-pro-preview"

    # High-volume quality agents (Flash tier ~1,500 RPD)
    assert mgr.resolve_model(AgentRole.CHIEF_CRITIC) == "gemini-3.6-flash"
    assert mgr.resolve_model(AgentRole.COPYWRITER) == "gemini-3.6-flash"
    assert mgr.resolve_model(AgentRole.SCRIPTWRITER) == "gemini-3.6-flash"
    assert mgr.resolve_model(AgentRole.IDEATOR) == "gemini-3.6-flash"

    # Budget tier (Flash-Lite ~1,500 RPD)
    assert mgr.resolve_model(AgentRole.BRAND_DESIGNER) == "gemini-3.5-flash-lite"
    assert mgr.resolve_model(AgentRole.STRATEGIST) == "gemini-3.5-flash-lite"


# ==============================================================================
# STATIC ASSIGNMENT: Round-Robin by Agent Role (Not Random Rotation)
# ==============================================================================

def test_static_assignment_round_robin_by_role(temp_state_file, test_notifier, mock_accounts):
    """Verifies deterministic, static per-agent assignment (Rule 14)."""
    mgr = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    # Role 0: Creative Director -> Account 1
    acc_cd_1 = mgr.get_assigned_account(AgentRole.CREATIVE_DIRECTOR)
    acc_cd_2 = mgr.get_assigned_account(AgentRole.CREATIVE_DIRECTOR)
    assert acc_cd_1.account_id == 1
    assert acc_cd_2.account_id == 1, "Assignment must be static, never random hopping"

    # Role 1: Brand Designer -> Account 2
    acc_bd = mgr.get_assigned_account(AgentRole.BRAND_DESIGNER)
    assert acc_bd.account_id == 2

    # Role 2: Chief Critic -> Account 3
    acc_cc = mgr.get_assigned_account(AgentRole.CHIEF_CRITIC)
    assert acc_cc.account_id == 3

    # Role 3: Copywriter -> Account 1 (round-robin wraps)
    acc_cw = mgr.get_assigned_account(AgentRole.COPYWRITER)
    assert acc_cw.account_id == 1

    # Role 4: Strategist -> Account 2
    acc_st = mgr.get_assigned_account(AgentRole.STRATEGIST)
    assert acc_st.account_id == 2


# ==============================================================================
# CRITERION 2: Simulated account-restricted response triggers quarantine + alert
# ==============================================================================

def test_simulated_account_restricted_response_triggers_quarantine_and_alert(
    temp_state_file, test_notifier, mock_accounts
):
    """Simulates an account restriction response (e.g. 403 or zero-quota 429)

    Verifies that:
    1. Account is quarantined immediately.
    2. Alert "[ACCOUNT_DOWN] Account N appears restricted - operating on remaining." is emitted.
    3. The request automatically degrades to the remaining healthy account.
    """
    mgr = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    # Initial state: all 3 accounts healthy
    assert len(mgr.get_healthy_accounts()) == 3

    # Brand Designer is statically assigned to Account 2
    target_account = mgr.get_assigned_account(AgentRole.BRAND_DESIGNER)
    assert target_account.account_id == 2

    # Create mock clients
    mock_client_1 = MagicMock()
    mock_client_2 = MagicMock()
    mock_client_3 = MagicMock()

    # Mock response for healthy accounts
    healthy_resp = MagicMock()
    healthy_resp.text = "Healthy response text"
    healthy_resp.parsed = None
    mock_client_1.models.generate_content.return_value = healthy_resp
    mock_client_3.models.generate_content.return_value = healthy_resp

    # Account 2 raises an account-level restriction: 403 PERMISSION_DENIED (Account Suspended)
    restriction_err = Exception("403 Forbidden: PERMISSION_DENIED. Account appears restricted or suspended.")
    mock_client_2.models.generate_content.side_effect = restriction_err

    # Inject mock clients
    mgr._clients[1] = mock_client_1
    mgr._clients[2] = mock_client_2
    mgr._clients[3] = mock_client_3

    # Execute generation for Brand Designer (which initially hits Account 2)
    raw_text, parsed = mgr.generate_content(
        role=AgentRole.BRAND_DESIGNER,
        contents="Generate design guidelines"
    )

    # 1. Verify Account 2 was quarantined
    acc_2 = next(a for a in mgr.accounts if a.account_id == 2)
    assert acc_2.is_quarantined is True
    assert "restriction" in acc_2.quarantine_reason.lower()
    assert len(mgr.get_healthy_accounts()) == 2

    # 2. Verify Notifier emitted the exact required alert
    alerts = test_notifier.get_alerts()
    expected_msg = "[ACCOUNT_DOWN] Account 2 appears restricted - operating on remaining."
    assert any(expected_msg in a["message"] for a in alerts), f"Expected alert '{expected_msg}' in {alerts}"

    # 3. Verify request degraded and succeeded on remaining healthy account
    assert raw_text == "Healthy response text"


def test_zero_quota_429_classified_as_account_restriction(temp_state_file, test_notifier, mock_accounts):
    """Verifies that 429 with 'limit: 0' (tier restriction) is classified as account restriction."""
    mgr = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    zero_quota_err = Exception(
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro"
    )
    classification = mgr.classify_error(zero_quota_err)
    assert classification == ErrorClassification.ACCOUNT_RESTRICTION


def test_normal_rate_limit_does_not_quarantine(temp_state_file, test_notifier, mock_accounts):
    """Verifies that normal 429 rate limit is classified as NORMAL_RATE_LIMIT and does not quarantine."""
    mgr = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    normal_429_err = Exception(
        "429 RESOURCE_EXHAUSTED. Rate limit exceeded for requests per minute. Please retry in 5s."
    )
    classification = mgr.classify_error(normal_429_err)
    assert classification == ErrorClassification.NORMAL_RATE_LIMIT


# ==============================================================================
# CRITERION 4: Quarantine state persists across process restart
# ==============================================================================

def test_quarantine_state_persists_across_process_restart(temp_state_file, test_notifier, mock_accounts):
    """Verifies that quarantined account status persists to JSON and survives instance re-creation."""
    # Instance 1: Quarantine Account 2 and Account 3
    mgr1 = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)
    mgr1.quarantine_account(2, reason="Simulated restriction test")
    mgr1.quarantine_account(3, reason="Simulated billing issue")

    assert temp_state_file.exists(), "State file must exist on disk"
    with open(temp_state_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["2"]["is_quarantined"] is True
    assert saved["3"]["is_quarantined"] is True
    assert saved["1"]["is_quarantined"] is False

    # Instance 2: Simulate fresh process restart by creating new accounts and new manager
    new_accounts = [
        LLMAccount(account_id=1, api_key="fresh_key_1"),
        LLMAccount(account_id=2, api_key="fresh_key_2"),
        LLMAccount(account_id=3, api_key="fresh_key_3"),
    ]
    mgr2 = LLMManager(accounts=new_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    # Verify state was restored
    healthy = mgr2.get_healthy_accounts()
    assert len(healthy) == 1
    assert healthy[0].account_id == 1
    assert mgr2.accounts[1].is_quarantined is True
    assert mgr2.accounts[2].is_quarantined is True


# ==============================================================================
# CRITERION 3: Full pipeline run completes with only 1 of 3 accounts healthy
# ==============================================================================

def test_full_pipeline_run_completes_with_only_1_of_3_accounts_healthy(
    temp_state_file, test_notifier, mock_accounts
):
    """Verifies that all agent roles cleanly degrade and execute when only 1 account is healthy."""
    mgr = LLMManager(accounts=mock_accounts, state_path=temp_state_file, notifier_instance=test_notifier)

    # Quarantine Accounts 2 and 3
    mgr.quarantine_account(2, reason="Restricted in simulation")
    mgr.quarantine_account(3, reason="Restricted in simulation")

    assert len(mgr.get_healthy_accounts()) == 1
    sole_healthy = mgr.get_healthy_accounts()[0]
    assert sole_healthy.account_id == 1

    # Verify ALL roles now resolve to Account 1
    for role in AgentRole:
        acc = mgr.get_assigned_account(role)
        assert acc.account_id == 1, f"Role {role} should degrade to healthy Account 1"

    # Set up mock client on Account 1
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"topic": "Quantum Encryption", "hook": "Your passwords are obsolete"}'
    mock_resp.parsed = None
    mock_client.models.generate_content.return_value = mock_resp
    mgr._clients[1] = mock_client

    # Execute calls across Creative Director, Copywriter, Strategist
    text1, _ = mgr.generate_content(AgentRole.CREATIVE_DIRECTOR, contents="Director concept")
    text2, _ = mgr.generate_content(AgentRole.COPYWRITER, contents="Copywriter script")
    text3, _ = mgr.generate_content(AgentRole.STRATEGIST, contents="Scheduling JSON")

    assert "Quantum Encryption" in text1
    assert "Quantum Encryption" in text2
    assert "Quantum Encryption" in text3
    assert mock_client.models.generate_content.call_count == 3
