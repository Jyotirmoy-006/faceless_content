"""Unit tests for ModelQuotaTracker and quota-aware auto-downgrade model routing."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.core.llm_manager import (
    AgentRole,
    AllModelsExhaustedError,
    LLMAccount,
    LLMManager,
    MODEL_DAILY_LIMITS,
    MODEL_FALLBACK_CHAIN,
    ModelQuotaTracker,
)


@pytest.fixture
def temp_quota_file(tmp_path: Path) -> Path:
    """Fixture providing an isolated temp path for model quota tracking."""
    return tmp_path / "test_model_quota_state.json"


@pytest.fixture
def mock_accounts() -> list[LLMAccount]:
    """Fixture providing test LLM accounts."""
    return [
        LLMAccount(account_id=1, api_key="test_key_account_1"),
        LLMAccount(account_id=2, api_key="test_key_account_2"),
        LLMAccount(account_id=3, api_key="test_key_account_3"),
    ]


def test_model_quota_tracker_records_and_persists(temp_quota_file: Path):
    """Verifies that ModelQuotaTracker accurately records calls and persists to disk."""
    tracker = ModelQuotaTracker(state_path=temp_quota_file)
    model = "gemini-3.1-pro-preview"

    assert tracker.get_used(model) == 0
    assert tracker.get_remaining(model) == MODEL_DAILY_LIMITS[model]
    assert not tracker.is_exhausted(model)

    tracker.record_call(model)
    tracker.record_call(model)

    assert tracker.get_used(model) == 2
    assert tracker.get_remaining(model) == MODEL_DAILY_LIMITS[model] - 2

    # Reload from disk into a fresh tracker instance
    tracker2 = ModelQuotaTracker(state_path=temp_quota_file)
    assert tracker2.get_used(model) == 2
    assert tracker2.get_remaining(model) == MODEL_DAILY_LIMITS[model] - 2


def test_model_quota_tracker_midnight_reset(temp_quota_file: Path):
    """Verifies that ModelQuotaTracker resets counters when date rolls over."""
    tracker = ModelQuotaTracker(state_path=temp_quota_file)
    model = "gemini-3.6-flash"

    tracker.record_call(model)
    assert tracker.get_used(model) == 1

    # Simulate date rollover by altering the in-memory date key
    tracker._date_key = "2020-01-01"
    tracker._check_reset()

    # Should be reset to 0 for the new day
    assert tracker.get_used(model) == 0
    assert tracker.get_remaining(model) == MODEL_DAILY_LIMITS[model]


def test_model_quota_status_structure(temp_quota_file: Path):
    """Verifies get_status returns expected structure for all configured models."""
    tracker = ModelQuotaTracker(state_path=temp_quota_file)
    tracker.record_call("gemini-3.5-flash-lite")

    status = tracker.get_status()
    assert "date" in status
    assert "models" in status

    lite_info = status["models"]["gemini-3.5-flash-lite"]
    assert lite_info["used"] == 1
    assert lite_info["limit"] == 1500
    assert lite_info["remaining"] == 1499
    assert not lite_info["exhausted"]


def test_resolve_model_auto_downgrade_when_pro_exhausted(
    temp_quota_file: Path,
    mock_accounts: list[LLMAccount],
    tmp_path: Path
):
    """Verifies that resolve_model downgrades from Pro to Flash when Pro limit is reached."""
    custom_tracker = ModelQuotaTracker(state_path=temp_quota_file)
    pro_model = "gemini-3.1-pro-preview"
    flash_model = "gemini-3.6-flash"

    # Fill Pro quota up to its limit
    pro_limit = MODEL_DAILY_LIMITS[pro_model]
    for _ in range(pro_limit):
        custom_tracker.record_call(pro_model)

    assert custom_tracker.is_exhausted(pro_model)

    mgr = LLMManager(
        accounts=mock_accounts,
        state_path=tmp_path / "llm_state.json",
        notifier_instance=MagicMock()
    )

    with patch("pipeline.core.llm_manager.model_quota_tracker", custom_tracker):
        # Creative Director prefers Pro, but should downgrade to Flash
        resolved = mgr.resolve_model(AgentRole.CREATIVE_DIRECTOR)
        assert resolved == flash_model


def test_resolve_model_auto_downgrade_to_flash_lite(
    temp_quota_file: Path,
    mock_accounts: list[LLMAccount],
    tmp_path: Path
):
    """Verifies that resolve_model downgrades to Flash-Lite when both Pro and Flash are exhausted."""
    custom_tracker = ModelQuotaTracker(state_path=temp_quota_file)
    pro_model = "gemini-3.1-pro-preview"
    flash_model = "gemini-3.6-flash"
    lite_model = "gemini-3.5-flash-lite"

    # Exhaust both Pro and Flash
    for _ in range(MODEL_DAILY_LIMITS[pro_model]):
        custom_tracker.record_call(pro_model)
    for _ in range(MODEL_DAILY_LIMITS[flash_model]):
        custom_tracker.record_call(flash_model)

    assert custom_tracker.is_exhausted(pro_model)
    assert custom_tracker.is_exhausted(flash_model)
    assert not custom_tracker.is_exhausted(lite_model)

    mgr = LLMManager(
        accounts=mock_accounts,
        state_path=tmp_path / "llm_state.json",
        notifier_instance=MagicMock()
    )

    with patch("pipeline.core.llm_manager.model_quota_tracker", custom_tracker):
        resolved = mgr.resolve_model(AgentRole.CREATIVE_DIRECTOR)
        assert resolved == lite_model


def test_all_models_exhausted_raises_error(
    temp_quota_file: Path,
    mock_accounts: list[LLMAccount],
    tmp_path: Path
):
    """Verifies that AllModelsExhaustedError is raised when entire fallback chain is exhausted."""
    custom_tracker = ModelQuotaTracker(state_path=temp_quota_file)

    for model in MODEL_FALLBACK_CHAIN:
        limit = MODEL_DAILY_LIMITS[model]
        for _ in range(limit):
            custom_tracker.record_call(model)
        assert custom_tracker.is_exhausted(model)

    mgr = LLMManager(
        accounts=mock_accounts,
        state_path=tmp_path / "llm_state.json",
        notifier_instance=MagicMock()
    )

    with patch("pipeline.core.llm_manager.model_quota_tracker", custom_tracker):
        with pytest.raises(AllModelsExhaustedError):
            mgr.resolve_model(AgentRole.CREATIVE_DIRECTOR)


def test_generate_content_records_quota_usage(
    temp_quota_file: Path,
    mock_accounts: list[LLMAccount],
    tmp_path: Path
):
    """Verifies that successful generate_content calls record usage in quota tracker."""
    custom_tracker = ModelQuotaTracker(state_path=temp_quota_file)
    mgr = LLMManager(
        accounts=mock_accounts,
        state_path=tmp_path / "llm_state.json",
        notifier_instance=MagicMock()
    )

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Concept text"
    mock_resp.parsed = None
    mock_client.models.generate_content.return_value = mock_resp
    mgr._clients[1] = mock_client

    with patch("pipeline.core.llm_manager.model_quota_tracker", custom_tracker):
        assert custom_tracker.get_used("gemini-3.1-pro-preview") == 0
        raw_text, _ = mgr.generate_content(
            role=AgentRole.CREATIVE_DIRECTOR,
            contents="Generate concepts"
        )
        assert raw_text == "Concept text"
        assert custom_tracker.get_used("gemini-3.1-pro-preview") == 1

