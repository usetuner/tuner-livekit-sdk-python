"""Tests for livekit_agents_tuner.config"""

import pytest

from livekit_agents_tuner.config import TunerConfig


def test_from_env_reads_env_vars(monkeypatch):
    monkeypatch.setenv("TUNER_API_KEY", "tr_api_test123")
    monkeypatch.setenv("TUNER_WORKSPACE_ID", "42")
    monkeypatch.setenv("TUNER_AGENT_ID", "my-agent")

    config = TunerConfig.from_env()

    assert config.api_key == "tr_api_test123"
    assert config.workspace_id == 42
    assert config.agent_id == "my-agent"
    assert config.base_url == "https://api.usetuner.ai"


def test_from_env_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("TUNER_API_KEY", "tr_api_from_env")
    monkeypatch.setenv("TUNER_WORKSPACE_ID", "1")
    monkeypatch.setenv("TUNER_AGENT_ID", "env-agent")

    config = TunerConfig.from_env(
        api_key="tr_api_explicit",
        workspace_id=99,
        agent_id="explicit-agent",
    )

    assert config.api_key == "tr_api_explicit"
    assert config.workspace_id == 99
    assert config.agent_id == "explicit-agent"


def test_from_env_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TUNER_API_KEY", raising=False)
    monkeypatch.setenv("TUNER_WORKSPACE_ID", "1")
    monkeypatch.setenv("TUNER_AGENT_ID", "agent")

    with pytest.raises(ValueError, match="TUNER_API_KEY"):
        TunerConfig.from_env()


def test_from_env_missing_workspace_id_raises(monkeypatch):
    monkeypatch.setenv("TUNER_API_KEY", "tr_api_x")
    monkeypatch.delenv("TUNER_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("TUNER_AGENT_ID", "agent")

    with pytest.raises(ValueError, match="TUNER_WORKSPACE_ID"):
        TunerConfig.from_env()


def test_from_env_missing_agent_id_raises(monkeypatch):
    monkeypatch.setenv("TUNER_API_KEY", "tr_api_x")
    monkeypatch.setenv("TUNER_WORKSPACE_ID", "1")
    monkeypatch.delenv("TUNER_AGENT_ID", raising=False)

    with pytest.raises(ValueError, match="TUNER_AGENT_ID"):
        TunerConfig.from_env()


def test_from_env_workspace_id_parsed_as_int(monkeypatch):
    monkeypatch.setenv("TUNER_API_KEY", "tr_api_x")
    monkeypatch.setenv("TUNER_WORKSPACE_ID", "123")
    monkeypatch.setenv("TUNER_AGENT_ID", "agent")

    config = TunerConfig.from_env()
    assert isinstance(config.workspace_id, int)
    assert config.workspace_id == 123


def test_from_env_custom_base_url(monkeypatch):
    monkeypatch.setenv("TUNER_API_KEY", "tr_api_x")
    monkeypatch.setenv("TUNER_WORKSPACE_ID", "1")
    monkeypatch.setenv("TUNER_AGENT_ID", "agent")
    monkeypatch.setenv("TUNER_BASE_URL", "https://staging.usetuner.ai")

    config = TunerConfig.from_env()
    assert config.base_url == "https://staging.usetuner.ai"


def test_from_env_defaults():
    config = TunerConfig.from_env(
        api_key="tr_api_x",
        workspace_id=1,
        agent_id="agent",
    )
    assert config.enabled is True
    assert config.max_retries == 3
    assert config.timeout_seconds == 30.0
    assert config.call_type is None
    assert config.recording_url_resolver is None
    assert config.cost_calculator is None
    assert config.extra_metadata is None


def test_direct_construction():
    config = TunerConfig(api_key="tr_api_x", workspace_id=5, agent_id="direct")
    assert config.api_key == "tr_api_x"
    assert config.workspace_id == 5
