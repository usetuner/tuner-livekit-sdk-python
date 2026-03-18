"""Tests for tuner.client"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tuner.client import _RETRY_STATUSES, submit_call
from tuner.config import TunerConfig


@pytest.fixture
def config():
    return TunerConfig(
        api_key="tr_api_test",
        workspace_id=1,
        agent_id="test-agent",
        max_retries=2,
        timeout_seconds=5.0,
    )


@pytest.fixture
def payload():
    return {"call_id": "job_123", "call_type": "web_call", "transcript_with_tool_calls": []}


def make_mock_response(status: int, body: dict | str = "") -> MagicMock:
    """Build an aiohttp-style async context manager mock response."""
    resp = AsyncMock()
    resp.status = status
    if isinstance(body, dict):
        resp.json = AsyncMock(return_value=body)
    resp.text = AsyncMock(return_value=str(body))

    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def make_mock_session(response_cm) -> MagicMock:
    """Build an aiohttp.ClientSession mock."""
    session = MagicMock()
    session.post = MagicMock(return_value=response_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    return session_cm


# ---------------------------------------------------------------------------
# Retry status codes
# ---------------------------------------------------------------------------


def test_retry_statuses_include_5xx():
    assert 500 in _RETRY_STATUSES
    assert 502 in _RETRY_STATUSES
    assert 503 in _RETRY_STATUSES
    assert 504 in _RETRY_STATUSES


def test_retry_statuses_exclude_4xx():
    assert 400 not in _RETRY_STATUSES
    assert 404 not in _RETRY_STATUSES
    assert 422 not in _RETRY_STATUSES


# ---------------------------------------------------------------------------
# Successful submission
# ---------------------------------------------------------------------------


async def test_successful_201(config, payload):
    resp_cm = make_mock_response(201, {"id": 42, "is_new": True})
    session_cm = make_mock_session(resp_cm)

    with patch("aiohttp.ClientSession", return_value=session_cm):
        await submit_call(payload, config)  # Should not raise


# ---------------------------------------------------------------------------
# Non-retryable 4xx
# ---------------------------------------------------------------------------


async def test_422_does_not_retry(config, payload):
    resp_cm = make_mock_response(422, {"detail": "bad request"})
    session_cm = make_mock_session(resp_cm)

    call_count = 0
    original_post = MagicMock(return_value=resp_cm)

    def counting_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return resp_cm

    with patch("aiohttp.ClientSession", return_value=session_cm):
        session_cm.__aenter__.return_value.post = counting_post
        await submit_call(payload, config)

    # Should only attempt once (no retries on 4xx)
    assert call_count == 1


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


async def test_url_contains_workspace_and_agent(config, payload):
    resp_cm = make_mock_response(201, {"id": 1, "is_new": True})
    session = MagicMock()
    session.post = MagicMock(return_value=resp_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=session_cm):
        await submit_call(payload, config)

    call_args = session.post.call_args
    url = call_args[0][0]
    assert "workspace_id=1" in url
    assert "agent_remote_identifier=test-agent" in url


# ---------------------------------------------------------------------------
# Authorization header
# ---------------------------------------------------------------------------


async def test_bearer_token_in_header(config, payload):
    resp_cm = make_mock_response(201, {"id": 1, "is_new": True})
    session = MagicMock()
    session.post = MagicMock(return_value=resp_cm)

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=session)
    session_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=session_cm):
        await submit_call(payload, config)

    call_kwargs = session.post.call_args[1]
    assert call_kwargs["headers"]["Authorization"] == "Bearer tr_api_test"
