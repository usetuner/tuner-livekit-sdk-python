"""Tests for livekit_agents_tuner.mapper"""

from __future__ import annotations

import pytest
from livekit.agents import AgentSession
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput

from livekit_agents_tuner.mapper import (
    map_history_to_segments,
    to_create_call_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def user_msg(text: str, created_at: float = 1_700_000_000.0) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=[text],
        created_at=created_at,
        metrics={
            "started_speaking_at": created_at,
            "stopped_speaking_at": created_at,
        },
    )


def agent_msg(text: str, created_at: float = 1_700_000_001.0) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=[text],
        created_at=created_at,
        metrics={
            "started_speaking_at": created_at,
            "stopped_speaking_at": created_at,
        },
    )


def system_msg(text: str = "You are helpful.") -> ChatMessage:
    return ChatMessage(role="system", content=[text])


def func_call(call_id: str, name: str, args: str = "{}") -> FunctionCall:
    return FunctionCall(call_id=call_id, name=name, arguments=args)


def func_output(call_id: str, output: str, is_error: bool = False) -> FunctionCallOutput:
    return FunctionCallOutput(call_id=call_id, output=output, is_error=is_error)



def test_map_history_to_segments_with_provided_restaurant_slot_values():
    """Maps the exact user-provided 4-item history: user, agent_function, agent_result, agent."""
    items = [
        ChatMessage(
            id="item_6fef49b68773",
            role="user",
            content=[
                "Hello. What is the first available slot for two persons tomorrow between eight and twelve?"
            ],
            interrupted=False,
            transcript_confidence=0.9951172,
            extra={},
            metrics={
                "started_speaking_at": 1772730374.590425,
                "stopped_speaking_at": 1772730379.283241,
                "transcription_delay": 0.20846891403198242,
                "end_of_turn_delay": 0.5001420974731445,
                "on_user_turn_completed_delay": 1.1541997082531452e-05,
            },
            created_at=1772730379.491848,
        ),
        FunctionCall(
            id="item_bab20b0ad46d/fnc_0",
            call_id="call_3DxJTGulkeLIKVqABS3oe2Ij",
            arguments='{"date":"2024-04-28","guests":2,"time":"08:00"}',
            name="check_table_availability",
            created_at=1772730380.4851992,
            extra={},
            group_id=None,
        ),
        FunctionCallOutput(
            id="item_9632d7a79fff",
            name="check_table_availability",
            call_id="call_3DxJTGulkeLIKVqABS3oe2Ij",
            output="Table is available on 2024-04-28 at 08:00 for 2 guests.",
            is_error=False,
            created_at=1772730380.495991,
        ),
        ChatMessage(
            id="item_4dd0f89440d5",
            role="assistant",
            content=[
                "The first available slot for two persons tomorrow is at 08:00. Would you like to"
            ],
            interrupted=True,
            transcript_confidence=None,
            extra={},
            metrics={
                "started_speaking_at": 1772730381.561763,
                "stopped_speaking_at": 1772730385.875136,
                "llm_node_ttft": 0.5310274169896729,
                "tts_node_ttfb": 0.512128375004977,
                "e2e_latency": 2.27852201461792,
            },
            created_at=1772730380.521417,
        ),
    ]

    segments = map_history_to_segments(
        items,
        session_start_ts=1772730379.491848,
        session_end_ts=1772730380.521417,
    )

    # FunctionCall and FunctionCallOutput each produce their own segment now
    assert len(segments) == 4  # user, agent_function, agent_result, agent

    # --- segment 0: user message ---
    assert segments[0]["role"] == "user"
    assert segments[0]["metadata"]["id"] == "item_6fef49b68773"
    assert segments[0]["metadata"]["interrupted"] is False
    assert (
        segments[0]["text"]
        == "Hello. What is the first available slot for two persons tomorrow between eight and twelve?"
    )

    # --- segment 1: function call (no result here; result is in the next segment) ---
    assert segments[1]["role"] == "agent_function"
    fn_tool = segments[1]["tool"]
    assert fn_tool["name"] == "check_table_availability"
    assert fn_tool["request_id"] == "call_3DxJTGulkeLIKVqABS3oe2Ij"
    assert fn_tool["params"] == {"date": "2024-04-28", "guests": 2, "time": "08:00"}
    expected_start_ms = max(0, int((items[1].created_at - 1772730379.491848) * 1000))
    assert segments[1]["start_ms"] == expected_start_ms
    assert "start_ms" not in fn_tool

    # --- segment 2: function output ---
    assert segments[2]["role"] == "agent_result"
    res_tool = segments[2]["tool"]
    assert res_tool["name"] == "check_table_availability"
    assert res_tool["request_id"] == "call_3DxJTGulkeLIKVqABS3oe2Ij"
    assert res_tool["is_error"] is False
    assert res_tool["output"] == "Table is available on 2024-04-28 at 08:00 for 2 guests."
    expected_result_ms = max(0, int((items[2].created_at - 1772730379.491848) * 1000))
    assert segments[2]["start_ms"] == expected_result_ms
    assert "start_ms" not in res_tool

    # --- segment 3: agent reply ---
    expected_agent_start_ms = max(0, int((items[3].metrics["started_speaking_at"] - 1772730379.491848) * 1000))
    expected_agent_end_ms = max(0, int((items[3].metrics["stopped_speaking_at"] - 1772730379.491848) * 1000))
    assert segments[3]["start_ms"] == expected_agent_start_ms
    assert segments[3]["end_ms"] == expected_agent_end_ms
    assert segments[3]["metadata"]["id"] == "item_4dd0f89440d5"
    assert segments[3]["metadata"]["interrupted"] is True
    assert (
        segments[3]["text"]
        == "The first available slot for two persons tomorrow is at 08:00. Would you like to"
    )


def test_chat_message_metadata_fields():
    """All metadata fields from ChatMessage are mapped correctly for both user and agent roles."""
    session_start = 1_700_000_000.0

    user = ChatMessage(
        id="user_id_001",
        role="user",
        content=["Test user message"],
        interrupted=False,
        transcript_confidence=0.987,
        created_at=session_start + 1.0,
        metrics={
            "started_speaking_at": session_start + 0.5,
            "stopped_speaking_at": session_start + 1.5,
            "transcription_delay": 0.21,
        },
    )
    agent = ChatMessage(
        id="agent_id_002",
        role="assistant",
        content=["Test agent reply"],
        interrupted=True,
        transcript_confidence=None,
        created_at=session_start + 2.0,
        metrics={
            "started_speaking_at": session_start + 2.0,
            "stopped_speaking_at": session_start + 3.5,
            "llm_node_ttft": 0.45,
            "tts_node_ttfb": 0.32,
            "e2e_latency": 1.1,
        },
    )

    segments = map_history_to_segments([user, agent], session_start_ts=session_start)

    assert len(segments) == 2

    # --- user segment ---
    u = segments[0]
    assert u["role"] == "user"
    assert u["text"] == "Test user message"
    assert u["start_ms"] == 500   # (session_start + 0.5 - session_start) * 1000
    assert u["end_ms"] == 1500
    assert u["metadata"]["id"] == "user_id_001"
    assert u["metadata"]["interrupted"] is False
    assert u["metadata"]["transcript_confidence"] == pytest.approx(0.987)
    # stt_node_ttfb is sourced from transcription_delay (user-only metric)
    assert u["metadata"]["stt_node_ttfb"] == pytest.approx(0.21)
    assert u["metadata"]["llm_node_ttft"] is None
    assert u["metadata"]["tts_node_ttfb"] is None
    assert u["metadata"]["e2e_latency"] is None

    # --- agent segment ---
    a = segments[1]
    assert a["role"] == "agent"
    assert a["text"] == "Test agent reply"
    assert a["start_ms"] == 2000
    assert a["end_ms"] == 3500
    assert a["metadata"]["id"] == "agent_id_002"
    assert a["metadata"]["interrupted"] is True
    assert a["metadata"]["transcript_confidence"] is None
    assert a["metadata"]["llm_node_ttft"] == pytest.approx(0.45)
    assert a["metadata"]["tts_node_ttfb"] == pytest.approx(0.32)
    assert a["metadata"]["e2e_latency"] == pytest.approx(1.1)
    # Agent messages have no transcription_delay, so stt_node_ttfb is None
    assert a["metadata"]["stt_node_ttfb"] is None


def test_function_call_metadata_fields():
    """FunctionCall segment has correct tool fields: name, request_id, params, start_ms."""
    session_start = 1_700_000_000.0
    call_ts = session_start + 3.5

    item = FunctionCall(
        call_id="call_abc123",
        name="book_table",
        arguments='{"date": "2024-06-15", "guests": 4, "time": "19:00"}',
        created_at=call_ts,
    )

    segments = map_history_to_segments([item], session_start_ts=session_start)

    assert len(segments) == 1
    seg = segments[0]
    assert seg["role"] == "agent_function"
    assert seg["start_ms"] == 3500

    tool = seg["tool"]
    assert tool["name"] == "book_table"
    assert tool["request_id"] == "call_abc123"
    assert tool["params"] == {"date": "2024-06-15", "guests": 4, "time": "19:00"}
    # No result fields on a call segment
    assert "output" not in tool
    assert "is_error" not in tool
    assert "start_ms" not in tool


def test_function_call_output_metadata_fields():
    """FunctionCallOutput produces an agent_result segment with correct tool fields."""
    session_start = 1_700_000_000.0

    # Successful output
    success = FunctionCallOutput(
        call_id="call_abc123",
        name="book_table",
        output="Booking confirmed for 4 guests on 2024-06-15 at 19:00.",
        is_error=False,
        created_at=session_start + 4.0,
    )
    # Error output
    failure = FunctionCallOutput(
        call_id="call_xyz999",
        name="book_table",
        output="No availability on that date.",
        is_error=True,
        created_at=session_start + 5.0,
    )

    segments = map_history_to_segments([success, failure], session_start_ts=session_start)

    assert len(segments) == 2

    # --- success segment ---
    ok = segments[0]
    assert ok["role"] == "agent_result"
    assert ok["start_ms"] == 4000
    assert ok["tool"]["name"] == "book_table"
    assert ok["tool"]["request_id"] == "call_abc123"
    assert ok["tool"]["is_error"] is False
    assert ok["tool"]["output"] == "Booking confirmed for 4 guests on 2024-06-15 at 19:00."
    assert "error" not in ok["tool"]
    assert "start_ms" not in ok["tool"]

    # --- error segment ---
    err = segments[1]
    assert err["role"] == "agent_result"
    assert err["start_ms"] == 5000
    assert err["tool"]["name"] == "book_table"
    assert err["tool"]["request_id"] == "call_xyz999"
    assert err["tool"]["is_error"] is True
    assert err["tool"]["error"] == "No availability on that date."
    assert "output" not in err["tool"]
    assert "start_ms" not in err["tool"]


# ---------------------------------------------------------------------------
# to_create_call_request
# ---------------------------------------------------------------------------


class MockJob:
    """Mock LiveKit JobContext.job."""

    def __init__(self, job_id: str = "job_12345"):
        self.id = job_id


class MockRemoteParticipant:
    """Mock LiveKit RemoteParticipant."""

    def __init__(self, kind: int = 0):  # kind=0 for regular participant
        self.kind = kind


class MockRoom:
    """Mock LiveKit Room."""

    def __init__(self, name: str = "console", participants: dict | None = None):
        self.name = name
        self.remote_participants = participants or {}


class MockJobContext:
    """Mock LiveKit JobContext."""

    def __init__(
        self, job_id: str = "job_12345", room_name: str = "console", participants: dict | None = None
    ):
        self.job = MockJob(job_id)
        self.room = MockRoom(room_name, participants)


class MockComponent:
    def __init__(self, model: str):
        self.model = model


def make_mock_session(
    stt_model: str | None = None,
    llm_model: str | None = None,
    tts_model: str | None = None,
) -> "AgentSession":
    """Return a MagicMock typed as AgentSession for use in tests."""
    from unittest.mock import MagicMock

    from livekit.agents import AgentSession

    mock = MagicMock(spec=AgentSession)
    mock.stt = MockComponent(stt_model) if stt_model else None
    mock.llm = MockComponent(llm_model) if llm_model else None
    mock.tts = MockComponent(tts_model) if tts_model else None
    return mock  # type: ignore[return-value]


def test_to_create_call_request_basic():
    """Test basic payload generation with web_call type."""
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    session_start = 1_700_000_000.0
    session_end = 1_700_000_010.0

    state = SessionState(start_timestamp=session_start, end_timestamp=session_end)
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="test_agent",
        call_type="web_call",
    )
    ctx = MockJobContext(job_id="test_job_1")
    session = make_mock_session()

    items = [
        user_msg("Hello", created_at=session_start),
        agent_msg("Hi there!", created_at=session_start + 1),
    ]

    payload = to_create_call_request(session, state, items, config, ctx)

    assert payload["call_id"] == "test_job_1"
    assert payload["call_type"] == "web_call"
    assert payload["start_timestamp"] == int(session_start)
    assert payload["end_timestamp"] == int(session_end)
    assert payload["duration_ms"] == 10000
    assert payload["call_status"] == "completed"
    assert payload["general_meta_data_raw"]["livekit_job_id"] == "test_job_1"
    assert payload["general_meta_data_raw"]["livekit_room_name"] == "console"
    assert "transcript" in payload
    assert "user: Hello" in payload["transcript"]
    assert "agent: Hi there!" in payload["transcript"]


def test_to_create_call_request_with_function_calls():
    """Test payload with function calls extracted from history."""
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    session_start = 1_700_000_000.0

    state = SessionState(
        start_timestamp=session_start,
        end_timestamp=session_start + 5,
    )
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="test_agent",
    )
    ctx = MockJobContext()
    session = make_mock_session()

    items = [
        user_msg("What's available?", created_at=session_start),
        func_call("call_1", "check_availability", '{"date": "2024-06-15"}'),
        func_output("call_1", "Two slots available", is_error=False),
        agent_msg("I found two available slots.", created_at=session_start + 1),
    ]

    payload = to_create_call_request(session, state, items, config, ctx)

    segments = payload["transcript_with_tool_calls"]
    assert len(segments) >= 3

    # Verify function call segment
    func_segments = [s for s in segments if s["role"] == "agent_function"]
    assert len(func_segments) == 1
    assert func_segments[0]["tool"]["name"] == "check_availability"
    assert func_segments[0]["tool"]["params"]["date"] == "2024-06-15"

    # Verify function output is a separate agent_result segment
    result_segments = [s for s in segments if s["role"] == "agent_result"]
    assert len(result_segments) == 1
    assert result_segments[0]["tool"]["output"] == "Two slots available"
    assert result_segments[0]["tool"]["is_error"] is False


def test_to_create_call_request_with_sip_detection():
    """Test that is_sip flag correctly sets call_type to 'phone_call'."""
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    state = SessionState(start_timestamp=100.0, end_timestamp=110.0, is_sip=True, caller_phone_number="+1234567890")
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="test_agent",
        call_type=None,  # Auto-detect
    )
    ctx = MockJobContext()
    session = make_mock_session()

    items = [user_msg("Hello from phone", created_at=100.0)]

    payload = to_create_call_request(session, state, items, config, ctx)

    assert payload["call_type"] == "phone_call"
    assert payload["caller_phone_number"] == "+1234567890"


def test_to_create_call_request_with_extra_metadata():
    """Test that extra_metadata is merged into general_meta_data_raw."""
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    state = SessionState(start_timestamp=100.0, end_timestamp=110.0)
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="test_agent",
        extra_metadata={"custom_field": "custom_value", "team": "support"},
    )
    ctx = MockJobContext()
    session = make_mock_session()

    items = [user_msg("Hi", created_at=100.0)]

    payload = to_create_call_request(session, state, items, config, ctx)

    meta = payload["general_meta_data_raw"]
    assert meta["custom_field"] == "custom_value"
    assert meta["team"] == "support"
    assert "usage_token" in meta


def test_to_create_call_request_with_error():
    """Test that close_error is reflected in call_successful field."""
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    state = SessionState(start_timestamp=100.0, end_timestamp=110.0, close_error=RuntimeError("Connection failed"))
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="test_agent",
    )
    ctx = MockJobContext()
    session = make_mock_session()

    items = [user_msg("Hi", created_at=100.0)]

    payload = to_create_call_request(session, state, items, config, ctx)

    assert payload["call_successful"] is False
    assert payload["call_status"] == "error"


def test_to_create_call_request_with_shutdown_reason():
    """Test that shutdown_reason appears in payload."""
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    state = SessionState(
        start_timestamp=100.0, end_timestamp=110.0, shutdown_reason="user_hang_up"
    )
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="test_agent",
    )
    ctx = MockJobContext()
    session = make_mock_session()

    items = [user_msg("Hi", created_at=100.0)]

    payload = to_create_call_request(session, state, items, config, ctx)

    assert payload["disconnection_reason"] == "user_hang_up"


def test_to_create_call_request_restaurant_booking_scenario():
    """
    Test with a representative conversation from the provided logs.
    This simulates a restaurant booking reservation scenario.
    """
    from livekit_agents_tuner.collector import SessionState
    from livekit_agents_tuner.config import TunerConfig

    session_start = 1_772_724_643.0

    # Create history items matching the provided log output
    items = [
        user_msg("Hello?", created_at=session_start + 2.3),
        agent_msg(
            "Hello! Thank you for calling. How can I assist you today? Are you looking to book a table?",
            created_at=session_start + 2.5,
        ),
        user_msg("Yes. I would like to book a table for two persons.", created_at=session_start + 14.7),
        agent_msg(
            "Great! Could you please provide me with the date and time you would like to book the table for two persons?",
            created_at=session_start + 15.1,
        ),
        user_msg("I want the first slot available tomorrow morning between eight and twelve.", created_at=session_start + 27.3),
        agent_msg(
            "To assist you better, could you please specify the exact date for tomorrow? Also, do you have a preferred time within the 8 AM to 12 PM range, or should I find the earliest available slot for you?",
            created_at=session_start + 27.7,
        ),
        user_msg("Select a salt between eight and twelve.", created_at=session_start + 43.0),
        func_call(
            "call_aQsizBZr6wQJxOdUvJBLMCa2",
            "check_table_availability",
            '{"date":"2024-06-13","guests":2,"time":"08:00"}',
        ),
        func_output("call_aQsizBZr6wQJxOdUvJBLMCa2", "Table is available on 2024-06-13 at 08:00 for 2 guests."),
        agent_msg("We have a table available tomorrow at 8:00 AM for two persons. Would you like me to book this slot for you? If so,", created_at=session_start + 50.5),
        user_msg("Yes.", created_at=session_start + 52.1),
        agent_msg("Could you please provide me with your full name to confirm the booking?", created_at=session_start + 52.1),
    ]

    session_end = session_start + 60.0

    state = SessionState(start_timestamp=session_start, end_timestamp=session_end)
    config = TunerConfig(
        api_key="test_key",
        workspace_id=123,
        agent_id="booking_agent",
    )
    ctx = MockJobContext(room_name="console")
    session = make_mock_session()

    payload = to_create_call_request(session, state, items, config, ctx)

    # Validate basic structure
    assert payload["call_id"]
    assert payload["call_type"] in ("web_call", "phone_call")
    assert payload["start_timestamp"] == int(session_start)
    assert payload["end_timestamp"] == int(session_end)
    assert payload["duration_ms"] == 60000
    assert payload["call_status"] == "completed"

    # Validate segments
    segments = payload["transcript_with_tool_calls"]
    assert len(segments) > 0

    # Check that we have user and agent messages
    user_segments = [s for s in segments if s["role"] == "user"]
    agent_segments = [s for s in segments if s["role"] == "agent"]
    func_segments = [s for s in segments if s["role"] == "agent_function"]

    assert len(user_segments) > 0
    assert len(agent_segments) > 0
    assert len(func_segments) == 1

    # Verify function call details
    func_seg = func_segments[0]
    assert func_seg["tool"]["name"] == "check_table_availability"
    assert func_seg["tool"]["params"]["date"] == "2024-06-13"
    assert func_seg["tool"]["params"]["guests"] == 2

    # Verify function output is a separate agent_result segment
    result_segments = [s for s in segments if s["role"] == "agent_result"]
    assert len(result_segments) == 1
    assert "2024-06-13" in result_segments[0]["tool"]["output"]

    # Verify transcript
    assert "transcript" in payload
    assert "user: Hello?" in payload["transcript"]
    assert "agent: Hello!" in payload["transcript"]
    assert "user: Yes" in payload["transcript"]

    # Verify metadata
    assert payload["general_meta_data_raw"]["livekit_job_id"]
    assert payload["general_meta_data_raw"]["livekit_room_name"] == "console"
    assert "usage_token" in payload["general_meta_data_raw"]
