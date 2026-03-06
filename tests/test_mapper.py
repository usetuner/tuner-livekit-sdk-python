"""Tests for livekit_agents_tuner.mapper"""

import pytest
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
    """Maps the exact user-provided 4-item history and merges tool output correctly."""
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

    assert len(segments) == 3

    assert segments[0]["role"] == "user"
    assert segments[0]["metadata"]["id"] == "item_6fef49b68773"
    assert segments[0]["metadata"]["interrupted"] is False
    assert (
        segments[0]["text"]
        == "Hello. What is the first available slot for two persons tomorrow between eight and twelve?"
    )

    assert segments[1]["role"] == "agent_function"
    tool = segments[1]["tool"]
    assert tool["name"] == "check_table_availability"
    assert tool["request_id"] == "call_3DxJTGulkeLIKVqABS3oe2Ij"
    assert tool["params"] == {"date": "2024-04-28", "guests": 2, "time": "08:00"}
    assert tool["result"] == {"message": "Table is available on 2024-04-28 at 08:00 for 2 guests."}
    assert tool["is_error"] is False
    assert tool["error"] is None
    expected_start_ms = max(0, int((items[1].created_at - 1772730379.491848) * 1000))
    expected_end_ms = max(0, int((items[2].created_at - 1772730379.491848) * 1000))
    assert tool["start_ms"] == expected_start_ms
    assert tool["end_ms"] == expected_end_ms

    expected_agent_start_ms = max(0, int((items[3].metrics["started_speaking_at"] - 1772730379.491848) * 1000))
    expected_agent_end_ms = max(0, int((items[3].metrics["stopped_speaking_at"] - 1772730379.491848) * 1000))
    assert segments[2]["start_ms"] == expected_agent_start_ms
    assert segments[2]["end_ms"] == expected_agent_end_ms
    assert segments[2]["metadata"]["id"] == "item_4dd0f89440d5"
    assert segments[2]["metadata"]["interrupted"] is True
    assert (
        segments[2]["text"]
        == "The first available slot for two persons tomorrow is at 08:00. Would you like to"
    )



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

    items = [
        user_msg("Hello", created_at=session_start),
        agent_msg("Hi there!", created_at=session_start + 1),
    ]

    payload = to_create_call_request(state, items, config, ctx)

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

    items = [
        user_msg("What's available?", created_at=session_start),
        func_call("call_1", "check_availability", '{"date": "2024-06-15"}'),
        func_output("call_1", "Two slots available", is_error=False),
        agent_msg("I found two available slots.", created_at=session_start + 1),
    ]

    payload = to_create_call_request(state, items, config, ctx)

    segments = payload["transcript_with_tool_calls"]
    assert len(segments) >= 3

    # Verify function call was included
    func_segments = [s for s in segments if s["role"] == "agent_function"]
    assert len(func_segments) == 1
    assert func_segments[0]["tool"]["name"] == "check_availability"
    assert func_segments[0]["tool"]["params"]["date"] == "2024-06-15"
    assert func_segments[0]["tool"]["result"]["message"] == "Two slots available"


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

    items = [user_msg("Hello from phone", created_at=100.0)]

    payload = to_create_call_request(state, items, config, ctx)

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

    items = [user_msg("Hi", created_at=100.0)]

    payload = to_create_call_request(state, items, config, ctx)

    meta = payload["general_meta_data_raw"]
    assert meta["custom_field"] == "custom_value"
    assert meta["team"] == "support"
    assert "usage_summary" in meta


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

    items = [user_msg("Hi", created_at=100.0)]

    payload = to_create_call_request(state, items, config, ctx)

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

    items = [user_msg("Hi", created_at=100.0)]

    payload = to_create_call_request(state, items, config, ctx)

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

    payload = to_create_call_request(state, items, config, ctx)

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
    assert "2024-06-13" in func_seg["tool"]["result"]["message"]

    # Verify transcript
    assert "transcript" in payload
    assert "user: Hello?" in payload["transcript"]
    assert "agent: Hello!" in payload["transcript"]
    assert "user: Yes" in payload["transcript"]

    # Verify metadata
    assert payload["general_meta_data_raw"]["livekit_job_id"]
    assert payload["general_meta_data_raw"]["livekit_room_name"] == "console"
    assert "usage_summary" in payload["general_meta_data_raw"]
