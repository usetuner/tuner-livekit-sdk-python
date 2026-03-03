"""Tests for livekit_agents_tuner.mapper"""

import pytest
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput

from livekit_agents_tuner.mapper import (
    build_plain_transcript,
    map_history_to_segments,
    to_create_call_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def user_msg(text: str, created_at: float = 1_700_000_000.0) -> ChatMessage:
    return ChatMessage(role="user", content=[text], created_at=created_at)


def agent_msg(text: str, created_at: float = 1_700_000_001.0) -> ChatMessage:
    return ChatMessage(role="assistant", content=[text], created_at=created_at)


def system_msg(text: str = "You are helpful.") -> ChatMessage:
    return ChatMessage(role="system", content=[text])


def func_call(call_id: str, name: str, args: str = "{}") -> FunctionCall:
    return FunctionCall(call_id=call_id, name=name, arguments=args)


def func_output(call_id: str, output: str, is_error: bool = False) -> FunctionCallOutput:
    return FunctionCallOutput(call_id=call_id, output=output, is_error=is_error)


# ---------------------------------------------------------------------------
# map_history_to_segments
# ---------------------------------------------------------------------------


def test_empty_history():
    assert map_history_to_segments([]) == []


def test_user_message_role():
    segments = map_history_to_segments([user_msg("Hello")])
    assert len(segments) == 1
    assert segments[0]["role"] == "user"
    assert segments[0]["text"] == "Hello"


def test_agent_message_role():
    segments = map_history_to_segments([agent_msg("Hi!")])
    assert len(segments) == 1
    assert segments[0]["role"] == "agent"
    assert segments[0]["text"] == "Hi!"


def test_system_message_is_skipped():
    segments = map_history_to_segments([system_msg()])
    assert segments == []


def test_developer_message_is_skipped():
    msg = ChatMessage(role="developer", content=["internal"])
    assert map_history_to_segments([msg]) == []


def test_timestamps_set_correctly():
    msg = user_msg("Hello", created_at=1_700_000_000.5)
    seg = map_history_to_segments([msg])[0]
    assert seg["start_ms"] == 1_700_000_000_500
    assert "end_ms" not in seg
    assert "duration_ms" not in seg


def test_interrupted_flag_in_metadata():
    msg = ChatMessage(role="user", content=["Hi"], interrupted=True)
    seg = map_history_to_segments([msg])[0]
    assert seg["metadata"]["interrupted"] is True


def test_tool_call_merge():
    fc = func_call("c1", "get_weather", '{"location": "London"}')
    fco = func_output("c1", "sunny")
    segments = map_history_to_segments([fc, fco])

    assert len(segments) == 1
    assert segments[0]["role"] == "agent_function"
    tool = segments[0]["tool"]
    assert tool["name"] == "get_weather"
    assert tool["request_id"] == "c1"
    assert tool["params"] == {"location": "London"}
    assert tool["result"] == "sunny"
    assert tool["is_error"] is False
    assert tool["error"] is None


def test_tool_call_error_merge():
    fc = func_call("c2", "bad_tool", "{}")
    fco = func_output("c2", "API failure", is_error=True)
    segments = map_history_to_segments([fc, fco])

    tool = segments[0]["tool"]
    assert tool["is_error"] is True
    assert tool["error"] == "API failure"
    assert tool["result"] == "API failure"


def test_invalid_json_arguments_passed_as_string():
    fc = func_call("c3", "my_tool", "not-json")
    segments = map_history_to_segments([fc])
    assert segments[0]["tool"]["params"] == "not-json"


def test_orphan_function_output():
    """FunctionCallOutput with no matching FunctionCall → agent_result segment."""
    fco = func_output("orphan", "some result")
    segments = map_history_to_segments([fco])
    assert len(segments) == 1
    assert segments[0]["role"] == "agent_result"
    assert segments[0]["tool"]["result"] == "some result"


def test_mixed_conversation():
    items = [
        user_msg("What's the weather?"),
        func_call("c1", "get_weather", '{"city": "Paris"}'),
        func_output("c1", "rainy"),
        agent_msg("It's rainy in Paris."),
    ]
    segments = map_history_to_segments(items)
    assert len(segments) == 3
    assert segments[0]["role"] == "user"
    assert segments[1]["role"] == "agent_function"
    assert segments[1]["tool"]["result"] == "rainy"
    assert segments[2]["role"] == "agent"


def test_multiple_tool_calls_merge_independently():
    items = [
        func_call("c1", "tool_a", "{}"),
        func_call("c2", "tool_b", "{}"),
        func_output("c2", "result_b"),
        func_output("c1", "result_a"),
    ]
    segments = map_history_to_segments(items)
    assert len(segments) == 2
    # c1 should get result_a
    c1 = next(s for s in segments if s["tool"]["request_id"] == "c1")
    c2 = next(s for s in segments if s["tool"]["request_id"] == "c2")
    assert c1["tool"]["result"] == "result_a"
    assert c2["tool"]["result"] == "result_b"


# ---------------------------------------------------------------------------
# build_plain_transcript
# ---------------------------------------------------------------------------


def test_plain_transcript_basic():
    items = [
        user_msg("Hello"),
        agent_msg("Hi!"),
    ]
    transcript = build_plain_transcript(items)
    assert transcript == "user: Hello\nagent: Hi!"


def test_plain_transcript_skips_system():
    items = [system_msg(), user_msg("Hello")]
    transcript = build_plain_transcript(items)
    assert transcript == "user: Hello"


def test_plain_transcript_empty():
    assert build_plain_transcript([]) == ""


def test_plain_transcript_skips_empty_text():
    """Messages with no text content (audio-only) should not appear."""
    msg = ChatMessage(role="user", content=[])
    transcript = build_plain_transcript([msg])
    assert transcript == ""
