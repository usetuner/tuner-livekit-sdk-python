"""Tests for KB retrieval capture: record_kb_retrieval, linking and mapper placement."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from uuid import uuid4

import pytest
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput

from tuner.collector import SessionState
from tuner.config import TunerConfig
from tuner.kb_retrieval import InvalidKbRetrieval, build_kb_retrieval, resolve_user_turn_id
from tuner.mapper import map_history_to_segments, to_create_call_request
from tuner.plugin import TunerPlugin

T0 = 1_700_000_000.0
SECRET = "Refunds within 30 days for unused items"


def user_msg(text: str, at: float, id: str | None = None) -> ChatMessage:
    kwargs = {"id": id} if id else {}
    return ChatMessage(
        role="user",
        content=[text],
        created_at=at,
        metrics={"started_speaking_at": at, "stopped_speaking_at": at + 0.2},
        **kwargs,
    )


def agent_msg(text: str, at: float) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=[text],
        created_at=at,
        metrics={"started_speaking_at": at, "stopped_speaking_at": at + 0.2},
    )


def record(msg, chunks, query=None, *, capture_text=True, at=None):
    return build_kb_retrieval(
        msg, chunks, query, capture_text=capture_text, recorded_at=at if at is not None else msg.created_at + 0.05
    )


def two_turn_history():
    u1 = user_msg("What is your refund window?", T0 + 1, id="item_u1")
    a1 = agent_msg("Thirty days.", T0 + 2)
    u2 = user_msg("And express shipping?", T0 + 3, id="item_u2")
    a2 = agent_msg("One to two days.", T0 + 4)
    return [u1, a1, u2, a2]


def two_turn_retrievals(items):
    u1, _, u2, _ = items
    return [
        record(u1, [{"text": SECRET, "source_id": "refund#1", "score": 0.82}], query="refund window", at=T0 + 1.5),
        record(u2, ["Express shipping arrives in 1 to 2 business days."], at=T0 + 3.5),
    ]


# --- build_kb_retrieval ------------------------------------------------------


def test_accepts_strings_and_dicts_and_copies_inputs():
    msg = user_msg("q", T0, id="item_q")
    chunks = [
        "plain chunk",
        {"text": "dict chunk", "source_id": "doc-1#2", "score": 0.5, "extra": "ignored"},
        {"text": "int id", "source_id": 7, "score": 1},
    ]
    before = msg.model_dump()

    r = record(msg, chunks, query="q")
    chunks[1]["text"] = "mutated after the call"
    chunks.append("added after the call")

    assert msg.model_dump() == before
    assert r.user_turn_id == "item_q"
    assert r.query == "q"
    assert list(r.chunks) == [
        {"text": "plain chunk"},
        {"text": "dict chunk", "source_id": "doc-1#2", "score": 0.5},
        {"text": "int id", "source_id": 7, "score": 1},
    ]
    assert r.chunk_text_omitted is False


def test_flag_off_keeps_source_id_and_score_but_no_text():
    msg = user_msg("q", T0, id="item_q")
    r = record(msg, [SECRET, {"text": SECRET, "source_id": "d#1", "score": 0.3}], capture_text=False)
    assert list(r.chunks) == [{}, {"source_id": "d#1", "score": 0.3}]
    assert r.chunk_text_omitted is True
    assert SECRET not in repr(r)


@pytest.mark.parametrize(
    "chunks",
    [
        "a string, not a list",
        None,
        [None],
        [{"source_id": "no text"}],
        [{"text": 5}],
        [{"text": "x", "score": "high"}],
        [{"text": "x", "score": True}],
        [{"text": "x", "score": math.nan}],
        [{"text": "x", "source_id": ["list"]}],
    ],
)
def test_bad_chunks_raise_invalid(chunks):
    with pytest.raises(InvalidKbRetrieval):
        record(user_msg("q", T0, id="item_q"), chunks)


def test_bad_query_or_message_raises_invalid():
    with pytest.raises(InvalidKbRetrieval):
        record(user_msg("q", T0, id="item_q"), ["x"], query=42)
    with pytest.raises(InvalidKbRetrieval):
        build_kb_retrieval(SimpleNamespace(id=None), ["x"], None, capture_text=True, recorded_at=T0)


# --- resolve_user_turn_id (preemptive fallback) ------------------------------


def test_resolves_by_id_when_present():
    items = two_turn_history()
    r = record(items[2], ["x"])
    assert resolve_user_turn_id(r, [items[0], items[2]]) == "item_u2"


def test_preemptive_id_mismatch_falls_back_to_same_text_nearest_timestamp():
    # history holds the preemptive copy (different id); "yes" was said twice
    yes_turn1 = user_msg("yes", T0 + 1, id="item_hist_1")
    yes_turn2 = user_msg("yes", T0 + 10, id="item_hist_2")
    other = user_msg("no", T0 + 10.1, id="item_hist_3")
    new_message = user_msg("yes", T0 + 10.3, id="item_new_msg")

    r = record(new_message, ["x"])
    assert resolve_user_turn_id(r, [yes_turn1, yes_turn2, other]) == "item_hist_2"


def test_unresolvable_returns_none():
    r = record(user_msg("never said", T0, id="item_gone"), ["x"])
    assert resolve_user_turn_id(r, two_turn_history()[::2]) is None


# --- mapper placement --------------------------------------------------------


def _roles(segments):
    return [s["role"] for s in segments]


def test_two_turn_order_without_langgraph():
    items = two_turn_history()
    segments = map_history_to_segments(items, T0, kb_retrievals=two_turn_retrievals(items))

    assert _roles(segments) == ["user", "kb_retrieval", "agent", "user", "kb_retrieval", "agent"]
    assert segments[0]["id"] == "item_u1" and segments[0]["metadata"]["id"] == "item_u1"
    assert segments[1] == {
        "role": "kb_retrieval",
        "user_turn_id": "item_u1",
        "start_ms": 1500,
        "query": "refund window",
        "chunks": [{"text": SECRET, "source_id": "refund#1", "score": 0.82}],
    }
    assert segments[4]["user_turn_id"] == "item_u2"
    assert segments[4]["chunks"] == [{"text": "Express shipping arrives in 1 to 2 business days."}]
    assert "query" not in segments[4]


def _accumulator_with_graph_between_turns():
    from tuner_langchain import TunerAccumulator

    acc = TunerAccumulator()
    ns = int(T0 * 1_000_000_000)
    for start_s in (1.3, 3.3):
        root, node = str(uuid4()), str(uuid4())
        acc.on_graph_start(root, ns + int(start_s * 1e9))
        acc.on_node_start("answer_node", node, root, ns + int((start_s + 0.1) * 1e9), {})
        acc.on_node_end(node, ns + int((start_s + 0.3) * 1e9), {})
        acc.on_graph_end(root, ns + int((start_s + 0.4) * 1e9))
    return acc


def test_two_turn_order_with_langgraph_accumulator():
    items = two_turn_history()
    segments = map_history_to_segments(
        items, T0, lg_acc=_accumulator_with_graph_between_turns(), kb_retrievals=two_turn_retrievals(items)
    )
    no_nodes = [s for s in segments if s["role"] != "node_transition"]

    assert "node_transition" in _roles(segments)
    assert _roles(no_nodes) == ["user", "kb_retrieval", "agent", "user", "kb_retrieval", "agent"]
    for i, seg in enumerate(segments):
        if seg["role"] == "kb_retrieval":
            assert segments[i - 1]["role"] == "user"
            assert segments[i - 1]["id"] == seg["user_turn_id"]


@pytest.mark.parametrize("use_langgraph", [False, True])
def test_timestamp_never_moves_a_retrieval_to_another_turn(use_langgraph):
    items = two_turn_history()
    # recorded long after turn 2, but linked to turn 1
    late = record(items[0], ["late chunk"], at=T0 + 50)
    lg_acc = _accumulator_with_graph_between_turns() if use_langgraph else None
    segments = map_history_to_segments(items, T0, lg_acc=lg_acc, kb_retrievals=[late])

    idx = _roles(segments).index("kb_retrieval")
    assert segments[idx - 1]["id"] == "item_u1"


def test_several_retrievals_on_one_turn_keep_recorded_order():
    items = two_turn_history()
    first = record(items[0], ["first"])
    second = record(items[0], ["second"])
    segments = map_history_to_segments(items, T0, kb_retrievals=[first, second])

    assert _roles(segments)[:4] == ["user", "kb_retrieval", "kb_retrieval", "agent"]
    assert segments[1]["chunks"] == [{"text": "first"}]
    assert segments[2]["chunks"] == [{"text": "second"}]


def test_tool_calls_unchanged_and_follow_the_retrieval():
    u1 = user_msg("Book a table", T0 + 1, id="item_u1")
    items = [
        u1,
        FunctionCall(call_id="c1", name="check", arguments="{}", created_at=T0 + 1.5),
        FunctionCallOutput(call_id="c1", name="check", output="ok", is_error=False, created_at=T0 + 1.6),
        agent_msg("Booked.", T0 + 2),
    ]
    plain = map_history_to_segments(items, T0)
    with_kb = map_history_to_segments(items, T0, kb_retrievals=[record(u1, ["menu"])])

    assert _roles(with_kb) == ["user", "kb_retrieval", "agent_function", "agent_result", "agent"]
    assert [s for s in with_kb if s["role"] != "kb_retrieval"] == plain


def test_preemptive_mismatch_links_to_history_id():
    items = two_turn_history()
    new_message = user_msg("And express shipping?", T0 + 3.4, id="item_preempt_new")
    segments = map_history_to_segments(items, T0, kb_retrievals=[record(new_message, ["x"])])

    idx = _roles(segments).index("kb_retrieval")
    assert segments[idx]["user_turn_id"] == "item_u2"
    assert segments[idx - 1]["id"] == "item_u2"


def test_unlinked_retrieval_warns_without_text_and_is_sent_with_original_id(caplog):
    items = two_turn_history()
    stray = user_msg("Something not in history", T0 + 5, id="item_stray")
    r = record(stray, [SECRET], query="secret query text")

    with caplog.at_level(logging.WARNING, logger="tuner.mapper"):
        segments = map_history_to_segments(items, T0, kb_retrievals=[r])

    warnings = [rec for rec in caplog.records if rec.name == "tuner.mapper" and rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "item_stray" in warnings[0].getMessage()
    assert SECRET not in caplog.text
    assert "secret query text" not in caplog.text
    assert "Something not in history" not in caplog.text

    assert segments[-1]["role"] == "kb_retrieval"
    assert segments[-1]["user_turn_id"] == "item_stray"
    assert segments[-1]["chunks"] == [{"text": SECRET}]


def test_flag_off_segment_has_no_text():
    items = two_turn_history()
    r = record(items[0], [SECRET, {"text": SECRET, "source_id": "d#1", "score": 0.4}], capture_text=False)
    seg = next(s for s in map_history_to_segments(items, T0, kb_retrievals=[r]) if s["role"] == "kb_retrieval")

    assert seg["chunk_text_omitted"] is True
    assert seg["chunks"] == [{}, {"source_id": "d#1", "score": 0.4}]
    assert SECRET not in repr(seg)


def test_empty_retrieval_is_sent_as_empty_chunks():
    items = two_turn_history()
    seg = next(
        s for s in map_history_to_segments(items, T0, kb_retrievals=[record(items[0], [])]) if s["role"] == "kb_retrieval"
    )
    assert seg["chunks"] == []
    assert "chunk_text_omitted" not in seg


def test_no_retrievals_keeps_payload_shape_except_user_id():
    items = two_turn_history()
    segments = map_history_to_segments(items, T0)
    assert _roles(segments) == ["user", "agent", "user", "agent"]
    assert segments[0]["id"] == "item_u1"
    assert "id" not in segments[1]


# --- plugin entrypoint -------------------------------------------------------


class _Emitter:
    def on(self, _event):
        return lambda fn: fn


def make_plugin(**kwargs) -> TunerPlugin:
    ctx = SimpleNamespace(room=_Emitter(), job=SimpleNamespace(id="job_1"), add_shutdown_callback=lambda _cb: None)
    return TunerPlugin(_Emitter(), ctx, api_key="k", workspace_id=1, agent_id="a", forward_traces=False, **kwargs)


def test_plugin_records_several_retrievals_per_turn():
    plugin = make_plugin()
    msg = user_msg("q", T0, id="item_q")
    plugin.record_kb_retrieval(msg, ["one"])
    plugin.record_kb_retrieval(msg, [{"text": "two", "score": 0.1}], query="q")
    assert [list(r.chunks) for r in plugin._kb_retrievals] == [[{"text": "one"}], [{"text": "two", "score": 0.1}]]


def test_plugin_never_raises_and_logs_ids_only(caplog):
    plugin = make_plugin()
    msg = user_msg("my card is 4111", T0, id="item_bad")
    with caplog.at_level(logging.WARNING):
        plugin.record_kb_retrieval(msg, [{"text": SECRET, "score": "bad"}], query="secret query text")
        plugin.record_kb_retrieval(object(), [SECRET])  # not a message at all

    assert plugin._kb_retrievals == []
    assert "item_bad" in caplog.text
    for leaked in (SECRET, "secret query text", "4111"):
        assert leaked not in caplog.text


def test_plugin_flag_off_and_disabled_plugin():
    off = make_plugin(capture_kb_chunk_text=False)
    off.record_kb_retrieval(user_msg("q", T0, id="item_q"), [SECRET])
    assert off._kb_retrievals[0].chunk_text_omitted is True
    assert off._kb_retrievals[0].chunks == ({},)

    disabled = make_plugin(enabled=False)
    disabled.record_kb_retrieval(user_msg("q", T0, id="item_q"), [SECRET])
    assert disabled._kb_retrievals == []


def test_to_create_call_request_includes_retrievals():
    from unittest.mock import MagicMock

    from livekit.agents import AgentSession

    items = two_turn_history()
    state = SessionState(start_timestamp=T0, end_timestamp=T0 + 5)
    config = TunerConfig(api_key="k", workspace_id=1, agent_id="a", call_type="web_call")
    ctx = SimpleNamespace(job=SimpleNamespace(id="job_1"), room=SimpleNamespace(name="r", remote_participants={}))
    session = MagicMock(spec=AgentSession)
    session.stt = session.llm = session.tts = None

    payload = to_create_call_request(session, state, items, config, ctx, kb_retrievals=two_turn_retrievals(items))

    assert _roles(payload["transcript_with_tool_calls"]) == [
        "user", "kb_retrieval", "agent", "user", "kb_retrieval", "agent",
    ]
    assert "kb_retrieval" not in payload["transcript"]
