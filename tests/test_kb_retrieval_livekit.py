"""KB retrieval capture against a real AgentSession / AgentActivity (text only, fake LLM).

session.run(user_input=) never calls on_user_turn_completed, so these tests drive the
AgentActivity hooks AudioRecognition calls after STT and end-of-turn (as in
docs/spikes/kb_retrieval_spike.py).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from livekit.agents import Agent, AgentSession, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice import audio_recognition as ar

import tuner.plugin as plugin_module
from tuner.plugin import TunerPlugin

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "kb_retrieval_two_turn.json").read_text())
TURNS = ["What is your refund window?", "How long does express shipping take?"]
# Same KB results the spike recorded in the fixture.
KB = {
    seg["query"]: seg["chunks"]
    for seg in FIXTURE["transcript_with_tool_calls"]
    if seg["role"] == "kb_retrieval"
}


def end_of_turn_info(text: str, now: float):
    timing = dict(
        started_speaking_at=now - 0.2, stopped_speaking_at=now, transcription_delay=0.1, end_of_turn_delay=0.2
    )
    base = dict(skip_reply=False, new_transcript=text, transcript_confidence=1.0)
    fields = {f.name for f in dataclasses.fields(ar._EndOfTurnInfo)}
    if "metrics" in fields:  # livekit-agents >= 1.6 groups the timing fields
        return ar._EndOfTurnInfo(**base, metrics=ar._EndOfTurnMetrics(**timing))
    return ar._EndOfTurnInfo(**base, **timing)


class FakeStream(llm.LLMStream):
    def __init__(self, fake: "FakeLLM", *, chat_ctx, tools, conn_options):
        super().__init__(fake, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._fake = fake

    async def _run(self) -> None:
        self._fake.calls.append(
            [(i.role, i.text_content) for i in self._chat_ctx.items if i.type == "message"]
        )
        await asyncio.sleep(0.02)
        self._event_ch.send_nowait(
            llm.ChatChunk(id="c", delta=llm.ChoiceDelta(role="assistant", content=f"Answer {len(self._fake.calls)}"))
        )


class FakeLLM(llm.LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[tuple[str, str | None]]] = []

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **_):
        return FakeStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class RagAgent(Agent):
    def __init__(self, plugin: TunerPlugin | None, modify_turn_ctx: bool) -> None:
        super().__init__(instructions="You are a support agent.")
        self.plugin = plugin
        self.modify_turn_ctx = modify_turn_ctx
        self.new_message_ids: list[str] = []

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        query = new_message.text_content or ""
        chunks = KB[query]
        self.new_message_ids.append(new_message.id)
        if self.modify_turn_ctx:
            turn_ctx.add_message(role="assistant", content="Context:\n" + "\n".join(c["text"] for c in chunks))
        if self.plugin is not None:
            self.plugin.record_kb_retrieval(new_message, chunks, query=query)


class _Room:
    name = "kb-room"
    remote_participants: dict = {}

    def on(self, _event):
        return lambda fn: fn


def make_plugin(session: AgentSession) -> TunerPlugin:
    ctx = SimpleNamespace(room=_Room(), job=SimpleNamespace(id="AJ_kb"), add_shutdown_callback=lambda _cb: None)
    return TunerPlugin(
        session, ctx, api_key="k", workspace_id=1, agent_id="a", call_type="web_call", forward_traces=False
    )


async def run_turns(*, with_plugin: bool, preemptive: bool, modify_turn_ctx: bool = True):
    fake = FakeLLM()
    session = AgentSession(llm=fake, turn_handling={"preemptive_generation": {"enabled": preemptive}})
    plugin = make_plugin(session) if with_plugin else None
    agent = RagAgent(plugin, modify_turn_ctx)
    await session.start(agent, record=False)
    act = session._activity
    for text in TURNS:
        await asyncio.sleep(0.2)
        now = time.time()
        if preemptive:
            act.on_preemptive_generation(
                ar._PreemptiveGenerationInfo(new_transcript=text, transcript_confidence=1.0, started_speaking_at=now - 0.2)
            )
            await asyncio.sleep(0.01)
        n_before = _assistant_count(session)
        act.on_end_of_turn(end_of_turn_info(text, now))
        for _ in range(200):
            await asyncio.sleep(0.02)
            if _assistant_count(session) > n_before:
                break
    await asyncio.sleep(0.1)
    return session, plugin, agent, fake


def _assistant_count(session: AgentSession) -> int:
    return sum(1 for i in session.history.items if i.type == "message" and i.role == "assistant")


async def submitted_payload(session, plugin, monkeypatch) -> dict:
    sent: list[dict] = []

    async def fake_submit(payload, _config):
        sent.append(payload)

    monkeypatch.setattr(plugin_module, "submit_call", fake_submit)
    await plugin._on_shutdown("test")
    await session.aclose()
    assert len(sent) == 1
    return sent[0]


@pytest.mark.parametrize("preemptive", [False, True])
async def test_llm_input_identical_with_and_without_plugin(preemptive):
    s1, _, _, without = await run_turns(with_plugin=False, preemptive=preemptive)
    await s1.aclose()
    s2, plugin, _, with_ = await run_turns(with_plugin=True, preemptive=preemptive)
    await s2.aclose()

    assert len(plugin._kb_retrievals) == 2
    # The preemptive path makes an extra (cancelled) call; compare what the replies were built from.
    assert without.calls[-1] == with_.calls[-1]
    if not preemptive:
        assert without.calls == with_.calls
    assert any("Express shipping costs" in (text or "") for _, text in with_.calls[-1])


async def test_two_turn_order_from_real_session_matches_fixture(monkeypatch):
    session, plugin, agent, _ = await run_turns(with_plugin=True, preemptive=False)
    payload = await submitted_payload(session, plugin, monkeypatch)
    segments = payload["transcript_with_tool_calls"]
    expected = FIXTURE["transcript_with_tool_calls"]

    assert [s["role"] for s in segments] == [s["role"] for s in expected]
    retrievals = [s for s in segments if s["role"] == "kb_retrieval"]
    users = [s for s in segments if s["role"] == "user"]
    assert [r["user_turn_id"] for r in retrievals] == [u["id"] for u in users] == agent.new_message_ids
    assert [r["chunks"] for r in retrievals] == [s["chunks"] for s in expected if s["role"] == "kb_retrieval"]
    assert [r["query"] for r in retrievals] == TURNS
    assert "Refunds" not in payload["transcript"]


async def test_two_turn_order_from_real_session_with_langgraph_accumulator(monkeypatch):
    from tuner_langchain import TunerAccumulator

    session, plugin, _, _ = await run_turns(with_plugin=True, preemptive=False)
    plugin._lg_accumulator = TunerAccumulator()
    payload = await submitted_payload(session, plugin, monkeypatch)

    assert [s["role"] for s in payload["transcript_with_tool_calls"]] == [
        "user", "kb_retrieval", "agent", "user", "kb_retrieval", "agent",
    ]


async def test_preemptive_reply_with_unmodified_turn_ctx_links_by_text(monkeypatch, caplog):
    # Scenario C in the spike: history keeps the preemptive message, whose id differs.
    session, plugin, agent, _ = await run_turns(with_plugin=True, preemptive=True, modify_turn_ctx=False)
    payload = await submitted_payload(session, plugin, monkeypatch)
    segments = payload["transcript_with_tool_calls"]
    users = [s for s in segments if s["role"] == "user"]

    assert not set(agent.new_message_ids) & {u["id"] for u in users}, "spike precondition: ids differ"
    assert [s["role"] for s in segments] == ["user", "kb_retrieval", "agent", "user", "kb_retrieval", "agent"]
    assert [s["user_turn_id"] for s in segments if s["role"] == "kb_retrieval"] == [u["id"] for u in users]
    assert not [r for r in caplog.records if r.name == "tuner.mapper" and r.levelname == "WARNING"]
