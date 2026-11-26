"""Stage 0 spike: per-turn RAG in livekit-agents 1.5.6, text only, fake LLM.

Drives the real AgentActivity turn hooks (on_preemptive_generation / on_end_of_turn),
which is what AudioRecognition calls after STT + end-of-turn. session.run(user_input=)
goes through generate_reply and never calls on_user_turn_completed, so it can't be used.

Run: .venv/bin/python docs/spikes/kb_retrieval_spike.py
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import Counter

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _PreemptiveGenerationInfo

from tuner.mapper import build_plain_transcript, map_history_to_segments

FIXTURE = "tests/fixtures/kb_retrieval_two_turn.json"

KB = [
    ("refund-policy#1", "Refunds are available within 30 days of purchase for unused items."),
    ("refund-policy#2", "Refunds are issued to the original payment method within 5 business days."),
    ("shipping#1", "Standard shipping takes 3 to 7 business days within the country."),
    ("shipping#2", "Express shipping costs 15 dollars and arrives in 1 to 2 business days."),
    ("hours#1", "Support is open Monday to Friday, 9am to 6pm local time."),
]


STOP = {"is", "are", "the", "a", "to", "of", "your", "what", "how", "does", "do", "in", "and", "for", "within"}


def _vec(text: str) -> Counter[str]:
    words = (w.rstrip("s") for w in re.findall(r"[a-z0-9]+", text.lower()))
    return Counter(w for w in words if w not in STOP)


def retrieve(query: str, k: int = 2) -> list[dict]:
    q = _vec(query)
    scored = []
    for source_id, text in KB:
        d = _vec(text)
        dot = sum(q[w] * d[w] for w in q)
        norm = math.sqrt(sum(v * v for v in q.values())) * math.sqrt(sum(v * v for v in d.values()))
        scored.append({"text": text, "source_id": source_id, "score": round(dot / norm, 3) if norm else 0.0})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]


class FakeStream(llm.LLMStream):
    def __init__(self, fake: FakeLLM, *, chat_ctx, tools, conn_options):
        super().__init__(fake, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._fake = fake

    async def _run(self) -> None:
        self._fake.calls.append([(i.role, i.text_content) for i in self._chat_ctx.items if i.type == "message"])
        await asyncio.sleep(0.05)
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
    def __init__(self, mode: str) -> None:
        super().__init__(instructions="You are a support agent.")
        self.mode = mode  # "turn_ctx" | "noop" | "update_chat_ctx"
        self.seen: list[dict] = []

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        chunks = retrieve(new_message.text_content or "")
        self.seen.append({"new_message_id": new_message.id, "chunks": chunks})
        rag = "Context:\n" + "\n".join(c["text"] for c in chunks)
        if self.mode == "turn_ctx":
            turn_ctx.add_message(role="assistant", content=rag)
        elif self.mode == "update_chat_ctx":
            ctx = self.chat_ctx.copy()
            ctx.add_message(role="system", content="UPDATE_CHAT_CTX " + rag)
            await self.update_chat_ctx(ctx)


def write_fixture(items: list, seen: list[dict], session_start: float) -> None:
    """Real history -> today's mapper -> add Stage 3 fields (top-level id, kb_retrieval after its user turn)."""
    segments = map_history_to_segments(items, session_start_ts=session_start)
    out, turn = [], 0
    for seg in segments:
        out.append(seg)
        if seg["role"] == "user":
            seg["id"] = seg["metadata"]["id"]
            out.append({
                "role": "kb_retrieval",
                "user_turn_id": seen[turn]["new_message_id"],
                "start_ms": seg["end_ms"] + 40,
                "query": seg["text"],
                "chunks": seen[turn]["chunks"],
            })
            turn += 1
    payload = {
        "call_id": "AJ_spikeKbRetrieval01",
        "call_type": "web_call",
        "transcript_with_tool_calls": out,
        "start_timestamp": int(session_start),
        "end_timestamp": int(session_start) + 30,
        "duration_ms": 30000,
        "call_status": "ended",
        "general_meta_data_raw": {"livekit_job_id": "AJ_spikeKbRetrieval01", "livekit_room_name": "spike-kb-room"},
        "transcript": build_plain_transcript(items),
    }
    with open(FIXTURE, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"  wrote {FIXTURE}")


async def run_scenario(name: str, mode: str, preemptive: bool, turns: list[str], fixture: bool = False) -> None:
    fake = FakeLLM()
    agent = RagAgent(mode)
    session = AgentSession(
        llm=fake, turn_handling={"preemptive_generation": {"enabled": preemptive}}
    )
    await session.start(agent, record=False)
    act = session._activity
    session_start = time.time()
    for t, text in enumerate(turns):
        await asyncio.sleep(0.3)
        now = time.time()
        if preemptive:
            act.on_preemptive_generation(
                _PreemptiveGenerationInfo(new_transcript=text, transcript_confidence=1.0, started_speaking_at=now - 0.2)
            )
            await asyncio.sleep(0.01)
        act.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False, new_transcript=text, transcript_confidence=1.0,
                started_speaking_at=now - 0.2, stopped_speaking_at=now,
                transcription_delay=0.1, end_of_turn_delay=0.2,
            )
        )
        n_before = sum(1 for i in session.history.items if i.type == "message" and i.role == "assistant")
        for _ in range(200):
            await asyncio.sleep(0.02)
            n = sum(1 for i in session.history.items if i.type == "message" and i.role == "assistant")
            if n > n_before:
                break
    await asyncio.sleep(0.1)

    print(f"\n=== {name} (mode={mode}, preemptive={preemptive}) ===")
    hist = [i for i in session.history.items if i.type == "message"]
    for i in hist:
        print(f"  history  {i.role:9s} id={i.id}  {i.text_content!r}")
    hist_users = [i for i in hist if i.role == "user"]
    for t, seen in enumerate(agent.seen):
        hid = hist_users[t].id if t < len(hist_users) else None
        print(f"  turn {t + 1}: new_message.id={seen['new_message_id']} history_user.id={hid} match={seen['new_message_id'] == hid}")
    leaked = [i.text_content for i in hist if "Context:" in (i.text_content or "")]
    print(f"  chunk text in session.history: {bool(leaked)}")
    agent_ctx = [i.text_content for i in agent.chat_ctx.items if i.type == "message" and "Context:" in (i.text_content or "")]
    print(f"  chunk text in agent.chat_ctx: {len(agent_ctx)} msgs")
    print(f"  LLM calls={len(fake.calls)}; last call saw context: {any('Context:' in (c or '') for _, c in fake.calls[-1])}")
    if fixture:
        write_fixture(list(session.history.items), agent.seen, session_start - 2.0)
    await session.aclose()


async def main() -> None:
    turns = ["What is your refund window?", "How long does express shipping take?"]
    await run_scenario("A: RAG into turn_ctx", "turn_ctx", False, turns, fixture=True)
    await run_scenario("B: preemptive on, turn_ctx modified", "turn_ctx", True, turns)
    await run_scenario("C: preemptive on, turn_ctx NOT modified", "noop", True, turns)
    await run_scenario("D: update_chat_ctx", "update_chat_ctx", False, turns)
    await run_scenario("E: preemptive on, update_chat_ctx", "update_chat_ctx", True, turns)


if __name__ == "__main__":
    asyncio.run(main())
