# Spike: KB retrievals in LiveKit transcripts

Stage 0 of the "KB retrievals in transcripts" plan. livekit-agents 1.5.6, SDK 0.1.12.

## How it was run

`docs/spikes/kb_retrieval_spike.py`, run with the SDK venv (`.venv/bin/python docs/spikes/kb_retrieval_spike.py`).

- In-memory KB (5 chunks, bag-of-words cosine, top-2) and a fake LLM. No API key, no room, no audio.
- `session.run(user_input=...)` cannot be used: it goes through `generate_reply` and never calls `on_user_turn_completed`. The script calls the real `AgentActivity.on_preemptive_generation` and `on_end_of_turn` hooks, which is what `AudioRecognition` calls after STT and end-of-turn.
- Nihal's `feat/rag-voice-agent` branch was not used. It needs a room, pgvector and real models, and it can't go through the text path for the reason above.

## Findings

| Scenario | chunks in `session.history` | `new_message.id` == history user id |
|---|---|---|
| A. RAG chunks added to `turn_ctx`, preemptive off | no | yes (2/2 turns) |
| B. preemptive on, `turn_ctx` modified | no | yes (2/2) |
| C. preemptive on, `turn_ctx` not modified | no | **no (0/2)** |
| D. `update_chat_ctx`, preemptive off | no | yes (2/2) |
| E. `update_chat_ctx`, preemptive on | no | **no (0/2)** |

1. Chunks added to `turn_ctx` never reach `session.history`. The LLM sees them, but they are not kept anywhere.
2. In the normal path, `new_message.id` is the id of the user message in `session.history`.
3. Preemptive generation is **on by default** in 1.5.6 (`_PREEMPTIVE_GENERATION_DEFAULTS`). At end of turn, the framework compares the preemptive `chat_ctx` with `turn_ctx` (`agent_activity.py:2021-2046`):
   - `turn_ctx` changed (the usual RAG case): the preemptive reply is cancelled ("chat context or tools have changed" warning). The reply uses `new_message`, so the ids match. The cancelled LLM call still ran (4 LLM calls for 2 turns in B).
   - `turn_ctx` unchanged (retrieval returned nothing, or the customer used `update_chat_ctx` or instructions instead): the preemptive reply is used. History gets the preemptive `ChatMessage`, which has a different id from `new_message`. The text is the same, so the plan's same-text, nearest-timestamp fallback works for this case.
4. `update_chat_ctx` content stays in `agent.chat_ctx`, so later LLM calls see it, but it does not reach `session.history`.

## Chunk size (open question 3)

These numbers are estimates from a typical setup (top-k 3-5, chunks of 500-1000 chars, about 4 chars per token). They were not measured on a real KB.

| | chars | tokens |
|---|---|---|
| per turn | 1,500-5,000 | ~375-1,250 |
| 10-turn call (one retrieval per turn) | 15,000-50,000 | ~3.75k-12.5k |
| 10-turn call, deduped (assume 40-60% repeats in a single-topic call) | ~6,000-30,000 | ~1.5k-7.5k |

The plain transcript of a 10-turn call is about 3-5k chars, so retrievals are 3-15x larger than the transcript. The judge budget (`transcript_budget.py`) needs a cap or dedup for `kb_retrieval` entries. JSON overhead per chunk (`source_id`, `score`, keys) is about 60 chars, which is small next to the text.

## Fixture

`tests/fixtures/kb_retrieval_two_turn.json` comes from run A. The real `session.history` goes through today's `map_history_to_segments`, and then the Stage 3 fields are added: a top-level `id` on user segments, and a `kb_retrieval` entry after each user turn whose `user_turn_id` is that turn's `new_message.id`. The agent text is fake LLM output.
