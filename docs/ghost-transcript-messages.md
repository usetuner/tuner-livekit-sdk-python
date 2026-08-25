# Ghost user messages in the transcript — analysis and remediation plan

**Status:** analysis complete, fixes not yet implemented
**Affects:** `tuner-livekit-sdk` ≤ 0.1.10, all versions of `map_history_to_segments`
**Reference LiveKit version for line numbers:** `livekit-agents` 1.7.0 (behaviour verified identical in 1.4.0 and 1.6.0 where cited)

---

## 1. Symptom

Calls arrive in Tuner with `role: "user"` segments the caller never said. The text is
agent configuration — persona rules, scenario setup, response-length limits — not speech.
From the reported payload:

```json
{
  "role": "user",
  "text": "Start Talking for less than 10 seconds.",
  "end_ms": 371,
  "metadata": { "id": "item_3d810712-d71", "interrupted": false },
  "start_ms": 371
}
```

The surrounding turns also read out of order: an agent greeting at `18494` sits *before*
the user turn at `18544` that prompted it.

## 2. What the SDK does today

`TunerPlugin._on_shutdown` snapshots the conversation once, at the end of the call
(`src/tuner/plugin.py:305`):

```python
history_items = list(self._session.history.items)
```

`map_history_to_segments` then emits a segment for every item
(`src/tuner/mapper.py:70-111`). The **only** guard is the role check:

```python
if item.role not in ("user", "assistant"):
    continue  # Skip system / developer instruction messages
role = "user" if item.role == "user" else "agent"
text = item.text_content or ""
```

That is the whole filter. Anything sitting in `session.history` with `role="user"` becomes
a user transcript turn, whatever put it there and however it got there.

`build_plain_transcript` (`src/tuner/mapper.py:159-171`) applies the same rule, so the
plain-text `transcript` field carries the ghosts too.

## 3. Evidence

### 3.1 The ghost was not minted by LiveKit

LiveKit mints chat item ids with `utils.shortuuid("item_")`, which is
`prefix + uuid4().hex[:12]` — twelve **hex** characters, no dash
(`livekit/agents/utils/misc.py:24-25`; identical in 1.4.0, 1.6.0 and 1.7.0):

```
item_17a160bf3d08
```

The ghost's id is `item_3d810712-d71`: twelve characters with a dash at index 8. That is
`"item_" + str(uuid.uuid4())[:12]` — the *dashed* form, truncated. LiveKit never produces
it. **Some other code constructed that message.**

The other ids in the payload (`item_EEuomlQqZ7VmQ6ho865Y5` — 21 base62 characters) aren't
LiveKit-minted either. They are provider item ids passed straight through: LiveKit reuses
the realtime provider's id for both user transcripts (`agent_activity.py:1982`,
`id=ev.item_id`) and assistant messages (`agent_activity.py:4085-4088`,
`id=message_id`). That shape matches OpenAI Realtime item ids.

**Conclusion: the reported call ran on a realtime (speech-to-speech) model, and one item in
its history was inserted by application or orchestrator code.**

### 3.2 Every segment in the payload lacks speech metrics

Each reported segment has `start_ms == end_ms` and no `llm_node_ttft`, `tts_node_ttfb`,
`stt_node_ttfb`, `eou_delay`, `e2e_latency` or `transcript_confidence`. The mapper always
emits those keys, so the API is stripping nulls — meaning every source `ChatMessage` had
an **empty `MetricsReport`**, and `start_ms` fell back to `created_at`
(`src/tuner/mapper.py:79-84`).

This matters because it is exactly the signal that separates a spoken turn from an
inserted one. A genuine pipeline user turn gets `started_speaking_at`,
`stopped_speaking_at`, `transcription_delay` and `end_of_turn_delay`
(`agent_activity.py:4569-4585`). A genuine assistant turn gets `started_speaking_at` /
`stopped_speaking_at` once audio plays (`agent_activity.py:2985-2987`). An inserted message
gets nothing.

### 3.3 Reproduced

`map_history_to_segments` fed a `ChatMessage(id="item_3d810712-d71", role="user",
content=["Start Talking for less than 10 seconds."], created_at=T0+0.371)` emits the
reported segment byte-for-byte:

```json
{
  "role": "user",
  "text": "Start Talking for less than 10 seconds.",
  "start_ms": 371,
  "end_ms": 371,
  "metadata": { "id": "item_3d810712-d71", "interrupted": false,
                "llm_node_ttft": null, "tts_node_ttfb": null, "stt_node_ttfb": null,
                "eou_delay": null, "e2e_latency": null, "transcript_confidence": null }
}
```

The same message with `role="system"` is correctly dropped. So the SDK is not *fabricating*
anything — it is faithfully reporting an item that LiveKit put in `session.history` with a
user role.

## 4. Root causes

### RC-1 — Role is the only guard, and several non-speech paths use `role="user"`

`session.history` is not the agent's prompt context. It is an event log seeded empty
(`agent_session.py:561`) and appended to by `_conversation_item_added`
(`agent_session.py:2049-2056`). The agent's `instructions` never land in it. But three
paths do put non-spoken text there with a user role:

| # | Path | Where |
|---|---|---|
| a | Application/orchestrator constructs a `ChatMessage(role="user", …)` and inserts it | outside LiveKit — this is the reported ghost |
| b | `session.generate_reply(user_input="…")` → `ChatMessage(role="user", content=[user_input])` inserted into history | `agent_session.py:1490-1493`, `agent_activity.py:3261-3262` |
| c | Anything sent over the room's **text stream**: the default text-input handler calls `generate_reply(user_input=ev.text)` | `voice/room_io/types.py:46-49` |

(c) is the quiet one. A control plane that pushes per-call instructions over the text
channel produces a history item indistinguishable, by role, from a spoken turn.

A fourth variant: developers targeting providers without a system role move the system
prompt into a `user` message by hand.

### RC-2 — Injected context is concatenated into a real user turn

`ChatMessage.text_content` joins every text content part with `\n`. LiveKit's documented
RAG hook, `Agent.on_user_turn_completed(turn_ctx, new_message)`, hands the developer the
**actual** `new_message` object that later goes into history
(`agent_activity.py:2504-2512`). Appending to `new_message.content` — a common pattern —
glues the instruction onto the end of a genuine utterance:

```
user: Hi, I'd like a table for two.
Additional context: the caller is a VIP. Keep replies under 10 seconds.
```

Role filtering cannot catch this one; the item *is* a real user turn.

### RC-3 — Ordering and zero-length segments make ghosts blend in

Two compounding issues:

1. **The mapper only sorts when LangGraph is active** (`src/tuner/mapper.py:150-154`).
   The plain path preserves `session.history` order, which is `created_at` order.
2. **For realtime models, `created_at` is the transcript *arrival* time.** LiveKit stamps
   the user message when the provider delivers the final transcript, and only corrects it
   when the plugin supplies `turn_started_at` (`agent_activity.py:1982-1988`). Providers
   that withhold the transcript until their reply finishes generating therefore produce a
   user turn stamped *after* the agent reply it caused — precisely the `18494` agent /
   `18544` user inversion in the report.

With no speech metrics, every segment collapses to `start_ms == end_ms`, so nothing in the
timeline distinguishes a 4-second utterance from a zero-width insertion.

### RC-4 — The SDK ignores two signals LiveKit already provides

- **Reserved ids.** LiveKit keys its own injected system messages by fixed ids:
  `lk.agent_task.instructions` (`voice/generation.py:1064`) and
  `lk.expressive.instructions` (`voice/generation.py:1117`). The mapper does not look at
  ids at all.
- **Speech metrics.** As established in §3.2, presence of `started_speaking_at` /
  `stopped_speaking_at` / `transcription_delay` / `transcript_confidence` is a reliable
  discriminator for audio sessions.

### RC-5 — Snapshot-at-shutdown loses provenance

Reading `session.history.items` once at shutdown (`src/tuner/plugin.py:305`) means the SDK
sees a flat list with no record of *how* each item arrived. Items inserted after the fact
are indistinguishable from items that came off the wire.

---

## 5. Proposed solutions

Each is independently shippable. Priorities in §6.

### S1 — Skip LiveKit's reserved `lk.*` item ids

Drop any item whose `id` starts with `lk.`. LiveKit owns that namespace for injected
system messages.

- **Cost:** ~3 lines in `map_history_to_segments` and `build_plain_transcript`.
- **Risk:** none. No false positives.
- **Catches:** RC-4 (reserved ids). Not the reported ghost.

### S2 — Classify spoken vs. injected, and make the policy configurable *(recommended core fix)*

Derive a `source` for every message segment:

```python
_USER_SPEECH_KEYS  = ("started_speaking_at", "stopped_speaking_at",
                      "transcription_delay", "end_of_turn_delay")
_AGENT_SPEECH_KEYS = ("started_speaking_at", "stopped_speaking_at",
                      "llm_node_ttft", "tts_node_ttfb")

def _segment_source(item) -> str:
    keys = _USER_SPEECH_KEYS if item.role == "user" else _AGENT_SPEECH_KEYS
    if any(k in item.metrics for k in keys):
        return "speech"
    if item.role == "user" and item.transcript_confidence is not None:
        return "speech"
    return "injected"
```

Expose it as `metadata.source` and add a plugin option:

```python
TunerPlugin(session, ctx, transcript_mode="annotate")
```

| mode | behaviour |
|---|---|
| `"raw"` | today's behaviour — emit everything, no `source` key |
| `"annotate"` | emit everything, tag `metadata.source` (**proposed default**) |
| `"spoken_only"` | drop `source == "injected"` message segments entirely |

**Critical caveat — do not default to `spoken_only`.** Text-only sessions, chat-mode
agents, and eval runs via `session.run(user_input=...)` legitimately have *no* speech
metrics on any turn; `spoken_only` would blank their transcripts entirely. Gate the
classifier on the session actually carrying audio (STT/TTS present, or
`session.output.audio_enabled`) and fall back to `"raw"` when it doesn't. `"annotate"` is
safe everywhere because it never drops data — it lets the Tuner UI grey out or collapse
injected turns, and lets us measure how often this fires in the fleet before changing any
default.

- **Cost:** ~40 lines in `mapper.py`, one config field, UI work on the Tuner side to
  render `source`.
- **Risk:** medium. Needs the text-session guard, or it silently deletes real transcripts.
- **Catches:** RC-1 (a, b, c) and RC-4.

### S3 — Give developers an explicit escape hatch

Three complementary mechanisms, cheapest first:

1. **`ChatMessage.extra` marker.** LiveKit 1.7 gives every `ChatMessage` an `extra: dict`
   field (`llm/chat_context.py:317`). Honour `extra={"tuner": {"exclude": True}}` and skip
   the item. Zero guesswork, but the customer must change their agent code.
2. **`transcript_filter` callback.** `TunerPlugin(session, ctx,
   transcript_filter=lambda item: not item.id.startswith("sysprompt_"))`. Full control,
   also just a few lines.
3. **Declarative excludes** for customers who can't touch the agent:
   `exclude_item_ids=[...]`, `exclude_text_patterns=[r"^Start Talking for"]`.

- **Cost:** small.
- **Risk:** low. Opt-in throughout.
- **Catches:** RC-1(a) — including the reported case — and RC-2 via (1).

### S4 — Collect turns as they happen instead of snapshotting

Subscribe to `session.on("conversation_item_added")` and `user_input_transcribed` at
plugin construction, record arrival time and provenance per item, and use the shutdown
snapshot only to reconcile (fill in final text, catch anything missed).

- **Gains:** real arrival ordering independent of `created_at`; a durable record of which
  items came through the speech pipeline; resilience to the developer mutating
  `session.history`; partial transcripts survive a crashed session.
- **Cost:** the largest of these — new collector state, careful dedup against the snapshot
  by item id, plus tests for interruption/upsert paths (LiveKit calls `_upsert_item` for
  realtime items, so the same id can be updated more than once).
- **Risk:** medium-high; touches the plugin's core data path.
- **Catches:** RC-3 and RC-5, and makes S2's classification far more reliable.

### S5 — Always sort, and stop claiming zero-length spoken turns

1. Move the `segments.sort(key=...)` out of the `lg_acc is not None` branch
   (`src/tuner/mapper.py:154`) so it always runs. Use a stable
   `(start_ms, original_index)` key so equal timestamps keep history order.
2. When `stopped_speaking_at` is missing but the turn is classified `speech`, leave
   `end_ms == start_ms` but set `metadata.duration_known = false`, so the UI can render a
   point event rather than a zero-width bar.

- **Cost:** ~10 lines.
- **Risk:** low. Ordering changes are visible in the UI, so ship with a changelog note.
- **Catches:** RC-3 (item 1 only — the realtime `created_at` skew itself is upstream).

### S6 — Stop concatenating extra content parts into `text`

Use the first text part as `text`; put any remaining parts under
`metadata.additional_content`. Nothing is lost, and appended RAG/instruction blocks stop
appearing as words the caller said.

- **Cost:** ~10 lines.
- **Risk:** low. Multi-part user messages are rare in voice; the data is still in the
  payload.
- **Catches:** RC-2.

### S7 — Ingest-side guard in the Tuner API

The SDK fix only helps customers who upgrade. Independently:

- Render `metadata.source` (once S2 ships) and collapse injected turns by default.
- Per-agent "known instruction text" deny list, so an affected customer can be cleaned up
  today without an SDK release.
- Flag calls whose first user turn has no speech metrics — a cheap fleet-wide detector for
  how widespread this is.

- **Catches:** everything, retroactively, for customers pinned to old SDK versions.

---

## 6. Recommended sequencing

**P0 — patch release (0.1.11), low risk:**
S1 + S5 + S3(2) `transcript_filter`. Ships an immediate, documented workaround for the
reporting customer and fixes the visible ordering bug.

**P1 — minor release (0.2.0):**
S2 with `transcript_mode="annotate"` as the default and the text-session guard, plus S6
and S3(1) `extra` marker. Coordinate with the Tuner UI to render `metadata.source`.

**P2 — once P1 data shows how often injection happens:**
S4 event-sourced collection, and consider promoting `spoken_only` to the default for
audio sessions. S7 in parallel, on the API side.

## 7. Immediate workaround for the affected customer

No SDK change needed. Any of these stops the ghost today:

1. **Inject with `role="system"` instead of `role="user"`.** The mapper already skips it
   (`src/tuner/mapper.py:73`), and every LLM provider LiveKit supports accepts a system
   turn mid-context.
2. **Use `session.generate_reply(instructions="…")` rather than `user_input="…"`.**
   LiveKit adds instructions as a `role="system"` message on a throwaway copy of the chat
   context (`agent_activity.py:3105-3113`); it never reaches `session.history`.
3. **If the text is being pushed over the room's text stream,** override
   `text_input_cb` in `RoomOptions` so it doesn't fall through to the default
   `generate_reply(user_input=…)` (`voice/room_io/types.py:46-49`).
4. **If context is being appended in `on_user_turn_completed`,** add it to `turn_ctx`
   (the copy) rather than to `new_message.content` (the object that lands in history).

## 8. Confirming which root cause applies

Before implementing, get one affected call to log its raw history. Add a debug dump to the
plugin behind `TUNER_DEBUG_TRANSCRIPT=1`:

```python
for i, item in enumerate(self._session.history.items):
    logger.info("tuner.history[%d] type=%s id=%s role=%s metrics=%s parts=%d",
                i, type(item).__name__, item.id, getattr(item, "role", None),
                sorted(getattr(item, "metrics", {})), len(getattr(item, "content", [])))
```

Then read it against §3.1: LiveKit-minted ids are `item_` + 12 hex; provider realtime ids
are `item_` + ~21 base62; anything else was constructed by application code. Items with an
empty `metrics` dict on an audio session were not spoken.

## 9. Test plan

Add to `tests/test_mapper.py`, one case per root cause:

- `role="user"` message with empty metrics and an app-minted id → asserted behaviour per
  `transcript_mode`.
- `role="system"` instruction → still dropped (regression guard).
- Item with `id="lk.agent_task.instructions"` → dropped (S1).
- Multi-part user content → `text` holds only the spoken part; the rest lands in
  `metadata.additional_content` (S6).
- Out-of-order `created_at` with no `lg_acc` → segments come back sorted (S5).
- Text-only session (no metrics on *any* turn) under `transcript_mode="spoken_only"` →
  transcript is **not** emptied (the S2 guard).
- Golden fixture built from the reported payload, asserting the ghost is gone and the four
  genuine turns survive in the right order.
