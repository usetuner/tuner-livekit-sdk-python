# Technical Spec: `livekit-agents-tuner`

> Integration library to ingest LiveKit Agents session data into the Tuner observability API with minimal developer effort.

---

## 1. Developer Experience (Target: 2 lines of code)

```python
from livekit_agents_tuner import TunerPlugin   # Line 1

# Inside your entrypoint, after creating AgentSession:
TunerPlugin(session, ctx)                       # Line 2
```

**Configuration via env vars (no code required):**
```bash
TUNER_API_KEY=tr_api_xxxxxxxxxxxx
TUNER_WORKSPACE_ID=12345
TUNER_AGENT_ID=my-agent
```

**Advanced configuration (optional):**
```python
TunerPlugin(
    session, ctx,
    api_key="...",
    workspace_id=12345,
    agent_id="my-agent",
    call_type=None,                    # auto-detect: SIP→phone_call, STANDARD→web_call
    recording_url_resolver=None,       # async callable to resolve recording URL
    cost_calculator=None,              # callable(UsageSummary) -> float (USD)
    extra_metadata={"env": "prod"},
    max_retries=3,
    timeout_seconds=30,
)
```

**Full integration example (agent.py):**
```python
import logging
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, cli, inference, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit_agents_tuner import TunerPlugin          # <--- LINE 1

logger = logging.getLogger("agent")
load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant...")

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en-US"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="..."),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    TunerPlugin(session, ctx)                          # <--- LINE 2
    await session.start(agent=Assistant(), room=ctx.room, room_options=...)
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
```

### What is automatic vs. what requires configuration

| Data Point | Automatic? | Notes |
|---|---|---|
| API key | Env var `TUNER_API_KEY` | Required |
| Workspace ID | Env var `TUNER_WORKSPACE_ID` | Required |
| Agent ID | Env var `TUNER_AGENT_ID` | Falls back to `ctx.job.agent_name` |
| `call_id` | Auto: `ctx.job.id` | Unique per job, provides idempotency |
| `call_type` | Auto-detected from participant kind | SIP = `phone_call`, STANDARD = `web_call` |
| `transcript_with_tool_calls` | Auto: from `session.history` at shutdown | Fully automatic |
| `start_timestamp` / `end_timestamp` | Auto: recorded at construct / shutdown | Fully automatic |
| `duration_ms` | Auto: computed from timestamps | Fully automatic |
| Metrics / usage | Auto: via `UsageCollector` on `metrics_collected` | Fully automatic |
| `recording_url` | Not automatic | Requires `recording_url_resolver` callback |
| `caller_phone_number` | Auto: from SIP participant attributes | Best effort |
| `call_cost` | Not automatic | Requires `cost_calculator` callback |
| `collected_dynamic_variables` | Not automatic | Pass via `extra_metadata` |

---

## 2. Technical Architecture

### Recommended: Constructor-based Plugin (Event Listener Pattern)

**Rejected alternatives:**
- **Decorator on `rtc_session`** — fragile, couples to server internals
- **Mixin on `AgentSession`** — invasive, requires class hierarchy change
- **Monkey-patching** — fragile across SDK upgrades

**Why constructor-based plugin wins:**
- Non-invasive (no subclassing, no patching)
- Idiomatic — uses the SDK's own `EventEmitter` pattern
- Explicit — developer sees exactly one line doing the work
- Survives SDK version upgrades

### Data Flow

```
AgentSession events                 TunerPlugin
  conversation_item_added  ──────►
  function_tools_executed  ──────►  SessionDataCollector (in-memory)
  metrics_collected        ──────►
  close                    ──────►

JobContext.add_shutdown_callback ──► [on shutdown]
                                          │
                                     finalize()
                                          │
                                     mapper.to_create_call_request()
                                          │
                                     TunerAPIClient.submit_call()
                                          │
                                     [awaited with 30s timeout]
```

### Library Structure

```
livekit_agents_tuner/
    __init__.py       # Exports TunerPlugin, TunerConfig
    plugin.py         # Core TunerPlugin class (~120 lines)
    config.py         # TunerConfig, env-var loading (~50 lines)
    collector.py      # Accumulates events in real-time (~100 lines)
    mapper.py         # LiveKit → Tuner data mapping (~150 lines)
    client.py         # Async HTTP with retries (~80 lines)
    types.py          # Pydantic models for API schemas (~100 lines)
```

---

## 3. Data Mapping Spec

### Top-Level Fields

| Tuner Field | Source | Logic |
|---|---|---|
| `call_id` | `ctx.job.id` | Unique per LiveKit job — provides idempotency |
| `call_type` | Participant kind | `PARTICIPANT_KIND_SIP` → `"phone_call"`, otherwise `"web_call"` |
| `transcript_with_tool_calls` | `session.history` at shutdown | See below |
| `start_timestamp` | `time.time()` at plugin construction | Unix epoch seconds |
| `end_timestamp` | `time.time()` in shutdown callback | Unix epoch seconds |
| `duration_ms` | Computed | `(end - start) * 1000` |
| `call_status` | `CloseEvent.error` | `None` → `"completed"`, otherwise `"error"` |
| `disconnection_reason` | Shutdown reason string | Passed through |
| `recording_url` | `recording_url_resolver(room, job_id)` | Optional callback; omitted if not provided |
| `caller_phone_number` | SIP participant attributes | `participant.attributes.get("sip.phoneNumber")` |
| `call_cost` | `cost_calculator(UsageSummary)` | Optional callback; omitted if not provided |
| `general_meta_data_raw` | Aggregated | Job ID, room name, usage summary (tokens/chars/duration), `extra_metadata` |

### transcript_with_tool_calls Mapping

LiveKit `ChatContext` items map as follows:

| LiveKit Type | Tuner Role | Key fields |
|---|---|---|
| `ChatMessage(role="user")` | `user` | `text`, `start_ms` from `created_at` |
| `ChatMessage(role="assistant")` | `agent` | `text`, `start_ms` from `created_at` |
| `FunctionCall` | `agent_function` | `tool.name`, `tool.request_id=call_id`, `tool.params=arguments` |
| `FunctionCallOutput` | Merges into `agent_function` | `tool.result`, `tool.is_error` |
| `ChatMessage(role="system")` | Skipped | Instructions only |

> **Note:** `words[]` array (word-level timing) cannot be populated — LiveKit's `ChatMessage` doesn't carry this data. Acceptable gap since Tuner's analysis works with segment-level timing.

> **Timing note:** `start_ms` and `end_ms` will be identical per segment (LiveKit provides message creation timestamp, not per-segment audio end timing). This is acceptable.

### Mapping pseudocode

```python
def map_history_to_transcript_segments(items: list[ChatItem]) -> list[dict]:
    segments = []
    for item in items:
        if isinstance(item, ChatMessage):
            if item.role in ("user", "assistant"):
                role = "user" if item.role == "user" else "agent"
                segments.append({
                    "role": role,
                    "text": item.text_content or "",
                    "start_ms": int(item.created_at * 1000),
                    "end_ms": int(item.created_at * 1000),
                    "duration_ms": 0,
                    "metadata": {"id": item.id, "interrupted": item.interrupted},
                })
            # Skip "system" and "developer" role messages

        elif isinstance(item, FunctionCall):
            segments.append({
                "role": "agent_function",
                "tool": {
                    "name": item.name,
                    "request_id": item.call_id,
                    "params": item.arguments,
                    "result": None,       # filled by matching FunctionCallOutput
                    "is_error": False,
                    "error": None,
                },
            })

        elif isinstance(item, FunctionCallOutput):
            # Find and update the matching agent_function segment
            for seg in reversed(segments):
                if seg.get("role") == "agent_function" and seg["tool"]["request_id"] == item.call_id:
                    seg["tool"]["result"] = item.output
                    seg["tool"]["is_error"] = item.is_error
                    if item.is_error:
                        seg["tool"]["error"] = item.output
                    break

    return segments
```

### call_type detection

```python
def detect_call_type(room: rtc.Room) -> str:
    for participant in room.remote_participants.values():
        if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            return "phone_call"
    return "web_call"
```

Detected lazily — on `participant_connected` event or at shutdown fallback.

---

## 4. Implementation Plan

### Phase 1: Core Library

| Step | Task | File |
|---|---|---|
| 1 | Project scaffolding | `pyproject.toml` |
| 2 | Pydantic request/response models | `types.py` |
| 3 | Config dataclass + env-var loading | `config.py` |
| 4 | Async HTTP client with retry logic | `client.py` |
| 5 | Event accumulator + UsageCollector | `collector.py` |
| 6 | Full data mapper | `mapper.py` |
| 7 | Plugin: wires events + shutdown hook | `plugin.py` |

### Phase 2: Tests

| Step | Task | File |
|---|---|---|
| 8 | Mapper unit tests (tool calls, interruptions, empty history) | `tests/test_mapper.py` |
| 9 | Collector unit tests | `tests/test_collector.py` |
| 10 | Client retry tests (mocked aiohttp) | `tests/test_client.py` |
| 11 | Config loading + validation tests | `tests/test_config.py` |
| 12 | Integration test (mock full session lifecycle) | `tests/test_integration.py` |

### Phase 3: Polish

| Step | Task |
|---|---|
| 13 | Logging: INFO on success, WARNING on retries, ERROR on failure |
| 14 | README with quickstart + configuration reference |

### Dependencies

```toml
[project]
name = "livekit-agents-tuner"
requires-python = ">=3.9"
dependencies = [
    "livekit-agents>=1.0.0",
    "aiohttp>=3.0",       # already a transitive dep of livekit-agents
    "pydantic>=2.0",
]
```

---

## 5. Edge Cases & Considerations

### Async / Non-Blocking Ingestion

The shutdown callback is `async` and awaited by the LiveKit SDK. Recommended approach: **awaited with timeout**.

```python
async def _on_shutdown(self, reason: str) -> None:
    self._collector.set_shutdown_reason(reason)
    data = self._collector.finalize()
    request = to_create_call_request(data, self._config, self._ctx)
    try:
        await asyncio.wait_for(
            self._client.submit_call(request),
            timeout=self._config.timeout_seconds,
        )
    except Exception:
        logger.exception("Failed to submit call to Tuner API")
```

This guarantees delivery before the process exits with zero latency impact on the user (session is already closed).

### Error Handling and Retries

- 3 retries with 1s / 2s / 4s exponential backoff + random jitter (0–500ms)
- Retry on: HTTP 5xx, `aiohttp.ClientError`, `asyncio.TimeoutError`
- Do not retry on: HTTP 4xx — log full request payload (excluding API key) for debugging
- After all retries exhausted: log at ERROR level

### Recording URL Availability

Recording URLs are available asynchronously via LiveKit's Egress API after session close. Use the optional resolver callback:

```python
async def my_resolver(room_name: str, job_id: str) -> str | None:
    egress_info = await lk_api.egress.list_egress(room_name=room_name)
    for egress in egress_info:
        if egress.status == "EGRESS_COMPLETE":
            return egress.file_results[0].download_url
    return None

TunerPlugin(session, ctx, recording_url_resolver=my_resolver)
```

If not provided, `recording_url` is omitted from the request.

### Cost Calculation

`UsageCollector` provides raw counts (tokens, characters, audio duration) — not USD. Strategy:

1. Always include raw usage in `general_meta_data_raw.usage_summary`
2. Optional `cost_calculator` callback for teams that know their model pricing:

```python
def my_cost_calc(summary: UsageSummary) -> float:
    return (
        summary.llm_prompt_tokens * 0.15 / 1_000_000
        + summary.llm_completion_tokens * 0.6 / 1_000_000
        + summary.tts_characters_count * 0.000015
        + summary.stt_audio_duration * 0.0059 / 60
    )

TunerPlugin(session, ctx, cost_calculator=my_cost_calc)
```

### call_type Detection Timing

Plugin listens to `ctx.room.on("participant_connected")` to lazily detect SIP participants. Falls back to checking `ctx.room.remote_participants` at shutdown. Defaults to `"web_call"` if no participants were ever detected.

### Multiple Sessions per Job

Each `TunerPlugin` instance is tied to one `AgentSession`. For v1, one session per job is the primary use case. If multiple sessions exist, create a `TunerPlugin` for each — the Tuner API's idempotency key (`call_id = ctx.job.id`) will handle deduplication.

### SDK Version Compatibility

Requires `livekit-agents>=1.0.0`. All APIs used are stable public APIs:
- `AgentSession.on()` — event subscription
- `AgentSession.history` — conversation data (`ChatContext`)
- `JobContext.add_shutdown_callback()` — shutdown hook
- `ChatMessage.created_at` — segment timestamps
- `FunctionCall` / `FunctionCallOutput` — tool call data
- `metrics.UsageCollector` — usage aggregation

---

## References

- [Tuner Public API](https://api.usetuner.ai/public/docs)
- [LiveKit Agents Data Hooks](https://docs.livekit.io/deploy/observability/data/)
- [LiveKit Agent Session](https://docs.livekit.io/agents/logic/sessions/)
- [LiveKit Job Lifecycle](https://docs.livekit.io/agents/server/job/)
