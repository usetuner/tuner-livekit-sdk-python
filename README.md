# tuner-livekit-sdk

Automatically ingest [LiveKit Agents](https://github.com/livekit/agents) session data into the [Tuner](https://usetuner.ai) observability API.

## Installation of the Library into your Livekit project

```bash
pip install tuner-livekit-sdk
```

## Quickstart

Set credentials via environment variables:

```bash
export TUNER_API_KEY="tr_api_..."
export TUNER_WORKSPACE_ID="123"
export TUNER_AGENT_ID="my-agent"
```

Then drop the plugin in right after creating your `AgentSession`:

```python
from tuner import TunerPlugin

async def entrypoint(ctx: JobContext):
    session = AgentSession(...)
    TunerPlugin(session, ctx)   # wires itself automatically
    await session.start(...)
```

That's it. The plugin listens to session events and submits call data to Tuner when the session ends.

## Configuration

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `TUNER_API_KEY` | ✅ | Bearer token (starts with `tr_api_`) |
| `TUNER_WORKSPACE_ID` | ✅ | Integer workspace ID |
| `TUNER_AGENT_ID` | ✅ | Agent identifier from Tuner Agent Settings |
| `TUNER_BASE_URL` | — | API base URL (default: `https://api.usetuner.ai`) |

### Credentials from code

Pass credentials directly instead of (or to override) environment variables:

```python
TunerPlugin(
    session, ctx,
    api_key="tr_api_...",
    workspace_id=123,
    agent_id="my-agent",
)
```

## Options

### Call type

By default the plugin auto-detects the call type (`phone_call` for SIP participants, `web_call` otherwise). Override it explicitly:

```python
TunerPlugin(session, ctx, call_type="phone_call")
TunerPlugin(session, ctx, call_type="web_call")
```

### Recipient (callee)

Pass the phone number or SIP URL of the called party when your agent initiates or routes outbound calls. This field is not auto-collected — supply it explicitly when known:

```python
# E.164 phone number
TunerPlugin(session, ctx, recipient="+15551234567")

# SIP URI
TunerPlugin(session, ctx, recipient="sip:alice@example.com")
```

`recipient` is optional. If omitted it is simply not included in the call record.

### Recording URL

Tuner requires a `recording_url` for every call. If you don't provide a resolver the plugin logs a warning and submits `"pending"` as a placeholder:

```python
# Static URL
async def my_resolver(room_name: str, job_id: str) -> str:
    return f"https://cdn.example.com/recordings/{job_id}.ogg"

TunerPlugin(session, ctx, recording_url_resolver=my_resolver)
```

```python
# LiveKit Egress → S3
async def egress_resolver(room_name: str, job_id: str) -> str:
    url = await my_egress_db.get_recording_url(room_name)
    return url or "pending"

TunerPlugin(session, ctx, recording_url_resolver=egress_resolver)
```

### Cost calculation

Provide a callable that receives a `UsageSummary` and returns the call cost in USD cents:

```python
def calculate_cost(usage) -> float:
    llm_cost  = usage.llm_prompt_tokens     * 0.000_003
    llm_cost += usage.llm_completion_tokens * 0.000_015
    tts_cost  = usage.tts_characters_count  * 0.000_030
    stt_cost  = usage.stt_audio_duration    * 0.000_006
    return llm_cost + tts_cost + stt_cost

TunerPlugin(session, ctx, cost_calculator=calculate_cost)
```

### Extra metadata

Attach arbitrary key-value data to every call record:

```python
TunerPlugin(
    session, ctx,
    extra_metadata={
        "env": "production",
        "region": "us-east-1",
        "deployment": "v2.3.1",
    },
)
```

### Retry and timeout

```python
TunerPlugin(
    session, ctx,
    timeout_seconds=15.0,   # per-request timeout (default: 30.0)
    max_retries=5,          # retries on 5xx / 429 / network errors (default: 3)
)
```

### Agent version tracking

Track which version of your agent handled each call — useful when you update a prompt, swap a model, or change your pipeline:

```bash
AGENT_VERSION=42 python agent.py start
```

Tuner reads it automatically. Bump the number on every deployment.

Override in code (takes priority over the env var):

```python
TunerPlugin(session, ctx, agent_version=42, ...)
```

### Disable the plugin

Useful for local development or test environments:

```python
import os

TunerPlugin(
    session, ctx,
    enabled=os.getenv("ENV") == "production",
)
```

## LangGraph / LangChain observability

If your agent uses LangGraph or LangChain as the orchestration layer, install
`tuner-langchain` and wire it in with `attach_langgraph()` or `attach_langchain()`:

```python
from tuner import TunerPlugin

plugin = TunerPlugin(session, ctx)
handler = plugin.attach_langgraph()

agent = Agent(
    llm=langchain.LLMAdapter(
        graph=my_graph,
        config={"callbacks": [handler]},
    ),
)
```

To limit what data is forwarded to Tuner, pass a `CaptureConfig`:

```python
from tuner import TunerPlugin
from tuner_langchain import CaptureConfig

plugin = TunerPlugin(session, ctx)
handler = plugin.attach_langgraph(
    capture=CaptureConfig(
        tool_inputs=False,
        node_instructions=False,
    )
)
```

## Simulation correlation (SIP)

Tuner simulations dial into your agent through the same SIP trunk that handles production phone calls. To match a simulation run with the session your agent submits, the SDK forwards LiveKit's `sip.callIDFull` attribute as a `sip_correlation_id`.

This section covers the **SDK wiring only**. For LiveKit platform setup (SIP URI, inbound trunk, dispatch rule, Tuner SIP settings), see:

→ **[docs.usetuner.ai/docs/api-and-integrations/connecting-to-livekit/simulation-setup](https://docs.usetuner.ai/docs/api-and-integrations/connecting-to-livekit/simulation-setup)**

### Requirements

- `tuner-livekit-sdk >= 0.1.5` (the `sip_correlation_id` argument was added in 0.1.5)

### Step 1 — The `_extract_sip_correlation_id` helper

This helper scans the LiveKit room for the SIP caller and returns their `sip.callIDFull` — the value Tuner uses to match a simulation run to the session your agent submits.

```python
from livekit import rtc


def _extract_sip_correlation_id(ctx: JobContext) -> str | None:
    for participant in ctx.room.remote_participants.values():
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            continue
        attributes = dict(getattr(participant, "attributes", {}) or {})
        sip_call_id_full = attributes.get("sip.callIDFull")
        if isinstance(sip_call_id_full, str) and sip_call_id_full:
            return sip_call_id_full
    return None
```

**How it works:**

- Loops through remote participants and keeps only the SIP one (rooms can hold web clients, observers, etc.).
- Reads `sip.callIDFull` from that participant's attributes — this is the full SIP `Call-ID` Tuner stamps on its outbound leg (not the shorter `sip.callID`).
- Returns `None` for web calls or non-simulation SIP calls; `TunerPlugin` accepts `None` and simply skips correlation.

### Step 2 — Pass it to `TunerPlugin`

Once you have the helper, the wiring in `entrypoint` is three lines: connect, extract, attach.

```python
async def entrypoint(ctx: JobContext):
    session = AgentSession(...)

    await ctx.connect()
    sip_correlation_id = _extract_sip_correlation_id(ctx)

    TunerPlugin(
        session,
        ctx,
        sip_correlation_id=sip_correlation_id,
        # ...other options
    )

    await session.start(...)
```

> **⚠️ Order matters:** `ctx.room.remote_participants` is empty until `await ctx.connect()` completes. If you call the helper too early it will always return `None` and you'll silently lose correlation for every simulation — no error, just missing data in Tuner. Always: build `AgentSession` → `await ctx.connect()` → extract ID → attach plugin → `await session.start(...)`.

### Step 3 — Full example

Putting the helper, the plugin wiring, and the usual options (cost, recording URL, metadata) together:

```python
import os
from livekit import rtc
from livekit.agents import JobContext, AgentSession
from tuner import TunerPlugin


def _extract_sip_correlation_id(ctx: JobContext) -> str | None:
    for participant in ctx.room.remote_participants.values():
        if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
            continue
        attributes = dict(getattr(participant, "attributes", {}) or {})
        sip_call_id_full = attributes.get("sip.callIDFull")
        if isinstance(sip_call_id_full, str) and sip_call_id_full:
            return sip_call_id_full
    return None


def calculate_cost(usage) -> float:
    return (
        usage.llm_prompt_tokens     * 0.000_003
        + usage.llm_completion_tokens * 0.000_015
        + usage.tts_characters_count  * 0.000_030
    )


async def get_recording_url(room_name: str, job_id: str) -> str:
    return await my_storage.get_url(job_id) or "pending"


async def entrypoint(ctx: JobContext):
    session = AgentSession(...)

    await ctx.connect()
    sip_correlation_id = _extract_sip_correlation_id(ctx)

    TunerPlugin(
        session,
        ctx,
        api_key=os.environ["TUNER_API_KEY"],
        workspace_id=int(os.environ["TUNER_WORKSPACE_ID"]),
        agent_id="customer-support-v3",
        call_type="phone_call",
        recording_url_resolver=get_recording_url,
        cost_calculator=calculate_cost,
        sip_correlation_id=sip_correlation_id,
        extra_metadata={"env": "prod", "region": "us-east-1"},
        timeout_seconds=20.0,
        max_retries=3,
        enabled=True,
    )

    await session.start(...)
```

## Requirements

- Python ≥ 3.10
- `livekit-agents >= 1.4`
- `tuner-livekit-sdk >= 0.1.5` (needed for `sip_correlation_id` / SIP correlation)
- `aiohttp >= 3.9`

## License

MIT

## Installation dependencies to build the library
uv sync --dev
source .venv/bin/activate


## Publish to Pypi
pip install build twine
python -m build
twine upload dist/*
