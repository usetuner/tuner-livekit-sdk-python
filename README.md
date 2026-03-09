# livekit-agents-tuner

Automatically ingest [LiveKit Agents](https://github.com/livekit/agents) session data into the [Tuner](https://usetuner.ai) observability API.

## Installation

```bash
pip install livekit-agents-tuner
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
from livekit_agents_tuner import TunerPlugin

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

Provide a callable that receives a `UsageSummary` and returns the call cost in USD:

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

### Disable the plugin

Useful for local development or test environments:

```python
import os

TunerPlugin(
    session, ctx,
    enabled=os.getenv("ENV") == "production",
)
```

## Full example

```python
import os
from livekit.agents import JobContext, AgentSession
from livekit_agents_tuner import TunerPlugin


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

    TunerPlugin(
        session, ctx,
        api_key=os.environ["TUNER_API_KEY"],
        workspace_id=int(os.environ["TUNER_WORKSPACE_ID"]),
        agent_id="customer-support-v3",
        call_type="phone_call",
        recording_url_resolver=get_recording_url,
        cost_calculator=calculate_cost,
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
- `aiohttp >= 3.9`

## License

MIT
