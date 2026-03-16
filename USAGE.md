# TunerPlugin Usage Guide

## Minimal setup (2 lines)

Configure credentials via environment variables, then drop the plugin in before `session.start()`:

```bash
export TUNER_API_KEY="tr_api_..."
export TUNER_WORKSPACE_ID="123"
export TUNER_AGENT_ID="my-agent"
```

```python
from tuner import TunerPlugin

async def entrypoint(ctx: JobContext):
    session = AgentSession(...)
    TunerPlugin(session, ctx)          # wires itself automatically
    await session.start(...)
```

---

## Credentials from code

Pass credentials directly instead of (or to override) environment variables:

```python
TunerPlugin(
    session, ctx,
    api_key="tr_api_...",
    workspace_id=123,
    agent_id="my-agent",
)
```

---

## Call type

By default the plugin detects the call type from room participants (`phone_call` for SIP, `web_call` otherwise).
Override it explicitly when you know ahead of time:

```python
TunerPlugin(session, ctx, call_type="phone_call")
TunerPlugin(session, ctx, call_type="web_call")
```

---

## Recording URL

Tuner requires a `recording_url` for every call. The plugin always calls a resolver to obtain it.

### Default (placeholder)

If you do not provide a resolver the plugin logs a warning and submits `"pending"` as the URL.
This keeps the API call from failing while you set up real recordings:

```python
TunerPlugin(session, ctx)
# WARN: No recording_url_resolver provided; submitting recording_url='pending'
```

### Static / pre-known URL

```python
async def my_resolver(room_name: str, job_id: str) -> str:
    return f"https://cdn.example.com/recordings/{job_id}.ogg"

TunerPlugin(session, ctx, recording_url_resolver=my_resolver)
```

### LiveKit Egress → S3

Use the LiveKit Egress API to start a room composite recording, then return the S3 URL:

```python
async def egress_resolver(room_name: str, job_id: str) -> str:
    # Query your egress service or database for the finished recording URL
    url = await my_egress_db.get_recording_url(room_name)
    return url or "pending"

TunerPlugin(session, ctx, recording_url_resolver=egress_resolver)
```

### Manual upload (LiveKit Cloud built-in recording)

LiveKit Cloud stores recordings locally. After manually uploading to a public host, you can
store and retrieve the URL from your own backend:

```python
async def manual_resolver(room_name: str, job_id: str) -> str | None:
    return await my_db.fetch_recording_url(job_id)  # populated after upload

TunerPlugin(session, ctx, recording_url_resolver=manual_resolver)
```

---

## Cost calculation

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

---

## Extra metadata

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

---

## Retry and timeout

```python
TunerPlugin(
    session, ctx,
    timeout_seconds=15.0,   # default: 30.0
    max_retries=5,          # default: 3  (retries on 5xx / 429 / network errors)
)
```

---

## Disable the plugin

Useful for local development or test environments:

```python
import os

TunerPlugin(
    session, ctx,
    enabled=os.getenv("ENV") == "production",
)
```

---

## Full example

```python
import os
from tuner import TunerPlugin

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
