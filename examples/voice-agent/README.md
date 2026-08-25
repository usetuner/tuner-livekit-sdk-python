# Example: restaurant receptionist voice agent

A complete, runnable [LiveKit Agents](https://docs.livekit.io/agents/) voice agent
instrumented with `tuner-livekit-sdk`. It answers calls for a restaurant, collects the
booking details, and exposes two function tools so you can see tool calls land in the
Tuner timeline.

The Tuner integration is a single call in [`src/agent.py`](src/agent.py):

```python
session = AgentSession(...)

TunerPlugin(session, ctx, cost_calculator=calculate_cost)

await session.start(...)
```

Everything else in this directory is stock LiveKit.

## Setup

This example is a standalone `uv` project. From this directory:

```bash
uv sync
cp .env.example .env.local
```

Fill in `.env.local`:

| Variable | Required | Notes |
|---|---|---|
| `LIVEKIT_URL` | ✅ | From [LiveKit Cloud](https://cloud.livekit.io/) |
| `LIVEKIT_API_KEY` | ✅ | |
| `LIVEKIT_API_SECRET` | ✅ | |
| `TUNER_API_KEY` | ✅ | Bearer token, starts with `tr_api_` |
| `TUNER_WORKSPACE_ID` | ✅ | Integer workspace ID |
| `TUNER_AGENT_ID` | ✅ | Agent identifier from Tuner Agent Settings |
| `TUNER_BASE_URL` | — | Defaults to `https://api.usetuner.ai` |
| `AGENT_VERSION` | — | Attached to every call record |

You can pull the LiveKit values automatically with the
[LiveKit CLI](https://docs.livekit.io/home/cli/cli-setup):

```bash
lk cloud auth
lk app env -w -d .env.local
```

`[tool.uv.sources]` in `pyproject.toml` points `tuner-livekit-sdk` at the repo checkout
two directories up, so local SDK edits take effect immediately. Delete that section to
install the published package from PyPI instead.

## Run

Download the Silero VAD and turn-detector models once:

```bash
uv run python src/agent.py download-files
```

Then talk to the agent in your terminal:

```bash
uv run python src/agent.py console
```

Or connect it to a room for a frontend or telephony:

```bash
uv run python src/agent.py dev     # development
uv run python src/agent.py start   # production
```

When the session ends, the plugin POSTs the call — transcript, tool calls, per-turn
timings, token usage and cost — to Tuner. Set `TUNER_BASE_URL` to a local endpoint if
you want to inspect the payload without sending it upstream.

## Tests

The evals under [`tests/`](tests/) use the LiveKit
[testing & evaluation framework](https://docs.livekit.io/agents/build/testing/). They
call a real LLM through LiveKit Inference, so the `LIVEKIT_*` credentials must be set:

```bash
uv run pytest
```

## Deploy

The included `Dockerfile` is production-ready. See
[deploying to production](https://docs.livekit.io/agents/ops/deployment/).

## What Tuner captures

| Payload field | Source |
|---|---|
| `transcript_with_tool_calls` | `session.history.items`, mapped to Tuner segments |
| `transcript` | Plain diarized text |
| per-segment `metadata` | LiveKit `MetricsReport` (STT/LLM/TTS latency, EOU delay, e2e latency) |
| `usage_token` | `UsageCollector` totals |
| `call_cost` | Your `cost_calculator` |
| `call_type` | `phone_call` when a SIP participant is present, else `web_call` |
| `disconnection_reason` | Participant disconnect reason / session close error |

See the [SDK README](../../README.md) for the full option list, including
`recording_url_resolver`, SIP simulation correlation, and LangGraph support.
