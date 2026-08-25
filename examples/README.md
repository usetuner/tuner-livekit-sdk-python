# Examples

Runnable projects showing `tuner-livekit-sdk` wired into a real LiveKit agent.

| Example | What it shows |
|---|---|
| [`voice-agent/`](voice-agent/) | Restaurant receptionist on the STT → LLM → TTS pipeline, with function tools, evals, and a production `Dockerfile`. Start here. |

Each example is a standalone `uv` project that installs the SDK from this checkout, so
edits to `src/tuner/` are picked up without a reinstall.
