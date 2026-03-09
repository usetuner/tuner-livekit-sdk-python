"""
livekit-agents-tuner
====================
Automatically ingest LiveKit Agents session data into the Tuner observability API.

Quickstart (2 lines):
    from livekit_agents_tuner import TunerPlugin

    TunerPlugin(session, ctx)  # after creating AgentSession

Set env vars:
    TUNER_API_KEY, TUNER_WORKSPACE_ID, TUNER_AGENT_ID
"""

from .config import TunerConfig
from .plugin import TunerPlugin

__version__ = "0.1.0"

__all__ = ["TunerPlugin", "TunerConfig", "__version__"]
