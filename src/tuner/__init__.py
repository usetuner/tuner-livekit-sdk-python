"""
livekit-agents-tuner
====================
Automatically ingest LiveKit Agents session data into the Tuner observability API.

Quickstart (2 lines):
    from tuner import TunerPlugin

    TunerPlugin(session, ctx)  # after creating AgentSession

Set env vars:
    TUNER_API_KEY, TUNER_WORKSPACE_ID, TUNER_AGENT_ID
"""

from typing import TYPE_CHECKING, Any

from .config import TunerConfig

if TYPE_CHECKING:
    from .plugin import TunerPlugin

__version__ = "0.1.0"

__all__ = ["TunerPlugin", "TunerConfig", "__version__"]


def __getattr__(name: str) -> Any:
    # Lazy so ``tuner.telephony`` (webhook servers) does not load livekit.agents.
    if name == "TunerPlugin":
        from .plugin import TunerPlugin

        return TunerPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
