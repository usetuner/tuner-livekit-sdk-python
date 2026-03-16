from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from livekit.agents.metrics import UsageSummary


_DEFAULT_BASE_URL = "https://api.usetuner.ai"


@dataclass
class TunerConfig:
    """Configuration for the Tuner integration plugin."""

    api_key: str
    workspace_id: int
    agent_id: str
    base_url: str = _DEFAULT_BASE_URL
    call_type: str | None = None  # None = auto-detect from participant kind
    recording_url_resolver: Callable[[str, str], Awaitable[str | None]] | None = None
    cost_calculator: Callable[[UsageSummary], float] | None = None
    extra_metadata: dict | None = None
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3

    @classmethod
    def from_env(
        cls,
        api_key: str | None = None,
        workspace_id: int | None = None,
        agent_id: str | None = None,
        base_url: str | None = None,
        call_type: str | None = None,
        recording_url_resolver: Callable[[str, str], Awaitable[str | None]] | None = None,
        cost_calculator: Callable[[UsageSummary], float] | None = None,
        extra_metadata: dict | None = None,
        enabled: bool = True,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> "TunerConfig":
        """
        Build config from keyword arguments, falling back to environment variables.

        Required env vars (when not passed explicitly):
            TUNER_API_KEY        - Bearer token (tr_api_ prefixed)
            TUNER_WORKSPACE_ID   - Integer workspace ID
            TUNER_AGENT_ID       - Agent identifier from Agent Settings

        Optional env vars:
            TUNER_BASE_URL       - API base URL (default: https://api.usetuner.ai)
        """
        resolved_api_key = api_key or os.environ.get("TUNER_API_KEY", "")
        if not resolved_api_key:
            raise ValueError(
                "TUNER_API_KEY is required. "
                "Set the TUNER_API_KEY env var or pass api_key= to TunerPlugin."
            )

        if workspace_id is None:
            ws_str = os.environ.get("TUNER_WORKSPACE_ID", "")
            if not ws_str:
                raise ValueError(
                    "TUNER_WORKSPACE_ID is required. "
                    "Set the TUNER_WORKSPACE_ID env var or pass workspace_id= to TunerPlugin."
                )
            try:
                workspace_id = int(ws_str)
            except ValueError:
                raise ValueError(
                    f"TUNER_WORKSPACE_ID must be an integer, got '{ws_str}'. "
                    "Set the TUNER_WORKSPACE_ID env var or pass workspace_id= to TunerPlugin."
                ) from None

        resolved_agent_id = agent_id or os.environ.get("TUNER_AGENT_ID", "")
        if not resolved_agent_id:
            raise ValueError(
                "TUNER_AGENT_ID is required. "
                "Set the TUNER_AGENT_ID env var or pass agent_id= to TunerPlugin."
            )

        resolved_base_url = base_url or os.environ.get("TUNER_BASE_URL", _DEFAULT_BASE_URL)

        return cls(
            api_key=resolved_api_key,
            workspace_id=workspace_id,
            agent_id=resolved_agent_id,
            base_url=resolved_base_url,
            call_type=call_type,
            recording_url_resolver=recording_url_resolver,
            cost_calculator=cost_calculator,
            extra_metadata=extra_metadata,
            enabled=enabled,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
