from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

from .client import submit_call
from .collector import SessionState
from .config import TunerConfig
from .mapper import to_create_call_request

if TYPE_CHECKING:
    from livekit.agents import AgentSession, JobContext
    from livekit.agents.metrics import UsageSummary

logger = logging.getLogger("livekit_agents_tuner")

_RECORDING_URL_PLACEHOLDER = "pending"


async def _default_recording_url_resolver(room_name: str, job_id: str) -> str:
    """
    Default resolver used when no recording_url_resolver is provided.

    Returns a placeholder value so the Tuner API call succeeds.
    Replace this by passing recording_url_resolver= to TunerPlugin with logic
    that returns the real publicly accessible recording URL for your setup.

    Example (Egress → S3):
        async def my_resolver(room_name: str, job_id: str) -> str:
            return f"https://my-bucket.s3.amazonaws.com/recordings/{job_id}.ogg"

        TunerPlugin(session, ctx, recording_url_resolver=my_resolver)
    """
    logger.warning(
        "No recording_url_resolver provided; submitting recording_url='%s'. "
        "Pass recording_url_resolver= to TunerPlugin to supply the real URL.",
        _RECORDING_URL_PLACEHOLDER,
    )
    return _RECORDING_URL_PLACEHOLDER


class TunerPlugin:
    """
    Automatically ingests LiveKit agent session data into the Tuner observability API.

    Usage (2 lines):
        from livekit_agents_tuner import TunerPlugin

        # After creating AgentSession, before session.start():
        TunerPlugin(session, ctx)

    Configuration via env vars:
        TUNER_API_KEY        Bearer token (tr_api_ prefixed)
        TUNER_WORKSPACE_ID   Integer workspace ID
        TUNER_AGENT_ID       Agent remote identifier from Tuner Agent Settings

    Optional env vars:
        TUNER_BASE_URL       API base URL (default: https://api.usetuner.ai)

    Advanced usage:
        TunerPlugin(
            session, ctx,
            api_key="tr_api_...",
            workspace_id=123,
            agent_id="my-agent",
            call_type="phone_call",          # override auto-detection
            recording_url_resolver=my_fn,    # async (room, job_id) -> str | None
            cost_calculator=my_cost_fn,      # (UsageSummary) -> float (cost in dollars)
            extra_metadata={"env": "prod"},
            max_retries=3,
            timeout_seconds=30,
        )
    """

    def __init__(
        self,
        session: "AgentSession",
        ctx: "JobContext",
        *,
        api_key: str | None = None,
        workspace_id: int | None = None,
        agent_id: str | None = None,
        base_url: str | None = None,
        call_type: str | None = None,
        recording_url_resolver: Callable | None = None,
        cost_calculator: Callable[[UsageSummary], float] | None = None,
        extra_metadata: dict | None = None,
        enabled: bool = True,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._session = session
        self._ctx = ctx
        self._state = SessionState()
        self._config: TunerConfig | None = None

        if not enabled:
            logger.debug("TunerPlugin is disabled; no data will be sent to Tuner")
            return

        try:
            self._config = TunerConfig.from_env(
                api_key=api_key,
                workspace_id=workspace_id,
                agent_id=agent_id,
                base_url=base_url,
                call_type=call_type,
                recording_url_resolver=recording_url_resolver,
                cost_calculator=cost_calculator,
                extra_metadata=extra_metadata,
                enabled=enabled,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except ValueError as exc:
            logger.error(
                "TunerPlugin is misconfigured and will not send data: %s", exc
            )
            return

        self._setup_event_listeners()
        self._register_shutdown_hook()
        logger.debug(
            "TunerPlugin initialized for workspace=%d agent=%s",
            self._config.workspace_id,
            self._config.agent_id,
        )

    def _setup_event_listeners(self) -> None:
        from livekit import rtc

        state = self._state

        @self._session.on("metrics_collected")
        def _on_metrics(ev) -> None:
            state.record_metrics(ev.metrics)

        @self._session.on("close")
        def _on_close(ev) -> None:
            state.record_close(ev.error)

        @self._ctx.room.on("participant_connected")
        def _on_participant(participant) -> None:
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                phone = (
                    participant.attributes.get("sip.phoneNumber")
                    or participant.attributes.get("phoneNumber")
                    or participant.attributes.get("phone_number")
                )
                state.record_sip_participant(phone)

    def _register_shutdown_hook(self) -> None:
        self._ctx.add_shutdown_callback(self._on_shutdown)

    async def _on_shutdown(self, reason: str) -> None:
        if self._config is None:
            return

        self._state.finalize(reason)

        # Resolve recording URL — always required by Tuner.
        # Falls back to _default_recording_url_resolver which returns a placeholder
        # if the developer has not provided their own resolver.
        resolver = self._config.recording_url_resolver or _default_recording_url_resolver
        try:
            recording_url = await resolver(self._ctx.room.name, str(self._ctx.job.id))
        except Exception:
            logger.warning(
                "recording_url_resolver raised an error; falling back to placeholder",
                exc_info=True,
            )
            recording_url = _RECORDING_URL_PLACEHOLDER

        if not recording_url:
            recording_url = _RECORDING_URL_PLACEHOLDER

        # Snapshot conversation history at shutdown
        history_items = list(self._session.history.items)

        # Build Tuner payload
        try:
            payload = to_create_call_request(
                self._session,
                self._state,
                history_items,
                self._config,
                self._ctx,
            )
        except Exception:
            logger.exception("Failed to build Tuner payload; skipping submission")
            return

        payload["recording_url"] = recording_url

        # Submit with timeout guard.
        # Budget = per-request timeout × attempts + cumulative backoff delays.
        # Backoff delays in client.py: 2^0 + 2^1 + ... + 2^(n-1) seconds, plus up to
        # 0.5 s jitter per retry, where n = max_retries.
        _max_jitter_s = 0.5
        backoff_budget = (
            sum(2.0**i for i in range(self._config.max_retries))
            + _max_jitter_s * self._config.max_retries
        )
        total_timeout = (
            self._config.timeout_seconds * (self._config.max_retries + 1) + backoff_budget
        )
        try:
            await asyncio.wait_for(
                submit_call(payload, self._config),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Tuner submission timed out after %.1fs for call_id=%s",
                total_timeout,
                payload.get("call_id"),
            )
        except Exception:
            logger.exception("Tuner submission raised an unexpected error")
