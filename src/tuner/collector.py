from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from livekit.agents.metrics import EOUMetrics, UsageCollector, UsageSummary


class DisconnectReason(str, Enum):
    """Standard disconnection reasons detected by the SDK."""

    USER_HANGUP = "user_hangup"
    AGENT_HANGUP = "agent_hangup"
    ERROR = "error"


@dataclass
class SessionState:
    """Accumulates LiveKit session data for Tuner ingestion."""

    start_timestamp: float = field(default_factory=time.time)
    end_timestamp: float | None = None
    is_sip: bool = False
    caller_phone_number: str | None = None
    sip_call_id: str | None = None
    close_error: Optional[Exception] = None
    eou_metrics: list[dict] = field(default_factory=list, repr=False)
    _shutdown_reason: str = field(default="", init=False, repr=False)
    _usage_collector: UsageCollector = field(default_factory=UsageCollector, repr=False)

    def record_metrics(self, metrics: Any) -> None:
        """Feed an AgentMetrics event into the usage collector.

        EOUMetrics (one per committed user turn) are kept separately —
        UsageCollector ignores them, but they carry the end-of-utterance
        delay used for per-call EOU latency rollups.
        """
        if isinstance(metrics, EOUMetrics):
            self.eou_metrics.append(
                {
                    "timestamp": metrics.timestamp,
                    "eou_delay_ms": int(metrics.end_of_utterance_delay * 1000),
                    "transcription_delay_ms": int(metrics.transcription_delay * 1000),
                    "speech_id": metrics.speech_id,
                }
            )
            return
        self._usage_collector.collect(metrics)

    def record_close(self, error: Optional[Exception]) -> None:
        """Record the session close error (None if clean close)."""
        self.close_error = error

    def record_sip_participant(
        self, phone_number: str | None, sip_call_id: str | None = None
    ) -> None:
        """Mark session as SIP (phone_call) and record SIP participant identifiers."""
        self.is_sip = True
        self.caller_phone_number = phone_number
        self.sip_call_id = sip_call_id

    @property
    def shutdown_reason(self) -> str:
        return self._shutdown_reason

    def set_shutdown_reason(self, reason: str) -> None:
        """Write-once: first meaningful value wins."""
        if not self._shutdown_reason and reason:
            self._shutdown_reason = reason

    def finalize(self, reason: str) -> None:
        """Seal the state at shutdown time."""
        self.end_timestamp = time.time()
        self.set_shutdown_reason(reason)

    def get_usage_summary(self) -> UsageSummary:
        return self._usage_collector.get_summary()

    @property
    def duration_ms(self) -> int:
        if self.end_timestamp is None:
            return 0
        return max(0, int((self.end_timestamp - self.start_timestamp) * 1000))

    @property
    def call_status(self) -> str:
        return "error" if self.close_error is not None else "completed"
