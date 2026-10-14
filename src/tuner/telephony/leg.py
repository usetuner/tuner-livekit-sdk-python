"""What the customer dials on the leg into LiveKit."""

from __future__ import annotations

from dataclasses import dataclass

from .correlation import CORRELATION_HEADER


@dataclass(frozen=True)
class LiveKitLeg:
    """The LiveKit leg to dial, carrying Tuner's correlation token when there is one.

    Use the field your carrier takes: ``sip_uri`` (e.g. Twilio ``<Dial><Sip>``) or
    ``custom_headers`` (e.g. Telnyx Call Control ``transfer``).
    """

    sip_uri: str
    correlation_id: str | None = None

    @property
    def headers(self) -> dict[str, str]:
        """``{"X-Correlation-Id": token}``, or ``{}`` when there is no token."""
        return {CORRELATION_HEADER: self.correlation_id} if self.correlation_id else {}

    @property
    def custom_headers(self) -> list[dict[str, str]]:
        """The headers as a ``[{"name", "value"}]`` list, the shape carrier APIs take."""
        return [{"name": name, "value": value} for name, value in self.headers.items()]
