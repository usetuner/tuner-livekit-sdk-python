"""``SipInfo`` (what a carrier payload carries) and ``Forward`` (the leg into LiveKit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

from .correlation import CORRELATION_HEADER, valid_correlation_id

if TYPE_CHECKING:
    from .providers.base import CarrierProvider


@dataclass(frozen=True)
class SipInfo:
    """SIP data read from one carrier payload."""

    sip_call_id: str | None = None
    sip_headers: dict[str, str] = field(default_factory=dict)
    # Carrier-native ID (CallSid, call_control_id); not a SIP Call-ID.
    carrier_call_id: str | None = None
    provider: str | None = field(default=None, compare=False)
    # Outbound only: the number or SIP URI the call dialed.
    recipient: str | None = None
    # The provider object extract_sip used, so an unregistered one still forwards.
    provider_impl: CarrierProvider | None = field(default=None, repr=False, compare=False)

    @property
    def is_fallback(self) -> bool:
        """True when ``sip_call_id`` is only the carrier ID."""
        return bool(self.sip_call_id) and self.sip_call_id == self.carrier_call_id

    @property
    def raw_correlation_id(self) -> str | None:
        """The ``X-Correlation-Id`` as received, unvalidated. Never log it."""
        name = CORRELATION_HEADER.lower()
        for key, value in self.sip_headers.items():
            if key.lower() == name and value:
                return value
        return None

    @property
    def correlation_id(self) -> str:
        """The validated ``X-Correlation-Id``, or ``""``."""
        return valid_correlation_id(self.raw_correlation_id)


@dataclass(frozen=True)
class Forward:
    """The leg to dial into LiveKit, carrying Tuner's correlation token when there is one.

    Use what your carrier takes: ``twiml`` / ``sip_uri`` (Twilio ``<Dial><Sip>``) or
    ``transfer`` / ``custom_headers`` (Telnyx Call Control ``transfer``).
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

    @property
    def twiml(self) -> str:
        """``<Response><Dial><Sip>sip_uri</Sip></Dial></Response>``."""
        return f"<Response><Dial><Sip>{escape(self.sip_uri)}</Sip></Dial></Response>"

    @property
    def transfer(self) -> dict[str, Any]:
        """The Telnyx ``transfer`` body: ``{"to", "custom_headers"}``."""
        body: dict[str, Any] = {"to": self.sip_uri}
        if self.custom_headers:
            body["custom_headers"] = self.custom_headers
        return body
