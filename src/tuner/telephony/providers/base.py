"""The contract every carrier provider implements."""

from __future__ import annotations

from typing import Any, Protocol

from ..info import Forward, SipInfo


class CarrierProvider(Protocol):
    """Knows where a carrier's webhook carries SIP data, and how its dial/transfer to
    LiveKit carries Tuner's correlation header."""

    name: str

    def extract(self, payload: Any) -> SipInfo:
        """Read SIP data from the carrier's webhook payload."""

    def forward_to_sip(self, info: SipInfo, sip_uri: str) -> Forward:
        """The leg to ``sip_uri`` carrying ``info.correlation_id`` the way this carrier needs."""
