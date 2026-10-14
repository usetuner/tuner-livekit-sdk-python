"""The contract every carrier provider implements."""

from __future__ import annotations

from typing import Any, Protocol

from ..leg import LiveKitLeg


class CarrierProvider(Protocol):
    """Knows where a carrier's webhook carries custom SIP headers, and how its
    dial/transfer to LiveKit accepts them."""

    name: str

    def read_correlation_id(self, event: Any) -> str | None:
        """Return the raw ``X-Correlation-Id`` from the carrier's webhook event, if any."""

    def build_leg(self, livekit_uri: str, correlation_id: str | None) -> LiveKitLeg:
        """Return the LiveKit leg carrying ``correlation_id`` the way this carrier needs."""
