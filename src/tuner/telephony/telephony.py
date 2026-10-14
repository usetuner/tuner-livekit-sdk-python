"""``TunerTelephony``: one call in the customer's own webhook, same for every carrier."""

from __future__ import annotations

import logging
from typing import Any

from .correlation import valid_correlation_id
from .leg import LiveKitLeg
from .providers.registry import get_provider

logger = logging.getLogger("tuner.telephony")


class TunerTelephony:
    """Carries Tuner's simulation correlation token onto the carrier's leg into LiveKit.

    Usage, inside the webhook where you already bridge the call to LiveKit::

        telephony = TunerTelephony(provider="twilio")
        leg = telephony.correlate(params, livekit_uri="sip:agent@xxxx.sip.livekit.cloud")
        # Twilio: dial leg.sip_uri in <Dial><Sip>

    Calls without a token (normal production calls) get the LiveKit URI unchanged.
    Your server keeps its routing, signature verification and call control.
    """

    def __init__(self, provider: str) -> None:
        self.provider = get_provider(provider)

    def correlate(self, event: Any, *, livekit_uri: str) -> LiveKitLeg:
        """Return the LiveKit leg for this call. ``event`` is what the carrier sent you."""
        raw = self.provider.read_correlation_id(event)
        correlation_id = valid_correlation_id(raw) or None
        if raw and not correlation_id:
            logger.warning("dropped a malformed correlation header (%s)", self.provider.name)
        else:
            logger.debug(
                "correlation header %s (%s)",
                "forwarded" if correlation_id else "absent",
                self.provider.name,
            )
        return self.provider.build_leg(livekit_uri, correlation_id)
