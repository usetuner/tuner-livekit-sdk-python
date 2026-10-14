"""Twilio: the webhook gets SIP X-headers as ``SipHeader_<Name>`` form params, and
``<Dial><Sip>`` sends URI headers (``sip:...?X-Name=value``) as SIP headers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..correlation import CORRELATION_HEADER
from ..leg import LiveKitLeg
from ..sip_uri import with_header

_PARAM = f"sipheader_{CORRELATION_HEADER}".lower()


class TwilioProvider:
    name = "twilio"

    def read_correlation_id(self, event: Any) -> str | None:
        """``event``: the webhook's form params (any mapping, e.g. ``await request.form()``)."""
        if not isinstance(event, Mapping):
            return None
        for key, value in event.items():
            if isinstance(key, str) and key.lower() == _PARAM and value:
                return str(value)
        return None

    def build_leg(self, livekit_uri: str, correlation_id: str | None) -> LiveKitLeg:
        sip_uri = (
            with_header(livekit_uri, CORRELATION_HEADER, correlation_id)
            if correlation_id
            else livekit_uri
        )
        return LiveKitLeg(sip_uri=sip_uri, correlation_id=correlation_id)
