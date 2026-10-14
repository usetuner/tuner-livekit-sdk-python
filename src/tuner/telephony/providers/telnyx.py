"""Telnyx Call Control: webhook events carry the INVITE's X-headers in
``data.payload.custom_headers``, and ``transfer`` sends ``custom_headers`` on the new
INVITE. Use ``leg.custom_headers`` in the transfer; ``leg.sip_uri`` is unchanged."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..correlation import CORRELATION_HEADER
from ..leg import LiveKitLeg

_NAME = CORRELATION_HEADER.lower()


class TelnyxProvider:
    name = "telnyx"

    def read_correlation_id(self, event: Any) -> str | None:
        """``event``: the webhook JSON (the full envelope, or its ``data.payload``)."""
        payload = _payload(event)
        headers = payload.get("custom_headers")
        if not isinstance(headers, list):
            return None
        for header in headers:
            if not isinstance(header, Mapping):
                continue
            name, value = header.get("name"), header.get("value")
            if isinstance(name, str) and name.lower() == _NAME and value:
                return str(value)
        return None

    def build_leg(self, livekit_uri: str, correlation_id: str | None) -> LiveKitLeg:
        return LiveKitLeg(sip_uri=livekit_uri, correlation_id=correlation_id)


def _payload(event: Any) -> Mapping[str, Any]:
    if not isinstance(event, Mapping):
        return {}
    data = event.get("data")
    if isinstance(data, Mapping) and isinstance(data.get("payload"), Mapping):
        return data["payload"]
    return event
