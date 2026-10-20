"""Telnyx Call Control: webhook events carry the INVITE's X-headers in
``data.payload.custom_headers``, and ``transfer`` sends ``custom_headers`` on the new
INVITE. Use ``fwd.transfer`` (or ``fwd.custom_headers``); ``fwd.sip_uri`` is unchanged."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..info import Forward, SipInfo


class TelnyxProvider:
    name = "telnyx"

    def extract(self, payload: Any) -> SipInfo:
        """``payload``: the webhook JSON (the full envelope, or its ``data.payload``)."""
        inner = _payload(payload)
        headers: dict[str, str] = {}
        for key in ("custom_headers", "sip_headers"):
            items = inner.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                name, value = item.get("name"), item.get("value")
                if isinstance(name, str) and value and name.lower() not in {
                    k.lower() for k in headers
                }:
                    headers[name] = str(value)
        carrier = inner.get("call_control_id")
        carrier_id = str(carrier) if carrier else None
        sip_call_id = inner.get("sip_call_id")
        recipient = inner.get("to") if inner.get("direction") == "outgoing" else None
        return SipInfo(
            sip_call_id=str(sip_call_id) if sip_call_id else carrier_id,
            sip_headers=headers,
            carrier_call_id=carrier_id,
            recipient=str(recipient) if recipient else None,
        )

    def forward_to_sip(self, info: SipInfo, sip_uri: str) -> Forward:
        return Forward(sip_uri=sip_uri, correlation_id=info.correlation_id or None)


def _payload(event: Any) -> Mapping[str, Any]:
    if not isinstance(event, Mapping):
        return {}
    data = event.get("data")
    if isinstance(data, Mapping) and isinstance(data.get("payload"), Mapping):
        return data["payload"]
    return event
