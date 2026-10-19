"""Twilio: the webhook gets SIP X-headers as ``SipHeader_<Name>`` form params, and
``<Dial><Sip>`` sends URI headers (``sip:...?X-Name=value``) as SIP headers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..correlation import CORRELATION_HEADER
from ..info import Forward, SipInfo
from ..sip_uri import with_header

_PREFIX = "sipheader_"


class TwilioProvider:
    name = "twilio"

    def extract(self, payload: Any) -> SipInfo:
        """``payload``: the webhook's form params (any mapping, e.g. ``await request.form()``)."""
        if not isinstance(payload, Mapping):
            return SipInfo()
        form = {str(k): str(v) for k, v in payload.items() if isinstance(k, str) and v}
        headers: dict[str, str] = {}
        for key, value in form.items():
            if key.lower().startswith(_PREFIX) and len(key) > len(_PREFIX):
                headers[key] = value
                headers.setdefault(key[len(_PREFIX) :], value)
        carrier = form.get("CallSid")
        outbound = form.get("Direction", "").startswith("outbound")
        return SipInfo(
            sip_call_id=form.get("SipCallId") or carrier,
            sip_headers=headers,
            carrier_call_id=carrier,
            recipient=form.get("To") if outbound else None,
        )

    def forward_to_sip(self, info: SipInfo, sip_uri: str) -> Forward:
        token = info.correlation_id or None
        uri = with_header(sip_uri, CORRELATION_HEADER, token) if token else sip_uri
        return Forward(sip_uri=uri, correlation_id=token)
