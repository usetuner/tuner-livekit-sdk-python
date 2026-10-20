"""Carry Tuner's simulation correlation header from your carrier's webhook to LiveKit.

The agent side needs nothing from here: ``TunerPlugin`` reads the header from the SIP
participant. In the webhook where you bridge calls to LiveKit::

    info = extract_sip(form, provider="twilio")
    fwd = forward_to_sip(info, sip_uri="sip:agent@xxxx.sip.livekit.cloud")
    # Twilio: return fwd.twiml. Telnyx: POST fwd.transfer to actions/transfer.
"""

from .api import extract_sip, forward_to_sip
from .info import Forward, SipInfo
from .providers.base import CarrierProvider
from .providers.registry import register_provider

__all__ = [
    "CarrierProvider",
    "Forward",
    "SipInfo",
    "extract_sip",
    "forward_to_sip",
    "register_provider",
]
