"""``extract_sip`` and ``forward_to_sip``: look up the provider and delegate."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from .info import Forward, SipInfo
from .providers.base import CarrierProvider
from .providers.registry import get_provider

logger = logging.getLogger("tuner.telephony")


def extract_sip(payload: Any, *, provider: str | CarrierProvider) -> SipInfo:
    """Read SIP data from what the carrier sent your webhook."""
    impl = get_provider(provider) if isinstance(provider, str) else provider
    info = impl.extract(payload)
    return dataclasses.replace(info, provider=info.provider or impl.name, provider_impl=impl)


def forward_to_sip(
    info: SipInfo, *, sip_uri: str, provider: str | CarrierProvider | None = None
) -> Forward:
    """The leg into LiveKit at ``sip_uri``, carrying Tuner's correlation token.

    Calls without a token (normal production calls) get ``sip_uri`` unchanged.
    """
    chosen = provider or info.provider_impl or info.provider
    if chosen is None:
        raise ValueError("pass provider= or a SipInfo from extract_sip")
    impl = get_provider(chosen) if isinstance(chosen, str) else chosen
    if info.raw_correlation_id and not info.correlation_id:
        logger.warning("dropped a malformed correlation header (%s)", impl.name)
    else:
        logger.debug(
            "correlation header %s (%s)",
            "forwarded" if info.correlation_id else "absent",
            impl.name,
        )
    return impl.forward_to_sip(info, sip_uri)
