"""Provider name -> provider. Built-ins are registered here; customers can add their own."""

from __future__ import annotations

from .base import CarrierProvider
from .twilio import TwilioProvider

_PROVIDERS: dict[str, CarrierProvider] = {}


def register_provider(provider: CarrierProvider) -> None:
    """Make ``provider`` available as ``TunerTelephony(provider=provider.name)``."""
    _PROVIDERS[provider.name.lower()] = provider


def get_provider(name: str) -> CarrierProvider:
    try:
        return _PROVIDERS[name.lower()]
    except KeyError:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider {name!r}. Available: {available}") from None


register_provider(TwilioProvider())
