"""Every built-in provider must leave production calls (no token) untouched."""

import pytest

from tuner.telephony.leg import LiveKitLeg
from tuner.telephony.providers import registry

LIVEKIT = "sip:agent@xxxx.sip.livekit.cloud"


@pytest.mark.parametrize("name", sorted(registry._PROVIDERS))
def test_no_token_means_unchanged_leg(name):
    provider = registry.get_provider(name)
    assert provider.build_leg(LIVEKIT, None) == LiveKitLeg(sip_uri=LIVEKIT)


@pytest.mark.parametrize("name", sorted(registry._PROVIDERS))
def test_empty_event_has_no_token(name):
    assert registry.get_provider(name).read_correlation_id({}) is None


@pytest.mark.parametrize("name", sorted(registry._PROVIDERS))
def test_token_reaches_the_leg(name):
    leg = registry.get_provider(name).build_leg(LIVEKIT, "abc123")
    assert leg.correlation_id == "abc123"
    assert leg.headers == {"X-Correlation-Id": "abc123"}
